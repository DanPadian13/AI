import os
import random
from collections import deque, namedtuple

import neurogym as ngym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from Question_2a import Net  # reuse the four recurrent architectures


# CRITICAL FIX 1: Store observation sequences, not single timesteps
Transition = namedtuple('Transition', ('obs_sequence', 'action', 'reward', 'next_obs_sequence', 'done'))


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


def select_action(policy_net, obs_sequence, hidden, epsilon, action_space, device):
    """
    CRITICAL FIX 1 & 2: Use observation sequence and maintain hidden state
    """
    if random.random() < epsilon:
        # Even during exploration, we need to update hidden state
        with torch.no_grad():
            _, _, hidden = policy_net(obs_sequence.to(device), hidden)
        return action_space.sample(), hidden
    with torch.no_grad():
        # obs_sequence shape: [seq_len, 1, obs_dim]
        q_vals, _, hidden = policy_net(obs_sequence.to(device), hidden)
        # Get Q-values from last timestep
        action = int(q_vals[-1, 0, :].argmax().item())
        return action, hidden


def pad_sequences(sequences, max_len):
    """
    Pad variable-length sequences to max_len by repeating the first observation.
    Args:
        sequences: list of tensors with shape [seq_len, 1, obs_dim]
        max_len: target sequence length
    Returns:
        padded tensor of shape [batch_size, max_len, obs_dim]
    """
    batch = []
    for seq in sequences:
        seq = seq.squeeze(1)  # [seq_len, obs_dim]
        seq_len = seq.shape[0]
        if seq_len < max_len:
            # Pad by repeating the first observation
            padding = seq[0:1].repeat(max_len - seq_len, 1)
            seq = torch.cat([padding, seq], dim=0)
        elif seq_len > max_len:
            # Take last max_len observations
            seq = seq[-max_len:]
        batch.append(seq)
    return torch.stack(batch)


def optimize_model(policy_net, target_net, buffer, optimizer, batch_size, gamma, device, model_type, max_seq_len=50):
    """
    CRITICAL FIX 1: Train on sequences, not single timesteps
    """
    if len(buffer) < batch_size:
        return None
    transitions = buffer.sample(batch_size)
    batch = Transition(*transitions)

    # Pad sequences to same length
    state_batch = pad_sequences(batch.obs_sequence, max_seq_len).to(device)  # [B, T, obs]
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_state_batch = pad_sequences(batch.next_obs_sequence, max_seq_len).to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    # Net expects [T, B, obs]
    state_batch = state_batch.permute(1, 0, 2)
    next_state_batch = next_state_batch.permute(1, 0, 2)

    # Forward through policy net: get Q at last timestep
    q_values, rnn_activity, _ = policy_net(state_batch)
    q_values = q_values[-1]  # [B, act_dim]
    q_sa = q_values.gather(1, action_batch)

    with torch.no_grad():
        q_next, _, _ = target_net(next_state_batch)
        q_next = q_next[-1]
        max_next_q = q_next.max(dim=1, keepdim=True)[0]
        target = reward_batch + gamma * max_next_q * (1 - done_batch)

    loss = nn.functional.smooth_l1_loss(q_sa, target)

    # Add L1/L2 regularization for bio-realistic model
    if model_type == 'bio_realistic':
        beta_L1 = 1e-5
        beta_L2 = 0.01
        l1_loss = beta_L1 * sum(torch.sum(torch.abs(p)) for p in policy_net.parameters())
        l2_loss = beta_L2 * torch.mean(rnn_activity ** 2)
        loss = loss + l1_loss + l2_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def plot_training_all(histories, params=None, output_path='images/question_2d_readysetgo_dqn_rewards.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    colors = {'vanilla': '#1f77b4', 'leaky': '#ff7f0e', 'leaky_fa': '#2ca02c', 'bio': '#d62728'}
    fig, ax = plt.subplots(figsize=(12, 7))
    for key, hist in histories.items():
        rewards = hist['rewards']
        smooth = np.convolve(rewards, np.ones(50) / 50, mode='same') if len(rewards) > 50 else rewards
        ax.plot(smooth, linewidth=2, label=f'{key} (50-ep smooth)', color=colors.get(key, None))
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Reward (50-ep smooth)', fontsize=13)
    ax.set_title('ReadySetGo-v0 DQN (FIXED): Reward (smoothed)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3)
    if params:
        text = (f"episodes={params.get('num_episodes')}, "
                f"max_steps={params.get('max_steps_per_episode')}, "
                f"lr={params.get('lr')}, gamma={params.get('gamma')}, "
                f"eps_start={params.get('epsilon_start')}, eps_end={params.get('epsilon_end')}, "
                f"eps_decay={params.get('epsilon_decay')}, "
                f"batch={params.get('batch_size')}, target_update_steps={params.get('target_update_steps')}")
        ax.text(0.01, -0.15, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', wrap=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training plot: {output_path}")


def train_dqn_models(task='ReadySetGo-v0', dt=20, num_episodes=2000, gamma=0.99,
                     epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=1000,
                     batch_size=32, target_update_steps=1000, lr=1e-3,
                     hidden_size=64, device='cpu', max_seq_len=20):
    """
    CRITICAL FIXES IMPLEMENTED:
    1. Use observation sequences (max_seq_len) instead of single timesteps
    2. Persist hidden states across episode steps
    3. Fix reward shaping - trust environment, reduce fixation penalty
    4. Update target network by steps, not episodes
    """
    env_kwargs = {'dt': dt}
    env = ngym.make(task, **env_kwargs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Instantiate four Q-networks via Net with different model types
    model_types = [('vanilla', 'vanilla'), ('leaky', 'leaky'),
                   ('leaky_fa', 'leaky_fa'), ('bio', 'bio_realistic')]
    policy_nets = {}
    target_nets = {}
    optimizers = {}
    buffers = {}
    histories = {}

    for key, mtype in model_types:
        net = Net(obs_dim, hidden_size, act_dim, model_type=mtype, dt=dt, tau=150, sigma_rec=0.1,
                  exc_ratio=0.8 if mtype == 'bio_realistic' else None, sparsity=0.2 if mtype == 'bio_realistic' else None).to(device)
        target = Net(obs_dim, hidden_size, act_dim, model_type=mtype, dt=dt, tau=150, sigma_rec=0.1,
                     exc_ratio=0.8 if mtype == 'bio_realistic' else None, sparsity=0.2 if mtype == 'bio_realistic' else None).to(device)
        target.load_state_dict(net.state_dict())
        target.eval()

        policy_nets[key] = net
        target_nets[key] = target
        optimizers[key] = optim.Adam(net.parameters(), lr=lr)
        buffers[key] = ReplayBuffer(capacity=100000)
        histories[key] = {'rewards': [], 'losses': [], 'eps': []}

    epsilon = epsilon_start
    # CRITICAL FIX: Reduce max steps - intervals are 500-2000ms with gain 1.5 = max ~3000ms = 150 steps
    max_steps_per_episode = 150

    # CRITICAL FIX 4: Track global steps for target network updates
    global_steps = {k: 0 for k in policy_nets.keys()}

    go_idx = env.action_space.n - 1 if hasattr(env, 'action_space') else 1

    print(f"Starting training for {num_episodes} episodes...")
    print(f"max_steps_per_episode = {max_steps_per_episode}")
    print(f"max_seq_len = {max_seq_len} (observation history window)")
    print(f"target_update_steps = {target_update_steps}")
    print(f"CRITICAL FIXES APPLIED:")
    print(f"  1. Using observation sequences (not single timesteps)")
    print(f"  2. Persisting hidden states across episode")
    print(f"  3. Fixed reward shaping (trust env, remove fixation penalty)")
    print(f"  4. Target network updates by steps (not episodes)")
    print()

    for ep in range(1, num_episodes + 1):
        ep_rewards = {k: 0.0 for k in policy_nets.keys()}

        # Debug: Show progress for first episode
        if ep == 1:
            print("Running first episode (this may take a moment)...")
            print("  Training 4 models sequentially: vanilla, leaky, leaky_fa, bio")
            print()

        # Step each model independently for this episode (separate trajectories)
        for model_idx, (key, mtype) in enumerate(model_types):
            if ep == 1:
                print(f"  [{model_idx+1}/4] Running {key}...", end='', flush=True)
            obs, _ = env.reset()
            done = False
            steps = 0
            has_pressed_go = False

            # CRITICAL FIX 1 & 2: Maintain observation history and hidden state
            obs_history = deque(maxlen=max_seq_len)
            obs_history.append(obs)
            hidden = None  # RNN hidden state - persisted across episode

            while not done and steps < max_steps_per_episode:
                # CRITICAL FIX 1: Create observation sequence from history
                # Shape: [seq_len, 1, obs_dim]
                obs_sequence = torch.from_numpy(np.array(list(obs_history))).float().unsqueeze(1)

                # CRITICAL FIX 2: Pass and receive hidden state
                action, hidden = select_action(policy_nets[key], obs_sequence, hidden, epsilon, env.action_space, device)

                next_obs, reward, done, _, info = env.step(action)

                # CRITICAL FIX 3: Improved reward shaping
                target_go_ms = env.trial.get('production', None)
                if target_go_ms is not None:
                    target_go = int(target_go_ms / env.dt)
                    if action == go_idx:
                        if has_pressed_go:
                            # Penalize multiple go presses
                            shaped_reward = -0.5
                        else:
                            has_pressed_go = True
                            # Use environment's reward structure, add small shaping bonus
                            shaped_reward = reward
                            # Small curriculum bonus for early episodes only
                            if ep < 500:
                                timing_error = abs(steps - target_go)
                                if timing_error <= 20:  # Within 400ms
                                    shaped_reward += 0.5 * (1.0 - timing_error / 20.0)
                    else:
                        # CRITICAL FIX 3: Trust fixation action - NO penalty
                        # The agent SHOULD fixate until the right time
                        shaped_reward = reward
                else:
                    shaped_reward = reward

                # Update observation history
                obs_history.append(next_obs)

                # Create next_obs_sequence for replay buffer
                next_obs_sequence = torch.from_numpy(np.array(list(obs_history))).float().unsqueeze(1)

                # Store transition with sequences
                buffers[key].push(obs_sequence, action, shaped_reward, next_obs_sequence, done)
                ep_rewards[key] += shaped_reward

                # Optimize
                loss = optimize_model(policy_nets[key], target_nets[key], buffers[key],
                                      optimizers[key], batch_size, gamma, device, mtype, max_seq_len)
                if loss is not None:
                    histories[key]['losses'].append(loss)

                # CRITICAL FIX 4: Update target network by steps, not episodes
                global_steps[key] += 1
                if global_steps[key] % target_update_steps == 0:
                    target_nets[key].load_state_dict(policy_nets[key].state_dict())

                obs = next_obs
                steps += 1

            if ep == 1:
                print(f" done ({steps} steps, reward={ep_rewards[key]:.2f})")

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)
        for key in histories.keys():
            histories[key]['rewards'].append(ep_rewards[key])
            histories[key]['eps'].append(epsilon)

        if ep == 1:
            print()
            print("Episode 1 complete! Continuing training...")
            print()

        if ep == 1 or ep % 50 == 0:
            msg = " | ".join([f"{k}: R(avg100)={np.mean(histories[k]['rewards'][-100:]):.3f}" for k in histories])
            print(f"Ep {ep:4d} | {msg} | ε={epsilon:.3f}")

    # Save combined reward plot
    plot_training_all(histories, params={
        'num_episodes': num_episodes,
        'max_steps_per_episode': max_steps_per_episode,
        'lr': lr,
        'gamma': gamma,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay': epsilon_decay,
        'batch_size': batch_size,
        'target_update_steps': target_update_steps
    }, output_path='images/question_2d_readysetgo_dqn_fixed_rewards.png')

    # Save checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'policy_states': {k: net.state_dict() for k, net in policy_nets.items()},
        'target_states': {k: net.state_dict() for k, net in target_nets.items()},
        'histories': histories,
        'env_kwargs': env_kwargs,
        'task': task,
    }, 'checkpoints/question_2d_readysetgo_dqn_fixed.pt')
    print("Saved checkpoint: checkpoints/question_2d_readysetgo_dqn_fixed.pt")


if __name__ == '__main__':
    # MPS is slower for small RNNs - stick with CPU
    use_mps = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    train_dqn_models(device=device)
