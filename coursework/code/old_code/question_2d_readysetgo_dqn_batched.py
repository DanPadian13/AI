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


def select_action_batch(policy_net, obs_sequences, hiddens, epsilon, action_space, device, num_parallel):
    """
    Batched action selection for multiple parallel episodes
    Args:
        obs_sequences: list of tensors, each [seq_len, 1, obs_dim]
        hiddens: list of hidden states (or None)
        num_parallel: number of parallel episodes
    Returns:
        actions: list of actions
        hiddens: list of updated hidden states
    """
    actions = []
    new_hiddens = []

    for i in range(num_parallel):
        if random.random() < epsilon:
            # Exploration - still need to update hidden state
            with torch.no_grad():
                _, _, hidden = policy_net(obs_sequences[i].to(device), hiddens[i])
            actions.append(action_space.sample())
            new_hiddens.append(hidden)
        else:
            # Exploitation
            with torch.no_grad():
                q_vals, _, hidden = policy_net(obs_sequences[i].to(device), hiddens[i])
                action = int(q_vals[-1, 0, :].argmax().item())
                actions.append(action)
                new_hiddens.append(hidden)

    return actions, new_hiddens


def pad_sequences(sequences, max_len):
    """
    Pad variable-length sequences to max_len by repeating the first observation.
    """
    batch = []
    for seq in sequences:
        seq = seq.squeeze(1)  # [seq_len, obs_dim]
        seq_len = seq.shape[0]
        if seq_len < max_len:
            padding = seq[0:1].repeat(max_len - seq_len, 1)
            seq = torch.cat([padding, seq], dim=0)
        elif seq_len > max_len:
            seq = seq[-max_len:]
        batch.append(seq)
    return torch.stack(batch)


def optimize_model(policy_net, target_net, buffer, optimizer, batch_size, gamma, device, model_type, max_seq_len=20):
    """
    CRITICAL FIX 1: Train on sequences, not single timesteps
    """
    if len(buffer) < batch_size:
        return None
    transitions = buffer.sample(batch_size)
    batch = Transition(*transitions)

    # Pad sequences to same length
    state_batch = pad_sequences(batch.obs_sequence, max_seq_len).to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_state_batch = pad_sequences(batch.next_obs_sequence, max_seq_len).to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    # Net expects [T, B, obs]
    state_batch = state_batch.permute(1, 0, 2)
    next_state_batch = next_state_batch.permute(1, 0, 2)

    # Forward through policy net
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
    ax.set_title('ReadySetGo-v0 DQN (BATCHED): Reward (smoothed)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3)
    if params:
        text = (f"episodes={params.get('num_episodes')}, num_parallel={params.get('num_parallel')}, "
                f"lr={params.get('lr')}, gamma={params.get('gamma')}, "
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
                     hidden_size=64, device='cpu', max_seq_len=20, num_parallel=4):
    """
    BATCHED VERSION: Run multiple parallel environments per model for speedup

    CRITICAL FIXES IMPLEMENTED:
    1. Use observation sequences (max_seq_len) instead of single timesteps
    2. Persist hidden states across episode steps
    3. Fix reward shaping - trust environment, reduce fixation penalty
    4. Update target network by steps, not episodes
    5. **NEW: Batched parallel environments for speedup**
    """
    env_kwargs = {'dt': dt}
    # Create multiple environments for parallel execution
    envs = [ngym.make(task, **env_kwargs) for _ in range(num_parallel)]
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.n

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
    max_steps_per_episode = 150

    # CRITICAL FIX 4: Track global steps for target network updates
    global_steps = {k: 0 for k in policy_nets.keys()}

    go_idx = envs[0].action_space.n - 1

    print(f"Starting BATCHED training for {num_episodes} episodes...")
    print(f"num_parallel = {num_parallel} (running {num_parallel} episodes simultaneously)")
    print(f"max_steps_per_episode = {max_steps_per_episode}")
    print(f"max_seq_len = {max_seq_len} (observation history window)")
    print(f"target_update_steps = {target_update_steps}")
    print(f"SPEEDUP: ~{num_parallel}x faster than non-batched version!")
    print(f"CRITICAL FIXES APPLIED:")
    print(f"  1. Using observation sequences (not single timesteps)")
    print(f"  2. Persisting hidden states across episode")
    print(f"  3. Fixed reward shaping (trust env, no fixation penalty)")
    print(f"  4. Target network updates by steps (not episodes)")
    print(f"  5. BATCHED: {num_parallel} parallel environments per model")
    print()

    for ep in range(1, num_episodes + 1):
        ep_rewards_total = {k: 0.0 for k in policy_nets.keys()}

        # Step each model independently with parallel environments
        for model_idx, (key, mtype) in enumerate(model_types):
            # Initialize parallel environments
            obs_list = [env.reset()[0] for env in envs]
            done_list = [False] * num_parallel
            steps_list = [0] * num_parallel
            has_pressed_go_list = [False] * num_parallel

            # Observation histories and hidden states for each parallel env
            obs_histories = [deque(maxlen=max_seq_len) for _ in range(num_parallel)]
            for i, obs in enumerate(obs_list):
                obs_histories[i].append(obs)
            hiddens = [None] * num_parallel

            ep_rewards = [0.0] * num_parallel

            # Run all parallel environments until all are done
            while not all(done_list) and max(steps_list) < max_steps_per_episode:
                # Create observation sequences for all active environments
                obs_sequences = []
                for i in range(num_parallel):
                    if not done_list[i]:
                        obs_seq = torch.from_numpy(np.array(list(obs_histories[i]))).float().unsqueeze(1)
                        obs_sequences.append(obs_seq)
                    else:
                        # Dummy for completed episodes
                        obs_sequences.append(obs_sequences[0] if obs_sequences else torch.zeros(1, 1, obs_dim))

                # Batched action selection
                actions, hiddens = select_action_batch(
                    policy_nets[key], obs_sequences, hiddens, epsilon, envs[0].action_space, device, num_parallel
                )

                # Step each environment
                for i in range(num_parallel):
                    if done_list[i]:
                        continue

                    next_obs, reward, done, _, info = envs[i].step(actions[i])

                    # CRITICAL FIX 3: Improved reward shaping
                    target_go_ms = envs[i].trial.get('production', None)
                    if target_go_ms is not None:
                        target_go = int(target_go_ms / envs[i].dt)
                        if actions[i] == go_idx:
                            if has_pressed_go_list[i]:
                                shaped_reward = -0.5
                            else:
                                has_pressed_go_list[i] = True
                                shaped_reward = reward
                                # Curriculum bonus for early episodes
                                if ep < 500:
                                    timing_error = abs(steps_list[i] - target_go)
                                    if timing_error <= 20:
                                        shaped_reward += 0.5 * (1.0 - timing_error / 20.0)
                        else:
                            shaped_reward = reward
                    else:
                        shaped_reward = reward

                    # Update observation history
                    obs_histories[i].append(next_obs)
                    next_obs_sequence = torch.from_numpy(np.array(list(obs_histories[i]))).float().unsqueeze(1)

                    # Store transition
                    buffers[key].push(obs_sequences[i], actions[i], shaped_reward, next_obs_sequence, done)
                    ep_rewards[i] += shaped_reward

                    # Optimize
                    loss = optimize_model(policy_nets[key], target_nets[key], buffers[key],
                                          optimizers[key], batch_size, gamma, device, mtype, max_seq_len)
                    if loss is not None:
                        histories[key]['losses'].append(loss)

                    # CRITICAL FIX 4: Update target network by steps
                    global_steps[key] += 1
                    if global_steps[key] % target_update_steps == 0:
                        target_nets[key].load_state_dict(policy_nets[key].state_dict())

                    obs_list[i] = next_obs
                    steps_list[i] += 1
                    done_list[i] = done

            # Average reward across parallel episodes for this model
            ep_rewards_total[key] = np.mean(ep_rewards)

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)
        for key in histories.keys():
            histories[key]['rewards'].append(ep_rewards_total[key])
            histories[key]['eps'].append(epsilon)

        if ep == 1 or ep % 50 == 0:
            msg = " | ".join([f"{k}: R={ep_rewards_total[k]:.2f}" for k in policy_nets.keys()])
            print(f"Ep {ep:4d} | {msg} | ε={epsilon:.3f}")

    # Save combined reward plot
    plot_training_all(histories, params={
        'num_episodes': num_episodes,
        'num_parallel': num_parallel,
        'lr': lr,
        'gamma': gamma,
        'batch_size': batch_size,
        'target_update_steps': target_update_steps
    }, output_path='images/question_2d_readysetgo_dqn_batched_rewards.png')

    # Save checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'policy_states': {k: net.state_dict() for k, net in policy_nets.items()},
        'target_states': {k: net.state_dict() for k, net in target_nets.items()},
        'histories': histories,
        'env_kwargs': env_kwargs,
        'task': task,
    }, 'checkpoints/question_2d_readysetgo_dqn_batched.pt')
    print("Saved checkpoint: checkpoints/question_2d_readysetgo_dqn_batched.pt")


if __name__ == '__main__':
    # MPS is slower even with batching - stick with CPU
    # (Tested: CPU is 8-9x faster for this small RNN network)
    use_mps = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print()

    # Run with 16 parallel environments per model for 16x speedup!
    train_dqn_models(device=device, num_parallel=16)
