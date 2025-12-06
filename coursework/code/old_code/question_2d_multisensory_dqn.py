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


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


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


def select_action(policy_net, state, epsilon, action_space, device):
    if random.random() < epsilon:
        return action_space.sample()
    with torch.no_grad():
        q_vals, _ = policy_net(state.to(device))
        return int(q_vals.squeeze(0).squeeze(0).argmax().item())


def optimize_model(policy_net, target_net, buffer, optimizer, batch_size, gamma, device):
    if len(buffer) < batch_size:
        return None
    transitions = buffer.sample(batch_size)
    batch = Transition(*transitions)

    state_batch = torch.cat(batch.state, dim=0).to(device)      # [B, T, obs] with T=1
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_state_batch = torch.cat(batch.next_state, dim=0).to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    # Net expects [T, B, obs]; current is [B, 1, obs]
    state_batch = state_batch.permute(1, 0, 2)
    next_state_batch = next_state_batch.permute(1, 0, 2)

    # Forward through policy net: get Q at last timestep
    q_values, _ = policy_net(state_batch)
    q_values = q_values[-1]  # [B, act_dim]
    q_sa = q_values.gather(1, action_batch)

    with torch.no_grad():
        q_next, _ = target_net(next_state_batch)
        q_next = q_next[-1]
        max_next_q = q_next.max(dim=1, keepdim=True)[0]
        target = reward_batch + gamma * max_next_q * (1 - done_batch)

    loss = nn.functional.smooth_l1_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def plot_training_all(histories, params=None, output_path='images/question_2d_multisensory_dqn_rewards.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    colors = {'vanilla': '#1f77b4', 'leaky': '#ff7f0e', 'leaky_fa': '#2ca02c', 'bio': '#d62728'}
    fig, ax = plt.subplots(figsize=(12, 7))
    for key, hist in histories.items():
        rewards = hist['rewards']
        smooth = np.convolve(rewards, np.ones(50) / 50, mode='same') if len(rewards) > 50 else rewards
        ax.plot(smooth, linewidth=2, label=f'{key} (50-ep smooth)', color=colors.get(key, None))
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Reward (50-ep smooth)', fontsize=13)
    ax.set_title('MultiSensoryIntegration-v0 DQN: Reward (smoothed)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3)
    if params:
        text = (f"episodes={params.get('num_episodes')}, "
                f"max_steps={params.get('max_steps_per_episode')}, "
                f"lr={params.get('lr')}, gamma={params.get('gamma')}, "
                f"eps_start={params.get('epsilon_start')}, eps_end={params.get('epsilon_end')}, "
                f"eps_decay={params.get('epsilon_decay')}, "
                f"batch={params.get('batch_size')}, target_update={params.get('target_update')}")
        ax.text(0.01, -0.15, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', wrap=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training plot: {output_path}")


def plot_accuracy_all(histories, output_path='images/question_2d_multisensory_dqn_accuracy.png'):
    """Plot accuracy over training episodes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    colors = {'vanilla': '#1f77b4', 'leaky': '#ff7f0e', 'leaky_fa': '#2ca02c', 'bio': '#d62728'}
    fig, ax = plt.subplots(figsize=(12, 7))
    for key, hist in histories.items():
        if 'accuracies' in hist and len(hist['accuracies']) > 0:
            accuracies = hist['accuracies']
            smooth = np.convolve(accuracies, np.ones(50) / 50, mode='same') if len(accuracies) > 50 else accuracies
            ax.plot(smooth, linewidth=2, label=f'{key} (50-ep smooth)', color=colors.get(key, None))
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Accuracy (50-ep smooth)', fontsize=13)
    ax.set_title('MultiSensoryIntegration-v0 DQN: Accuracy', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot: {output_path}")


def train_dqn_models(task='MultiSensoryIntegration-v0', dt=100, num_episodes=3000, gamma=0.99,
                     epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=2000,
                     batch_size=64, target_update=100, lr=1e-3,
                     hidden_size=64, device='cpu'):
    env_kwargs = {'dt': dt}
    env = ngym.make(task, **env_kwargs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print(f"Task: {task}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    print(f"Actions: 0=fixate, 1=left, 2=right")
    print()

    # Instantiate four Q-networks via Net with different model types
    model_types = [('vanilla', 'vanilla'), ('leaky', 'leaky'),
                   ('leaky_fa', 'leaky_fa'), ('bio', 'bio_realistic')]
    policy_nets = {}
    target_nets = {}
    optimizers = {}
    buffers = {}
    histories = {}

    for key, mtype in model_types:
        net = Net(obs_dim, hidden_size, act_dim, model_type=mtype, dt=dt, tau=100, sigma_rec=0.1,
                  exc_ratio=0.8 if mtype == 'bio_realistic' else None,
                  sparsity=0.2 if mtype == 'bio_realistic' else None).to(device)
        target = Net(obs_dim, hidden_size, act_dim, model_type=mtype, dt=dt, tau=100, sigma_rec=0.1,
                     exc_ratio=0.8 if mtype == 'bio_realistic' else None,
                     sparsity=0.2 if mtype == 'bio_realistic' else None).to(device)
        target.load_state_dict(net.state_dict())
        target.eval()

        policy_nets[key] = net
        target_nets[key] = target
        optimizers[key] = optim.Adam(net.parameters(), lr=lr)
        buffers[key] = ReplayBuffer(capacity=100000)
        histories[key] = {'rewards': [], 'losses': [], 'eps': [], 'accuracies': []}

    epsilon = epsilon_start
    # MultiSensory trials are very short (~11 steps at dt=100)
    max_steps_per_episode = 50  # Cap episode length

    for ep in range(1, num_episodes + 1):
        ep_rewards = {k: 0.0 for k in policy_nets.keys()}
        ep_correct = {k: 0 for k in policy_nets.keys()}
        ep_decisions = {k: 0 for k in policy_nets.keys()}

        # Step each model independently for this episode (separate trajectories)
        for key in policy_nets.keys():
            obs, _ = env.reset()
            done = False
            steps = 0

            while not done and steps < max_steps_per_episode:
                state = torch.from_numpy(obs).float().unsqueeze(0)
                state = state.unsqueeze(1)  # [1,1,obs]
                action = select_action(policy_nets[key], state, epsilon, env.action_space, device)
                next_obs, reward, done, _, info = env.step(action)

                # Reward shaping for MultiSensory task
                # Encourage correct decisions (left/right) and penalize wrong fixations
                shaped_reward = reward

                # Get ground truth if available
                gt = env.gt_now if hasattr(env, 'gt_now') else None
                if gt is not None:
                    # Decision actions are 1 (left) and 2 (right)
                    if action in [1, 2]:
                        ep_decisions[key] += 1
                        if action == gt:
                            shaped_reward = reward + 1.0  # Bonus for correct decision
                            ep_correct[key] += 1
                        else:
                            shaped_reward = reward - 0.5  # Penalty for wrong decision
                    # Action 0 is fixate
                    elif action == 0:
                        if gt == 0:
                            shaped_reward = reward + 0.01  # Small reward for correct fixation
                        else:
                            shaped_reward = reward - 0.1  # Penalty for fixating when should decide

                next_state = torch.from_numpy(next_obs).float().unsqueeze(0).unsqueeze(1)
                buffers[key].push(state, action, shaped_reward, next_state, done)
                ep_rewards[key] += shaped_reward

                loss = optimize_model(policy_nets[key], target_nets[key], buffers[key],
                                      optimizers[key], batch_size, gamma, device)
                if loss is not None:
                    histories[key]['losses'].append(loss)

                obs = next_obs
                steps += 1

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)

        for key in histories.keys():
            histories[key]['rewards'].append(ep_rewards[key])
            histories[key]['eps'].append(epsilon)
            # Calculate accuracy for this episode
            acc = ep_correct[key] / ep_decisions[key] if ep_decisions[key] > 0 else 0.0
            histories[key]['accuracies'].append(acc)

        # Target updates
        if ep % target_update == 0:
            for key in policy_nets.keys():
                target_nets[key].load_state_dict(policy_nets[key].state_dict())

        if ep % 100 == 0:
            msg = " | ".join([
                f"{k}: R={np.mean(histories[k]['rewards'][-100:]):.2f} "
                f"Acc={np.mean(histories[k]['accuracies'][-100:]):.3f}"
                for k in histories
            ])
            print(f"Ep {ep:4d} | {msg} | ε={epsilon:.3f}")

    # Save plots
    plot_training_all(histories, params={
        'num_episodes': num_episodes,
        'max_steps_per_episode': max_steps_per_episode,
        'lr': lr,
        'gamma': gamma,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay': epsilon_decay,
        'batch_size': batch_size,
        'target_update': target_update
    }, output_path='images/question_2d_multisensory_dqn_rewards.png')

    plot_accuracy_all(histories, output_path='images/question_2d_multisensory_dqn_accuracy.png')

    # Save checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'policy_states': {k: net.state_dict() for k, net in policy_nets.items()},
        'target_states': {k: net.state_dict() for k, net in target_nets.items()},
        'histories': histories,
        'env_kwargs': env_kwargs,
        'task': task,
    }, 'checkpoints/question_2d_multisensory_dqn.pt')
    print("Saved checkpoint: checkpoints/question_2d_multisensory_dqn.pt")

    # Print final statistics
    print("\n" + "="*70)
    print("Training Complete - Final Performance (last 100 episodes):")
    print("="*70)
    for key in policy_nets.keys():
        avg_reward = np.mean(histories[key]['rewards'][-100:])
        avg_acc = np.mean(histories[key]['accuracies'][-100:])
        print(f"{key:12s}: Avg Reward = {avg_reward:6.2f}, Avg Accuracy = {avg_acc:.3f}")
    print("="*70)


if __name__ == '__main__':
    use_mps = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")

    train_dqn_models(device=device, num_episodes=3000)
