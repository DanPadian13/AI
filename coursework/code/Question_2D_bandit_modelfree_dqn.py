import random
import os
from collections import deque, namedtuple

import neurogym as ngym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def select_action(policy_net, state, epsilon, action_space, device):
    if random.random() < epsilon:
        return action_space.sample()
    with torch.no_grad():
        q_vals = policy_net(state.to(device))
        return int(q_vals.argmax(dim=1).item())


def optimize_model(policy_net, target_net, buffer, optimizer, batch_size, gamma, device):
    if len(buffer) < batch_size:
        return None

    transitions = buffer.sample(batch_size)
    batch = Transition(*transitions)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_state_batch = torch.cat(batch.next_state).to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = policy_net(state_batch).gather(1, action_batch)

    with torch.no_grad():
        max_next_q = target_net(next_state_batch).max(dim=1, keepdim=True)[0]
        target = reward_batch + gamma * max_next_q * (1 - done_batch)

    loss = nn.functional.smooth_l1_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def plot_rewards(rewards, output_path='images/question_2d_bandit_dqn_rewards.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    smooth = np.convolve(rewards, np.ones(50) / 50, mode='same') if len(rewards) > 50 else rewards
    plt.plot(rewards, alpha=0.4, label='Reward')
    plt.plot(smooth, linewidth=2, label='Reward (smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Bandit-v0 Q-learning (ε-greedy + target net + replay)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reward plot: {output_path}")


def plot_metrics(rewards, losses, eps_history, output_path='images/question_2d_bandit_dqn_metrics.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Rewards
    smooth = np.convolve(rewards, np.ones(50) / 50, mode='same') if len(rewards) > 50 else rewards
    axes[0].plot(rewards, alpha=0.4, label='Reward per episode')
    axes[0].plot(smooth, linewidth=2, label='Reward (50-ep smooth)')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Bandit-v0 DQN performance')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Losses
    if losses:
        axes[1].plot(losses, color='#d62728', alpha=0.8)
        axes[1].set_ylabel('Loss')
        axes[1].set_title('TD loss over updates')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No loss data', ha='center', va='center')
        axes[1].set_axis_off()

    # Epsilon schedule
    axes[2].plot(eps_history, color='#1f77b4')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Epsilon')
    axes[2].set_title('Exploration schedule (ε-greedy)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics plot: {output_path}")


def main():
    # Device selection
    use_mps = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Environment
    task = 'Bandit-v0'
    env_kwargs = {'dt': 100, 'n': 2, 'p': (0.8, 0.2)}
    env = ngym.make(task, **env_kwargs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Q-networks
    policy_net = QNetwork(obs_dim, act_dim, hidden=128).to(device)
    target_net = QNetwork(obs_dim, act_dim, hidden=128).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=50000)

    # Hyperparameters
    num_episodes = 10000
    batch_size = 64
    gamma = 0.99
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.1, 5000
    target_update = 100  # episodes

    rewards = []
    losses = []
    eps_history = []

    epsilon = epsilon_start
    max_steps_per_episode = 10  # Bandit is a single-step task; cap steps to avoid infinite loops if env never signals done

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        for _ in range(max_steps_per_episode):
            if done:
                break
            state = torch.from_numpy(obs).float().unsqueeze(0)
            action = select_action(policy_net, state, epsilon, env.action_space, device)
            next_obs, reward, done, _, _ = env.step(action)

            next_state = torch.from_numpy(next_obs).float().unsqueeze(0)
            buffer.push(state, action, reward, next_state, done)
            ep_reward += reward

            loss = optimize_model(policy_net, target_net, buffer, optimizer, batch_size, gamma, device)
            if loss is not None:
                losses.append(loss)

            obs = next_obs

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)
        eps_history.append(epsilon)

        # Update target network
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards.append(ep_reward)
        if ep % 250 == 0:
            avg = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
            print(f"Ep {ep:4d} | Reward: {ep_reward:.3f} | Avg(last): {avg:.3f} | ε={epsilon:.3f}")

    # Plot and save
    plot_rewards(rewards)
    plot_metrics(rewards, losses, eps_history)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'policy_state_dict': policy_net.state_dict(),
        'target_state_dict': target_net.state_dict(),
        'rewards': rewards,
        'losses': losses,
        'env_kwargs': env_kwargs,
        'task': task,
        'eps_history': eps_history
    }, 'checkpoints/question_2d_bandit_dqn.pt')
    print("Saved checkpoint: checkpoints/question_2d_bandit_dqn.pt")


if __name__ == '__main__':
    main()
