"""
SIMPLE, FAST DQN for ReadySetGo
No fancy batching, just straightforward implementation
Target: ~30-60 minutes for 4 models

PARAMETERS MATCHED TO QUESTION 2A FOR FAIR COMPARISON:
- Hidden size: 64 (you'll update 2a to match)
- Tau: 100 (same as 2a)
- Sigma_rec: 0.15 (same as 2a)
- Learning rate: 0.0005 (same as 2a)
- Bio-realistic regularization: L1=0.0005 on weights, L2=0.01 on activity
- Exc_ratio: 0.8 (80% excitatory, 20% inhibitory)
"""
import os
import random
from collections import deque, namedtuple
import neurogym as ngym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Question_2a import Net

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=10000):  # Smaller buffer
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


def train_one_model(model_name, model_type, task='ReadySetGo-v0', dt=20,
                    num_episodes=5000, device='cpu'):
    """Train a single model - simple and fast"""

    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")

    # Setup
    env = ngym.make(task, dt=dt)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Network - MATCHED TO QUESTION 2A PARAMETERS (except hidden_size=64)
    # Hidden size: 64, tau: 100, sigma_rec: 0.15
    net = Net(obs_dim, 64, act_dim, model_type=model_type, dt=dt, tau=100,
              sigma_rec=0.15, exc_ratio=0.8 if model_type == 'bio_realistic' else None).to(device)
    target_net = Net(obs_dim, 64, act_dim, model_type=model_type, dt=dt, tau=100,
                     sigma_rec=0.15, exc_ratio=0.8 if model_type == 'bio_realistic' else None).to(device)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    # Optimizer - MATCHED TO QUESTION 2A: lr=0.0005 (not 0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    buffer = ReplayBuffer(capacity=10000)

    # Regularization params - MATCHED TO QUESTION 2A (only for bio-realistic model)
    # L1: 0.0005 on weights, L2: 0.01 on ACTIVITY (not weights)
    use_regularization = (model_type == 'bio_realistic')
    l1_lambda = 0.0005 if use_regularization else 0.0
    l2_lambda = 0.01 if use_regularization else 0.0

    # Training params
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay_end = int(num_episodes * 0.7)  # Reach min epsilon at 70% of training
    epsilon_decay_per_episode = (epsilon - epsilon_min) / epsilon_decay_end  # Linear decay
    # Linear decay: ε goes from 1.0 → 0.1 over first 70% of episodes, then stays at 0.1
    gamma = 0.99
    batch_size = 32

    rewards_history = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        step = 0

        while not done and step < 150:
            # Simple state: just current observation (no sequences!)
            state = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)  # [1, 1, obs]

            # Action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals, _, _ = net(state.to(device))
                    action = int(q_vals[0, 0].argmax().item())

            # Step
            next_obs, reward, done, _, _ = env.step(action)
            next_state = torch.from_numpy(next_obs).float().unsqueeze(0).unsqueeze(0)

            # Store
            buffer.push(state, action, reward, next_state, done)
            ep_reward += reward

            # Optimize (only every 4 steps AND if buffer big enough)
            if step % 4 == 0 and len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)

                # Stack states
                state_batch = torch.cat(batch.state).to(device)  # [batch, 1, obs]
                action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
                next_state_batch = torch.cat(batch.next_state).to(device)
                done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

                # Q-learning update
                q_vals, activity, _ = net(state_batch)
                q_vals = q_vals[:, 0, :]  # [batch, act_dim]
                q_sa = q_vals.gather(1, action_batch)

                with torch.no_grad():
                    q_next, _, _ = target_net(next_state_batch)
                    q_next = q_next[:, 0, :]
                    max_next_q = q_next.max(dim=1, keepdim=True)[0]
                    target = reward_batch + gamma * max_next_q * (1 - done_batch)

                loss = nn.functional.smooth_l1_loss(q_sa, target)

                # Add L1 and L2 regularization for bio-realistic model
                # MATCHED TO QUESTION 2A: L1 on weights, L2 on ACTIVITY (firing rates)
                if use_regularization:
                    # L1 regularization on weights (sparse connectivity)
                    l1_penalty = sum(p.abs().sum() for p in net.parameters())
                    # L2 regularization on firing rates (low firing rates) - SAME AS 2A
                    l2_penalty = torch.mean(activity ** 2)
                    loss = loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            obs = next_obs
            step += 1

        # Update target network every 10 episodes
        if ep % 10 == 0:
            target_net.load_state_dict(net.state_dict())

        # Linear epsilon decay
        if ep <= epsilon_decay_end:
            epsilon = max(epsilon_min, epsilon - epsilon_decay_per_episode)
        else:
            epsilon = epsilon_min

        rewards_history.append(ep_reward)

        if ep % 50 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"  Ep {ep:4d} | Avg reward: {avg_reward:6.2f} | ε={epsilon:.3f}")

    return rewards_history, net


def main():
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Train all 4 models
    models_to_train = [
        ('Vanilla RNN', 'vanilla'),
        ('Leaky RNN', 'leaky'),
        ('Leaky + FA', 'leaky_fa'),
        ('Bio-Realistic', 'bio_realistic')
    ]

    all_histories = {}
    all_nets = {}

    for name, mtype in models_to_train:
        history, net = train_one_model(name, mtype, num_episodes=3000, device=device)
        all_histories[name] = history
        all_nets[name] = net

    # Plot
    os.makedirs('images', exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (name, _), color in zip(models_to_train, colors):
        rewards = all_histories[name]
        smooth = np.convolve(rewards, np.ones(50)/50, mode='same')
        plt.plot(smooth, label=name, color=color, linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward (smoothed)')
    plt.title('DQN on ReadySetGo-v0 (Parameters Matched to Question 2a)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/question_2d_simple_dqn_rewards.png', dpi=150)
    print(f"\n✓ Saved plot: images/question_2d_simple_dqn_rewards.png")

    # Save models
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'histories': all_histories,
        'models': {k: v.state_dict() for k, v in all_nets.items()}
    }, 'checkpoints/question_2d_simple_dqn.pt')
    print(f"✓ Saved checkpoint: checkpoints/question_2d_simple_dqn.pt")


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"Total training time: {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    print("\nPARAMETERS MATCHED TO QUESTION 2A:")
    print("  ✓ Hidden size: 64 (update 2a to match this)")
    print("  ✓ Time constant τ: 100ms (same as supervised learning)")
    print("  ✓ Recurrent noise σ: 0.15 (same as supervised learning)")
    print("  ✓ Learning rate: 0.0005 (same as supervised learning)")
    print("  ✓ Bio-realistic regularization:")
    print("    - L1 (β=0.0005) on weights → sparse connectivity")
    print("    - L2 (β=0.01) on activity → low firing rates")
    print("  ✓ Dale's principle: 80% excitatory, 20% inhibitory")
    print("\nNow you can fairly compare RL (DQN) vs Supervised Learning!")
    print("="*70)
