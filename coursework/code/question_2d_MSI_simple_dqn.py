"""
SIMPLE, FAST DQN for MultiSensoryIntegration
Model-free RL approach to multi-sensory integration
Target: ~20-40 minutes for 4 models

PARAMETERS MATCHED TO QUESTION 2A FOR FAIR COMPARISON:
- Hidden size: 64
- Tau: 100
- Sigma_rec: 0.15
- Learning rate: 0.0005
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
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

def train_one_model(model_name, model_type, task='MultiSensoryIntegration-v0', dt=100,
                    num_episodes=3000, device='cpu'):
    """Train a single model on MultiSensoryIntegration task."""

    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")

    # Setup
    env = ngym.make(task, dt=dt)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Network - MATCHED TO QUESTION 2A PARAMETERS
    net = Net(obs_dim, 64, act_dim, model_type=model_type, dt=dt, tau=100,
              sigma_rec=0.15, exc_ratio=0.8 if model_type == 'bio_realistic' else None).to(device)
    target_net = Net(obs_dim, 64, act_dim, model_type=model_type, dt=dt, tau=100,
                     sigma_rec=0.15, exc_ratio=0.8 if model_type == 'bio_realistic' else None).to(device)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    # Optimizer - MATCHED TO QUESTION 2A
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    buffer = ReplayBuffer(capacity=10000)

    # Regularization params - MATCHED TO QUESTION 2A (only for bio-realistic model)
    use_regularization = (model_type == 'bio_realistic')
    l1_lambda = 0.0005 if use_regularization else 0.0
    l2_lambda = 0.01 if use_regularization else 0.0

    # Training params
    epsilon = 1.0
    epsilon_min = 0.10  # INCREASED from 0.1 to encourage more exploration
    epsilon_decay_end = int(num_episodes * 0.8)  # EXTENDED decay period to 80%
    epsilon_decay_per_episode = (epsilon - epsilon_min) / epsilon_decay_end  # Linear decay
    gamma = 0.99
    batch_size = 16
    
    # Track FINAL DECISIONS only (not every timestep action)
    final_action_counts = [0, 0, 0]  # fixate, left, right - only track what ends trials

    rewards_history = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        step = 0
        
        # CRITICAL FIX: Track history for temporal integration
        obs_history = []
        last_action = None  # Track the action that ended the episode

        while not done and step < 100:  # MSI has shorter episodes than RSG
            # CRITICAL FIX: Use full observation history (temporal context)
            obs_history.append(obs)
            state = torch.from_numpy(np.array(obs_history)).float().unsqueeze(1)  # [T, 1, obs]

            # Action selection - use LAST timestep Q-values
            if random.random() < epsilon:
                # During exploration, after step 10 (past fixation/stimulus period),
                # force left/right choices instead of allowing fixation
                if step > 10:  # Decision period - don't allow fixation
                    action = env.action_space.sample()
                    while action == 0:  # Reject fixation, resample
                        action = env.action_space.sample()
                else:
                    action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals, _, _ = net(state.to(device))
                    # After step 10, restrict to left/right choices only
                    if step > 10:
                        # Set fixation Q-value very negative so it's never chosen
                        q_vals_copy = q_vals[-1, 0].clone()
                        q_vals_copy[0] = -1e9  # Make fixation impossible
                        action = int(q_vals_copy.argmax().item())
                    else:
                        action = int(q_vals[-1, 0].argmax().item())  # Use final timestep
            
            last_action = action  # Keep track of last action taken

            # Step
            next_obs, reward, done, _, _ = env.step(action)
            
            # CRITICAL: If episode ends with fixation (action=0), that's wrong!
            # The task requires choosing left (1) or right (2) to end the trial
            # Apply STRONG penalty - must overwhelm any positive reward from env
            if done and action == 0:
                # Override the environment's reward completely
                # This is a failure state - the model should NEVER end on fixation
                reward = -5.0  # Strong penalty to discourage fixation endings
            
            # CRITICAL FIX: Next state includes next observation in history
            next_obs_history = obs_history + [next_obs]
            next_state = torch.from_numpy(np.array(next_obs_history)).float().unsqueeze(1)

            # Store
            buffer.push(state, action, reward, next_state, done)
            ep_reward += reward

            # Optimize (every 16 steps AND if buffer big enough)
            if step % 16 == 0 and len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)

                # CRITICAL FIX: Pad sequences to same length for batching
                # Find max length in batch
                max_len = max(s.shape[0] for s in batch.state)
                
                # Pad all sequences
                state_batch = []
                next_state_batch = []
                for s, ns in zip(batch.state, batch.next_state):
                    # Pad with zeros at the beginning (preserve final timestep position)
                    if s.shape[0] < max_len:
                        pad_len = max_len - s.shape[0]
                        s_padded = torch.cat([torch.zeros(pad_len, 1, s.shape[2]), s], dim=0)
                        ns_padded = torch.cat([torch.zeros(pad_len, 1, ns.shape[2]), ns], dim=0)
                    else:
                        s_padded = s
                        ns_padded = ns
                    # Squeeze out the batch dimension (currently always 1)
                    state_batch.append(s_padded.squeeze(1))  # [T, obs]
                    next_state_batch.append(ns_padded.squeeze(1))  # [T, obs]
                
                # Stack along batch dimension: list of [T, obs] -> [T, batch, obs]
                state_batch = torch.stack(state_batch, dim=1).to(device)  # [T, batch, obs]
                next_state_batch = torch.stack(next_state_batch, dim=1).to(device)
                
                action_batch = torch.tensor(batch.action, dtype=torch.long, device=device)  # [batch]
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)  # [batch]
                done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device)  # [batch]

                # Q-learning update - use FINAL timestep
                q_vals, activity, _ = net(state_batch)  # [T, batch, act_dim]
                q_vals = q_vals[-1, :, :]  # Take LAST time step -> [batch, act_dim]
                
                q_sa = q_vals.gather(1, action_batch.unsqueeze(1))  # [batch, 1]

                with torch.no_grad():
                    q_next, _, _ = target_net(next_state_batch)  # [T, batch, act_dim]
                    q_next = q_next[-1, :, :]  # Take LAST time step -> [batch, act_dim]
                    max_next_q = q_next.max(dim=1, keepdim=True)[0]  # [batch, 1]
                    target = reward_batch.unsqueeze(1) + gamma * max_next_q * (1 - done_batch.unsqueeze(1))  # [batch, 1]

                loss = nn.functional.smooth_l1_loss(q_sa, target)

                # Add L1 and L2 regularization for bio-realistic model
                # MATCHED TO QUESTION 2A: L1 on weights, L2 on ACTIVITY (firing rates)
                if use_regularization:
                    # L1 regularization on weights (sparse connectivity)
                    l1_penalty = sum(p.abs().sum() for p in net.parameters())
                    # L2 regularization on firing rates (low firing rates) - SAME AS 2A
                    # Average over time dimension as well
                    l2_penalty = torch.mean(activity ** 2)
                    loss = loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            obs = next_obs
            step += 1
        
        # Episode ended - record the FINAL action (the decision)
        if last_action is not None:
            final_action_counts[last_action] += 1

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
            # Show FINAL DECISION distribution (what action ended trials)
            total_decisions = sum(final_action_counts)
            if total_decisions > 0:
                decision_dist = [c/total_decisions for c in final_action_counts]
                print(f"  Ep {ep:4d} | Avg reward: {avg_reward:6.2f} | ε={epsilon:.3f} | "
                      f"Final: fix={decision_dist[0]:.2f} L={decision_dist[1]:.2f} R={decision_dist[2]:.2f}")
            else:
                print(f"  Ep {ep:4d} | Avg reward: {avg_reward:6.2f} | ε={epsilon:.3f}")
            
            # Reset final action counts periodically
            final_action_counts = [0, 0, 0]

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
        history, net = train_one_model(name, mtype, num_episodes=5000, device=device)
        all_histories[name] = history
        all_nets[name] = net

    # Plot
    os.makedirs('images', exist_ok=True)
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (name, _), color in zip(models_to_train, colors):
        rewards = all_histories[name]
        smooth = np.convolve(rewards, np.ones(50)/50, mode='same')
        # Exclude final 50 datapoints
        smooth = smooth[:-50]
        plt.plot(smooth, label=name, color=color, linewidth=3.0)

    plt.xlabel('Episode', fontsize=24, fontweight='bold')
    plt.ylabel('Reward (smoothed)', fontsize=24, fontweight='bold')
    plt.title('DQN on MultiSensoryIntegration-v0 (Parameters Matched to Question 2a)', fontsize=26, fontweight='bold')
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/question_2d_MSI_dqn_rewards.png', dpi=150)
    print(f"\n✓ Saved plot: images/question_2d_MSI_dqn_rewards.png")

    # Save models
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'histories': all_histories,
        'models': {k: v.state_dict() for k, v in all_nets.items()},
        'task': 'MultiSensoryIntegration-v0',
        'env_kwargs': {'dt': 100}
    }, 'checkpoints/question_2d_MSI_dqn.pt')
    print(f"✓ Saved checkpoint: checkpoints/question_2d_MSI_dqn.pt")


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"Total training time: {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    print("\nPARAMETERS MATCHED TO QUESTION 2A:")
    print("  ✓ Hidden size: 64")
    print("  ✓ Time constant τ: 100ms")
    print("  ✓ Recurrent noise σ: 0.15")
    print("  ✓ Learning rate: 0.0005")
    print("  ✓ Bio-realistic regularization:")
    print("    - L1 (β=0.0005) on weights → sparse connectivity")
    print("    - L2 (β=0.01) on activity → low firing rates")
    print("  ✓ Dale's principle: 80% excitatory, 20% inhibitory")
    print("\nTask: MultiSensoryIntegration-v0")
    print("  - Multi-modal sensory integration")
    print("  - 5 inputs: fixation + 4 sensory modalities")
    print("  - 3 outputs: fixate, left, right")
    print("  - Model-free RL learns to integrate evidence from multiple sources")
    print("="*70)
