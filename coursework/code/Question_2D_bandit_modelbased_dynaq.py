import os
import random
import numpy as np
import matplotlib.pyplot as plt
import neurogym as ngym


def plot_rewards(rewards, output_path='images/question_2d_bandit_dyna_q_rewards.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    smooth = np.convolve(rewards, np.ones(50) / 50, mode='same') if len(rewards) > 50 else rewards
    plt.plot(rewards, alpha=0.4, label='Reward')
    plt.plot(smooth, linewidth=2, label='Reward (50-ep smooth)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Bandit-v0 Dyna-Q (model-based planning updates)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reward plot: {output_path}")


def main():
    task = 'Bandit-v0'
    env_kwargs = {'dt': 100, 'n': 2, 'p': (0.8, 0.2)}
    env = ngym.make(task, **env_kwargs)
    act_dim = env.action_space.n

    # Q-table and simple model (empirical mean reward per action)
    Q = np.zeros(act_dim, dtype=np.float32)
    model_reward_sum = np.zeros(act_dim, dtype=np.float32)
    model_counts = np.zeros(act_dim, dtype=np.int32)

    # Hyperparameters
    num_episodes = 20000
    alpha = 0.5
    gamma = 0.0  # bandit has no future reward
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.01, 1000
    planning_steps = 100  # number of model-based updates per real step

    rewards = []
    epsilon = epsilon_start

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        # Choose action via epsilon-greedy over Q
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q))

        next_obs, reward, done, _, _ = env.step(action)

        # Model-free Q update (single-step)
        td_target = reward + gamma * 0.0
        Q[action] += alpha * (td_target - Q[action])

        # Update simple reward model
        model_reward_sum[action] += reward
        model_counts[action] += 1

        # Planning updates: sample synthetic transitions from learned model
        for _ in range(planning_steps):
            a_plan = random.randrange(act_dim)
            if model_counts[a_plan] == 0:
                continue  # skip unseen actions
            r_plan = model_reward_sum[a_plan] / model_counts[a_plan]
            td_plan = r_plan + gamma * 0.0
            Q[a_plan] += alpha * (td_plan - Q[a_plan])

        # Track reward
        rewards.append(reward)

        # Decay epsilon linearly
        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)

        if ep % 250 == 0:
            avg = np.mean(rewards[-50:])
            print(f"Ep {ep:4d} | R: {reward:.2f} | Avg(50): {avg:.2f} | ε={epsilon:.3f} | Q={Q}")

    plot_rewards(rewards)

    os.makedirs('checkpoints', exist_ok=True)
    np.savez('checkpoints/question_2d_bandit_dyna_q.npz',
             Q=Q,
             rewards=np.array(rewards),
             env_kwargs=env_kwargs,
             task=task,
             model_reward_sum=model_reward_sum,
             model_counts=model_counts)
    print("Saved checkpoint: checkpoints/question_2d_bandit_dyna_q.npz")


if __name__ == '__main__':
    main()
