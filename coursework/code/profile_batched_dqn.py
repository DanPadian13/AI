"""
Profile the batched DQN to find bottlenecks
"""
import time
import numpy as np
import torch
import neurogym as ngym
from collections import deque
from Question_2a import Net

print("Profiling batched DQN execution...")
print("="*70)
print()

# Setup
task = 'ReadySetGo-v0'
dt = 20
env_kwargs = {'dt': dt}
num_parallel = 16
max_seq_len = 20
max_steps_per_episode = 150

# Create environments
envs = [ngym.make(task, **env_kwargs) for _ in range(num_parallel)]
obs_dim = envs[0].observation_space.shape[0]
act_dim = envs[0].action_space.n
hidden_size = 64
device = torch.device('cpu')

# Create network
net = Net(obs_dim, hidden_size, act_dim, model_type='leaky', dt=dt, tau=150, sigma_rec=0.1).to(device)

# Initialize parallel environments
obs_list = [env.reset()[0] for env in envs]
obs_histories = [deque(maxlen=max_seq_len) for _ in range(num_parallel)]
for i, obs in enumerate(obs_list):
    obs_histories[i].append(obs)
hiddens = [None] * num_parallel
done_list = [False] * num_parallel
steps_list = [0] * num_parallel

print(f"Running 1 batched episode with {num_parallel} parallel environments...")
print()

# Profile different operations
times = {
    'obs_sequence_creation': 0,
    'forward_pass': 0,
    'env_step': 0,
    'reward_shaping': 0,
    'total': 0
}

start_total = time.time()
step_count = 0

go_idx = envs[0].action_space.n - 1

while not all(done_list) and max(steps_list) < max_steps_per_episode:
    step_count += 1

    # 1. Create observation sequences
    t1 = time.time()
    obs_sequences = []
    for i in range(num_parallel):
        if not done_list[i]:
            obs_seq = torch.from_numpy(np.array(list(obs_histories[i]))).float().unsqueeze(1)
            obs_sequences.append(obs_seq)
        else:
            obs_sequences.append(obs_sequences[0] if obs_sequences else torch.zeros(1, 1, obs_dim))
    times['obs_sequence_creation'] += time.time() - t1

    # 2. Forward passes (action selection)
    t2 = time.time()
    actions = []
    new_hiddens = []
    for i in range(num_parallel):
        with torch.no_grad():
            q_vals, _, hidden = net(obs_sequences[i].to(device), hiddens[i])
            action = int(q_vals[-1, 0, :].argmax().item())
            actions.append(action)
            new_hiddens.append(hidden)
    hiddens = new_hiddens
    times['forward_pass'] += time.time() - t2

    # 3. Environment steps
    t3 = time.time()
    for i in range(num_parallel):
        if done_list[i]:
            continue
        next_obs, reward, done, _, info = envs[i].step(actions[i])
        obs_list[i] = next_obs
        done_list[i] = done
    times['env_step'] += time.time() - t3

    # 4. Reward shaping and observation history update
    t4 = time.time()
    for i in range(num_parallel):
        if not done_list[i]:
            obs_histories[i].append(obs_list[i])
    times['reward_shaping'] += time.time() - t4

    # Update steps
    for i in range(num_parallel):
        if not done_list[i]:
            steps_list[i] += 1

times['total'] = time.time() - start_total

print(f"Completed in {times['total']:.2f}s ({step_count} steps)")
print()
print("Time breakdown:")
print("-"*70)
for key, value in times.items():
    if key != 'total':
        pct = (value / times['total']) * 100
        print(f"  {key:30s}: {value:6.2f}s ({pct:5.1f}%)")
print("-"*70)
print(f"  {'TOTAL':30s}: {times['total']:6.2f}s")
print()

# Calculate what's happening
steps_per_env = np.mean([s for s in steps_list])
print(f"Average steps per environment: {steps_per_env:.1f}")
print(f"Time per step (all {num_parallel} envs): {times['total']/step_count*1000:.1f}ms")
print(f"Time per step per environment: {times['total']/step_count/num_parallel*1000:.1f}ms")
print()

# The bottleneck
print("="*70)
print("ANALYSIS:")
print("="*70)
max_time_key = max([(k, v) for k, v in times.items() if k != 'total'], key=lambda x: x[1])
print(f"Bottleneck: {max_time_key[0]} ({max_time_key[1]/times['total']*100:.1f}% of time)")
print()

if max_time_key[0] == 'forward_pass':
    print("Issue: Forward passes are sequential, not truly batched!")
    print("Solution: We need to batch the forward passes together.")
    print()
    print("Currently doing:")
    print("  for i in range(16):")
    print("      net(obs_sequences[i])  # 16 sequential forward passes")
    print()
    print("Should do:")
    print("  net(torch.stack(obs_sequences))  # 1 batched forward pass")
    print()
    print("This would give ~16x speedup on forward passes!")

elif max_time_key[0] == 'obs_sequence_creation':
    print("Issue: Creating observation sequences is slow")
    print("Solution: Optimize the numpy array conversion")

elif max_time_key[0] == 'env_step':
    print("Issue: Environment stepping is the bottleneck")
    print("Solution: This is hard to optimize - it's inherent to the task")
