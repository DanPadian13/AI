import numpy as np
import torch
import neurogym as ngym

# Load checkpoint to get environment config
checkpoint = torch.load('checkpoints/question_2a_models_and_data.pt', weights_only=False)
env_config = checkpoint['env_config']
task = env_config['task']
dt = env_config['dt']

# Setup environment
kwargs = {'dt': dt}
seq_len = 100
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=1, seq_len=seq_len)
env = dataset.env

# Generate a trial and inspect the inputs
env.new_trial()
ob, gt = env.ob, env.gt

print("Input shape:", ob.shape)
print("Number of input channels:", ob.shape[1])
print("\nFirst 20 timesteps of each input channel:")
print("=" * 60)
for i in range(ob.shape[1]):
    print(f"\nChannel {i}:")
    print(f"  Max value: {np.max(ob[:, i]):.4f}")
    print(f"  Min value: {np.min(ob[:, i]):.4f}")
    print(f"  Non-zero values: {np.sum(ob[:, i] > 0)}")
    print(f"  Values > 0.5: {np.sum(ob[:, i] > 0.5)}")
    if np.any(ob[:, i] > 0.5):
        first_idx = np.where(ob[:, i] > 0.5)[0][0]
        print(f"  First activation at timestep {first_idx} (time={first_idx * dt:.1f}ms)")

print("\n" + "=" * 60)
print("Ground truth (target):")
print(f"  Max value: {np.max(gt):.4f}")
print(f"  Non-zero values: {np.sum(gt > 0)}")
if np.any(gt > 0):
    first_gt = np.where(gt > 0)[0][0]
    print(f"  First target Go at timestep {first_gt} (time={first_gt * dt:.1f}ms)")
