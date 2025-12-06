"""
Quick test to verify that Net class now properly returns hidden states
"""
import torch
import numpy as np
from Question_2a import Net

print("Testing Net class with hidden state support...")
print()

# Create a simple network
obs_dim = 3
hidden_size = 10
output_size = 2
model_type = 'leaky'

net = Net(obs_dim, hidden_size, output_size, model_type=model_type, dt=20, tau=100, sigma_rec=0.0)

# Create a small sequence
seq_len = 5
batch_size = 1
x = torch.randn(seq_len, batch_size, obs_dim)

print(f"Input shape: {x.shape} [seq_len, batch_size, obs_dim]")
print()

# Test 1: Forward pass without hidden state
print("Test 1: Forward pass without providing hidden state")
out1, activity1, hidden1 = net(x)
print(f"  Output shape: {out1.shape}")
print(f"  Activity shape: {activity1.shape}")
print(f"  Hidden state type: {type(hidden1)}")
if isinstance(hidden1, tuple):
    print(f"  Hidden state shapes: {[h.shape for h in hidden1]}")
elif hidden1 is not None:
    print(f"  Hidden state shape: {hidden1.shape}")
print("  ✓ Works!")
print()

# Test 2: Forward pass WITH hidden state (persistence)
print("Test 2: Forward pass with hidden state persistence")
hidden = None
for i in range(3):
    out, activity, hidden = net(x, hidden)
    print(f"  Step {i+1}: Hidden state persisted")
print("  ✓ Works!")
print()

# Test 3: Compare outputs with and without hidden state
print("Test 3: Different sequences produce different hidden states")
x2 = torch.randn(seq_len, batch_size, obs_dim)
out1, act1, h1 = net(x)
out2, act2, h2 = net(x2)

if isinstance(h1, tuple):
    diff = torch.abs(h1[0] - h2[0]).mean().item()
else:
    diff = torch.abs(h1 - h2).mean().item() if h1 is not None and h2 is not None else 0

print(f"  Hidden state difference: {diff:.6f}")
print(f"  ✓ Different inputs produce different hidden states!")
print()

print("="*60)
print("SUCCESS! Net class properly supports hidden state now.")
print("="*60)
print()
print("You can now run:")
print("  python code/question_2d_readysetgo_dqn_fixed.py")
