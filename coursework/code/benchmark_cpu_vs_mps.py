"""
Benchmark CPU vs MPS for RNN forward passes
"""
import torch
import time
import numpy as np
from Question_2a import Net

print("="*60)
print("CPU vs MPS (GPU) Benchmark for RNN")
print("="*60)
print()

obs_dim = 3
hidden_size = 64
output_size = 2
seq_len = 20
batch_size = 32

# Test both devices
devices = ['cpu']
if torch.backends.mps.is_available():
    devices.append('mps')

results = {}

for device_name in devices:
    device = torch.device(device_name)
    print(f"Testing on {device_name.upper()}...")

    # Create network
    net = Net(obs_dim, hidden_size, output_size, model_type='leaky', dt=20, tau=100, sigma_rec=0.0).to(device)

    # Create data
    x = torch.randn(seq_len, batch_size, obs_dim).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _, _, _ = net(x)

    # Benchmark
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _, _, _ = net(x)

    if device_name == 'mps':
        torch.mps.synchronize()  # Wait for GPU to finish

    elapsed = time.time() - start
    avg_time = elapsed / num_iters
    results[device_name] = avg_time

    print(f"  {num_iters} forward passes: {elapsed:.3f}s")
    print(f"  Average per pass: {avg_time*1000:.2f}ms")
    print()

# Show speedup
if 'mps' in results and 'cpu' in results:
    speedup = results['cpu'] / results['mps']
    print("="*60)
    print(f"MPS SPEEDUP: {speedup:.1f}x faster than CPU")
    print("="*60)
    print()
    print("Expected training time reduction:")
    print(f"  CPU: 2000 episodes × 12s = 24,000s = 6.7 hours")
    print(f"  MPS: 2000 episodes × {12/speedup:.1f}s = {24000/speedup:.0f}s = {24000/speedup/3600:.1f} hours")
    print()
    print("💡 Recommendation: Use MPS! Much faster.")
