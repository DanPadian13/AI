"""
Benchmark CPU vs MPS for BATCHED RNN forward passes (simulating parallel episodes)
"""
import torch
import time
import numpy as np
from Question_2a import Net

print("="*70)
print("CPU vs MPS Benchmark for BATCHED RNN (Parallel Episodes)")
print("="*70)
print()

obs_dim = 3
hidden_size = 64
output_size = 2
seq_len = 20

# Test different batch sizes (simulating num_parallel)
batch_sizes = [1, 4, 8, 16, 32]

devices = ['cpu']
if torch.backends.mps.is_available():
    devices.append('mps')

print("Testing forward passes with different num_parallel values...")
print()

results = {device: {} for device in devices}

for batch_size in batch_sizes:
    print(f"Batch size (num_parallel) = {batch_size}")
    print("-" * 70)

    for device_name in devices:
        device = torch.device(device_name)

        # Create network
        net = Net(obs_dim, hidden_size, output_size, model_type='leaky', dt=20, tau=100, sigma_rec=0.0).to(device)

        # Create data
        x = torch.randn(seq_len, batch_size, obs_dim).to(device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _, _, _ = net(x)

        # Benchmark
        num_iters = 50
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                _, _, _ = net(x)

        if device_name == 'mps':
            torch.mps.synchronize()

        elapsed = time.time() - start
        avg_time = elapsed / num_iters
        results[device_name][batch_size] = avg_time

        print(f"  {device_name.upper()}: {avg_time*1000:.2f}ms per forward pass")

    if 'mps' in results and 'cpu' in results:
        speedup = results['cpu'][batch_size] / results['mps'][batch_size]
        if speedup > 1:
            print(f"  → MPS is {speedup:.2f}x FASTER ✓")
        else:
            print(f"  → CPU is {1/speedup:.2f}x faster (MPS slower)")
    print()

print("="*70)
print("SUMMARY: When does MPS become worthwhile?")
print("="*70)
print()

if 'mps' in results:
    for batch_size in batch_sizes:
        speedup = results['cpu'][batch_size] / results['mps'][batch_size]
        if speedup > 1:
            status = f"✓ USE MPS ({speedup:.2f}x faster)"
        else:
            status = f"✗ Use CPU ({1/speedup:.2f}x faster)"
        print(f"num_parallel={batch_size:2d}: {status}")

    print()
    print("="*70)

    # Find crossover point
    for batch_size in batch_sizes:
        speedup = results['cpu'][batch_size] / results['mps'][batch_size]
        if speedup > 1.1:  # >10% faster
            print(f"💡 RECOMMENDATION: Use MPS with num_parallel >= {batch_size}")
            print(f"   At num_parallel=16: {results['cpu'][16] / results['mps'][16]:.2f}x speedup")

            # Estimate training time
            base_time_cpu = 12  # seconds per episode on CPU with num_parallel=1
            episodes = 2000

            time_cpu_batched = (base_time_cpu / 16) * episodes
            time_mps_batched = time_cpu_batched / (results['cpu'][16] / results['mps'][16])

            print()
            print(f"Estimated training time for 2000 episodes:")
            print(f"  CPU (num_parallel=16): {time_cpu_batched:.0f}s = {time_cpu_batched/3600:.2f} hours")
            print(f"  MPS (num_parallel=16): {time_mps_batched:.0f}s = {time_mps_batched/3600:.2f} hours")
            break
    else:
        print("💡 RECOMMENDATION: Stick with CPU even at num_parallel=16")
        print("   MPS overhead still dominates for this small network")
