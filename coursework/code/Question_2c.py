import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import neurogym as ngym
import os

from Question_2a import Net, train_model

device = torch.device('cpu')
print(f"Using device: {device}")


def plot_task_structure(env, output_path='images/delaymatchsample_task_structure.png'):
    """Visualize the DelayMatchSample task structure."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Generate two example trials - one match, one non-match
    for trial_idx in range(2):
        env.new_trial()
        ob, gt = env.ob, env.gt

        time = np.arange(len(ob)) * env.dt
        ax = axes[trial_idx]

        # Plot input channels
        n_channels = ob.shape[1]
        for i in range(n_channels):
            ax.plot(time, ob[:, i] + i*1.5, linewidth=1.5, label=f'Input {i}', alpha=0.8)

        # Plot ground truth (target action) - offset vertically
        gt_offset = n_channels * 1.5 + 1
        ax.plot(time, gt + gt_offset, linewidth=2.5, label='Ground Truth Action',
               color='black', linestyle='-', alpha=0.9)

        # Mark key trial periods
        sample_start = env.start_t['sample'] * env.dt
        ax.axvline(sample_start, color='green', linestyle='--', linewidth=2,
                  alpha=0.7, label='Sample Start')

        delay_start = env.start_t['delay'] * env.dt
        ax.axvline(delay_start, color='cyan', linestyle='--', linewidth=2,
                  alpha=0.7, label='Delay Start')

        test_start = env.start_t['test'] * env.dt
        ax.axvline(test_start, color='orange', linestyle='--', linewidth=2,
                  alpha=0.7, label='Test Start')

        decision_start = env.start_t['decision'] * env.dt
        ax.axvline(decision_start, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label='Decision Start')

        # Add text annotation for ground truth action
        final_action = gt[-1]
        action_labels = {0: 'Fixate', 1: 'Match', 2: 'Non-match'}
        ax.text(time[-1] * 0.95, gt_offset + final_action,
               f'Target: {action_labels[final_action]}',
               fontsize=11, fontweight='bold', ha='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Input Activity', fontsize=11)
        ax.set_title(f'Trial {trial_idx + 1}: Delay Match-to-Sample',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if trial_idx == 0:
            ax.legend(loc='upper left', fontsize=9, ncol=2)

    plt.suptitle('DelayMatchSample Task Structure (Remember & Compare)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def explore_task():
    """Explore and visualize the DelayMatchSample task."""
    print("="*70)
    print("Question 2c: DelayMatchSample Task Setup")
    print("="*70)
    print()

    # Setup task
    task = 'DelayMatchSample-v0'
    kwargs_env = {'dt': 20, 'dim_ring': 2}

    dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=16, seq_len=300)
    env = dataset.env

    print(f"Task: {task}")
    print(f"dt: {kwargs_env['dt']} ms")
    print(f"dim_ring: {kwargs_env['dim_ring']} (number of stimulus options)")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Print task description
    print("Task Description:")
    print("-" * 70)
    print("Delayed match-to-sample task testing working memory.")
    print("1. Sample stimulus shown (e.g., orientation)")
    print("2. Delay period - must hold sample in memory")
    print("3. Test stimulus shown")
    print("4. Decide: Match or Non-match?")
    print()

    # Run a few sample trials to understand structure
    print("Sample trial structure:")
    print("-" * 70)
    for i in range(3):
        env.new_trial()
        ob, gt = env.ob, env.gt
        print(f"\nTrial {i+1}:")
        print(f"  Trial length: {len(ob)} timesteps ({len(ob) * env.dt} ms)")
        print(f"  Observation shape: {ob.shape}")
        print(f"  Ground truth shape: {gt.shape}")
        print(f"  Unique actions in trial: {np.unique(gt)}")

    print()
    print("Generating task visualization...")
    plot_task_structure(env)

    print()
    print("="*70)
    print("Task exploration complete!")
    print("="*70)


if __name__ == '__main__':
    explore_task()
