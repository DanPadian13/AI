import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import neurogym as ngym
from sklearn.decomposition import PCA
import os
import logging

logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)

# Import model classes from Question_2a
from Question_2a import Net, VanillaRNN, LeakyRNN, LeakyRNNFeedbackAlignment, BiologicallyRealisticRNN


def load_models_and_config(checkpoint_path='checkpoints/question_2a_models_and_data.pt'):
    """Load trained models and environment configuration."""
    print(f"Loading models from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    env_config = checkpoint['env_config']
    task = env_config['task']
    dt = env_config['dt']

    # Setup environment
    kwargs = {'dt': dt}
    seq_len = 100
    dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=1, seq_len=seq_len)
    env = dataset.env

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = 50

    # Initialize models
    models = {}

    # Vanilla RNN
    net_vanilla = Net(input_size, hidden_size, output_size, model_type='vanilla')
    net_vanilla.load_state_dict(checkpoint['vanilla_model'])
    net_vanilla.eval()
    models['Vanilla'] = net_vanilla

    # Leaky RNN
    net_leaky = Net(input_size, hidden_size, output_size, model_type='leaky',
                    dt=dt, tau=100, sigma_rec=0.15)
    net_leaky.load_state_dict(checkpoint['leaky_model'])
    net_leaky.eval()
    models['Leaky'] = net_leaky

    # Leaky + FA
    net_leaky_fa = Net(input_size, hidden_size, output_size, model_type='leaky_fa',
                       dt=dt, tau=100, sigma_rec=0.15)
    net_leaky_fa.load_state_dict(checkpoint['leaky_fa_model'])
    net_leaky_fa.eval()
    models['Leaky+FA'] = net_leaky_fa

    # Bio-Realistic
    net_bio = Net(input_size, hidden_size, output_size, model_type='bio_realistic',
                  dt=dt, tau=100, sigma_rec=0.15, exc_ratio=0.8, sparsity=0.2)
    net_bio.load_state_dict(checkpoint['bio_model'])
    net_bio.eval()
    models['Bio-Realistic'] = net_bio

    return models, env, checkpoint


def collect_trial_data(models, env, num_trials=100):
    """Collect neural activities and timing data from all models."""
    trial_data = {name: {'activities': [], 'outputs': [], 'stimuli': [], 'targets': []}
                  for name in models.keys()}

    print(f"\nCollecting data from {num_trials} trials...")

    for trial_idx in range(num_trials):
        env.new_trial()
        ob, gt = env.ob, env.gt

        inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)

        for name, model in models.items():
            with torch.no_grad():
                output, activity = model(inputs)

                trial_data[name]['activities'].append(activity[:, 0, :].cpu().numpy())
                trial_data[name]['outputs'].append(output[:, 0, :].cpu().numpy())
                trial_data[name]['stimuli'].append(ob)
                trial_data[name]['targets'].append(gt)

    print("Data collection complete!")
    return trial_data


def plot_task_structure(env, models=None, output_path='images/readysetgo_task_structure.png'):
    """Visualize the ReadySetGo task structure with example trials and model predictions."""
    fig = plt.figure(figsize=(14, 10))

    if models is None:
        # Original layout without predictions
        gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
        n_rows = 3
    else:
        # Layout with predictions
        gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        n_rows = 3

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']

    # Generate 3 example trials
    for idx, trial_idx in enumerate([0, 50, 100]):
        # Left column: Task structure
        if models is None:
            ax_input = fig.add_subplot(gs[idx])
        else:
            ax_input = fig.add_subplot(gs[idx, 0])

        env.new_trial()
        ob, gt = env.ob, env.gt

        time = np.arange(len(ob)) * env.dt

        # Plot input channels
        for i in range(ob.shape[1]):
            ax_input.plot(time, ob[:, i] + i*1.2, linewidth=1.5, label=f'Input {i+1}')

        # Mark key events
        ready_time = np.where(ob[:, 0] > 0)[0][0] * env.dt if np.any(ob[:, 0] > 0) else None
        set_time = np.where(ob[:, 1] > 0)[0][0] * env.dt if np.any(ob[:, 1] > 0) else None
        go_time = np.where(gt > 0)[0][0] * env.dt if np.any(gt > 0) else None

        if ready_time:
            ax_input.axvline(ready_time, color='green', linestyle='--', alpha=0.5, label='Ready' if idx == 0 else '')
        if set_time:
            ax_input.axvline(set_time, color='orange', linestyle='--', alpha=0.5, label='Set' if idx == 0 else '')
        if go_time:
            ax_input.axvline(go_time, color='red', linestyle='--', alpha=0.5, label='Target Go' if idx == 0 else '')

        ax_input.set_xlabel('Time (ms)', fontsize=10)
        ax_input.set_ylabel('Input Activity', fontsize=10)

        # Calculate sample interval
        if ready_time and set_time:
            sample_interval = f'{set_time - ready_time:.0f}ms'
        else:
            sample_interval = 'N/A'

        ax_input.set_title(f'Trial {idx+1}: Inputs (Interval = {sample_interval})',
                          fontsize=11, fontweight='bold')
        if idx == 0:
            ax_input.legend(loc='upper right', fontsize=7)
        ax_input.grid(True, alpha=0.3)

        # Right column: Model predictions
        if models is not None:
            ax_pred = fig.add_subplot(gs[idx, 1])

            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)

            # Plot target
            ax_pred.plot(time, gt, 'k-', linewidth=2.5, label='Target', alpha=0.7)

            # Plot each model's predictions
            for model_idx, (name, model) in enumerate(models.items()):
                with torch.no_grad():
                    output, _ = model(inputs)
                    predictions = torch.argmax(output[:, 0, :], dim=1).cpu().numpy()

                ax_pred.plot(time, predictions, linewidth=2, color=colors[model_idx],
                           label=name, alpha=0.8)

            # Mark key events
            if ready_time:
                ax_pred.axvline(ready_time, color='green', linestyle='--', alpha=0.3)
            if set_time:
                ax_pred.axvline(set_time, color='orange', linestyle='--', alpha=0.3, linewidth=2)
                ax_pred.text(set_time, 1.9, 'Set', fontsize=8, ha='center')
            if go_time:
                ax_pred.axvline(go_time, color='red', linestyle='--', alpha=0.3, linewidth=2)
                ax_pred.text(go_time, 1.9, 'Target Go', fontsize=8, ha='center')

            # Zoom in very close to Target Go time
            if go_time:
                window = 200  # Show ±200ms around Go time
                ax_pred.set_xlim([go_time - window, go_time + window])

            ax_pred.set_xlabel('Time (ms)', fontsize=10)
            ax_pred.set_ylabel('Action (0=fixate, 1+=go)', fontsize=10)
            ax_pred.set_title(f'Trial {idx+1}: Predictions (±200ms around Target Go)',
                            fontsize=11, fontweight='bold')
            if idx == 0:
                ax_pred.legend(loc='upper left', fontsize=7)
            ax_pred.grid(True, alpha=0.3)
            ax_pred.set_ylim([-0.1, 2.1])

    if models is None:
        plt.suptitle('ReadySetGo Task Structure', fontsize=14, fontweight='bold', y=0.995)
    else:
        plt.suptitle('ReadySetGo Task: Inputs and Model Predictions', fontsize=14, fontweight='bold', y=0.995)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_neural_trajectories(trial_data, output_path='images/readysetgo_trajectories.png'):
    """Plot neural trajectories in PCA space for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']

    for idx, name in enumerate(model_names):
        ax = axes[idx]

        # Concatenate all trials for PCA
        all_activities = np.concatenate(trial_data[name]['activities'][:20], axis=0)

        # Fit PCA
        pca = PCA(n_components=3)
        pca_activities = pca.fit_transform(all_activities)

        # Plot first 10 trials
        for trial_idx in range(10):
            activity = trial_data[name]['activities'][trial_idx]
            activity_pca = pca.transform(activity)

            # Plot trajectory
            ax.plot(activity_pca[:, 0], activity_pca[:, 1],
                   alpha=0.6, linewidth=1.5, color=colors[idx])

            # Mark start
            ax.scatter(activity_pca[0, 0], activity_pca[0, 1],
                      s=50, color='green', marker='o', alpha=0.8, zorder=5)

            # Mark end
            ax.scatter(activity_pca[-1, 0], activity_pca[-1, 1],
                      s=50, color='red', marker='s', alpha=0.8, zorder=5)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.set_title(f'{name} RNN', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add legend to first plot
        if idx == 0:
            ax.scatter([], [], s=50, color='green', marker='o', label='Start (Ready)')
            ax.scatter([], [], s=50, color='red', marker='s', label='End (Go)')
            ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('Neural Trajectories in PCA Space (10 trials)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_activity_heatmaps(trial_data, trial_idx=0,
                           output_path='images/readysetgo_heatmaps.png'):
    """Plot heatmaps of neural activity over time for a single trial."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']

    for idx, name in enumerate(model_names):
        ax = axes[idx]

        activity = trial_data[name]['activities'][trial_idx]

        # Plot heatmap
        im = ax.imshow(activity.T, aspect='auto', cmap='viridis',
                      interpolation='nearest')

        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Neuron Index', fontsize=10)
        ax.set_title(f'{name} RNN', fontsize=12, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Activity')

    plt.suptitle(f'Neural Activity Heatmaps (Trial {trial_idx+1})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_ramping_activity(trial_data, output_path='images/readysetgo_ramping.png'):
    """Plot average ramping activity during Set->Go interval."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']

    for idx, name in enumerate(model_names):
        # Find minimum trial length
        min_length = min(len(trial_data[name]['activities'][i])
                        for i in range(min(50, len(trial_data[name]['activities']))))

        # Average activity across neurons and trials
        avg_activities = []
        for trial_idx in range(min(50, len(trial_data[name]['activities']))):
            activity = trial_data[name]['activities'][trial_idx][:min_length]  # Truncate to min length
            avg_activity = np.mean(activity, axis=1)
            avg_activities.append(avg_activity)

        avg_activities = np.array(avg_activities)
        mean_activity = np.mean(avg_activities, axis=0)
        std_activity = np.std(avg_activities, axis=0)

        time_steps = np.arange(len(mean_activity))

        ax.plot(time_steps, mean_activity, linewidth=2.5, color=colors[idx], label=name)
        ax.fill_between(time_steps, mean_activity - std_activity,
                       mean_activity + std_activity, alpha=0.2, color=colors[idx])

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Average Neural Activity (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Average Neural Activity Over Time (50 trials)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_timing_accuracy(trial_data, env, output_path='images/readysetgo_timing.png'):
    """Analyze and plot timing accuracy across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']

    for idx, name in enumerate(model_names):
        produced_times = []
        target_times = []

        for trial_idx in range(len(trial_data[name]['outputs'])):
            output = trial_data[name]['outputs'][trial_idx]
            target = trial_data[name]['targets'][trial_idx]

            # Find when model produces Go response
            predictions = np.argmax(output, axis=1)
            go_pred = np.where(predictions > 0)[0]
            produced_time = go_pred[0] if len(go_pred) > 0 else len(output)

            # Find target Go time
            target_go = np.where(target > 0)[0]
            target_time = target_go[0] if len(target_go) > 0 else len(target)

            produced_times.append(produced_time * env.dt)
            target_times.append(target_time * env.dt)

        produced_times = np.array(produced_times)
        target_times = np.array(target_times)

        # Scatter plot
        ax1.scatter(target_times, produced_times, alpha=0.5, s=30,
                   color=colors[idx], label=name)

        # Error distribution
        errors = produced_times - target_times
        ax2.hist(errors, bins=20, alpha=0.5, color=colors[idx], label=name)

    # Ideal line
    max_time = max([max(trial_data[name]['targets'][i]) for name in model_names
                    for i in range(len(trial_data[name]['targets']))]) * env.dt * 1.1
    ax1.plot([0, max_time], [0, max_time], 'k--', linewidth=2, label='Perfect timing')

    ax1.set_xlabel('Target Time (ms)', fontsize=11)
    ax1.set_ylabel('Produced Time (ms)', fontsize=11)
    ax1.set_title('Timing Accuracy', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Timing Error (ms)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_firing_rate_comparison(trial_data, output_path='images/readysetgo_firing_rates.png'):
    """Compare firing rates across models (effect of L2 regularization)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']

    mean_rates = []
    max_rates = []

    for name in model_names:
        all_activities = np.concatenate(trial_data[name]['activities'][:50], axis=0)
        mean_rate = np.mean(all_activities)
        max_rate = np.max(np.mean(all_activities, axis=0))

        mean_rates.append(mean_rate)
        max_rates.append(max_rate)

    # Bar plot of mean firing rates
    ax1.bar(model_names, mean_rates, color=colors, alpha=0.8)
    ax1.set_ylabel('Mean Firing Rate', fontsize=11)
    ax1.set_title('Average Neural Activity', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    # Bar plot of max firing rates
    ax2.bar(model_names, max_rates, color=colors, alpha=0.8)
    ax2.set_ylabel('Max Firing Rate', fontsize=11)
    ax2.set_title('Peak Neural Activity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    plt.suptitle('Firing Rate Comparison (Effect of L2 Regularization)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("ReadySetGo Task Analysis")
    print("="*70)

    # Load models
    models, env, checkpoint = load_models_and_config()

    # Collect trial data
    trial_data = collect_trial_data(models, env, num_trials=100)

    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    # Generate all plots
    plot_task_structure(env, models)
    plot_neural_trajectories(trial_data)
    plot_activity_heatmaps(trial_data, trial_idx=5)
    plot_ramping_activity(trial_data)
    plot_timing_accuracy(trial_data, env)
    plot_firing_rate_comparison(trial_data)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("1. images/readysetgo_task_structure.png - Task structure examples")
    print("2. images/readysetgo_trajectories.png - Neural trajectories in PCA space")
    print("3. images/readysetgo_heatmaps.png - Activity heatmaps")
    print("4. images/readysetgo_ramping.png - Ramping activity patterns")
    print("5. images/readysetgo_timing.png - Timing accuracy analysis")
    print("6. images/readysetgo_firing_rates.png - Firing rate comparison")
    print("="*70)
