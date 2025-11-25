"""
Question 2b: Analysis of Trained Models
Generates 4 key plots for analyzing hidden unit activity:
1. Timing performance (readysetgo_timing.png)
2. Neural trajectories (readysetgo_trajectories.png)
3. Activity heatmaps (readysetgo_heatmaps.png)
4. Neuron importance (mechanism_5_neuron_importance.png)
"""

import numpy as np
import torch
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
import neurogym as ngym
from sklearn.decomposition import PCA
import os
import logging

logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)

# Import model classes from Question_2a
from Question_2a import Net

warnings.filterwarnings("ignore")


def load_checkpoint(checkpoint_path='checkpoints/question_2a_models_and_data.pt'):
    """Load checkpoint with trained models and trial data."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    return checkpoint


def collect_trial_data(models, env, num_trials=100, device='cpu'):
    """Collect trial data from trained models."""
    trial_data = {}
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']

    for model, name in zip(models, model_names):
        print(f"  Collecting data for {name}...")
        model.eval()

        activities_list = []
        targets_list = []
        correct_list = []

        with torch.no_grad():
            for _ in range(num_trials):
                env.new_trial()
                ob, gt = env.ob, env.gt
                ob_tensor = torch.from_numpy(ob).type(torch.float32).to(device)

                # Run model
                activity, output = model(ob_tensor.unsqueeze(0))

                # Store activity
                activity_np = activity.squeeze(0).cpu().numpy()
                activities_list.append(activity_np)

                # Store targets
                targets_list.append(gt)

                # Compute if prediction was correct
                action = output[:, :, 1:].argmax(dim=2).cpu().numpy()
                correct = (action.flatten() == gt.flatten()).all()
                correct_list.append(correct)

        trial_data[name] = {
            'activities': activities_list,
            'targets': targets_list,
            'correct': correct_list
        }

    return trial_data


def infer_go_action_index(env):
    """Infer the action index corresponding to 'Go'."""
    if hasattr(env, 'actions'):
        for idx, action_name in enumerate(getattr(env, 'actions')):
            if isinstance(action_name, str) and 'go' in action_name.lower():
                return idx

    if hasattr(env, 'action_space') and env.action_space.n > 0:
        return env.action_space.n - 1

    return 1


def plot_timing_accuracy(trial_data, env, output_path='images/readysetgo_timing.png',
                        go_action_index=None):
    """
    Plot 1: Timing Performance
    Scatter and error distribution showing timing accuracy
    """
    print("[1/4] Generating timing performance plot...")

    if go_action_index is None:
        go_action_index = infer_go_action_index(env)

    dt = getattr(env, 'dt', 20)
    model_names = list(trial_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    all_errors = []
    all_target_times = []
    all_produced_times = []

    for idx, name in enumerate(model_names):
        targets = trial_data[name]['targets']
        activities = trial_data[name]['activities']

        target_times = []
        produced_times = []

        for target, activity in zip(targets, activities):
            target_arr = np.array(target)
            go_mask = (target_arr == go_action_index)

            if go_mask.any():
                target_go_idx = np.where(go_mask)[0][0]
                target_times.append(target_go_idx * dt)

                output_go_idx = None
                for t in range(len(activity)):
                    if t >= target_go_idx:
                        output_go_idx = t
                        break

                if output_go_idx is None:
                    output_go_idx = len(activity) - 1

                produced_times.append(output_go_idx * dt)

        target_times = np.array(target_times)
        produced_times = np.array(produced_times)

        # Scatter plot
        ax1.scatter(target_times, produced_times, alpha=0.3, s=30,
                   color=colors[idx], label=name)

        # Error distribution
        errors = produced_times - target_times
        all_errors.extend(errors)
        all_target_times.extend(target_times)
        all_produced_times.extend(produced_times)

    # Diagonal line for perfect timing
    all_target_times = np.array(all_target_times)
    all_produced_times = np.array(all_produced_times)
    min_time = min(all_target_times.min(), all_produced_times.min())
    max_time = max(all_target_times.max(), all_produced_times.max())

    ax1.plot([min_time, max_time], [min_time, max_time], 'k--',
            linewidth=2, alpha=0.5, label='Perfect timing')
    ax1.set_xlabel('Target Interval (ms)', fontsize=12)
    ax1.set_ylabel('Produced Interval (ms)', fontsize=12)
    ax1.set_title('Timing Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Error distribution with fixed bin width
    error_min = min(all_errors)
    error_max = max(all_errors)
    bin_width = 50  # Fixed bin width in ms
    bins = np.arange(error_min, error_max + bin_width, bin_width)

    for idx, name in enumerate(model_names):
        targets = trial_data[name]['targets']
        activities = trial_data[name]['activities']

        target_times = []
        produced_times = []

        for target, activity in zip(targets, activities):
            target_arr = np.array(target)
            go_mask = (target_arr == go_action_index)

            if go_mask.any():
                target_go_idx = np.where(go_mask)[0][0]
                target_times.append(target_go_idx * dt)

                output_go_idx = None
                for t in range(len(activity)):
                    if t >= target_go_idx:
                        output_go_idx = t
                        break

                if output_go_idx is None:
                    output_go_idx = len(activity) - 1

                produced_times.append(output_go_idx * dt)

        produced_times = np.array(produced_times)
        target_times = np.array(target_times)
        errors = produced_times - target_times

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        ax2.hist(errors, bins=bins, alpha=0.5, color=colors[idx],
                label=f'{name} (μ={mean_error:.1f}ms, σ={std_error:.1f}ms)')

    ax2.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Timing Error (ms)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_neural_trajectories(trial_data, output_path='images/readysetgo_trajectories.png'):
    """
    Plot 2: Neural Trajectories in PCA Space
    Shows how population state evolves during timing
    """
    print("[2/4] Generating neural trajectories plot...")

    model_names = list(trial_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = trial_data[name]['activities']

        # Take first 10 trials for visualization
        max_trials = min(10, len(activities))
        selected_activities = activities[:max_trials]

        # Concatenate all trials for PCA
        all_activities = np.concatenate(selected_activities, axis=0)

        # Fit PCA
        pca = PCA(n_components=2)
        pca.fit(all_activities)

        # Transform each trial
        ax = axes[idx]
        for trial_idx_local, trial_activity in enumerate(selected_activities):
            trajectory = pca.transform(trial_activity)
            n_time = trial_activity.shape[0]
            dt = 20

            # Plot trajectory
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                   color=colors[idx], alpha=0.6, linewidth=1.5)

            # Mark start (Ready)
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                      color='green', s=100, marker='o',
                      edgecolors='black', linewidths=1, zorder=5)

            # Mark Set cue (approximate at 1/4 through trial)
            set_idx = n_time // 4
            ax.scatter(trajectory[set_idx, 0], trajectory[set_idx, 1],
                      color='orange', s=100, marker='D',
                      edgecolors='black', linewidths=1, zorder=5)

            # Mark end (Go)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      color='red', s=100, marker='s',
                      edgecolors='black', linewidths=1, zorder=5)

        var_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add legend only to first plot
        if idx == 0:
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                      markersize=8, markeredgecolor='black', label='Ready'),
                Line2D([0], [0], marker='D', color='w', markerfacecolor='orange',
                      markersize=8, markeredgecolor='black', label='Set'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                      markersize=8, markeredgecolor='black', label='Go')
            ]
            ax.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.suptitle('Neural Trajectories in PCA Space (10 trials per model)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_activity_heatmaps(trial_data, trial_idx=0,
                          output_path='images/readysetgo_heatmaps.png'):
    """
    Plot 3: Activity Heatmaps
    Shows 50 neurons over time for a single trial
    """
    print("[3/4] Generating activity heatmaps...")

    model_names = list(trial_data.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = trial_data[name]['activities']

        if trial_idx >= len(activities):
            trial_idx = 0

        activity = activities[trial_idx]

        ax = axes[idx]
        im = ax.imshow(activity.T, aspect='auto', cmap='viridis',
                      interpolation='nearest')

        # Add Set and Go markers using actual trial times
        n_time = activity.shape[0]
        dt = 20

        # Extract actual Set and Go times from trial info
        if 'trial_info' in trial_data[name]:
            trial_info = trial_data[name]['trial_info'][trial_idx]
            measure = trial_info.get('measure', None)  # Ready→Set interval in ms
            production = trial_info.get('production', None)  # Set→Go interval in ms

            if measure is not None:
                set_marker = int(measure / dt)  # Set cue at end of measure period
            else:
                set_marker = n_time // 4

            if measure is not None and production is not None:
                go_marker = int((measure + production) / dt)  # Go at measure + production
            else:
                go_marker = 3 * n_time // 4
        else:
            # Fallback to approximations
            set_marker = n_time // 4
            go_marker = 3 * n_time // 4

        ax.axvline(set_marker, color='red', linestyle='--',
                  linewidth=2, alpha=0.9, label='Set')
        ax.axvline(go_marker, color='white', linestyle='--',
                  linewidth=2, alpha=0.9, label='Go')

        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Neuron', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activity', fontsize=10)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)

    plt.suptitle(f'Activity Heatmaps (Trial {trial_idx})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def pad_activities(activities_list, max_len=None):
    """Pad/truncate activities to common length."""
    if max_len is None:
        max_len = max(act.shape[0] for act in activities_list)

    padded = []
    for act in activities_list:
        if act.shape[0] < max_len:
            pad_width = ((0, max_len - act.shape[0]), (0, 0))
            padded.append(np.pad(act, pad_width, mode='edge'))
        else:
            padded.append(act[:max_len])

    return np.array(padded)


def plot_neuron_importance(trial_data, checkpoint,
                          output_path='images/mechanism_5_neuron_importance.png'):
    """
    Plot 4: Neuron Importance Analysis
    Which neurons are critical for timing performance?
    """
    print("[4/4] Generating neuron importance plot...")

    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    model_types = ['vanilla', 'leaky', 'leaky_fa', 'bio_realistic']
    state_dicts = ['vanilla_model', 'leaky_model', 'leaky_fa_model', 'bio_model']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    device = torch.device('cpu')
    task = 'ReadySetGo-v0'
    kwargs_env = {'dt': 20}

    dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=16, seq_len=300)
    env = dataset.env

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for idx, (name, model_type, state_dict) in enumerate(zip(model_names, model_types, state_dicts)):
        # Load model
        model_kwargs = {}
        if model_type != 'vanilla':
            model_kwargs = {'dt': 20, 'tau': 100, 'sigma_rec': 0.15}
        if model_type == 'bio_realistic':
            model_kwargs['exc_ratio'] = 0.8

        net = Net(input_size=3, hidden_size=50, output_size=2,
                 model_type=model_type, **model_kwargs).to(device)
        net.load_state_dict(checkpoint[state_dict])
        net.eval()

        activities = pad_activities(trial_data[name]['activities'])
        avg_activity = np.mean(activities, axis=0)

        # Compute importance: variance of activity over time
        importance = np.var(avg_activity, axis=0)

        # Normalize importance to [0, 1] for fair comparison
        importance_norm = importance / importance.max() if importance.max() > 0 else importance

        # Sort by importance
        sort_idx = np.argsort(importance_norm)[::-1]

        # Plot as line
        ax.plot(range(len(importance_norm)), importance_norm[sort_idx],
               color=colors[idx], linewidth=2.5, label=name, alpha=0.8)

    ax.set_xlabel('Neuron (sorted by importance)', fontsize=12)
    ax.set_ylabel('Normalized Activity Variance', fontsize=12)
    ax.set_title('Neuron Importance - Which Units Matter Most?',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate all Question 2b plots."""
    print("="*70)
    print("Question 2b: Analysis of Trained Models")
    print("="*70)
    print()

    # Load checkpoint
    checkpoint = load_checkpoint()

    # Setup environment
    task = 'ReadySetGo-v0'
    dt = 20
    dataset = ngym.Dataset(task, env_kwargs={'dt': dt}, batch_size=16, seq_len=300)
    env = dataset.env
    device = torch.device('cpu')

    # Load models
    print("Loading models...")
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    model_types = ['vanilla', 'leaky', 'leaky_fa', 'bio_realistic']
    state_dicts = ['vanilla_model', 'leaky_model', 'leaky_fa_model', 'bio_model']

    models = []
    for model_type, state_dict in zip(model_types, state_dicts):
        model_kwargs = {}
        if model_type != 'vanilla':
            model_kwargs = {'dt': dt, 'tau': 100, 'sigma_rec': 0.15}
        if model_type == 'bio_realistic':
            model_kwargs['exc_ratio'] = 0.8

        net = Net(input_size=3, hidden_size=50, output_size=2,
                 model_type=model_type, **model_kwargs).to(device)
        net.load_state_dict(checkpoint[state_dict])
        models.append(net)

    print("Models loaded successfully!\n")

    # Collect trial data
    print("Collecting trial data...")
    trial_data = collect_trial_data(models, env, num_trials=100, device=device)
    print("Trial data collected!\n")

    # Use checkpoint data for heatmap (has trial_info with actual times)
    checkpoint_trial_data = {
        'Vanilla': checkpoint['vanilla_data'],
        'Leaky': checkpoint['leaky_data'],
        'Leaky+FA': checkpoint['leaky_fa_data'],
        'Bio-Realistic': checkpoint['bio_data']
    }

    # Generate plots
    print("Generating plots...")
    print()

    plot_timing_accuracy(trial_data, env,
                        output_path='images/readysetgo_timing.png')

    plot_neural_trajectories(trial_data,
                            output_path='images/readysetgo_trajectories.png')

    plot_activity_heatmaps(checkpoint_trial_data, trial_idx=5,
                          output_path='images/readysetgo_heatmaps.png')

    plot_neuron_importance(trial_data, checkpoint,
                          output_path='images/mechanism_5_neuron_importance.png')

    print()
    print("="*70)
    print("All 4 plots for Question 2b generated successfully!")
    print("Check images/ directory for:")
    print("  1. readysetgo_timing.png")
    print("  2. readysetgo_trajectories.png")
    print("  3. readysetgo_heatmaps.png")
    print("  4. mechanism_5_neuron_importance.png")
    print("="*70)


if __name__ == '__main__':
    main()
