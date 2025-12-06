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
from Question_2a import Net, VanillaRNN, LeakyRNN, LeakyRNNFeedbackAlignment, BiologicallyRealisticRNN

warnings.filterwarnings("ignore")


def infer_go_action_index(env, trial_data=None):
    """Infer the action index corresponding to 'Go'."""
    if hasattr(env, 'actions'):
        for idx, action_name in enumerate(getattr(env, 'actions')):
            if isinstance(action_name, str) and 'go' in action_name.lower():
                return idx

    if trial_data:
        sample_name = next(iter(trial_data.keys()))
        targets = trial_data[sample_name]['targets']
        if targets:
            concatenated = np.concatenate(targets)
            unique_vals = sorted({int(round(val)) for val in concatenated})
            positive = [val for val in unique_vals if val > 0]
            if positive:
                return positive[-1]

    if hasattr(env, 'action_space') and env.action_space.n > 0:
        return env.action_space.n - 1

    return 1


def load_models_and_config(checkpoint_path=None):
    """Load trained models and environment configuration."""
    if checkpoint_path is None:
        # Try multiple possible locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, '..', 'checkpoints', 'question_2a_models_and_data.pt'),
            os.path.join(script_dir, 'checkpoints', 'question_2a_models_and_data.pt'),
            'checkpoints/question_2a_models_and_data.pt',
            '../checkpoints/question_2a_models_and_data.pt'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find checkpoint file. Tried:\n" + 
                "\n".join(f"  - {p}" for p in possible_paths)
            )
    
    print(f"Loading models from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    env_config = checkpoint['env_config']
    task = env_config['task']
    dt = env_config['dt']

    # Setup environment
    kwargs = {'dt': dt}
    seq_len = env_config.get('seq_len', 100)
    dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=1, seq_len=seq_len)
    env = dataset.env

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # Infer hidden size from checkpoint (use vanilla model as reference)
    vanilla_state = checkpoint['vanilla_model']
    if 'rnn.h2h.weight' in vanilla_state:
        hidden_size = vanilla_state['rnn.h2h.weight'].shape[0]
    else:
        # Fallback for other architectures
        hidden_size = vanilla_state['fc.weight'].shape[1]

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
    # Infer hidden size from bio model (might have different architecture)
    bio_state = checkpoint['bio_model']
    if 'rnn.h2h.weight' in bio_state:
        bio_hidden_size = bio_state['rnn.h2h.weight'].shape[0]
    elif 'rnn.h2h_exc.weight' in bio_state:
        bio_hidden_size = bio_state['rnn.h2h_exc.weight'].shape[1] + bio_state['rnn.h2h_inh.weight'].shape[1]
    else:
        bio_hidden_size = hidden_size

    net_bio = Net(input_size, bio_hidden_size, output_size, model_type='bio_realistic',
                  dt=dt, tau=100, sigma_rec=0.15, exc_ratio=0.8)
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
                output, activity, _ = model(inputs)

                trial_data[name]['activities'].append(activity[:, 0, :].cpu().numpy())
                trial_data[name]['outputs'].append(output[:, 0, :].cpu().numpy())
                trial_data[name]['stimuli'].append(ob)
                trial_data[name]['targets'].append(gt)

    print("Data collection complete!")
    return trial_data


def plot_task_structure(env, models=None, output_path='images/readysetgo_task_structure.png',
                        go_action_idx=2):
    """Visualize the ReadySetGo task structure with one example trial."""
    fig = plt.figure(figsize=(10, 4))

    # Create single subplot
    ax_input = fig.add_subplot(111)

    # Generate one trial
    env.new_trial()
    ob, gt = env.ob, env.gt

    time = np.arange(len(ob)) * env.dt

    # Plot input channels
    for i in range(ob.shape[1]):
        ax_input.plot(time, ob[:, i] + i*1.2, linewidth=1.5, label=f'Input {i+1}')

    # Mark key events - Ready is channel 1, Set is channel 2
    ready_time = np.where(ob[:, 1] > 0.5)[0][0] * env.dt if np.any(ob[:, 1] > 0.5) else None
    set_time = np.where(ob[:, 2] > 0.5)[0][0] * env.dt if np.any(ob[:, 2] > 0.5) else None
    go_time = None
    if np.any(gt == go_action_idx):
        go_time = np.where(gt == go_action_idx)[0][0] * env.dt if np.any(gt == go_action_idx) else None

    if ready_time:
        ax_input.axvline(ready_time, color='green', linestyle='--', alpha=0.5, label='Ready')
    if set_time:
        ax_input.axvline(set_time, color='orange', linestyle='--', alpha=0.5, label='Set')
    if go_time:
        ax_input.axvline(go_time, color='red', linestyle='--', alpha=0.5, label='Target Go')

    ax_input.set_xlabel('Time (ms)', fontsize=11)
    ax_input.set_ylabel('Input Activity', fontsize=11)

    # Calculate sample interval
    if ready_time and set_time:
        sample_interval = f'{set_time - ready_time:.0f}ms'
    else:
        sample_interval = 'N/A'

    ax_input.set_title(f'ReadySetGo Task Structure (Sample Interval = {sample_interval})',
                      fontsize=13, fontweight='bold')
    ax_input.legend(loc='upper right', fontsize=9)
    ax_input.grid(True, alpha=0.3)

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
            n_time = activity.shape[0]

            # Plot trajectory
            ax.plot(activity_pca[:, 0], activity_pca[:, 1],
                   alpha=0.4, linewidth=1.5, color='black')

            # Mark start (Ready)
            ax.scatter(activity_pca[0, 0], activity_pca[0, 1],
                      s=50, color='green', marker='o', alpha=0.8, zorder=5)

            # Mark Set cue (approximate at 1/4 through trial)
            set_idx = n_time // 4
            ax.scatter(activity_pca[set_idx, 0], activity_pca[set_idx, 1],
                      s=50, color='blue', marker='D', alpha=0.8, zorder=5)

            # Mark end (Go)
            ax.scatter(activity_pca[-1, 0], activity_pca[-1, 1],
                      s=50, color='red', marker='s', alpha=0.8, zorder=5)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.set_title(f'{name} RNN', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add legend to first plot
        if idx == 0:
            ax.scatter([], [], s=50, color='green', marker='o', label='Ready')
            ax.scatter([], [], s=50, color='orange', marker='D', label='Set')
            ax.scatter([], [], s=50, color='red', marker='s', label='Go')
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
        n_time = activity.shape[0]

        # Plot heatmap
        im = ax.imshow(activity.T, aspect='auto', cmap='viridis',
                      interpolation='nearest')

        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Neuron Index', fontsize=10)
        ax.set_title(f'{name} RNN', fontsize=12, fontweight='bold')

        # Extract actual Set and Go times from trial info if available
        # trial_info is a dict with keys:
        #   'measure' = Ready→Set interval (ms)
        #   'production' = Set→Go interval (ms)
        #   'gain' = production / measure ratio
        if 'trial_info' in trial_data[name]:
            trial_info = trial_data[name]['trial_info'][trial_idx]
            measure = trial_info.get('measure', None)  # Ready→Set interval in ms
            production = trial_info.get('production', None)  # Set→Go interval in ms

            # Convert from ms to timesteps (dt=20ms)
            dt = 20
            if measure is not None:
                set_marker = int(measure / dt)  # Set cue at end of measure period
            else:
                set_marker = n_time // 4

            if measure is not None and production is not None:
                go_marker = int((measure + production) / dt)  # Go cue at measure + production
            else:
                go_marker = 3 * n_time // 4
        else:
            # Fall back to approximations if trial_info not available
            set_marker = n_time // 4
            go_marker = 3 * n_time // 4

        ax.axvline(set_marker, color='red', linestyle='--', linewidth=2, alpha=0.9, label='Set')
        ax.axvline(go_marker, color='white', linestyle='--', linewidth=2, alpha=0.9, label='Go')
        ax.legend(loc='upper right', fontsize=9)

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


def plot_timing_accuracy(trial_data, env, output_path='images/readysetgo_timing.png',
                         go_action_idx=None):
    """Analyze and plot timing accuracy across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if go_action_idx is None:
        go_action_idx = infer_go_action_index(env, trial_data)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']

    all_target_times = []
    all_errors = []

    # First pass: collect all data to determine common bin range
    model_data = {}
    for name in model_names:
        produced_times = []
        target_times = []

        for trial_idx in range(len(trial_data[name]['outputs'])):
            output = trial_data[name]['outputs'][trial_idx]
            target = np.asarray(trial_data[name]['targets'][trial_idx], dtype=int)

            produced_idx = detect_go_time(output, go_action_idx)
            produced_time = produced_idx

            target_go = np.where(target == go_action_idx)[0]
            target_time = int(target_go[0]) if len(target_go) > 0 else len(target)

            produced_times.append(produced_time * env.dt)
            target_times.append(target_time * env.dt)

        produced_times = np.array(produced_times)
        target_times = np.array(target_times)
        errors = produced_times - target_times

        model_data[name] = {
            'produced': produced_times,
            'target': target_times,
            'errors': errors
        }

        all_target_times.extend(target_times.tolist())
        all_errors.extend(errors.tolist())

    # Determine common bin edges with fixed width for all models
    all_errors = np.array(all_errors)
    error_min, error_max = all_errors.min(), all_errors.max()
    bin_width = 50  # Fixed bin width in ms for direct comparison
    bins = np.arange(error_min, error_max + bin_width, bin_width)

    # Second pass: plot with common bin width
    for idx, name in enumerate(model_names):
        produced_times = model_data[name]['produced']
        target_times = model_data[name]['target']
        errors = model_data[name]['errors']

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # Scatter plot without stats
        ax1.scatter(target_times, produced_times, alpha=0.5, s=30,
                   color=colors[idx], label=name)

        # Error distribution with stats in legend using common bin edges
        ax2.hist(errors, bins=bins, alpha=0.5, color=colors[idx],
                label=f'{name} (μ={mean_error:.1f}ms, σ={std_error:.1f}ms)')

    # Ideal line
    max_time = max(all_target_times) * 1.1 if all_target_times else env.dt * trial_data[model_names[0]]['outputs'][0].shape[0]
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


def compute_performance_metrics(trial_data, env, timing_tolerance=40.0, sparsity_threshold=0.05,
                                go_action_idx=None):
    """Compute quantitative metrics for table in the report."""
    dt = env.dt
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    metrics = {}

    if go_action_idx is None:
        go_action_idx = infer_go_action_index(env, trial_data)

    for name in model_names:
        produced_times = []
        target_times = []

        for trial_idx in range(len(trial_data[name]['outputs'])):
            output = trial_data[name]['outputs'][trial_idx]
            target = np.asarray(trial_data[name]['targets'][trial_idx], dtype=int)

            produced_idx = detect_go_time(output, go_action_idx)
            target_go = np.where(target == go_action_idx)[0]
            target_time = int(target_go[0]) if len(target_go) > 0 else len(target)

            produced_times.append(produced_idx * dt)
            target_times.append(target_time * dt)

        produced_times = np.array(produced_times)
        target_times = np.array(target_times)
        errors = produced_times - target_times

        # Accuracy = fraction of trials with timing error within tolerance (ms)
        accuracy = np.mean(np.abs(errors) <= timing_tolerance)
        mean_abs_error = np.mean(np.abs(errors))

        # Activity statistics
        all_activities = np.concatenate(trial_data[name]['activities'], axis=0)
        mean_activity = np.mean(np.abs(all_activities))
        sparsity = np.mean(np.abs(all_activities) < sparsity_threshold)

        metrics[name] = {
            'accuracy': float(accuracy),
            'mean_abs_timing_error': float(mean_abs_error),
            'mean_signed_error': float(np.mean(errors)),
            'std_timing_error': float(np.std(errors)),
            'mean_activity': float(mean_activity),
            'sparsity': float(sparsity)
        }

    return metrics


def print_performance_metrics(metrics):
    """Pretty-print metrics for quick copy into the LaTeX table."""
    print("\nPerformance summary (timing accuracy within ±40 ms):")
    header = "{:<15s} {:>10s} {:>12s} {:>12s} {:>12s} {:>15s} {:>12s}"
    row = "{:<15s} {:>10.3f} {:>12.2f} {:>12.2f} {:>12.2f} {:>15.4f} {:>12.3f}"
    print(header.format("Model", "Accuracy", "|Error|", "Error μ", "Error σ",
                        "Mean Activity", "Sparsity"))
    for name, stats in metrics.items():
        print(row.format(
            name,
            stats['accuracy'],
            stats['mean_abs_timing_error'],
            stats['mean_signed_error'],
            stats['std_timing_error'],
            stats['mean_activity'],
            stats['sparsity']))


def detect_go_time(logits, go_action_idx):
    """Return earliest timestep where Go is the predicted action."""
    logits = np.asarray(logits)
    preds = np.argmax(logits, axis=1)

    first = np.where(preds == go_action_idx)[0]
    if len(first) > 0:
        return int(first[0])

    go_channel = logits[:, go_action_idx]
    return int(np.argmax(go_channel))


def inspect_go_predictions(trial_data, go_action_idx, num_trials=3):
    """Print detection diagnostics for first few trials."""
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    print("\nGo detection diagnostics (argmax-based):")
    for trial_idx in range(min(num_trials, len(trial_data[model_names[0]]['targets']))):
        target = np.asarray(trial_data[model_names[0]]['targets'][trial_idx], dtype=int)
        target_go = np.where(target == go_action_idx)[0]
        target_time = int(target_go[0]) if len(target_go) > 0 else None
        print(f"\nTrial {trial_idx+1}: target Go index = {target_time}")
        for name in model_names:
            logits = trial_data[name]['outputs'][trial_idx]
            preds = np.argmax(logits, axis=1)
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            go_probs = probs[:, go_action_idx]
            det = detect_go_time(logits, go_action_idx)
            first_pred = np.where(preds == go_action_idx)[0]
            print(f"  {name:12s} detect={det:3d} first_pred={first_pred[0] if len(first_pred)>0 else 'none'} "
                  f"max_prob={go_probs.max():.2f}")


def compute_pca_statistics(trial_data, num_trials=20):
    """Return explained variance for first two PCs."""
    stats = {}
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    for name in model_names:
        activities = np.concatenate(trial_data[name]['activities'][:num_trials], axis=0)
        pca = PCA(n_components=2)
        pca.fit(activities)
        stats[name] = {
            'pc1_var': float(pca.explained_variance_ratio_[0]),
            'pc2_var': float(pca.explained_variance_ratio_[1])
        }
    return stats


def compute_ramp_slopes(trial_data, num_trials=50):
    """Compute slope of average population activity."""
    slopes = {}
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    for name in model_names:
        trial_slopes = []
        for trial_idx in range(min(num_trials, len(trial_data[name]['activities']))):
            activity = trial_data[name]['activities'][trial_idx]
            mean_trace = np.mean(activity, axis=1)
            time = np.linspace(0, 1, len(mean_trace))
            coeffs = np.polyfit(time, mean_trace, 1)
            trial_slopes.append(coeffs[0])
        slopes[name] = {
            'mean_slope': float(np.mean(trial_slopes)),
            'std_slope': float(np.std(trial_slopes))
        }
    return slopes


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
    go_action_idx = infer_go_action_index(env, trial_data)
    print(f"Detected Go action index: {go_action_idx}")
    actions = getattr(env, 'actions', None)
    if actions is not None:
        print(f"Action labels: {actions}")

    plot_task_structure(env, models=None, go_action_idx=go_action_idx)
    plot_neural_trajectories(trial_data)

    # Use checkpoint data for heatmap (has trial_info with actual times)
    checkpoint_trial_data = {
        'Vanilla': checkpoint['vanilla_data'],
        'Leaky': checkpoint['leaky_data'],
        'Leaky+FA': checkpoint['leaky_fa_data'],
        'Bio-Realistic': checkpoint['bio_data']
    }
    plot_activity_heatmaps(checkpoint_trial_data, trial_idx=5)

    plot_ramping_activity(trial_data)
    plot_timing_accuracy(trial_data, env, go_action_idx=go_action_idx)
    plot_firing_rate_comparison(trial_data)

    # Summary metrics for coursework_report.tex tables
    metrics = compute_performance_metrics(trial_data, env, go_action_idx=go_action_idx)
    print_performance_metrics(metrics)
    inspect_go_predictions(trial_data, go_action_idx)

    pca_stats = compute_pca_statistics(trial_data)
    ramp_stats = compute_ramp_slopes(trial_data)

    print("\nPCA explained variance (PC1/PC2):")
    for name, stats in pca_stats.items():
        print(f"{name:<15s} {stats['pc1_var']*100:5.1f}% / {stats['pc2_var']*100:5.1f}%")

    print("\nMean ramp slopes (activity vs normalised time):")
    for name, stats in ramp_stats.items():
        print(f"{name:<15s} slope={stats['mean_slope']:.4f} ± {stats['std_slope']:.4f}")

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
