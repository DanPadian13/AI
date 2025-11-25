import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load trained models and trial data."""
    checkpoint = torch.load('checkpoints/question_2a_models_and_data.pt', weights_only=False)
    return checkpoint


def pad_activities(activities, max_len=None):
    """Pad activities to common length."""
    if max_len is None:
        max_len = max(len(a) for a in activities)

    padded = []
    for act in activities:
        if len(act) < max_len:
            pad_width = ((0, max_len - len(act)), (0, 0))
            padded.append(np.pad(act, pad_width, mode='edge'))
        else:
            padded.append(act[:max_len])
    return np.array(padded)


def extract_set_intervals(trial_data):
    """Extract set period intervals from trial info."""
    intervals = []
    for info in trial_data['trial_info']:
        if hasattr(info, 'set_period'):
            intervals.append(info.set_period)
        else:
            intervals.append(50)  # Default
    return np.array(intervals)


def plot_1_time_cell_analysis(trial_data, output_path='images/mechanism_1_time_cells.png'):
    """
    Plot 1: Time Cell Analysis
    Show neurons tuned to specific time points (sequential activation).
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = pad_activities(trial_data[name]['activities'])
        avg_activity = np.mean(activities, axis=0)  # (time, neurons)

        # Find peak time for each neuron
        peak_times = np.argmax(avg_activity, axis=0)
        peak_amplitudes = np.max(avg_activity, axis=0)

        # Sort neurons by their peak time
        sort_idx = np.argsort(peak_times)

        # Only show neurons with meaningful activity (peak > threshold)
        threshold = np.percentile(peak_amplitudes, 50)
        active_neurons = peak_amplitudes > threshold
        sort_idx = sort_idx[active_neurons[sort_idx]]

        # Plot time cells heatmap
        im = axes[idx].imshow(avg_activity[:, sort_idx].T, aspect='auto',
                             cmap='hot', interpolation='nearest')

        axes[idx].set_xlabel('Time Step', fontsize=11)
        axes[idx].set_ylabel('Neuron (sorted by peak time)', fontsize=11)
        axes[idx].set_title(f'{name}\n{len(sort_idx)} active neurons',
                           fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[idx], label='Activity')

        # Add task phase markers (approximate)
        n_time = avg_activity.shape[0]
        axes[idx].axvline(n_time//4, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='Set')
        axes[idx].axvline(3*n_time//4, color='lime', linestyle='--', linewidth=2, alpha=0.7, label='Go')
        axes[idx].legend(loc='upper right', fontsize=9)

    plt.suptitle('Plot 1: Time Cell Organization - Sequential Neural Activation',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_2_timing_info_readout(trial_data, output_path='images/mechanism_2_timing_readout.png'):
    """
    Plot 2: Timing Information Readout Over Time
    When can we decode the target Go time from neural activity?
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, name in enumerate(model_names):
        activities = pad_activities(trial_data[name]['activities'])
        set_intervals = extract_set_intervals(trial_data[name])

        n_trials, n_time, n_neurons = activities.shape

        # Train decoder at each timepoint
        r2_scores = []
        n_train = int(0.8 * n_trials)

        for t in range(0, n_time, 3):  # Every 3 steps for speed
            X_train = activities[:n_train, t, :]
            y_train = set_intervals[:n_train]
            X_test = activities[n_train:, t, :]
            y_test = set_intervals[n_train:]

            decoder = Ridge(alpha=1.0)
            decoder.fit(X_train, y_train)
            y_pred = decoder.predict(X_test)

            r2 = 1 - np.sum((y_test - y_pred)**2) / (np.sum((y_test - y_test.mean())**2) + 1e-10)
            r2_scores.append(max(r2, 0))

        time_points = np.arange(0, n_time, 3)
        axes[0].plot(time_points, r2_scores, label=name, color=colors[idx], linewidth=2.5)

    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Decoding R² (Set Interval)', fontsize=12)
    axes[0].set_title('When is Timing Information Available?', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Peak decoding time for each model
    peak_times = []
    peak_r2 = []
    for idx, name in enumerate(model_names):
        activities = pad_activities(trial_data[name]['activities'])
        set_intervals = extract_set_intervals(trial_data[name])

        n_trials, n_time, n_neurons = activities.shape
        n_train = int(0.8 * n_trials)

        best_r2 = 0
        best_t = 0
        for t in range(0, n_time, 3):
            X_train = activities[:n_train, t, :]
            y_train = set_intervals[:n_train]
            X_test = activities[n_train:, t, :]
            y_test = set_intervals[n_train:]

            decoder = Ridge(alpha=1.0)
            decoder.fit(X_train, y_train)
            y_pred = decoder.predict(X_test)

            r2 = 1 - np.sum((y_test - y_pred)**2) / (np.sum((y_test - y_test.mean())**2) + 1e-10)
            if r2 > best_r2:
                best_r2 = max(r2, 0)
                best_t = t

        peak_times.append(best_t)
        peak_r2.append(best_r2)

    axes[1].bar(model_names, peak_r2, color=colors, alpha=0.7)
    axes[1].set_ylabel('Peak Decoding R²', fontsize=12)
    axes[1].set_title('Best Timing Information Readout', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1])

    # Add peak time as text
    for i, (t, r2) in enumerate(zip(peak_times, peak_r2)):
        axes[1].text(i, r2 + 0.02, f't={t}', ha='center', fontsize=9)

    plt.suptitle('Plot 2: Temporal Decoding - When Can We Read Out Timing?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_3_input_output_transformation(trial_data, output_path='images/mechanism_3_io_transform.png'):
    """
    Plot 3: Input-Output Transformation Analysis
    How do neural trajectories differ for different set intervals?
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = pad_activities(trial_data[name]['activities'])
        set_intervals = extract_set_intervals(trial_data[name])

        # Bin intervals into short, medium, long
        percentiles = np.percentile(set_intervals, [33, 67])
        short_trials = set_intervals < percentiles[0]
        medium_trials = (set_intervals >= percentiles[0]) & (set_intervals < percentiles[1])
        long_trials = set_intervals >= percentiles[1]

        # PCA on all data
        n_trials, n_time, n_neurons = activities.shape
        activities_flat = activities.reshape(-1, n_neurons)
        pca = PCA(n_components=2)
        pca.fit(activities_flat)

        # Project each interval group
        for trials_mask, color, label in [
            (short_trials, '#2ca02c', f'Short ({short_trials.sum()} trials)'),
            (medium_trials, '#ff7f0e', f'Medium ({medium_trials.sum()} trials)'),
            (long_trials, '#d62728', f'Long ({long_trials.sum()} trials)')
        ]:
            if trials_mask.sum() == 0:
                continue

            group_activities = activities[trials_mask]
            avg_trajectory = np.mean(group_activities, axis=0)

            projected = pca.transform(avg_trajectory)

            axes[idx].plot(projected[:, 0], projected[:, 1],
                          color=color, linewidth=2.5, label=label, alpha=0.8)
            axes[idx].scatter(projected[0, 0], projected[0, 1],
                            color=color, s=100, marker='o', edgecolors='k', linewidths=2, zorder=5)
            axes[idx].scatter(projected[-1, 0], projected[-1, 1],
                            color=color, s=100, marker='s', edgecolors='k', linewidths=2, zorder=5)

        axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9, loc='best')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Plot 3: Neural Trajectories for Different Set Intervals\n(Circle=start, Square=end)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_4_error_trial_analysis(trial_data, output_path='images/mechanism_4_error_trials.png'):
    """
    Plot 4: Error Trial Analysis
    What goes wrong when timing fails?
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = pad_activities(trial_data[name]['activities'])
        correct = np.array(trial_data[name]['correct'])

        if correct.sum() == 0 or (~correct).sum() == 0:
            axes[idx].text(0.5, 0.5, 'All trials correct\nor all incorrect',
                          ha='center', va='center', fontsize=12)
            axes[idx].set_title(f'{name}\nAcc: {correct.mean():.2%}',
                              fontsize=12, fontweight='bold')
            continue

        # Average activity for correct vs incorrect
        correct_activity = np.mean(activities[correct], axis=0)
        incorrect_activity = np.mean(activities[~correct], axis=0)

        # Plot mean activity over time
        time_steps = np.arange(correct_activity.shape[0])

        axes[idx].plot(time_steps, np.mean(correct_activity, axis=1),
                      color='#2ca02c', linewidth=2.5, label=f'Correct (n={correct.sum()})')
        axes[idx].fill_between(time_steps,
                              np.mean(correct_activity, axis=1) - np.std(correct_activity, axis=1),
                              np.mean(correct_activity, axis=1) + np.std(correct_activity, axis=1),
                              color='#2ca02c', alpha=0.2)

        axes[idx].plot(time_steps, np.mean(incorrect_activity, axis=1),
                      color='#d62728', linewidth=2.5, label=f'Error (n={(~correct).sum()})')
        axes[idx].fill_between(time_steps,
                              np.mean(incorrect_activity, axis=1) - np.std(incorrect_activity, axis=1),
                              np.mean(incorrect_activity, axis=1) + np.std(incorrect_activity, axis=1),
                              color='#d62728', alpha=0.2)

        axes[idx].set_xlabel('Time Step', fontsize=11)
        axes[idx].set_ylabel('Mean Population Activity', fontsize=11)
        axes[idx].set_title(f'{name}\nAccuracy: {correct.mean():.1%}',
                           fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Plot 4: Correct vs Error Trials - What Goes Wrong?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_5_neuron_importance(trial_data, checkpoint, output_path='images/mechanism_5_neuron_importance.png'):
    """
    Plot 5: Neuron Importance Analysis
    Which neurons are critical for timing performance?
    """
    from Question_2a import Net
    import neurogym as ngym

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


def plot_6_set_interval_encoding(trial_data, output_path='images/mechanism_6_interval_encoding.png'):
    """
    Plot 6: Set Interval Encoding
    How does the network represent different Ready→Set intervals?
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = pad_activities(trial_data[name]['activities'])
        set_intervals = extract_set_intervals(trial_data[name])

        # Take activity snapshot at Set cue (approximate: 1/4 through trial)
        set_timepoint = activities.shape[1] // 4
        set_activities = activities[:, set_timepoint, :]

        # Compute mean population activity for each interval
        unique_intervals = np.unique(set_intervals)
        mean_activities = []

        for interval in unique_intervals:
            mask = set_intervals == interval
            if mask.sum() > 0:
                mean_activities.append(np.mean(set_activities[mask], axis=0))

        mean_activities = np.array(mean_activities)

        # Plot as heatmap
        im = axes[idx].imshow(mean_activities.T, aspect='auto', cmap='viridis')
        axes[idx].set_xlabel('Set Interval (binned)', fontsize=11)
        axes[idx].set_ylabel('Neuron', fontsize=11)
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[idx], label='Activity at Set')

        # Add interval values as x-ticks
        if len(unique_intervals) < 20:
            axes[idx].set_xticks(range(len(unique_intervals)))
            axes[idx].set_xticklabels([f'{int(i)}' for i in unique_intervals],
                                     rotation=45, fontsize=8)

    plt.suptitle('Plot 6: Set Interval Encoding - How is Time Represented?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_7_output_readiness(trial_data, checkpoint, output_path='images/mechanism_7_output_readiness.png'):
    """
    Plot 7: Output Readiness Trajectory
    When does "Go" probability start rising?
    """
    from Question_2a import Net
    import neurogym as ngym

    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    model_types = ['vanilla', 'leaky', 'leaky_fa', 'bio_realistic']
    state_dicts = ['vanilla_model', 'leaky_model', 'leaky_fa_model', 'bio_model']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    device = torch.device('cpu')
    task = 'ReadySetGo-v0'
    kwargs_env = {'dt': 20}

    dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=1, seq_len=300)
    env = dataset.env

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Run one trial through each model
    env.new_trial()
    ob, gt = env.ob, env.gt
    inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)

    for idx, (name, model_type, state_dict) in enumerate(zip(model_names, model_types, state_dicts)):
        # Load model
        model_kwargs = {}
        if model_type != 'vanilla':
            model_kwargs = {'dt': 20, 'tau': 100, 'sigma_rec': 0.0}  # No noise for visualization
        if model_type == 'bio_realistic':
            model_kwargs['exc_ratio'] = 0.8

        net = Net(input_size=3, hidden_size=50, output_size=2,
                 model_type=model_type, **model_kwargs).to(device)
        net.load_state_dict(checkpoint[state_dict])
        net.eval()

        with torch.no_grad():
            output, _ = net(inputs)
            output = output.detach().cpu().numpy()[:, 0, :]

            # Apply softmax to get probabilities
            output_probs = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
            go_prob = output_probs[:, 1]  # Probability of Go action

            axes[0].plot(go_prob, label=name, color=colors[idx], linewidth=2.5)

            # Find when Go prob exceeds 0.5
            go_onset = np.where(go_prob > 0.5)[0]
            if len(go_onset) > 0:
                axes[0].scatter(go_onset[0], go_prob[go_onset[0]],
                              color=colors[idx], s=100, zorder=5, edgecolors='k', linewidths=2)

    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('P(Go action)', fontsize=12)
    axes[0].set_title('Output Readiness: When Does Go Probability Rise?',
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0.5, color='k', linestyle='--', alpha=0.5, linewidth=2)
    axes[0].set_ylim([0, 1])

    # Show target Go time
    target_go = np.where(gt == 1)[0]
    if len(target_go) > 0:
        axes[0].axvline(target_go[0], color='red', linestyle='--',
                       linewidth=3, alpha=0.7, label='Target Go')
        axes[0].legend(fontsize=10)

    # Statistical summary across multiple trials
    go_onset_times = {name: [] for name in model_names}

    for trial_num in range(min(50, len(trial_data['Vanilla']['activities']))):
        env.new_trial()
        ob, gt = env.ob, env.gt
        inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
        target_go = np.where(gt == 1)[0]
        target_go_time = target_go[0] if len(target_go) > 0 else len(gt)

        for idx, (name, model_type, state_dict) in enumerate(zip(model_names, model_types, state_dicts)):
            model_kwargs = {}
            if model_type != 'vanilla':
                model_kwargs = {'dt': 20, 'tau': 100, 'sigma_rec': 0.0}
            if model_type == 'bio_realistic':
                model_kwargs['exc_ratio'] = 0.8

            net = Net(input_size=3, hidden_size=50, output_size=2,
                     model_type=model_type, **model_kwargs).to(device)
            net.load_state_dict(checkpoint[state_dict])
            net.eval()

            with torch.no_grad():
                output, _ = net(inputs)
                output = output.detach().cpu().numpy()[:, 0, :]
                output_probs = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
                go_prob = output_probs[:, 1]

                go_onset = np.where(go_prob > 0.5)[0]
                if len(go_onset) > 0:
                    error = go_onset[0] - target_go_time
                    go_onset_times[name].append(error)

    # Plot error distribution
    for idx, name in enumerate(model_names):
        if len(go_onset_times[name]) > 0:
            axes[1].hist(go_onset_times[name], bins=15, alpha=0.5,
                        label=f'{name} (μ={np.mean(go_onset_times[name]):.1f})',
                        color=colors[idx])

    axes[1].set_xlabel('Go Onset Error (timesteps)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Go Timing Precision Distribution', fontsize=13, fontweight='bold')
    axes[1].axvline(0, color='k', linestyle='--', linewidth=2, alpha=0.7, label='Perfect')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Plot 7: Output Decision Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_8_cross_interval_generalization(checkpoint, output_path='images/mechanism_8_generalization.png'):
    """
    Plot 8: Cross-Interval Generalization
    Do models generalize to novel intervals?
    """
    from Question_2a import Net, evaluate_model
    import neurogym as ngym

    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    model_types = ['vanilla', 'leaky', 'leaky_fa', 'bio_realistic']
    state_dicts = ['vanilla_model', 'leaky_model', 'leaky_fa_model', 'bio_model']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    device = torch.device('cpu')

    # Test on different gain factors
    gains = [1.0, 1.2, 1.5, 1.8, 2.0]  # Trained on 1.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    performances = {name: [] for name in model_names}

    for gain in gains:
        print(f"  Testing gain={gain}...")
        task = 'ReadySetGo-v0'
        kwargs_env = {'dt': 20, 'gain': gain}

        try:
            dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=16, seq_len=300)
            env = dataset.env

            for idx, (name, model_type, state_dict) in enumerate(zip(model_names, model_types, state_dicts)):
                model_kwargs = {}
                if model_type != 'vanilla':
                    model_kwargs = {'dt': 20, 'tau': 100, 'sigma_rec': 0.15}
                if model_type == 'bio_realistic':
                    model_kwargs['exc_ratio'] = 0.8

                net = Net(input_size=3, hidden_size=50, output_size=2,
                         model_type=model_type, **model_kwargs).to(device)
                net.load_state_dict(checkpoint[state_dict])

                perf, _ = evaluate_model(net, env, num_trials=100)
                performances[name].append(perf)
        except Exception as e:
            print(f"    Error with gain={gain}: {e}")
            for name in model_names:
                performances[name].append(np.nan)

    # Plot generalization curves
    for idx, name in enumerate(model_names):
        axes[0].plot(gains, performances[name], marker='o', markersize=8,
                    label=name, color=colors[idx], linewidth=2.5)

    axes[0].set_xlabel('Gain Factor', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Generalization to Novel Gain Factors\n(Trained on gain=1.5)',
                     fontsize=13, fontweight='bold')
    axes[0].axvline(1.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Training gain')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Bar plot of average generalization
    avg_gen = []
    for name in model_names:
        perfs = np.array(performances[name])
        # Average excluding the training gain
        train_idx = gains.index(1.5)
        gen_perfs = np.concatenate([perfs[:train_idx], perfs[train_idx+1:]])
        avg_gen.append(np.nanmean(gen_perfs))

    axes[1].bar(model_names, avg_gen, color=colors, alpha=0.7)
    axes[1].set_ylabel('Average Accuracy\n(excluding training gain)', fontsize=12)
    axes[1].set_title('Generalization Performance', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.05])

    plt.suptitle('Plot 8: Cross-Interval Generalization - Do Models Learn the Rule?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all 8 task-focused analysis plots."""
    print("="*70)
    print("Timing Mechanism Analysis - How Do Models Solve the Task?")
    print("="*70)

    checkpoint = load_data()

    trial_data = {
        'Vanilla': checkpoint['vanilla_data'],
        'Leaky': checkpoint['leaky_data'],
        'Leaky+FA': checkpoint['leaky_fa_data'],
        'Bio-Realistic': checkpoint['bio_data']
    }

    print("\n[1/8] Time Cell Analysis...")
    plot_1_time_cell_analysis(trial_data)

    print("\n[2/8] Timing Information Readout...")
    plot_2_timing_info_readout(trial_data)

    print("\n[3/8] Input-Output Transformation...")
    plot_3_input_output_transformation(trial_data)

    print("\n[4/8] Error Trial Analysis...")
    plot_4_error_trial_analysis(trial_data)

    print("\n[5/8] Neuron Importance...")
    plot_5_neuron_importance(trial_data, checkpoint)

    print("\n[6/8] Set Interval Encoding...")
    plot_6_set_interval_encoding(trial_data)

    print("\n[7/8] Output Readiness Trajectory...")
    plot_7_output_readiness(trial_data, checkpoint)

    print("\n[8/8] Cross-Interval Generalization...")
    plot_8_cross_interval_generalization(checkpoint)

    print("\n" + "="*70)
    print("All 8 mechanism plots generated successfully!")
    print("Review images/mechanism_*.png files and select your 2 favorites!")
    print("="*70)


if __name__ == "__main__":
    # Need to import checkpoint for some plots
    checkpoint = load_data()
    main()
