"""
Question 2b: Analysis of Trained Models
Generates 5 key plots for analyzing hidden unit activity:
1. Timing performance (readysetgo_timing.png)
2. Neural trajectories (readysetgo_trajectories.png)
3. Activity heatmaps (readysetgo_heatmaps.png)
4. Neuron importance (mechanism_5_neuron_importance.png)
5. Connectivity matrices (readysetgo_connectivity.png)
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


def load_checkpoint(checkpoint_path=None):
    """Load checkpoint with trained models and trial data."""
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
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    return checkpoint


def collect_trial_data(models, env, num_trials=100, device='cpu'):
    """Collect trial data from trained models."""
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']

    # Initialize data structure for all models
    trial_data = {name: {'activities': [], 'outputs': [], 'targets': [],
                         'correct': [], 'trial_info': []}
                  for name in model_names}

    print(f"  Collecting data from {num_trials} trials...")

    # Key fix: Generate each trial ONCE, then run all models on the SAME trial
    for trial_idx in range(num_trials):
        env.new_trial()
        ob, gt = env.ob, env.gt

        # Correct input shape: [time, batch, features]
        inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)

        # Run all models on the same trial
        for model, name in zip(models, model_names):
            model.eval()

            with torch.no_grad():
                # Run model - Net returns (output, activity, hidden)
                output, activity, _ = model(inputs)

                # Store activity: [time, batch, hidden] -> [time, hidden]
                activity_np = activity[:, 0, :].cpu().numpy()
                trial_data[name]['activities'].append(activity_np)

                # Store outputs: [time, batch, actions] -> [time, actions]
                output_np = output[:, 0, :].cpu().numpy()
                trial_data[name]['outputs'].append(output_np)

                # Store targets
                trial_data[name]['targets'].append(gt)

                # Store trial info (ground truth timing parameters)
                trial_data[name]['trial_info'].append(env.unwrapped.trial.copy())

                # Compute if prediction was correct
                action = output[:, 0, :].argmax(dim=1).cpu().numpy()
                correct = (action == gt).all()
                trial_data[name]['correct'].append(correct)

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


def detect_go_time(logits, go_action_idx):
    """Return earliest timestep where Go is the predicted action."""
    logits = np.asarray(logits)
    preds = np.argmax(logits, axis=1)

    first = np.where(preds == go_action_idx)[0]
    if len(first) > 0:
        return int(first[0])

    # Fallback: return timestep with max Go activation
    go_channel = logits[:, go_action_idx]
    return int(np.argmax(go_channel))


def plot_timing_accuracy(trial_data, env, output_path='images/readysetgo_timing.png',
                        go_action_index=None):
    """
    Plot 1: Timing Performance
    Scatter and error distribution showing timing accuracy

    For ReadySetGo task:
    - X-axis: target Go time (absolute timestep when Go should occur)
    - Y-axis: produced Go time (absolute timestep when model predicted Go)
    """
    print("[1/4] Generating timing performance plot...")

    if go_action_index is None:
        go_action_index = infer_go_action_index(env)

    dt = getattr(env, 'dt', 20)
    model_names = list(trial_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    all_errors = []
    all_target_times = []
    all_produced_times = []

    for idx, name in enumerate(model_names):
        outputs = trial_data[name].get('outputs', [])
        targets = trial_data[name].get('targets', [])

        if len(outputs) == 0 or len(targets) == 0:
            print(f"Warning: {name} missing outputs or targets, skipping...")
            continue

        produced_times = []
        target_times = []

        for trial_idx in range(len(outputs)):
            output = outputs[trial_idx]
            target = targets[trial_idx]

            # Find when model predicted Go (absolute timestep)
            produced_idx = detect_go_time(output, go_action_index)
            produced_time = produced_idx * dt

            # Find when target says Go (absolute timestep)
            target_go_idx = np.where(target == go_action_index)[0]
            target_time = (target_go_idx[0] * dt) if len(target_go_idx) > 0 else (len(target) * dt)

            produced_times.append(produced_time)
            target_times.append(target_time)

        if len(produced_times) > 0:
            produced_times = np.array(produced_times)
            target_times = np.array(target_times)

            # Scatter plot
            ax1.scatter(target_times, produced_times, alpha=0.3, s=50,
                       color=colors[idx], label=name)

            # Error distribution
            errors = produced_times - target_times
            all_errors.extend(errors)
            all_target_times.extend(target_times)
            all_produced_times.extend(produced_times)

    # Diagonal line for perfect timing
    if len(all_target_times) > 0:
        all_target_times = np.array(all_target_times)
        all_produced_times = np.array(all_produced_times)
        min_time = min(all_target_times.min(), all_produced_times.min())
        max_time = max(all_target_times.max(), all_produced_times.max())

        ax1.plot([min_time, max_time], [min_time, max_time], 'k--',
                linewidth=2.5, alpha=0.5, label='Perfect timing')

    ax1.set_xlabel('Target Go Time (ms)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Produced Go Time (ms)', fontsize=16, fontweight='bold')
    ax1.set_title('Timing Performance', fontsize=18, fontweight='bold')
    ax1.legend(fontsize=13)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(True, alpha=0.3)

    # Error distribution with fixed bin width
    if len(all_errors) > 0:
        error_array = np.array(all_errors)
        error_min = error_array.min()
        error_max = error_array.max()
        bin_width = 50  # Fixed bin width in ms
        bins = np.arange(error_min, error_max + bin_width, bin_width)

        for idx, name in enumerate(model_names):
            outputs = trial_data[name].get('outputs', [])
            targets = trial_data[name].get('targets', [])

            if len(outputs) == 0:
                continue

            produced_times = []
            target_times = []

            for trial_idx in range(len(outputs)):
                output = outputs[trial_idx]
                target = targets[trial_idx]

                produced_idx = detect_go_time(output, go_action_index)
                produced_time = produced_idx * dt

                target_go_idx = np.where(target == go_action_index)[0]
                target_time = (target_go_idx[0] * dt) if len(target_go_idx) > 0 else (len(target) * dt)

                produced_times.append(produced_time)
                target_times.append(target_time)

            if len(produced_times) > 0:
                produced_times = np.array(produced_times)
                target_times = np.array(target_times)
                errors = produced_times - target_times

                mean_error = np.mean(errors)
                std_error = np.std(errors)

                ax2.hist(errors, bins=bins, alpha=0.5, color=colors[idx],
                        label=f'{name} (μ={mean_error:.1f}ms, σ={std_error:.1f}ms)')

    ax2.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Timing Error (ms)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax2.set_title('Error Distribution', fontsize=18, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_neural_trajectories(trial_data, output_path='images/readysetgo_trajectories.png'):
    """
    Plot 2: Neural Trajectories in PCA Space
    Shows how population state evolves during timing
    """
    print("[2/4] Generating neural trajectories plot...")

    model_names = list(trial_data.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        ax = axes[idx]

        # Concatenate all trials for PCA (first 20 trials)
        all_activities = np.concatenate(trial_data[name]['activities'][:20], axis=0)

        # Fit PCA with 3 components
        pca = PCA(n_components=3)
        pca_activities = pca.fit_transform(all_activities)

        # Plot first 10 trials
        for trial_idx in range(10):
            activity = trial_data[name]['activities'][trial_idx]
            activity_pca = pca.transform(activity)
            n_time = activity.shape[0]

            # Plot trajectory as a single continuous line
            ax.plot(activity_pca[:, 0], activity_pca[:, 1],
                   alpha=0.4, linewidth=1.5, color='black')

        # Only add markers for the first trial to avoid clutter
        activity = trial_data[name]['activities'][0]
        activity_pca = pca.transform(activity)
        n_time = activity.shape[0]

        # Mark start (Ready)
        ax.scatter(activity_pca[0, 0], activity_pca[0, 1],
                  s=120, color='green', marker='o', alpha=0.9,
                  edgecolors='white', linewidths=2, zorder=10)

        # Mark Set cue (approximate at 1/4 through trial)
        set_idx = n_time // 4
        ax.scatter(activity_pca[set_idx, 0], activity_pca[set_idx, 1],
                  s=120, color='blue', marker='D', alpha=0.9,
                  edgecolors='white', linewidths=2, zorder=10)

        # Mark end (Go)
        ax.scatter(activity_pca[-1, 0], activity_pca[-1, 1],
                  s=120, color='red', marker='s', alpha=0.9,
                  edgecolors='white', linewidths=2, zorder=10)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
        ax.set_title(f'{name} RNN', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Add legend to first plot
        if idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                      markersize=11, markeredgecolor='white', markeredgewidth=2, label='Ready (start)'),
                Line2D([0], [0], marker='D', color='w', markerfacecolor='blue',
                      markersize=11, markeredgecolor='white', markeredgewidth=2, label='Set (cue)'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                      markersize=11, markeredgecolor='white', markeredgewidth=2, label='Go (end)')
            ]
            ax.legend(handles=legend_elements, loc='best', fontsize=12, framealpha=0.9)

    plt.suptitle('Neural Trajectories in PCA Space - Timing Task Dynamics',
                fontsize=18, fontweight='bold', y=0.995)
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

    # Store handles for shared legend
    legend_handles = []
    
    # Store the last image for shared colorbar
    last_im = None

    for idx, name in enumerate(model_names):
        activities = trial_data[name]['activities']

        if trial_idx >= len(activities):
            trial_idx = 0

        activity = activities[trial_idx]

        ax = axes[idx]
        im = ax.imshow(activity.T, aspect='auto', cmap='viridis',
                      interpolation='nearest')
        
        # Keep reference to last image for colorbar
        last_im = im

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

        # Only add labels to first subplot for shared legend
        if idx == 0:
            line_set = ax.axvline(set_marker, color='red', linestyle='--',
                      linewidth=2, alpha=0.9, label='Set')
            line_go = ax.axvline(go_marker, color='white', linestyle='--',
                      linewidth=2, alpha=0.9, label='Go')
            legend_handles = [line_set, line_go]
        else:
            # No labels for other subplots
            ax.axvline(set_marker, color='red', linestyle='--',
                      linewidth=2, alpha=0.9)
            ax.axvline(go_marker, color='white', linestyle='--',
                      linewidth=2, alpha=0.9)

        ax.set_xlabel('Time Step', fontsize=14)
        ax.set_ylabel('Neuron', fontsize=14)
        ax.set_title(f'{name}', fontsize=16, fontweight='bold')
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.suptitle(f'Activity Heatmaps (Trial {trial_idx})',
                fontsize=18, fontweight='bold')
    
    # Add single shared colorbar at the bottom
    fig.subplots_adjust(bottom=0.15, right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label('Activity', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Add single shared legend at the bottom
    fig.legend(handles=legend_handles, loc='lower center', 
               ncol=2, fontsize=13, frameon=True,
               bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 0.85, 0.96])  # Make room for colorbar and legend

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
        # Infer hidden size from saved state dict
        saved_state = checkpoint[state_dict]

        # Handle different architectures
        if 'rnn.h2h.weight' in saved_state:
            hidden_size = saved_state['rnn.h2h.weight'].shape[0]
        elif 'rnn.h2h_exc.weight' in saved_state:
            hidden_size = saved_state['rnn.h2h_exc.weight'].shape[1] + saved_state['rnn.h2h_inh.weight'].shape[1]
        else:
            raise ValueError(f"Cannot infer hidden size from state_dict keys")

        # Load model
        model_kwargs = {}
        if model_type != 'vanilla':
            model_kwargs = {'dt': 20, 'tau': 100, 'sigma_rec': 0.15}
        if model_type == 'bio_realistic':
            model_kwargs['exc_ratio'] = 0.8

        net = Net(input_size=3, hidden_size=hidden_size, output_size=2,
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


def plot_connectivity_matrices(checkpoint, output_path='images/readysetgo_connectivity.png'):
    """
    Plot 5: Recurrent Connectivity Matrices
    Shows how neurons are connected to each other in each model
    """
    print("[5/5] Generating connectivity matrices plot...")

    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    model_types = ['vanilla', 'leaky', 'leaky_fa', 'bio_realistic']
    state_dicts = ['vanilla_model', 'leaky_model', 'leaky_fa_model', 'bio_model']

    device = torch.device('cpu')

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # Store the last image for shared colorbar
    last_im = None

    for idx, (name, model_type, state_dict) in enumerate(zip(model_names, model_types, state_dicts)):
        ax = axes[idx]

        # Infer hidden size from saved state dict
        saved_state = checkpoint[state_dict]

        # Handle different architectures
        if 'rnn.h2h.weight' in saved_state:
            hidden_size = saved_state['rnn.h2h.weight'].shape[0]
        elif 'rnn.h2h_exc.weight' in saved_state:
            hidden_size = saved_state['rnn.h2h_exc.weight'].shape[1] + saved_state['rnn.h2h_inh.weight'].shape[1]
        else:
            raise ValueError(f"Cannot infer hidden size from state_dict keys")

        # Load model
        model_kwargs = {}
        if model_type != 'vanilla':
            model_kwargs = {'dt': 20, 'tau': 100, 'sigma_rec': 0.15}
        if model_type == 'bio_realistic':
            model_kwargs['exc_ratio'] = 0.8

        net = Net(input_size=3, hidden_size=hidden_size, output_size=2,
                 model_type=model_type, **model_kwargs).to(device)
        net.load_state_dict(checkpoint[state_dict])

        # Extract recurrent weight matrix
        if model_type == 'vanilla':
            # Vanilla RNN: h2h weight matrix
            weight_matrix = net.rnn.h2h.weight.detach().cpu().numpy()
        elif model_type == 'bio_realistic':
            # Bio-realistic: reconstruct full weight matrix from separate E/I weights
            n_exc = net.rnn.n_exc
            n_inh = net.rnn.n_inh

            # Create full weight matrix
            weight_matrix = np.zeros((hidden_size, hidden_size))

            # Excitatory connections (positive)
            exc_weights = torch.relu(net.rnn.h2h_exc.weight).detach().cpu().numpy()
            weight_matrix[:, :n_exc] = exc_weights

            # Inhibitory connections (negative)
            inh_weights = torch.relu(net.rnn.h2h_inh.weight).detach().cpu().numpy()
            weight_matrix[:, n_exc:] = -inh_weights
        else:
            # Leaky and Leaky+FA
            weight_matrix = net.rnn.h2h.weight.detach().cpu().numpy()

        # Plot connectivity matrix
        vmax = np.abs(weight_matrix).max()
        im = ax.imshow(weight_matrix, aspect='auto', cmap='RdBu_r',
                      interpolation='nearest', vmin=-vmax, vmax=vmax)
        
        last_im = im

        ax.set_xlabel('From Neuron', fontsize=16)
        ax.set_ylabel('To Neuron', fontsize=16)
        ax.set_title(f'{name}', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Add statistics
        sparsity = np.mean(np.abs(weight_matrix) < 0.01)
        mean_weight = np.mean(np.abs(weight_matrix))

        if model_type == 'bio_realistic':
            # For bio-realistic, also show E/I neuron counts
            text_str = f'Sparsity: {sparsity:.2%}\nMean |W|: {mean_weight:.3f}\nE/I: {n_exc}/{n_inh}'
            # Add line to separate excitatory and inhibitory
            ax.axhline(y=n_exc-0.5, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
            ax.axvline(x=n_exc-0.5, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
        else:
            text_str = f'Sparsity: {sparsity:.2%}\nMean |W|: {mean_weight:.3f}'
        text_str = f'Sparsity: {sparsity:.2%}\nMean |W|: {mean_weight:.3f}'
        ax.text(0.98, 0.98, text_str, transform=ax.transAxes,
               fontsize=13, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Recurrent Connectivity Matrices',
                fontsize=20, fontweight='bold')

    # Add single shared colorbar on the right
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label('Weight', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

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
        # Infer hidden size from saved state dict
        saved_state = checkpoint[state_dict]

        # Handle different architectures
        if 'rnn.h2h.weight' in saved_state:
            # Old architecture (vanilla, leaky, leaky_fa) or old bio_realistic
            hidden_size = saved_state['rnn.h2h.weight'].shape[0]
        elif 'rnn.h2h_exc.weight' in saved_state:
            # New bio_realistic architecture with separate exc/inh weights
            hidden_size = saved_state['rnn.h2h_exc.weight'].shape[1] + saved_state['rnn.h2h_inh.weight'].shape[1]
        else:
            raise ValueError(f"Cannot infer hidden size from state_dict keys: {saved_state.keys()}")

        model_kwargs = {}
        if model_type != 'vanilla':
            model_kwargs = {'dt': dt, 'tau': 100, 'sigma_rec': 0.15}
        if model_type == 'bio_realistic':
            model_kwargs['exc_ratio'] = 0.8

        net = Net(input_size=3, hidden_size=hidden_size, output_size=2,
                 model_type=model_type, **model_kwargs).to(device)
        net.load_state_dict(checkpoint[state_dict])
        models.append(net)

    print("Models loaded successfully!\n")

    # Collect trial data
    print("Collecting trial data...")
    trial_data = collect_trial_data(models, env, num_trials=100, device=device)
    print("Trial data collected!\n")

    # Generate plots
    print("Generating plots...")
    print()

    # Get script directory and set output to parent coursework/images/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'images')

    # Use freshly collected trial data (has outputs for accurate timing detection)
    plot_timing_accuracy(trial_data, env,
                        output_path=os.path.join(output_dir, 'readysetgo_timing.png'))

    plot_neural_trajectories(trial_data,
                            output_path=os.path.join(output_dir, 'readysetgo_trajectories.png'))

    plot_activity_heatmaps(trial_data, trial_idx=5,
                          output_path=os.path.join(output_dir, 'readysetgo_heatmaps.png'))

    plot_neuron_importance(trial_data, checkpoint,
                          output_path=os.path.join(output_dir, 'mechanism_5_neuron_importance.png'))

    plot_connectivity_matrices(checkpoint,
                              output_path=os.path.join(output_dir, 'readysetgo_connectivity.png'))

    print()
    print("="*70)
    print("All 5 plots for Question 2b generated successfully!")
    print("Check images/ directory for:")
    print("  1. readysetgo_timing.png")
    print("  2. readysetgo_trajectories.png")
    print("  3. readysetgo_heatmaps.png")
    print("  4. mechanism_5_neuron_importance.png")
    print("  5. readysetgo_connectivity.png")
    print("="*70)


if __name__ == '__main__':
    main()
