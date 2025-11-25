import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
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


def plot_1_selectivity_analysis(trial_data, output_path='images/analysis_1_selectivity.png'):
    """
    Plot 1: Selectivity Analysis
    Show which neurons are selective for different task epochs.
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = trial_data[name]['activities']

        # Pad to common length
        activities = pad_activities(activities)

        # Average across trials
        avg_activity = np.mean(activities, axis=0)  # (time, neurons)

        # Define task epochs (approximate)
        n_time = avg_activity.shape[0]
        ready_period = slice(0, n_time // 4)
        set_period = slice(n_time // 4, n_time // 2)
        delay_period = slice(n_time // 2, 3 * n_time // 4)
        go_period = slice(3 * n_time // 4, n_time)

        # Calculate selectivity index for each neuron in each epoch
        epochs = [ready_period, set_period, delay_period, go_period]
        epoch_names = ['Ready', 'Set', 'Delay', 'Go']

        selectivity_matrix = np.zeros((avg_activity.shape[1], 4))
        for i, epoch in enumerate(epochs):
            epoch_activity = avg_activity[epoch, :]
            selectivity_matrix[:, i] = np.mean(epoch_activity, axis=0)

        # Normalize by max to get selectivity
        max_activity = selectivity_matrix.max(axis=1, keepdims=True)
        max_activity[max_activity == 0] = 1
        selectivity_matrix = selectivity_matrix / max_activity

        # Sort neurons by preferred epoch
        preferred_epoch = np.argmax(selectivity_matrix, axis=1)
        sort_idx = np.argsort(preferred_epoch)

        im = axes[idx].imshow(selectivity_matrix[sort_idx, :].T, aspect='auto',
                             cmap='viridis', vmin=0, vmax=1)
        axes[idx].set_xlabel('Neuron Index (sorted)', fontsize=10)
        axes[idx].set_ylabel('Task Epoch', fontsize=10)
        axes[idx].set_yticks(range(4))
        axes[idx].set_yticklabels(epoch_names)
        axes[idx].set_title(f'{name}', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=axes[idx], label='Normalized Activity')

    plt.suptitle('Plot 1: Neuron Selectivity for Task Epochs', fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_2_neuron_type_classification(trial_data, checkpoint, output_path='images/analysis_2_neuron_types.png'):
    """
    Plot 2: Neuron Type Classification (Excitatory vs Inhibitory)
    For bio-realistic model only.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Get Dale's mask from bio model
    from Question_2a import Net
    import torch

    device = torch.device('cpu')
    net_bio = Net(input_size=3, hidden_size=50, output_size=2,
                  model_type='bio_realistic', dt=20, tau=100,
                  sigma_rec=0.15, exc_ratio=0.8).to(device)
    net_bio.load_state_dict(checkpoint['bio_model'])

    dale_mask = net_bio.rnn.dale_mask.cpu().numpy().flatten()
    exc_idx = np.where(dale_mask > 0)[0]
    inh_idx = np.where(dale_mask < 0)[0]

    activities = trial_data['Bio-Realistic']['activities']
    activities = pad_activities(activities)
    avg_activity = np.mean(activities, axis=0)

    # Plot 1: Average activity over time
    axes[0, 0].plot(np.mean(avg_activity[:, exc_idx], axis=1),
                   label=f'Excitatory (n={len(exc_idx)})', color='#d62728', linewidth=2)
    axes[0, 0].plot(np.mean(avg_activity[:, inh_idx], axis=1),
                   label=f'Inhibitory (n={len(inh_idx)})', color='#1f77b4', linewidth=2)
    axes[0, 0].set_xlabel('Time Step', fontsize=10)
    axes[0, 0].set_ylabel('Mean Activity', fontsize=10)
    axes[0, 0].set_title('Mean Activity by Type', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Activity distribution
    exc_activity = avg_activity[:, exc_idx].flatten()
    inh_activity = avg_activity[:, inh_idx].flatten()
    axes[0, 1].hist(exc_activity, bins=30, alpha=0.6, label='Excitatory', color='#d62728')
    axes[0, 1].hist(inh_activity, bins=30, alpha=0.6, label='Inhibitory', color='#1f77b4')
    axes[0, 1].set_xlabel('Activity', fontsize=10)
    axes[0, 1].set_ylabel('Count', fontsize=10)
    axes[0, 1].set_title('Activity Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Heatmap of excitatory neurons
    im1 = axes[1, 0].imshow(avg_activity[:, exc_idx].T, aspect='auto', cmap='Reds')
    axes[1, 0].set_xlabel('Time Step', fontsize=10)
    axes[1, 0].set_ylabel('Excitatory Neuron', fontsize=10)
    axes[1, 0].set_title('Excitatory Neurons Activity', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=axes[1, 0])

    # Plot 4: Heatmap of inhibitory neurons
    im2 = axes[1, 1].imshow(avg_activity[:, inh_idx].T, aspect='auto', cmap='Blues')
    axes[1, 1].set_xlabel('Time Step', fontsize=10)
    axes[1, 1].set_ylabel('Inhibitory Neuron', fontsize=10)
    axes[1, 1].set_title('Inhibitory Neurons Activity', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 1])

    plt.suptitle('Plot 2: Excitatory vs Inhibitory Neuron Analysis (Bio-Realistic Model)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_3_ramping_neurons(trial_data, output_path='images/analysis_3_ramping.png'):
    """
    Plot 3: Ramping Neuron Detection
    Identify and visualize neurons that show ramping activity.
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = trial_data[name]['activities']
        activities = pad_activities(activities)
        avg_activity = np.mean(activities, axis=0)

        # Detect ramping: fit linear regression to each neuron's activity
        n_time, n_neurons = avg_activity.shape
        time_points = np.arange(n_time)
        slopes = np.zeros(n_neurons)

        for i in range(n_neurons):
            slope, _ = np.polyfit(time_points, avg_activity[:, i], 1)
            slopes[i] = slope

        # Find top ramping neurons (positive and negative)
        n_top = 5
        top_pos_idx = np.argsort(slopes)[-n_top:]
        top_neg_idx = np.argsort(slopes)[:n_top]

        # Plot top positive rampers
        for i in top_pos_idx:
            axes[idx].plot(avg_activity[:, i], alpha=0.6, linewidth=1.5, color='#d62728')

        # Plot top negative rampers
        for i in top_neg_idx:
            axes[idx].plot(avg_activity[:, i], alpha=0.6, linewidth=1.5, color='#1f77b4')

        axes[idx].set_xlabel('Time Step', fontsize=10)
        axes[idx].set_ylabel('Activity', fontsize=10)
        axes[idx].set_title(f'{name}\n(Red=pos ramp, Blue=neg ramp)', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

        # Add text with stats
        n_pos = np.sum(slopes > 0.001)
        n_neg = np.sum(slopes < -0.001)
        axes[idx].text(0.05, 0.95, f'Pos ramp: {n_pos}\nNeg ramp: {n_neg}',
                      transform=axes[idx].transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round',
                      facecolor='wheat', alpha=0.5))

    plt.suptitle('Plot 3: Ramping Neurons Detection', fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_4_time_constants(trial_data, output_path='images/analysis_4_time_constants.png'):
    """
    Plot 4: Effective Time Constant Estimation
    Measure autocorrelation to estimate effective time constants.
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_tau_estimates = {}

    for idx, name in enumerate(model_names):
        activities = trial_data[name]['activities']
        activities = pad_activities(activities)

        # Average across trials
        avg_activity = np.mean(activities, axis=0)

        # Compute autocorrelation for each neuron
        tau_estimates = []
        for i in range(avg_activity.shape[1]):
            signal = avg_activity[:, i]
            if np.std(signal) > 0.01:  # Only for active neurons
                autocorr = np.correlate(signal - signal.mean(),
                                       signal - signal.mean(), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]

                # Find time to decay to 1/e
                try:
                    tau_idx = np.where(autocorr < 1/np.e)[0][0]
                    tau_estimates.append(tau_idx * 20)  # Convert to ms (dt=20)
                except:
                    tau_estimates.append(np.nan)

        tau_estimates = np.array(tau_estimates)
        tau_estimates = tau_estimates[~np.isnan(tau_estimates)]
        all_tau_estimates[name] = tau_estimates

        # Plot distribution
        axes[0].hist(tau_estimates, bins=20, alpha=0.5, label=name, color=colors[idx])

    axes[0].set_xlabel('Effective Time Constant (ms)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Distribution of Neuron Time Constants', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(100, color='k', linestyle='--', linewidth=2, label='τ=100ms (architectural)')

    # Box plot comparison
    data_to_plot = [all_tau_estimates[name] for name in model_names]
    bp = axes[1].boxplot(data_to_plot, labels=model_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1].set_ylabel('Effective Time Constant (ms)', fontsize=11)
    axes[1].set_title('Time Constant Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(100, color='k', linestyle='--', linewidth=2, alpha=0.5)

    plt.suptitle('Plot 4: Effective Time Constant Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_5_preparatory_vs_movement(trial_data, output_path='images/analysis_5_prep_vs_move.png'):
    """
    Plot 5: Preparatory vs Movement Activity
    Separate neurons by their response timing.
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = trial_data[name]['activities']
        activities = pad_activities(activities)
        avg_activity = np.mean(activities, axis=0)

        n_time = avg_activity.shape[0]
        prep_period = slice(0, 2 * n_time // 3)
        move_period = slice(2 * n_time // 3, n_time)

        # Calculate preference for each neuron
        prep_activity = np.mean(avg_activity[prep_period, :], axis=0)
        move_activity = np.mean(avg_activity[move_period, :], axis=0)

        # Classify neurons
        prep_neurons = prep_activity > move_activity
        move_neurons = ~prep_neurons

        # Scatter plot
        axes[idx].scatter(prep_activity[prep_neurons], move_activity[prep_neurons],
                         alpha=0.6, s=40, label=f'Prep (n={prep_neurons.sum()})', color='#2ca02c')
        axes[idx].scatter(prep_activity[move_neurons], move_activity[move_neurons],
                         alpha=0.6, s=40, label=f'Move (n={move_neurons.sum()})', color='#d62728')

        # Diagonal
        max_val = max(prep_activity.max(), move_activity.max())
        axes[idx].plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5)

        axes[idx].set_xlabel('Preparatory Activity', fontsize=10)
        axes[idx].set_ylabel('Movement Activity', fontsize=10)
        axes[idx].set_title(f'{name}', fontsize=11, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Plot 5: Preparatory vs Movement Neuron Classification',
                fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_6_weight_matrices(checkpoint, output_path='images/analysis_6_weight_matrices.png'):
    """
    Plot 6: Recurrent Weight Matrix Visualization
    Show structure of learned connectivity.
    """
    from Question_2a import Net
    import torch

    device = torch.device('cpu')
    model_configs = [
        ('vanilla', {}),
        ('leaky', {'dt': 20, 'tau': 100, 'sigma_rec': 0.15}),
        ('leaky_fa', {'dt': 20, 'tau': 100, 'sigma_rec': 0.15}),
        ('bio_realistic', {'dt': 20, 'tau': 100, 'sigma_rec': 0.15, 'exc_ratio': 0.8})
    ]

    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    state_dicts = ['vanilla_model', 'leaky_model', 'leaky_fa_model', 'bio_model']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (model_type, kwargs) in enumerate(model_configs):
        net = Net(input_size=3, hidden_size=50, output_size=2,
                 model_type=model_type, **kwargs).to(device)
        net.load_state_dict(checkpoint[state_dicts[idx]])

        # Get recurrent weights
        if hasattr(net.rnn, 'h2h'):
            if hasattr(net.rnn.h2h, 'weight'):
                W = net.rnn.h2h.weight.detach().cpu().numpy()
            else:
                W = net.rnn.h2h.weight.detach().cpu().numpy()
        else:
            W = net.rnn.h2h.weight.detach().cpu().numpy()

        # For bio model, show E/I structure
        if model_type == 'bio_realistic':
            dale_mask = net.rnn.dale_mask.cpu().numpy().flatten()
            exc_idx = np.where(dale_mask > 0)[0]
            inh_idx = np.where(dale_mask < 0)[0]

            # Reorder to show E/I blocks
            new_order = np.concatenate([exc_idx, inh_idx])
            W = W[new_order, :][:, new_order]

        im = axes[idx].imshow(W, cmap='RdBu_r', vmin=-np.abs(W).max(), vmax=np.abs(W).max())
        axes[idx].set_xlabel('Pre-synaptic Neuron', fontsize=10)
        axes[idx].set_ylabel('Post-synaptic Neuron', fontsize=10)
        axes[idx].set_title(f'{model_names[idx]}\nSparsity: {(np.abs(W) < 0.01).mean():.2%}',
                           fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=axes[idx], label='Weight')

        # Add E/I boundary for bio model
        if model_type == 'bio_realistic':
            n_exc = len(exc_idx)
            axes[idx].axhline(n_exc, color='k', linewidth=2)
            axes[idx].axvline(n_exc, color='k', linewidth=2)
            axes[idx].text(5, n_exc-5, 'E', fontsize=14, fontweight='bold', color='red')
            axes[idx].text(5, n_exc+5, 'I', fontsize=14, fontweight='bold', color='blue')

    plt.suptitle('Plot 6: Recurrent Weight Matrix Structure', fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_7_noise_correlations(trial_data, output_path='images/analysis_7_noise_corr.png'):
    """
    Plot 7: Noise Correlation Analysis
    Trial-to-trial variability and neuron pair correlations.
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = pad_activities(trial_data[name]['activities'])

        # Calculate noise correlations
        # For each time point, compute correlation across trials
        n_trials, n_time, n_neurons = activities.shape

        # Average over time to get trial-averaged activity per neuron
        trial_avg = np.mean(activities, axis=1)  # (trials, neurons)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(trial_avg.T)

        # Plot correlation matrix
        im = axes[idx].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[idx].set_xlabel('Neuron', fontsize=10)
        axes[idx].set_ylabel('Neuron', fontsize=10)
        axes[idx].set_title(f'{name}\nMean |corr|: {np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]).mean():.3f}',
                           fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=axes[idx], label='Correlation')

    plt.suptitle('Plot 7: Noise Correlations (Trial-to-Trial Variability)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_8_dimensionality_over_time(trial_data, output_path='images/analysis_8_dimensionality.png'):
    """
    Plot 8: Dimensionality Over Time
    Effective dimensionality at each timepoint using participation ratio.
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, name in enumerate(model_names):
        activities = trial_data[name]['activities']
        activities = pad_activities(activities)
        avg_activity = np.mean(activities, axis=0)  # (time, neurons)

        # Calculate participation ratio at each time point
        n_time = avg_activity.shape[0]
        pr = np.zeros(n_time)
        explained_var = []

        for t in range(n_time):
            activity_t = avg_activity[t, :]
            if np.std(activity_t) > 0:
                # Participation ratio
                normalized = activity_t / (np.sum(activity_t**2) + 1e-10)
                pr[t] = 1 / np.sum(normalized**2)

        # Overall PCA
        pca = PCA()
        pca.fit(avg_activity)
        explained_var = pca.explained_variance_ratio_

        # Plot participation ratio over time
        axes[0].plot(pr, label=name, color=colors[idx], linewidth=2)

        # Plot cumulative explained variance
        cumsum = np.cumsum(explained_var)
        axes[1].plot(cumsum[:20], label=name, color=colors[idx], linewidth=2, marker='o')

    axes[0].set_xlabel('Time Step', fontsize=11)
    axes[0].set_ylabel('Participation Ratio', fontsize=11)
    axes[0].set_title('Effective Dimensionality Over Time', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Principal Component', fontsize=11)
    axes[1].set_ylabel('Cumulative Variance Explained', fontsize=11)
    axes[1].set_title('Overall Dimensionality', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0.9, color='k', linestyle='--', alpha=0.5)

    plt.suptitle('Plot 8: Neural Dimensionality Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_9_single_trial_decoding(trial_data, output_path='images/analysis_9_decoding.png'):
    """
    Plot 9: Single Trial Decoding
    Train linear decoder to predict trial duration from neural activity.
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, name in enumerate(model_names):
        activities = pad_activities(trial_data[name]['activities'])
        trial_info = trial_data[name]['trial_info']

        # Extract set period duration from trial info (proxy for timing)
        go_times = []
        for info in trial_info:
            if hasattr(info, 'set_period') and info.set_period is not None:
                go_times.append(info.set_period)
            else:
                # Use activity length as proxy
                go_times.append(activities.shape[1])

        go_times = np.array(go_times)

        # Train decoder for each time point
        n_trials, n_time, n_neurons = activities.shape

        # Use first 80% for training, last 20% for testing
        n_train = int(0.8 * n_trials)
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_trials)

        decoding_acc = []
        r2_scores = []

        for t in range(0, n_time, 5):  # Every 5 timesteps for speed
            X_train = activities[train_idx, t, :]
            y_train = go_times[train_idx]
            X_test = activities[test_idx, t, :]
            y_test = go_times[test_idx]

            # Ridge regression
            decoder = Ridge(alpha=1.0)
            decoder.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = decoder.predict(X_test)
            r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
            r2_scores.append(max(r2, 0))  # Clip to 0

        time_points = np.arange(0, n_time, 5)
        axes[0].plot(time_points, r2_scores, label=name, color=colors[idx], linewidth=2)

    axes[0].set_xlabel('Time Step', fontsize=11)
    axes[0].set_ylabel('R² Score', fontsize=11)
    axes[0].set_title('Decoding Go Time from Neural Activity', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Bar plot of average decoding performance
    avg_performance = []
    for idx, name in enumerate(model_names):
        activities = pad_activities(trial_data[name]['activities'])
        trial_info = trial_data[name]['trial_info']

        go_times = []
        for info in trial_info:
            if hasattr(info, 'set_period') and info.set_period is not None:
                go_times.append(info.set_period)
            else:
                go_times.append(activities.shape[1])
        go_times = np.array(go_times)

        # Use middle timepoint
        t_mid = activities.shape[1] // 2
        X = activities[:, t_mid, :]
        y = go_times

        n_train = int(0.8 * len(X))
        decoder = Ridge(alpha=1.0)
        decoder.fit(X[:n_train], y[:n_train])
        y_pred = decoder.predict(X[n_train:])
        r2 = 1 - np.sum((y[n_train:] - y_pred)**2) / np.sum((y[n_train:] - y[n_train:].mean())**2)
        avg_performance.append(max(r2, 0))

    axes[1].bar(model_names, avg_performance, color=colors, alpha=0.7)
    axes[1].set_ylabel('R² Score (mid-trial)', fontsize=11)
    axes[1].set_title('Average Decoding Performance', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1])

    plt.suptitle('Plot 9: Linear Decoding of Timing Information', fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_10_neuron_clustering(trial_data, output_path='images/analysis_10_clustering.png'):
    """
    Plot 10: Neuron Clustering by Activity Patterns
    Hierarchical clustering reveals functional modules.
    """
    model_names = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        activities = trial_data[name]['activities']
        activities = pad_activities(activities)
        avg_activity = np.mean(activities, axis=0)  # (time, neurons)

        # Transpose for clustering (neurons x time)
        X = avg_activity.T

        # Compute linkage
        linkage_matrix = linkage(X, method='ward')

        # Reorder based on clustering
        from scipy.cluster.hierarchy import leaves_list
        order = leaves_list(linkage_matrix)

        # Plot clustered heatmap
        im = axes[idx].imshow(X[order, :], aspect='auto', cmap='viridis')
        axes[idx].set_xlabel('Time Step', fontsize=10)
        axes[idx].set_ylabel('Neuron (clustered)', fontsize=10)
        axes[idx].set_title(f'{name}', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=axes[idx], label='Activity')

    plt.suptitle('Plot 10: Hierarchical Clustering of Neurons by Activity',
                fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all 10 analysis plots."""
    print("="*70)
    print("Hidden Unit Analysis - Generating All 10 Plots")
    print("="*70)

    checkpoint = load_data()

    trial_data = {
        'Vanilla': checkpoint['vanilla_data'],
        'Leaky': checkpoint['leaky_data'],
        'Leaky+FA': checkpoint['leaky_fa_data'],
        'Bio-Realistic': checkpoint['bio_data']
    }

    print("\n[1/10] Generating selectivity analysis...")
    plot_1_selectivity_analysis(trial_data)

    print("\n[2/10] Generating neuron type classification...")
    plot_2_neuron_type_classification(trial_data, checkpoint)

    print("\n[3/10] Generating ramping neuron detection...")
    plot_3_ramping_neurons(trial_data)

    print("\n[4/10] Generating time constant analysis...")
    plot_4_time_constants(trial_data)

    print("\n[5/10] Generating preparatory vs movement analysis...")
    plot_5_preparatory_vs_movement(trial_data)

    print("\n[6/10] Generating weight matrix visualization...")
    plot_6_weight_matrices(checkpoint)

    print("\n[7/10] Generating noise correlation analysis...")
    plot_7_noise_correlations(trial_data)

    print("\n[8/10] Generating dimensionality analysis...")
    plot_8_dimensionality_over_time(trial_data)

    print("\n[9/10] Generating decoding analysis...")
    plot_9_single_trial_decoding(trial_data)

    print("\n[10/10] Generating clustering analysis...")
    plot_10_neuron_clustering(trial_data)

    print("\n" + "="*70)
    print("All 10 plots generated successfully!")
    print("Review images/analysis_*.png files and select your 2 favorites!")
    print("="*70)


if __name__ == "__main__":
    main()
