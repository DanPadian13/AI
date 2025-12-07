import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import neurogym as ngym
import os
import matplotlib.ticker as mticker
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from Question_2a import Net

device = torch.device('cpu')
print(f"Using device: {device}")

def plot_training_curves(loss_dict, lr_schedule=None, output_path='images/q2_multisensory_training_curves.png'):
    """Plot training curves for all models."""
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))

    colors = {'vanilla': '#1f77b4', 'leaky': '#ff7f0e',
              'leaky_fa': '#2ca02c', 'bio': '#d62728'}
    labels = {'vanilla': 'Vanilla RNN', 'leaky': 'Leaky RNN',
              'leaky_fa': 'Leaky RNN + FA', 'bio': 'Bio-Realistic RNN'}

    for model_name, loss_history in loss_dict.items():
        steps = np.arange(len(loss_history)) * 200 + 200
        ax1.semilogy(steps, loss_history, label=labels[model_name],
                     color=colors[model_name], linewidth=2.5)

    ax1.set_xlabel('Training Steps', fontsize=14)
    ax1.set_ylabel('Loss (log scale)', fontsize=14)
    ax1.set_title('MultiSensoryIntegration-v0: Training Curves', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_ylim(0.05, 0.2)
    ax1.set_yticks([0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax1.minorticks_off()
    ax1.grid(True, alpha=0.3, which='both')

    if lr_schedule is not None and len(lr_schedule) == len(next(iter(loss_dict.values()))):
        ax2 = ax1.twinx()
        steps = np.arange(len(lr_schedule)) * 200 + 200
        ax2.plot(steps, lr_schedule, color='gray', linestyle='--', linewidth=2, label='Learning Rate')
        ax2.set_ylabel('Learning Rate', fontsize=13, color='gray')
        ax2.tick_params(axis='y', labelsize=12, colors='gray')
        ax2.grid(False)
        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper right')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_task_structure(env, output_path='images/q2_multisensory_task_structure.png'):
    """Visualize task structure as a single continuous heatmap across trials."""
    n_trials = 3
    row_labels = ['Fixation', 'Mod1 Left', 'Mod1 Right', 'Mod2 Left', 'Mod2 Right']
    data_list = []
    decisions = []

    for _ in range(n_trials):
        env.new_trial()
        ob, gt = env.ob, env.gt
        trial = env.trial

        fix = ob[:, 0]
        m1L, m1R = ob[:, 1], ob[:, 2]
        m2L, m2R = ob[:, 3], ob[:, 4]
        decision = np.where(gt == 1, 1.0, np.where(gt == 2, -1.0, 0.0))
        data = np.stack([fix, m1L, m1R, m2L, m2R], axis=0)
        data_list.append((data, trial, gt))
        decisions.append(decision)

    total_len = sum(d.shape[1] for d, _, _ in data_list)
    combined = np.zeros((len(row_labels), total_len))
    boundaries = []
    titles = []
    decision_track = np.zeros(total_len)

    cursor = 0
    for idx, ((data, trial, gt), decision) in enumerate(zip(data_list, decisions)):
        L = data.shape[1]
        combined[:, cursor:cursor + L] = data
        decision_track[cursor:cursor + L] = decision
        boundaries.append(cursor)
        coh = trial.get('coh', 0)
        coh_prop = trial.get('coh_prop', 0)
        final_action = {0: 'Fixate', 1: 'Left', 2: 'Right'}.get(int(gt[-1]), str(gt[-1]))
        titles.append(f'Trial {idx + 1}: coh={coh}, weight={coh_prop:.2f}, choice={final_action}')
        cursor += L
    boundaries.append(total_len)

    time = np.arange(total_len) * env.dt

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(combined, aspect='auto', cmap='viridis', origin='lower',
                   extent=[time[0], time[-1], -0.5, combined.shape[0] - 0.5])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel('Time (ms)', fontsize=16)
    ax.set_xticklabels(ax.get_xticks(), fontsize=14)
    ax.set_title('MultiSensoryIntegration-v0 task structure', fontsize=18, fontweight='bold')

    # Horizontal white grid lines separating rows
    for y in np.arange(-0.5, combined.shape[0] + 0.5, 1.0):
        ax.axhline(y, color='white', linewidth=0.8, alpha=0.7)

    # Add a decision row as its own band (green=Left, red=Right, purple=0)
    dec_row = np.zeros_like(decision_track)
    dec_row = np.where(decision_track > 0, 1.0, np.where(decision_track < 0, -1.0, 0.0))
    # Build a combined plot with an extra row for decision
    dec_extent = [time[0], time[-1], combined.shape[0] - 0.5, combined.shape[0] + 0.5]
    ax.imshow(dec_row[np.newaxis, :], aspect='auto', cmap='PiYG', origin='lower',
              extent=[time[0], time[-1], combined.shape[0] - 0.5, combined.shape[0] + 0.5])
    ax.axhline(combined.shape[0] - 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylim(-0.5, combined.shape[0] + 1.0)
    yticks = list(ax.get_yticks())
    ax.set_yticks(np.arange(len(row_labels) + 1))
    ax.set_yticklabels(row_labels + ['Decision'])

    # Removed vertical boundary lines for a cleaner view

    # Annotate trials near top
    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        t_mid = time[start] + 0.5 * (time[end - 1] - time[start])
        ax.text(t_mid, len(row_labels) - 0.05, titles[idx], ha='center', va='top', fontsize=15, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Activation', fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_comparison(perf_dict, bal_acc_dict, output_path='images/q2_multisensory_performance.png'):
    """Compare final performance across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = ['vanilla', 'leaky', 'leaky_fa', 'bio']
    labels = ['Vanilla\nRNN', 'Leaky\nRNN', 'Leaky RNN\n+ FA', 'Bio-Realistic\nRNN']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    perfs = [perf_dict[m] for m in models]
    bal_accs = [bal_acc_dict[m] for m in models]

    # Plot accuracy
    ax = axes[0]
    bars = ax.bar(labels, perfs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, perf in zip(bars, perfs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{perf:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (80%)')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # Plot balanced accuracy
    ax = axes[1]
    bars = ax.bar(labels, bal_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, bal_acc in zip(bars, bal_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{bal_acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (80%)')
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    plt.suptitle('MultiSensoryIntegration-v0: Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_hidden_activity(trial_data, env, model_name, output_path_prefix='images/q2_multisensory'):
    """Analyze hidden unit activity during trials."""
    activities = np.array([trial_data['activities'][i] for i in range(len(trial_data['activities']))])
    correct_trials = np.array(trial_data['correct'])
    correct_activities = activities[correct_trials]
    avg_activity = np.mean(correct_activities, axis=0)
    time = np.arange(avg_activity.shape[0]) * env.dt

    # Heatmap-only grid (2x2) saved separately
    def plot_heatmaps():
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
        titles = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
        model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']
        for ax, key, title in zip(axes.flatten(), model_keys, titles):
            td = trial_data_dict[key]
            acts = np.array(td['activities'])[np.array(td['correct'])]
            avg_act = acts.mean(axis=0)
            im = ax.imshow(avg_act.T, aspect='auto', cmap='viridis',
                           extent=[0, (avg_act.shape[0]-1)*env.dt, 0, avg_act.shape[1]])
            ax.set_title(title, fontsize=20, fontweight='bold')
            ax.set_xlabel('Time (ms)', fontsize=18, fontweight='bold')
            ax.set_ylabel('Hidden Unit', fontsize=18, fontweight='bold')
            ax.tick_params(axis='both', labelsize=15)
        # Move colorbar to bottom spanning all axes
        cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Activity', fontsize=18, fontweight='bold')
        cbar.ax.tick_params(labelsize=15)
        # Add main title
        fig.suptitle('Hidden Unit Activity Heatmaps', fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
        plt.savefig('images/q2_multisensory_heatmaps.png', dpi=200, bbox_inches='tight')
        plt.close()

    # Mean-activity grid (2x2) saved separately
    def plot_mean_traces():
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        titles = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
        model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']
        for ax, key, title in zip(axes.flatten(), model_keys, titles):
            td = trial_data_dict[key]
            acts = np.array(td['activities'])[np.array(td['correct'])]
            avg_act = acts.mean(axis=0)
            mean_activity = np.mean(avg_act, axis=1)
            std_activity = np.std(avg_act, axis=1)
            t = np.arange(avg_act.shape[0]) * env.dt
            ax.plot(t, mean_activity, linewidth=2.2, color='blue', label='Mean')
            ax.fill_between(t, mean_activity - std_activity, mean_activity + std_activity,
                            alpha=0.25, color='blue', label='± 1 SD')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Activity', fontsize=12)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.3)
        axes[0,0].legend(fontsize=11, loc='upper left')
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12)
        os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
        plt.savefig('images/q2_multisensory_mean_activity.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Defer plotting to the main routine where trial_data_dict is available
    return plot_heatmaps, plot_mean_traces


def plot_pca_hidden_states_subplots(trial_data_dict, output_path='images/q2_multisensory_pca.png'):
    """PCA scatter for each model on a 2x2 grid, colored by final action."""
    model_order = [
        ('vanilla', 'Vanilla RNN'),
        ('leaky', 'Leaky RNN'),
        ('leaky_fa', 'Leaky RNN + FA'),
        ('bio', 'Bio-Realistic RNN'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    colors = {0: '#7f7f7f', 1: '#1f77b4', 2: '#d62728'}
    labels = {0: 'Fixate', 1: 'Left', 2: 'Right'}

    for ax, (key, name) in zip(axes.flatten(), model_order):
        if key not in trial_data_dict:
            ax.axis('off')
            continue

        trial_data = trial_data_dict[key]
        activities = np.array(trial_data['activities'])
        targets = np.array(trial_data['ground_truths'])

        if activities.ndim == 2:
            activities = activities[None, :, :]

        mean_states = activities.mean(axis=1)

        pca = PCA(n_components=2)
        proj = pca.fit_transform(mean_states)
        var_exp = pca.explained_variance_ratio_ * 100

        for cls in np.unique(targets):
            mask = targets == cls
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=colors.get(int(cls), '#999999'), label=labels.get(int(cls), f'Class {cls}'),
                       alpha=0.7, s=50, edgecolors='k', linewidths=0.5)

        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var)', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var)', fontsize=16, fontweight='bold')
        ax.set_title(name, fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper right', fontsize=15, frameon=True, shadow=True)

    # Add main title
    fig.suptitle('PCA of Hidden State Representations', fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA subplot figure: {output_path}")


def plot_choice_decoding_timecourse(trial_data_dict, output_path='images/q2_multisensory_choice_decoding.png'):
    """Train linear decoders per timestep to track when choice becomes linearly separable."""
    fig, ax = plt.subplots(figsize=(10, 6))
    model_order = [
        ('vanilla', 'Vanilla RNN', '#1f77b4'),
        ('leaky', 'Leaky RNN', '#ff7f0e'),
        ('leaky_fa', 'Leaky RNN + FA', '#2ca02c'),
        ('bio', 'Bio-Realistic RNN', '#d62728'),
    ]

    for key, name, color in model_order:
        if key not in trial_data_dict:
            continue

        trial_data = trial_data_dict[key]
        activities = np.array(trial_data['activities'])  # [N, T, H]
        labels = np.array(trial_data['ground_truths']).astype(int)

        if activities.ndim == 2:
            activities = activities[None, :, :]

        num_trials, num_steps, _ = activities.shape
        split = int(0.7 * num_trials)
        idx = np.arange(num_trials)
        np.random.shuffle(idx)
        train_idx, test_idx = idx[:split], idx[split:]

        accs = []
        for t in range(num_steps):
            X_train = activities[train_idx, t, :]
            y_train = labels[train_idx]
            X_test = activities[test_idx, t, :]
            y_test = labels[test_idx]

            clf = LogisticRegression(max_iter=200, multi_class='auto')
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            accs.append(accuracy_score(y_test, preds))

        ax.plot(np.arange(num_steps), accs, color=color, label=name, linewidth=2)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Choice decoding accuracy')
    ax.set_title('Linear choice decoding over time')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved choice decoding plot: {output_path}")


def plot_unit_selectivity_heatmap(trial_data_dict, env, output_path='images/q2_multisensory_unit_selectivity.png'):
    """Sort units by left/right d' and show mean activity traces per model."""
    model_order = [
        ('vanilla', 'Vanilla RNN'),
        ('leaky', 'Leaky RNN'),
        ('leaky_fa', 'Leaky RNN + FA'),
        ('bio', 'Bio-Realistic RNN'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (key, name) in zip(axes.flatten(), model_order):
        if key not in trial_data_dict:
            ax.axis('off')
            continue

        trial_data = trial_data_dict[key]
        activities = np.array(trial_data['activities'])
        targets = np.array(trial_data['ground_truths']).astype(int)

        if activities.ndim == 2:
            activities = activities[None, :, :]

        mask_lr = (targets == 1) | (targets == 2)
        if mask_lr.sum() < 2:
            ax.axis('off')
            continue

        acts_lr = activities[mask_lr]
        lbls_lr = targets[mask_lr]

        mean_left = acts_lr[lbls_lr == 1].mean(axis=(0, 1))
        mean_right = acts_lr[lbls_lr == 2].mean(axis=(0, 1))
        var_left = acts_lr[lbls_lr == 1].var(axis=(0, 1))
        var_right = acts_lr[lbls_lr == 2].var(axis=(0, 1))
        d_prime = (mean_left - mean_right) / np.sqrt(0.5 * (var_left + var_right) + 1e-8)

        sort_idx = np.argsort(-np.abs(d_prime))
        acts_sorted = acts_lr[:, :, sort_idx].mean(axis=0).T  # [H, T]

        im = ax.imshow(acts_sorted, aspect='auto', cmap='bwr',
                       extent=[0, acts_sorted.shape[1] * env.dt, 0, acts_sorted.shape[0]])
        ax.set_title(name)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel("Units (sorted by |d'|)")
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Avg activity')

    plt.suptitle("Unit selectivity: left vs right (sorted by d' magnitude)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved unit selectivity heatmap: {output_path}")


def plot_example_predictions(trial_data_dict, models_dict, env, output_path='images/q2_multisensory_example_predictions.png'):
    """Plot example trials showing model predictions."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        net = models_dict[model_key]
        net.eval()
        ax = axes[idx]

        # Generate a trial
        env.new_trial()
        ob, gt = env.ob, env.gt
        trial = env.trial

        with torch.no_grad():
            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
            action_pred, _, _ = net(inputs)
            action_pred_np = action_pred.detach().cpu().numpy()[:, 0, :]

        # Calculate probabilities
        pred_probs = np.exp(action_pred_np) / np.exp(action_pred_np).sum(axis=1, keepdims=True)

        time = np.arange(len(ob)) * env.dt

        # Plot ground truth
        ax.plot(time, gt * 1.5, 'k-', linewidth=3, label='Ground Truth', alpha=0.7)

        # Plot predicted probabilities
        ax.plot(time, pred_probs[:, 0], 'gray', linewidth=2, label='P(Fixate)', alpha=0.6, linestyle='--')
        ax.plot(time, pred_probs[:, 1], 'b-', linewidth=2, label='P(Left)', alpha=0.8)
        ax.plot(time, pred_probs[:, 2], 'r-', linewidth=2, label='P(Right)', alpha=0.8)

        # Check if correct
        final_pred = np.argmax(pred_probs[-1, :])
        final_true = gt[-1]
        correct = '✓' if final_pred == final_true else '✗'

        coh = trial.get('coh', 0)
        coh_prop = trial.get('coh_prop', 0)

        ax.set_ylabel('Probability / Action', fontsize=11)
        ax.set_ylim([-0.1, 2.0])
        ax.set_title(f'{model_name} {correct} | Coherence={coh}, Weight={coh_prop:.2f} | Pred={final_pred}, True={final_true}',
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, ncol=4)
        ax.grid(True, alpha=0.3)

        if idx == 3:
            ax.set_xlabel('Time (ms)', fontsize=11)

    plt.suptitle('Example Trial: Model Predictions Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_coherence_difficulty(trial_data_dict, output_path='images/q2_multisensory_coherence_analysis.png'):
    """Analyze accuracy as a function of coherence (task difficulty)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        trial_data = trial_data_dict[model_key]

        # Extract coherences and correctness
        coherences = []
        correct_list = []

        for i, trial_info in enumerate(trial_data['trial_info']):
            coh = trial_info.get('coh', 0)
            coherences.append(coh)
            correct_list.append(trial_data['correct'][i])

        coherences = np.array(coherences)
        correct_list = np.array(correct_list)

        # Calculate accuracy for each coherence level
        unique_cohs = sorted(np.unique(coherences))
        accuracies = []
        counts = []

        for coh in unique_cohs:
            mask = coherences == coh
            if np.sum(mask) > 0:
                acc = np.mean(correct_list[mask])
                accuracies.append(acc)
                counts.append(np.sum(mask))

        ax = axes[idx]
        ax.plot(unique_cohs, accuracies, 'o-', markersize=8, linewidth=2,
               color=plt.cm.viridis(idx/3), label=model_name)

        # Add counts as text
        for coh, acc, count in zip(unique_cohs, accuracies, counts):
            ax.text(coh, acc + 0.02, f'n={count}', ha='center', fontsize=8)

        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Chance')
        ax.set_xlabel('Coherence (Difficulty)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle('Accuracy vs Coherence (Higher = Easier)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(trial_data_dict, output_path='images/q2_multisensory_confusion_matrices.png'):
    """Plot confusion matrices for all models (Left vs Right only, excluding Fixate)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']
    action_names = ['Left (1)', 'Right (2)']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        trial_data = trial_data_dict[model_key]

        predictions = np.array(trial_data['predictions'])
        ground_truths = np.array(trial_data['ground_truths'])

        # Filter out fixation trials (class 0) - only keep Left (1) and Right (2)
        decision_mask = (ground_truths == 1) | (ground_truths == 2)
        predictions_filtered = predictions[decision_mask]
        ground_truths_filtered = ground_truths[decision_mask]

        # Create 2x2 confusion matrix for Left vs Right only
        confusion = np.zeros((2, 2))
        for i, true_val in enumerate([1, 2]):  # Left=1, Right=2
            for j, pred_val in enumerate([1, 2]):
                confusion[i, j] = np.sum((ground_truths_filtered == true_val) & (predictions_filtered == pred_val))

        # Normalize by row (ground truth)
        confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-10)

        # Calculate accuracy for decision trials only
        decision_accuracy = np.sum(predictions_filtered == ground_truths_filtered) / len(predictions_filtered) if len(predictions_filtered) > 0 else 0

        # Calculate balanced accuracy (average of Left recall and Right recall)
        left_recall = confusion_norm[0, 0] if confusion.sum(axis=1)[0] > 0 else 0
        right_recall = confusion_norm[1, 1] if confusion.sum(axis=1)[1] > 0 else 0
        balanced_acc = (left_recall + right_recall) / 2

        ax = axes[idx]
        im = ax.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)

        # Add text annotations
        for i in range(2):
            for j in range(2):
                count = int(confusion[i, j])
                pct = confusion_norm[i, j]
                text = f'{count}\n({pct:.1%})'
                ax.text(j, i, text, ha='center', va='center', fontsize=16,
                       color='white' if pct > 0.5 else 'black', fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(action_names, fontsize=15)
        ax.set_yticklabels(action_names, fontsize=15)
        ax.set_xlabel('Predicted', fontsize=17, fontweight='bold')
        ax.set_ylabel('Ground Truth', fontsize=17, fontweight='bold')

        ax.set_title(f'{model_name}\nDecision Acc: {decision_accuracy:.3f} | Bal Acc: {balanced_acc:.3f}',
                    fontsize=18, fontweight='bold')

        # Add colorbar with larger font
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)

    plt.suptitle('Confusion Matrices: Left vs Right Decision Performance', fontsize=20, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_modality_weighting_analysis(trial_data_dict, output_path='images/q2_multisensory_modality_weighting.png'):
    """Analyze how models perform across different modality weightings."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        trial_data = trial_data_dict[model_key]

        # Extract modality weights and correctness
        coh_props = []
        correct_list = []

        for i, trial_info in enumerate(trial_data['trial_info']):
            coh_prop = trial_info.get('coh_prop', 0.5)
            coh_props.append(coh_prop)
            correct_list.append(trial_data['correct'][i])

        coh_props = np.array(coh_props)
        correct_list = np.array(correct_list)

        # Bin modality weights
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

        accuracies = []
        counts = []
        bin_centers = []

        for i in range(len(bins)-1):
            mask = (coh_props >= bins[i]) & (coh_props < bins[i+1])
            if np.sum(mask) > 0:
                acc = np.mean(correct_list[mask])
                accuracies.append(acc)
                counts.append(np.sum(mask))
                bin_centers.append((bins[i] + bins[i+1]) / 2)

        ax = axes[idx]
        bars = ax.bar(range(len(bin_centers)), accuracies, alpha=0.7,
                     color=plt.cm.viridis(idx/3), edgecolor='black', linewidth=1.5)

        # Add counts
        for i, (acc, count) in enumerate(zip(accuracies, counts)):
            ax.text(i, acc + 0.02, f'n={count}', ha='center', fontsize=8)

        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Chance')
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.set_xlabel('Modality Weight (coh_prop)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)

    plt.suptitle('Accuracy vs Modality Weighting (Multi-Sensory Integration)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_decision_confidence(trial_data_dict, models_dict, env, output_path='images/q2_multisensory_decision_confidence.png'):
    """Analyze decision confidence (output probability distribution)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        net = models_dict[model_key]
        net.eval()

        correct_confidences = []
        incorrect_confidences = []

        with torch.no_grad():
            for trial_idx in range(200):
                env.new_trial()
                ob, gt = env.ob, env.gt

                inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
                action_pred, _, _ = net(inputs)

                action_pred_np = action_pred.detach().cpu().numpy()[-1, 0, :]
                pred_probs = np.exp(action_pred_np) / np.sum(np.exp(action_pred_np))

                final_pred = np.argmax(pred_probs)
                final_true = gt[-1]
                confidence = pred_probs[final_pred]

                if final_pred == final_true:
                    correct_confidences.append(confidence)
                else:
                    incorrect_confidences.append(confidence)

        ax = axes[idx]

        # Histogram of confidences
        bins = np.linspace(0, 1, 20)
        ax.hist(correct_confidences, bins=bins, alpha=0.7, color='green',
               label=f'Correct (n={len(correct_confidences)})', edgecolor='black')
        ax.hist(incorrect_confidences, bins=bins, alpha=0.7, color='red',
               label=f'Incorrect (n={len(incorrect_confidences)})', edgecolor='black')

        # Add mean lines
        if correct_confidences:
            ax.axvline(np.mean(correct_confidences), color='darkgreen', linestyle='--',
                      linewidth=2.5, label=f'Mean Correct: {np.mean(correct_confidences):.3f}')
        if incorrect_confidences:
            ax.axvline(np.mean(incorrect_confidences), color='darkred', linestyle='--',
                      linewidth=2.5, label=f'Mean Incorrect: {np.mean(incorrect_confidences):.3f}')

        ax.set_xlabel('Confidence (Max Probability)', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title(f'{model_name}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Decision Confidence: Correct vs Incorrect Predictions', fontsize=17, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_temporal_dynamics(trial_data_dict, models_dict, env, output_path='images/q2_multisensory_temporal_dynamics.png'):
    """Show how decision confidence evolves over time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        net = models_dict[model_key]
        net.eval()

        # Collect trajectories for correct trials
        left_trajectories = []
        right_trajectories = []

        with torch.no_grad():
            for trial_idx in range(100):
                env.new_trial()
                ob, gt = env.ob, env.gt

                inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
                action_pred, _, _ = net(inputs)

                action_pred_np = action_pred.detach().cpu().numpy()[:, 0, :]
                pred_probs = np.exp(action_pred_np) / np.exp(action_pred_np).sum(axis=1, keepdims=True)

                # Check if correct
                final_pred = np.argmax(pred_probs[-1, :])
                final_true = gt[-1]

                if final_pred == final_true:
                    if final_true == 1:  # Left
                        left_trajectories.append(pred_probs[:, 1])
                    elif final_true == 2:  # Right
                        right_trajectories.append(pred_probs[:, 2])

        ax = axes[idx]

        # Plot average trajectories
        if left_trajectories:
            left_avg = np.mean(left_trajectories, axis=0)
            left_std = np.std(left_trajectories, axis=0)
            time = np.arange(len(left_avg)) * env.dt
            ax.plot(time, left_avg, 'b-', linewidth=2, label=f'P(Left) | True=Left (n={len(left_trajectories)})')
            ax.fill_between(time, left_avg - left_std, left_avg + left_std, alpha=0.3, color='blue')

        if right_trajectories:
            right_avg = np.mean(right_trajectories, axis=0)
            right_std = np.std(right_trajectories, axis=0)
            time = np.arange(len(right_avg)) * env.dt
            ax.plot(time, right_avg, 'r-', linewidth=2, label=f'P(Right) | True=Right (n={len(right_trajectories)})')
            ax.fill_between(time, right_avg - right_std, right_avg + right_std, alpha=0.3, color='red')

        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Probability', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Temporal Dynamics: How Decisions Emerge Over Time (Correct Trials)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_analysis(trial_data_dict, output_path='images/q2_multisensory_error_analysis.png'):
    """Analyze when and how models make errors."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        trial_data = trial_data_dict[model_key]

        # Analyze errors by coherence
        coherences = []
        errors = []

        for i, trial_info in enumerate(trial_data['trial_info']):
            coh = trial_info.get('coh', 0)
            coherences.append(coh)
            errors.append(0 if trial_data['correct'][i] else 1)

        coherences = np.array(coherences)
        errors = np.array(errors)

        # Calculate error rate per coherence
        unique_cohs = sorted(np.unique(coherences))
        error_rates = []
        counts = []

        for coh in unique_cohs:
            mask = coherences == coh
            if np.sum(mask) > 0:
                error_rate = np.mean(errors[mask])
                error_rates.append(error_rate * 100)  # Convert to percentage
                counts.append(np.sum(mask))

        ax = axes[idx]
        bars = ax.bar(unique_cohs, error_rates, width=3, alpha=0.7,
                     color=plt.cm.Reds(idx/3), edgecolor='black', linewidth=1.5)

        # Add counts
        for coh, err_rate, count in zip(unique_cohs, error_rates, counts):
            if err_rate > 1:  # Only show text if error rate > 1%
                ax.text(coh, err_rate + 1, f'{int(err_rate * count / 100)}/{count}',
                       ha='center', fontsize=8)

        ax.set_xlabel('Coherence (Task Difficulty)', fontsize=11)
        ax.set_ylabel('Error Rate (%)', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, max(max(error_rates) * 1.2, 5)])
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Error Analysis: Where Do Models Fail?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_architecture_comparison_radar(perf_dict, bal_acc_dict, trial_data_dict,
                                       output_path='images/q2_multisensory_radar_comparison.png'):
    """Radar plot comparing models across multiple metrics."""
    from math import pi

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Define metrics
    categories = ['Accuracy', 'Balanced Acc', 'Left Recall', 'Right Recall', 'Consistency']
    N = len(categories)

    # Calculate metrics for each model
    for idx, (model_key, model_name, color) in enumerate(zip(model_keys, model_names, colors)):
        predictions = np.array(trial_data_dict[model_key]['predictions'])
        ground_truths = np.array(trial_data_dict[model_key]['ground_truths'])

        # Calculate recalls
        left_mask = ground_truths == 1
        right_mask = ground_truths == 2
        left_recall = np.sum((predictions == 1) & left_mask) / np.sum(left_mask) if np.sum(left_mask) > 0 else 0
        right_recall = np.sum((predictions == 2) & right_mask) / np.sum(right_mask) if np.sum(right_mask) > 0 else 0

        # Consistency: 1 - std of per-trial confidence
        consistency = 1 - (abs(left_recall - right_recall) / 2)  # How balanced are the recalls

        values = [
            perf_dict[model_key],
            bal_acc_dict[model_key],
            left_recall,
            right_recall,
            consistency
        ]

        # Compute angle for each axis
        angles = [n / float(N) * 2 * pi for n in range(N)]
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.title('Multi-Dimensional Model Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_connectivity_matrices(checkpoint, output_path='images/q2_multisensory_connectivity.png'):
    """
    Plot connectivity heatmaps showing recurrent weight matrices for all models.
    Shows how neurons are connected to each other in each model.
    """
    print("Generating connectivity matrices plot...")

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
            model_kwargs = {'dt': 20, 'tau': 100, 'sigma_rec': 0.1}
        if model_type == 'bio_realistic':
            model_kwargs['exc_ratio'] = 0.8

        # Get input/output sizes from environment
        input_size = 5  # MultiSensoryIntegration has 5 inputs
        output_size = 3  # MultiSensoryIntegration has 3 outputs

        # Check if bio model uses old architecture
        if model_type == 'bio_realistic' and 'rnn.h2h.weight' in saved_state:
            # Old architecture - load as leaky_fa
            model_type = 'leaky_fa'

        net = Net(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                 model_type=model_type, **model_kwargs).to(device)
        net.load_state_dict(checkpoint[state_dict])

        # Extract recurrent weight matrix
        if model_type == 'vanilla':
            weight_matrix = net.rnn.h2h.weight.detach().cpu().numpy()
        elif model_type == 'bio_realistic':
            # Bio-realistic: reconstruct full weight matrix from separate E/I weights
            n_exc = net.rnn.n_exc
            n_inh = net.rnn.n_inh

            weight_matrix = np.zeros((hidden_size, hidden_size))
            exc_weights = torch.relu(net.rnn.h2h_exc.weight).detach().cpu().numpy()
            weight_matrix[:, :n_exc] = exc_weights
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

        ax.set_xlabel('From Neuron', fontsize=18, fontweight='bold')
        ax.set_ylabel('To Neuron', fontsize=18, fontweight='bold')
        ax.set_title(f'{name}', fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=15)

        # Add statistics
        sparsity = np.mean(np.abs(weight_matrix) < 0.01)
        mean_weight = np.mean(np.abs(weight_matrix))

        if model_type == 'bio_realistic' and 'n_exc' in locals():
            # For bio-realistic, show E/I neuron counts
            text_str = f'Sparsity: {sparsity:.2%}\nMean |W|: {mean_weight:.3f}\nE/I: {n_exc}/{n_inh}'
            # Add line to separate excitatory and inhibitory
            ax.axhline(y=n_exc-0.5, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
            ax.axvline(x=n_exc-0.5, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
        else:
            text_str = f'Sparsity: {sparsity:.2%}\nMean |W|: {mean_weight:.3f}'

        ax.text(0.98, 0.98, text_str, transform=ax.transAxes,
               fontsize=14, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Recurrent Connectivity Matrices', fontsize=24, fontweight='bold', y=0.98)

    # Add single shared colorbar
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label('Weight Strength', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout(rect=[0, 0, 0.90, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_confusion_matrix_stats(trial_data_dict):
    """Print detailed confusion matrix statistics to console."""
    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for model_key, model_name in zip(model_keys, model_names):
        trial_data = trial_data_dict[model_key]
        predictions = np.array(trial_data['predictions'])
        ground_truths = np.array(trial_data['ground_truths'])

        # Create confusion matrix
        num_classes = 3
        confusion = np.zeros((num_classes, num_classes))
        for true_val in range(num_classes):
            for pred_val in range(num_classes):
                confusion[true_val, pred_val] = np.sum((ground_truths == true_val) & (predictions == pred_val))

        # Normalize by row
        confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-10)

        # Calculate metrics
        per_class_recalls = []
        for cls in range(num_classes):
            mask = ground_truths == cls
            if np.sum(mask) > 0:
                recall = np.sum((predictions == cls) & mask) / np.sum(mask)
                per_class_recalls.append(recall)

        balanced_acc = np.mean(per_class_recalls) if per_class_recalls else 0.0
        accuracy = np.sum(predictions == ground_truths) / len(predictions)

        print(f'\n{model_name} Confusion Matrix:')
        print('='*70)
        print(f'{"":15s} {"Pred Fixate":>14s} {"Pred Left":>14s} {"Pred Right":>14s}')
        print('-'*70)
        for i, true_label in enumerate(['True Fixate', 'True Left', 'True Right']):
            print(f'{true_label:15s}', end='')
            for j in range(num_classes):
                count = int(confusion[i, j])
                pct = confusion_norm[i, j]
                print(f' {count:4d} ({pct:5.1%})', end='')
            print()

        print(f'\nAccuracy: {accuracy:.3f}')
        print(f'Balanced Accuracy: {balanced_acc:.3f}')

        # Print per-class recalls (only for classes that exist in ground truth)
        class_names = ['Fixate', 'Left', 'Right']
        unique_classes = sorted(np.unique(ground_truths))
        print(f'Per-class Recall: ', end='')
        recall_strs = []
        for i, cls in enumerate(unique_classes):
            if i < len(per_class_recalls):
                recall_strs.append(f'{class_names[int(cls)]}={per_class_recalls[i]:.3f}')
        print(', '.join(recall_strs))

        # Show which classes are present
        print(f'Classes in test set: {[class_names[int(c)] for c in unique_classes]}')


if __name__ == '__main__':
    print("="*70)
    print("Question 2: Analysis of MultiSensoryIntegration-v0 Results")
    print("="*70)
    print()

    print("[1] Loading saved models and data...")
    checkpoint = torch.load('checkpoints/question_2_multisensory_models_and_data.pt', weights_only=False)

    env_config = checkpoint['env_config']
    task = env_config['task']
    kwargs_env = {'dt': env_config['dt']}
    seq_len = env_config['seq_len']

    dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=16, seq_len=seq_len)
    env = dataset.env

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # Infer hidden size from checkpoint
    vanilla_fc_weight_shape = checkpoint['vanilla_model']['fc.weight'].shape
    hidden_size = vanilla_fc_weight_shape[1]

    print(f"Task: {task}")
    print(f"Loaded checkpoint with {len(checkpoint.keys())} items")
    print(f"Detected hidden_size: {hidden_size}")
    print()

    print("[2] Reconstructing models...")
    models_dict = {}

    net_vanilla = Net(input_size, hidden_size, output_size, model_type='vanilla').to(device)
    net_vanilla.load_state_dict(checkpoint['vanilla_model'])
    models_dict['vanilla'] = net_vanilla

    net_leaky = Net(input_size, hidden_size, output_size, model_type='leaky',
                    dt=env.dt, tau=100, sigma_rec=0.1).to(device)
    net_leaky.load_state_dict(checkpoint['leaky_model'])
    models_dict['leaky'] = net_leaky

    net_leaky_fa = Net(input_size, hidden_size, output_size, model_type='leaky_fa',
                       dt=env.dt, tau=100, sigma_rec=0.1).to(device)
    net_leaky_fa.load_state_dict(checkpoint['leaky_fa_model'])
    models_dict['leaky_fa'] = net_leaky_fa

    # Check if bio model uses old or new architecture
    bio_state_dict = checkpoint['bio_model']
    if 'rnn.h2h_exc.weight' in bio_state_dict:
        # New architecture with separate exc/inh weights
        net_bio = Net(input_size, hidden_size, output_size, model_type='bio_realistic',
                      dt=env.dt, tau=100, sigma_rec=0.1, exc_ratio=0.8).to(device)
        net_bio.load_state_dict(bio_state_dict)
    else:
        # Old architecture with Dale mask - load as leaky_fa instead
        print("  Note: Bio model checkpoint uses old architecture (Dale mask)")
        print("        Loading as Leaky+FA model instead")
        net_bio = Net(input_size, hidden_size, output_size, model_type='leaky_fa',
                      dt=env.dt, tau=100, sigma_rec=0.1).to(device)
        net_bio.load_state_dict(bio_state_dict)

    models_dict['bio'] = net_bio

    print("Models reconstructed successfully")
    print()

    trial_data_dict = checkpoint['trial_data_dict']
    loss_dict = checkpoint['loss_dict']
    perf_dict = checkpoint['perf_dict']
    bal_acc_dict = checkpoint['bal_acc_dict']

    print("[3] Performance Summary:")
    print("-"*70)
    print(f"Vanilla RNN:          Acc: {perf_dict['vanilla']:.3f}, Bal Acc: {bal_acc_dict['vanilla']:.3f}")
    print(f"Leaky RNN:            Acc: {perf_dict['leaky']:.3f}, Bal Acc: {bal_acc_dict['leaky']:.3f}")
    print(f"Leaky RNN + FA:       Acc: {perf_dict['leaky_fa']:.3f}, Bal Acc: {bal_acc_dict['leaky_fa']:.3f}")
    print(f"Bio-Realistic RNN:    Acc: {perf_dict['bio']:.3f}, Bal Acc: {bal_acc_dict['bio']:.3f}")
    print()

    print("[4] Visualizing task structure...")
    plot_task_structure(env)
    print()

    print("[5] Generating training curves...")
    plot_training_curves(loss_dict)
    print()

    print("[6] Generating performance comparison...")
    plot_performance_comparison(perf_dict, bal_acc_dict)
    print()

    print("[7] Analyzing hidden unit activity...")
    # Generate 2x2 grids for heatmaps and mean traces
    plot_heatmaps_fn, plot_mean_traces_fn = analyze_hidden_activity(trial_data_dict['vanilla'], env, 'Vanilla RNN')
    # The above returns closures that need trial_data_dict; call them here
    plot_heatmaps_fn()
    plot_mean_traces_fn()
    print()

    print("[7b] PCA of hidden states (per-trial means)...")
    plot_pca_hidden_states_subplots(trial_data_dict)
    print()

    print("[7c] Choice decoding timecourse...")
    plot_choice_decoding_timecourse(trial_data_dict)
    print()

    print("[7d] Unit selectivity heatmaps...")
    plot_unit_selectivity_heatmap(trial_data_dict, env)
    print()

    print("[8] Plotting example predictions...")
    plot_example_predictions(trial_data_dict, models_dict, env)
    print()

    print("[9] Analyzing coherence-difficulty relationship...")
    analyze_coherence_difficulty(trial_data_dict)
    print()

    print("[10] Generating confusion matrices...")
    plot_confusion_matrices(trial_data_dict)
    print()

    print("[11] Printing detailed confusion matrix statistics...")
    print_confusion_matrix_stats(trial_data_dict)
    print()

    print("[12] Analyzing modality weighting effects...")
    plot_modality_weighting_analysis(trial_data_dict)
    print()

    print("[13] Analyzing decision confidence...")
    plot_decision_confidence(trial_data_dict, models_dict, env)
    print()

    print("[14] Analyzing temporal dynamics...")
    plot_temporal_dynamics(trial_data_dict, models_dict, env)
    print()

    print("[15] Analyzing error patterns...")
    plot_error_analysis(trial_data_dict)
    print()

    print("[16] Creating radar comparison plot...")
    plot_architecture_comparison_radar(perf_dict, bal_acc_dict, trial_data_dict)
    print()

    print("[17] Creating connectivity heatmaps...")
    plot_connectivity_matrices(checkpoint)
    print()

    print("="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nGenerated plots:")
    print("  - images/q2_multisensory_task_structure.png")
    print("  - images/q2_multisensory_training_curves.png")
    print("  - images/q2_multisensory_performance.png")
    print("  - images/q2_multisensory_*_activity.png (4 plots)")
    print("  - images/q2_multisensory_example_predictions.png")
    print("  - images/q2_multisensory_coherence_analysis.png")
    print("  - images/q2_multisensory_confusion_matrices.png")
    print("  - images/q2_multisensory_modality_weighting.png")
    print("  - images/q2_multisensory_decision_confidence.png")
    print("  - images/q2_multisensory_temporal_dynamics.png")
    print("  - images/q2_multisensory_error_analysis.png")
    print("  - images/q2_multisensory_radar_comparison.png")
    print("  - images/q2_multisensory_connectivity.png")
    print("\nKey findings to discuss:")
    print("  - Do all models achieve high accuracy (>80%)?")
    print("  - How does performance vary with coherence?")
    print("  - Which architectures best integrate multi-modal information?")
    print("  - Compare biological constraints vs performance")
    print("  - How do models integrate weighted sensory inputs?")
    print("  - Are models well-calibrated (confidence matches accuracy)?")
    print("  - How do decisions emerge over time?")
    print("="*70)
