import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import neurogym as ngym
import os
from sklearn.decomposition import PCA

from Question_2a import Net

device = torch.device('cpu')
print(f"Using device: {device}")

def plot_training_curves(loss_dict, output_path='images/q2_multisensory_training_curves.png'):
    """Plot training curves for all models."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'vanilla': '#1f77b4', 'leaky': '#ff7f0e',
              'leaky_fa': '#2ca02c', 'bio': '#d62728'}
    labels = {'vanilla': 'Vanilla RNN', 'leaky': 'Leaky RNN',
              'leaky_fa': 'Leaky RNN + FA', 'bio': 'Bio-Realistic RNN'}

    for model_name, loss_history in loss_dict.items():
        steps = np.arange(len(loss_history)) * 200 + 200
        ax.plot(steps, loss_history, label=labels[model_name],
                color=colors[model_name], linewidth=2)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('MultiSensoryIntegration-v0: Training Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_task_structure(env, output_path='images/q2_multisensory_task_structure.png'):
    """Visualize the MultiSensoryIntegration task structure."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    for trial_idx in range(3):
        env.new_trial()
        ob, gt = env.ob, env.gt
        trial = env.trial

        time = np.arange(len(ob)) * env.dt
        ax = axes[trial_idx]

        # Plot each sensory channel
        colors = ['gray', 'blue', 'red', 'green', 'purple']
        labels = ['Fixation', 'Modality 1', 'Modality 2', 'Modality 3', 'Modality 4']

        for i in range(ob.shape[1]):
            offset = i * 0.8
            ax.plot(time, ob[:, i] + offset, color=colors[i],
                   linewidth=2, label=labels[i], alpha=0.8)

        # Plot ground truth action
        gt_offset = ob.shape[1] * 0.8 + 0.5
        ax.plot(time, gt + gt_offset, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)

        # Add trial info
        coh = trial.get('coh', 0)
        coh_prop = trial.get('coh_prop', 0)
        final_action = gt[-1]

        action_labels = {0: 'Fixate', 1: 'Left', 2: 'Right'}
        title_str = f'Trial {trial_idx + 1}: Coherence={coh}, Modality Weight={coh_prop:.2f}'

        ax.text(time[-1] * 0.95, gt_offset + final_action,
               f'→ {action_labels[final_action]}',
               fontsize=11, fontweight='bold', ha='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Input Activity', fontsize=11)
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if trial_idx == 0:
            ax.legend(loc='upper left', fontsize=9, ncol=3)

    plt.suptitle('MultiSensoryIntegration-v0: Combining Multiple Sensory Inputs',
                fontsize=14, fontweight='bold')
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

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    time = np.arange(avg_activity.shape[0]) * env.dt

    # Heatmap
    im = axes[0].imshow(avg_activity.T, aspect='auto', cmap='viridis',
                        extent=[0, time[-1], 0, avg_activity.shape[1]])
    axes[0].set_xlabel('Time (ms)', fontsize=11)
    axes[0].set_ylabel('Hidden Unit', fontsize=11)
    axes[0].set_title(f'{model_name}: Average Hidden Unit Activity (Correct Trials)',
                     fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='Activity')

    # Time series
    mean_activity = np.mean(avg_activity, axis=1)
    std_activity = np.std(avg_activity, axis=1)

    axes[1].plot(time, mean_activity, linewidth=2, color='blue', label='Mean Activity')
    axes[1].fill_between(time,
                         mean_activity - std_activity,
                         mean_activity + std_activity,
                         alpha=0.3, color='blue', label='± 1 SD')
    axes[1].set_xlabel('Time (ms)', fontsize=11)
    axes[1].set_ylabel('Activity', fontsize=11)
    axes[1].set_title(f'{model_name}: Mean Activity Across Units', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f'{output_path_prefix}_{model_name.lower().replace(" ", "_")}_activity.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_pca_hidden_states(trial_data, model_name, output_path_prefix='images/q2_multisensory'):
    """Project per-trial mean hidden states onto first two PCs and color by action."""
    activities = np.array(trial_data['activities'])  # [num_trials, T, H]
    targets = np.array(trial_data['ground_truths'])

    if activities.ndim == 2:
        activities = activities[None, :, :]

    mean_states = activities.mean(axis=1)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(mean_states)
    var_exp = pca.explained_variance_ratio_ * 100

    colors = {0: '#7f7f7f', 1: '#1f77b4', 2: '#d62728'}
    labels = {0: 'Fixate', 1: 'Left', 2: 'Right'}

    plt.figure(figsize=(7, 6))
    for cls in np.unique(targets):
        mask = targets == cls
        plt.scatter(proj[mask, 0], proj[mask, 1],
                    c=colors.get(int(cls), '#999999'), label=labels.get(int(cls), f'Class {cls}'),
                    alpha=0.7, s=35, edgecolors='k', linewidths=0.3)

    plt.xlabel(f'PC1 ({var_exp[0]:.1f}% var)')
    plt.ylabel(f'PC2 ({var_exp[1]:.1f}% var)')
    plt.title(f'{model_name}: PCA of Trial-Averaged Hidden States')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
    out_path = f'{output_path_prefix}_{model_name.lower().replace(" ", "_")}_pca.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot: {out_path}")
    print(f"Explained variance: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%")


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
            action_pred, _ = net(inputs)
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
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']
    action_names = ['Fixate (0)', 'Left (1)', 'Right (2)']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        trial_data = trial_data_dict[model_key]

        predictions = np.array(trial_data['predictions'])
        ground_truths = np.array(trial_data['ground_truths'])

        # Create confusion matrix (3x3 for 3 classes)
        num_classes = 3
        confusion = np.zeros((num_classes, num_classes))
        for true_val in range(num_classes):
            for pred_val in range(num_classes):
                confusion[true_val, pred_val] = np.sum((ground_truths == true_val) & (predictions == pred_val))

        # Normalize by row (ground truth)
        confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-10)

        # Calculate balanced accuracy
        per_class_recalls = []
        for cls in range(num_classes):
            mask = ground_truths == cls
            if np.sum(mask) > 0:
                recall = np.sum((predictions == cls) & mask) / np.sum(mask)
                per_class_recalls.append(recall)
        balanced_acc = np.mean(per_class_recalls) if per_class_recalls else 0.0

        ax = axes[idx]
        im = ax.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)

        # Add text annotations
        for i in range(num_classes):
            for j in range(num_classes):
                count = int(confusion[i, j])
                pct = confusion_norm[i, j]
                text = f'{count}\n({pct:.1%})'
                ax.text(j, i, text, ha='center', va='center', fontsize=10,
                       color='white' if pct > 0.5 else 'black', fontweight='bold')

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(action_names, fontsize=9)
        ax.set_yticklabels(action_names, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Ground Truth', fontsize=10)

        accuracy = np.sum(predictions == ground_truths) / len(predictions)
        ax.set_title(f'{model_name}\nAcc: {accuracy:.3f} | Bal Acc: {balanced_acc:.3f}',
                    fontsize=11, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Confusion Matrices: What Do Models Actually Predict?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
                action_pred, _ = net(inputs)

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
                      linewidth=2, label=f'Mean Correct: {np.mean(correct_confidences):.3f}')
        if incorrect_confidences:
            ax.axvline(np.mean(incorrect_confidences), color='darkred', linestyle='--',
                      linewidth=2, label=f'Mean Incorrect: {np.mean(incorrect_confidences):.3f}')

        ax.set_xlabel('Confidence (Max Probability)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Decision Confidence: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
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
                action_pred, _ = net(inputs)

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

    net_bio = Net(input_size, hidden_size, output_size, model_type='bio_realistic',
                  dt=env.dt, tau=100, sigma_rec=0.1, exc_ratio=0.8).to(device)
    net_bio.load_state_dict(checkpoint['bio_model'])
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
    analyze_hidden_activity(trial_data_dict['vanilla'], env, 'Vanilla RNN')
    analyze_hidden_activity(trial_data_dict['leaky'], env, 'Leaky RNN')
    analyze_hidden_activity(trial_data_dict['leaky_fa'], env, 'Leaky RNN + FA')
    analyze_hidden_activity(trial_data_dict['bio'], env, 'Bio-Realistic RNN')
    print()

    print("[7b] PCA of hidden states (per-trial means)...")
    plot_pca_hidden_states(trial_data_dict['vanilla'], 'Vanilla RNN')
    plot_pca_hidden_states(trial_data_dict['leaky'], 'Leaky RNN')
    plot_pca_hidden_states(trial_data_dict['leaky_fa'], 'Leaky RNN + FA')
    plot_pca_hidden_states(trial_data_dict['bio'], 'Bio-Realistic RNN')
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
    print("\nKey findings to discuss:")
    print("  - Do all models achieve high accuracy (>80%)?")
    print("  - How does performance vary with coherence?")
    print("  - Which architectures best integrate multi-modal information?")
    print("  - Compare biological constraints vs performance")
    print("  - How do models integrate weighted sensory inputs?")
    print("  - Are models well-calibrated (confidence matches accuracy)?")
    print("  - How do decisions emerge over time?")
    print("="*70)
