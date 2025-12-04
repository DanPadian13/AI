import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import neurogym as ngym
import os

from Question_2a import Net

device = torch.device('cpu')
print(f"Using device: {device}")


def plot_training_curves(loss_dict, output_path='images/q2c_training_curves.png'):
    """Plot training curves for all models on DelayMatchSampleDistractor1D."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'vanilla': '#1f77b4', 'leaky': '#ff7f0e',
              'leaky_fa': '#2ca02c', 'bio': '#d62728'}
    labels = {'vanilla': 'Vanilla RNN', 'leaky': 'Leaky RNN',
              'leaky_fa': 'Leaky RNN + FA', 'bio': 'Bio-Realistic RNN'}

    for model_name, loss_history in loss_dict.items():
        steps = np.arange(len(loss_history)) * 50 + 50
        ax.plot(steps, loss_history, label=labels[model_name],
                color=colors[model_name], linewidth=2)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('DelayMatchSampleDistractor1D: Training Curves', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 0.5])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_task_structure(env, output_path='images/q2c_task_structure.png'):
    """Visualize the DelayMatchSampleDistractor1D task structure."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    for trial_idx in range(2):
        env.new_trial()
        ob, gt = env.ob, env.gt

        time = np.arange(len(ob)) * env.dt
        ax = axes[trial_idx]

        n_channels = ob.shape[1]
        for i in range(n_channels):
            ax.plot(time, ob[:, i] + i*1.5, linewidth=1.5, label=f'Input {i}', alpha=0.8)

        gt_offset = n_channels * 1.5 + 1
        ax.plot(time, gt + gt_offset, linewidth=2.5, label='Ground Truth Action',
               color='black', linestyle='-', alpha=0.9)

        if 'sample' in env.start_t:
            sample_start = env.start_t['sample'] * env.dt
            ax.axvline(sample_start, color='green', linestyle='--', linewidth=2,
                      alpha=0.7, label='Sample')

        if 'test1' in env.start_t:
            for i, test_key in enumerate(['test1', 'test2', 'test3', 'test4', 'test5']):
                if test_key in env.start_t:
                    test_start = env.start_t[test_key] * env.dt
                    ax.axvline(test_start, color='orange', linestyle='--', linewidth=1.5,
                              alpha=0.5, label=f'Test {i+1}' if i == 0 else '')

        final_action = gt[-1]
        action_labels = {0: 'Fixate', 1: 'Match', 2: 'Non-match'}
        ax.text(time[-1] * 0.95, gt_offset + final_action,
               f'Target: {action_labels[final_action]}',
               fontsize=11, fontweight='bold', ha='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Input Activity', fontsize=11)
        ax.set_title(f'Trial {trial_idx + 1}: Delay Match-to-Sample with Distractors',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if trial_idx == 0:
            ax.legend(loc='upper left', fontsize=8, ncol=2)

    plt.suptitle('DelayMatchSampleDistractor1D Task Structure (Working Memory + Distractors)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_comparison(perf_dict, output_path='images/q2c_performance.png'):
    """Compare final performance across models."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    models = ['vanilla', 'leaky', 'leaky_fa', 'bio']
    labels = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    perfs = [perf_dict[m] for m in models]

    bars = ax.bar(labels, perfs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, perf in zip(bars, perfs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{perf:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('DelayMatchSampleDistractor1D: Final Performance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_hidden_activity(trial_data, env, model_name, output_path_prefix='images/q2c'):
    """Analyze hidden unit activity during DelayMatchSampleDistractor1D trials."""

    activities = np.array([trial_data['activities'][i] for i in range(len(trial_data['activities']))])
    correct_trials = np.array(trial_data['correct'])

    correct_activities = activities[correct_trials]

    avg_activity = np.mean(correct_activities, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    time = np.arange(avg_activity.shape[0]) * env.dt

    im = axes[0].imshow(avg_activity.T, aspect='auto', cmap='viridis',
                        extent=[0, time[-1], 0, avg_activity.shape[1]])
    axes[0].set_xlabel('Time (ms)', fontsize=11)
    axes[0].set_ylabel('Hidden Unit', fontsize=11)
    axes[0].set_title(f'{model_name}: Average Hidden Unit Activity (Correct Trials)',
                     fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='Activity')

    mean_activity_over_units = np.mean(avg_activity, axis=1)
    std_activity_over_units = np.std(avg_activity, axis=1)

    axes[1].plot(time, mean_activity_over_units, linewidth=2, color='blue', label='Mean Activity')
    axes[1].fill_between(time,
                         mean_activity_over_units - std_activity_over_units,
                         mean_activity_over_units + std_activity_over_units,
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


def plot_example_trials(all_trial_data, models_dict, env, output_path='images/q2c_example_predictions.png'):
    """Plot example trials showing model predictions over time."""

    fig, axes = plt.subplots(4, 1, figsize=(16, 12))

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        net = models_dict[model_key]
        net.eval()

        ax = axes[idx]

        # Get one example trial
        env.new_trial()
        ob, gt = env.ob, env.gt

        with torch.no_grad():
            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
            action_pred, _ = net(inputs)
            action_pred_np = action_pred.detach().cpu().numpy()[:, 0, :]

        # Calculate probabilities over time
        pred_probs = np.exp(action_pred_np) / np.exp(action_pred_np).sum(axis=1, keepdims=True)

        time = np.arange(len(ob)) * env.dt

        # Plot ground truth action
        ax.plot(time, gt, 'k-', linewidth=3, label='Ground Truth', alpha=0.7)

        # Plot model's predicted probabilities for each action
        ax.plot(time, pred_probs[:, 0], 'b-', linewidth=2, label='P(Fixate)', alpha=0.8)
        ax.plot(time, pred_probs[:, 1], 'r-', linewidth=2, label='P(Match)', alpha=0.8)

        # Mark key trial periods
        if hasattr(env, 'start_t'):
            if 'sample' in env.start_t:
                sample_start = env.start_t['sample'] * env.dt
                ax.axvline(sample_start, color='green', linestyle='--', linewidth=1.5,
                          alpha=0.6, label='Sample')

            # Mark test stimuli
            for i, test_key in enumerate(['test1', 'test2', 'test3', 'test4', 'test5']):
                if test_key in env.start_t:
                    test_start = env.start_t[test_key] * env.dt
                    ax.axvline(test_start, color='orange', linestyle='--', linewidth=1,
                              alpha=0.4, label='Test' if i == 0 else '')

        # Calculate accuracy on this trial
        final_pred = np.argmax(pred_probs[-1, :])
        final_true = gt[-1]
        correct = '✓' if final_pred == final_true else '✗'

        ax.set_ylabel('Probability / Action', fontsize=11)
        ax.set_ylim([-0.1, 1.5])
        ax.set_title(f'{model_name} {correct} (Pred: {final_pred}, True: {final_true})',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, ncol=3)
        ax.grid(True, alpha=0.3)

        if idx == 3:
            ax.set_xlabel('Time (ms)', fontsize=11)

    plt.suptitle('Example Trial: Model Predictions Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_prediction_confidence(all_trial_data, models_dict, env, output_path='images/q2c_prediction_behavior.png'):
    """Visualize prediction behavior: probability of predicting Match for each ground truth class."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        net = models_dict[model_key]
        net.eval()

        match_probs_when_fixate = []  # P(Match) when truth is Fixate
        match_probs_when_match = []   # P(Match) when truth is Match

        with torch.no_grad():
            for trial_idx in range(min(200, len(all_trial_data[model_key]['trial_info']))):
                env.new_trial()
                ob, gt = env.ob, env.gt

                inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
                action_pred, _ = net(inputs)

                action_pred_np = action_pred.detach().cpu().numpy()
                pred_probs = np.exp(action_pred_np[-1, 0, :]) / np.sum(np.exp(action_pred_np[-1, 0, :]))

                true_action = gt[-1]
                prob_match = pred_probs[1]  # Probability of predicting Match

                if true_action == 0:
                    match_probs_when_fixate.append(prob_match)
                else:
                    match_probs_when_match.append(prob_match)

        ax = axes[idx]

        # Create violin/box plots
        data_to_plot = [match_probs_when_fixate, match_probs_when_match]
        positions = [0, 1]

        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=True,
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(marker='D', markerfacecolor='orange', markersize=8))

        # Color the boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')

        # Add horizontal reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')

        # Calculate accuracy
        fixate_correct = np.sum(np.array(match_probs_when_fixate) < 0.5)
        match_correct = np.sum(np.array(match_probs_when_match) >= 0.5)
        total_correct = fixate_correct + match_correct
        total_trials = len(match_probs_when_fixate) + len(match_probs_when_match)
        accuracy = total_correct / total_trials

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['True: Fixate\n(should predict low)', 'True: Match\n(should predict high)'],
                          fontsize=10)
        ax.set_ylabel('P(Match) - Model Output Probability', fontsize=10)
        ax.set_ylim([-0.05, 1.05])
        ax.set_title(f'{model_name}\nAccuracy: {accuracy:.3f}',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)

        # Add text annotations for mean probabilities
        mean_fixate = np.mean(match_probs_when_fixate)
        mean_match = np.mean(match_probs_when_match)
        ax.text(0, 1.02, f'μ={mean_fixate:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(1, 1.02, f'μ={mean_match:.3f}', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Model Predictions: P(Match) for Each Ground Truth Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def calculate_balanced_accuracy(predictions, ground_truths):
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    unique_classes = np.unique(ground_truths)
    per_class_recalls = []

    for cls in unique_classes:
        mask = ground_truths == cls
        if np.sum(mask) > 0:
            recall = np.sum((predictions == cls) & mask) / np.sum(mask)
            per_class_recalls.append(recall)

    balanced_acc = np.mean(per_class_recalls) if per_class_recalls else 0.0
    return balanced_acc


def analyze_prediction_confusion(all_trial_data, models_dict, env, output_path='images/q2c_prediction_confusion.png'):
    """Analyze what models actually predict vs ground truth."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    action_names = ['Fixate (0)', 'Match (1)', 'Non-Match (2)']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        net = models_dict[model_key]
        net.eval()

        predictions = []
        ground_truths = []

        with torch.no_grad():
            for trial_idx in range(min(200, len(all_trial_data[model_key]['trial_info']))):
                env.new_trial()
                ob, gt = env.ob, env.gt

                inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
                action_pred, _ = net(inputs)

                action_pred_np = action_pred.detach().cpu().numpy()
                pred_action = np.argmax(action_pred_np[-1, 0, :])
                true_action = gt[-1]

                predictions.append(pred_action)
                ground_truths.append(true_action)

        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)

        # Create confusion matrix (3x3 for 3 classes)
        num_classes = 3
        confusion = np.zeros((num_classes, num_classes))
        for true_val in range(num_classes):
            for pred_val in range(num_classes):
                confusion[true_val, pred_val] = np.sum((ground_truths == true_val) & (predictions == pred_val))

        # Normalize by row (ground truth)
        confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)

        # Print confusion matrix to console
        balanced_acc = calculate_balanced_accuracy(predictions, ground_truths)
        print(f'\n{model_name} Confusion Matrix:')
        print('='*70)
        print(f'{"":15s} {"Pred Fixate":>14s} {"Pred Match":>14s} {"Pred Non-Match":>18s}')
        print('-'*70)
        for i, true_label in enumerate(['True Fixate', 'True Match', 'True Non-Match']):
            print(f'{true_label:15s}', end='')
            for j in range(num_classes):
                count = int(confusion[i, j])
                pct = confusion_norm[i, j]
                print(f' {count:4d} ({pct:5.1%})', end='')
            print()
        accuracy = np.sum(predictions == ground_truths) / len(predictions)
        print(f'\nAccuracy: {accuracy:.3f}')
        print(f'Balanced Accuracy: {balanced_acc:.3f}')
        print(f'Predicted Fixate:    {np.sum(predictions == 0)}/{len(predictions)} ({np.mean(predictions == 0):.1%})')
        print(f'Predicted Match:     {np.sum(predictions == 1)}/{len(predictions)} ({np.mean(predictions == 1):.1%})')
        print(f'Predicted Non-Match: {np.sum(predictions == 2)}/{len(predictions)} ({np.mean(predictions == 2):.1%})')

        ax = axes[idx]
        im = ax.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)

        # Add text annotations
        for i in range(num_classes):
            for j in range(num_classes):
                count = int(confusion[i, j])
                pct = confusion_norm[i, j]
                text = f'{count}\n({pct:.1%})'
                ax.text(j, i, text, ha='center', va='center', fontsize=9,
                       color='white' if pct > 0.5 else 'black', fontweight='bold')

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(action_names, fontsize=9)
        ax.set_yticklabels(action_names, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Ground Truth', fontsize=10)

        accuracy = np.sum(predictions == ground_truths) / len(predictions)
        bal_acc_plot = calculate_balanced_accuracy(predictions, ground_truths)
        ax.set_title(f'{model_name}\nAcc: {accuracy:.3f} | Bal Acc: {bal_acc_plot:.3f}', fontsize=11, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Confusion Matrices: What Do Models Actually Predict?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("Question 2c: Analysis of DelayMatchSample Results")
    print("="*70)
    print()

    print("[1] Loading saved models and data...")
    checkpoint = torch.load('checkpoints/question_2c_models_and_data.pt', weights_only=False)

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
                    dt=env.dt, tau=100, sigma_rec=0.15).to(device)
    net_leaky.load_state_dict(checkpoint['leaky_model'])
    models_dict['leaky'] = net_leaky

    net_leaky_fa = Net(input_size, hidden_size, output_size, model_type='leaky_fa',
                       dt=env.dt, tau=100, sigma_rec=0.15).to(device)
    net_leaky_fa.load_state_dict(checkpoint['leaky_fa_model'])
    models_dict['leaky_fa'] = net_leaky_fa

    net_bio = Net(input_size, hidden_size, output_size, model_type='bio_realistic',
                  dt=env.dt, tau=100, sigma_rec=0.15, exc_ratio=0.8).to(device)
    net_bio.load_state_dict(checkpoint['bio_model'])
    models_dict['bio'] = net_bio

    print("Models reconstructed successfully")
    print()

    trial_data_dict = checkpoint['trial_data_dict']
    loss_dict = checkpoint['loss_dict']
    perf_dict = checkpoint['perf_dict']

    print("[3] Performance Summary:")
    print("-"*70)
    print(f"Vanilla RNN:          {perf_dict['vanilla']:.3f}")
    print(f"Leaky RNN:            {perf_dict['leaky']:.3f}")
    print(f"Leaky RNN + FA:       {perf_dict['leaky_fa']:.3f}")
    print(f"Bio-Realistic RNN:    {perf_dict['bio']:.3f}")
    print()

    print("[4] Visualizing task structure...")
    plot_task_structure(env)
    print()

    print("[5] Generating training curves...")
    plot_training_curves(loss_dict)
    print()

    print("[6] Generating performance comparison...")
    plot_performance_comparison(perf_dict)
    print()

    print("[7] Analyzing hidden unit activity for each model...")
    analyze_hidden_activity(trial_data_dict['vanilla'], env, 'Vanilla RNN')
    analyze_hidden_activity(trial_data_dict['leaky'], env, 'Leaky RNN')
    analyze_hidden_activity(trial_data_dict['leaky_fa'], env, 'Leaky RNN + FA')
    analyze_hidden_activity(trial_data_dict['bio'], env, 'Bio-Realistic RNN')
    print()

    print("[8] Plotting example trial predictions...")
    plot_example_trials(trial_data_dict, models_dict, env)

    print("\nAnalyzing prediction confusion matrices...")
    analyze_prediction_confusion(trial_data_dict, models_dict, env)

    print("\nAnalyzing prediction behavior...")
    analyze_prediction_confidence(trial_data_dict, models_dict, env)
    print()

    print("="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nGenerated plots:")
    print("  - images/q2c_task_structure.png")
    print("  - images/q2c_training_curves.png")
    print("  - images/q2c_performance.png")
    print("  - images/q2c_*_activity.png (4 plots)")
    print("  - images/q2c_prediction_errors.png")
    print("\nKey findings to discuss:")
    print("  - Compare performance across models on distractor task")
    print("  - How do brain-inspired features help with distractor resistance?")
    print("  - Do conclusions from ReadySetGo (Q2b) hold for working memory?")
    print("  - Analyze prediction confidence: are models well-calibrated?")
    print("="*70)
