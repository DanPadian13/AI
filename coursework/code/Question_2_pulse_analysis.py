import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import neurogym as ngym
import os

from Question_2a import Net

device = torch.device('cpu')
print(f"Using device: {device}")


def plot_training_curves(loss_dict, output_path='images/q2_pulse_training_curves.png'):
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
    ax.set_title('PulseDecisionMaking-v0: Training Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_task_structure(env, output_path='images/q2_pulse_task_structure.png'):
    """Visualize the PulseDecisionMaking task structure."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for trial_idx in range(3):
        env.new_trial()
        ob, gt = env.ob, env.gt
        trial = env.trial

        time = np.arange(len(ob)) * env.dt
        ax = axes[trial_idx]

        # Plot input channels
        ax.plot(time, ob[:, 0] * 3, 'gray', linewidth=2, label='Fixation', alpha=0.7)
        ax.plot(time, ob[:, 1] * 2, 'b-', linewidth=2, label='Pulse Channel 1', alpha=0.8)
        ax.plot(time, ob[:, 2] * 2, 'r-', linewidth=2, label='Pulse Channel 2', alpha=0.8)

        # Mark pulses
        pulse1_times = np.where(ob[:, 1] > 0)[0] * env.dt
        pulse2_times = np.where(ob[:, 2] > 0)[0] * env.dt

        for pt in pulse1_times:
            ax.axvline(pt, color='blue', alpha=0.2, linewidth=1)
        for pt in pulse2_times:
            ax.axvline(pt, color='red', alpha=0.2, linewidth=1)

        # Plot ground truth action
        gt_offset = 4
        ax.plot(time, gt + gt_offset, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)

        # Add pulse counts
        pulse1_count = trial.get('pulse1', 0)
        pulse2_count = trial.get('pulse2', 0)

        action_labels = {0: 'Fixate', 1: 'Choose Left', 2: 'Choose Right'}
        final_action = gt[-1]

        title_str = f'Trial {trial_idx + 1}: Pulse Counts - Ch1: {pulse1_count}, Ch2: {pulse2_count}'
        ax.text(time[-1] * 0.95, gt_offset + final_action,
               f'→ {action_labels[final_action]}',
               fontsize=11, fontweight='bold', ha='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Input Activity', fontsize=11)
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if trial_idx == 0:
            ax.legend(loc='upper left', fontsize=9, ncol=2)

    plt.suptitle('PulseDecisionMaking-v0: Count Pulses and Decide',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_comparison(perf_dict, bal_acc_dict, output_path='images/q2_pulse_performance.png'):
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

    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (90%)')
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

    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (90%)')
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    plt.suptitle('PulseDecisionMaking-v0: Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_hidden_activity(trial_data, env, model_name, output_path_prefix='images/q2_pulse'):
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


def plot_example_predictions(trial_data_dict, models_dict, env, output_path='images/q2_pulse_example_predictions.png'):
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

        # Show pulse counts
        pulse1 = trial.get('pulse1', 0)
        pulse2 = trial.get('pulse2', 0)

        # Check if correct
        final_pred = np.argmax(pred_probs[-1, :])
        final_true = gt[-1]
        correct = '✓' if final_pred == final_true else '✗'

        ax.set_ylabel('Probability / Action', fontsize=11)
        ax.set_ylim([-0.1, 2.0])
        ax.set_title(f'{model_name} {correct} | Pulses: Ch1={pulse1}, Ch2={pulse2} | Pred={final_pred}, True={final_true}',
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


def analyze_pulse_difficulty(trial_data_dict, output_path='images/q2_pulse_difficulty_analysis.png'):
    """Analyze accuracy as a function of pulse difference."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        trial_data = trial_data_dict[model_key]

        # Extract pulse differences and correctness
        pulse_diffs = []
        correct_list = []

        for i, trial_info in enumerate(trial_data['trial_info']):
            pulse1 = trial_info.get('pulse1', 0)
            pulse2 = trial_info.get('pulse2', 0)
            pulse_diff = abs(pulse1 - pulse2)
            pulse_diffs.append(pulse_diff)
            correct_list.append(trial_data['correct'][i])

        pulse_diffs = np.array(pulse_diffs)
        correct_list = np.array(correct_list)

        # Calculate accuracy for each pulse difference
        unique_diffs = np.unique(pulse_diffs)
        accuracies = []
        counts = []

        for diff in unique_diffs:
            mask = pulse_diffs == diff
            if np.sum(mask) > 0:
                acc = np.mean(correct_list[mask])
                accuracies.append(acc)
                counts.append(np.sum(mask))

        ax = axes[idx]
        ax.bar(unique_diffs, accuracies, alpha=0.7, color=plt.cm.viridis(idx/3),
              edgecolor='black', linewidth=1.5)

        # Add counts on bars
        for diff, acc, count in zip(unique_diffs, accuracies, counts):
            ax.text(diff, acc + 0.02, f'n={count}', ha='center', fontsize=8)

        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Chance')
        ax.set_xlabel('|Pulse1 - Pulse2|', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)

    plt.suptitle('Accuracy vs Pulse Difference (Task Difficulty)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("Question 2: Analysis of PulseDecisionMaking-v0 Results")
    print("="*70)
    print()

    print("[1] Loading saved models and data...")
    checkpoint = torch.load('checkpoints/question_2_pulse_models_and_data.pt', weights_only=False)

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

    print("[8] Plotting example predictions...")
    plot_example_predictions(trial_data_dict, models_dict, env)
    print()

    print("[9] Analyzing task difficulty (pulse difference)...")
    analyze_pulse_difficulty(trial_data_dict)
    print()

    print("="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nGenerated plots:")
    print("  - images/q2_pulse_task_structure.png")
    print("  - images/q2_pulse_training_curves.png")
    print("  - images/q2_pulse_performance.png")
    print("  - images/q2_pulse_*_activity.png (4 plots)")
    print("  - images/q2_pulse_example_predictions.png")
    print("  - images/q2_pulse_difficulty_analysis.png")
    print("\nKey findings to discuss:")
    print("  - Do all models achieve high accuracy (>90%)?")
    print("  - How does performance vary with pulse difference (difficulty)?")
    print("  - Compare brain-inspired features across architectures")
    print("  - Evidence accumulation strategies")
    print("="*70)
