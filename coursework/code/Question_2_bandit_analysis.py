import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import neurogym as ngym
import os

from Question_2a import Net

device = torch.device('cpu')
print(f"Using device: {device}")


def plot_training_curves(loss_dict, output_path='images/q2_bandit_training_curves.png'):
    """Plot training curves for all models on Bandit-v0."""
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
    ax.set_title('Bandit-v0: Training Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_task_structure(env, output_path='images/q2_bandit_task_structure.png'):
    """Visualize the Bandit-v0 task structure."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for trial_idx in range(3):
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

        trial_info = env.trial
        optimal_arm = trial_info.get('ground_truth', gt[-1])
        action_labels = {0: 'Fixate', 1: 'Arm 1', 2: 'Arm 2'}

        ax.text(time[-1] * 0.95, gt_offset + gt[-1],
               f'Target: {action_labels.get(gt[-1], "Unknown")}',
               fontsize=11, fontweight='bold', ha='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Input Activity', fontsize=11)
        ax.set_title(f'Trial {trial_idx + 1}: Bandit Decision',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if trial_idx == 0:
            ax.legend(loc='upper left', fontsize=8, ncol=2)

    plt.suptitle('Bandit-v0 Task Structure (Multi-Armed Bandit)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_comparison(perf_dict, reward_dict, output_path='images/q2_bandit_performance.png'):
    """Compare final performance and rewards across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = ['vanilla', 'leaky', 'leaky_fa', 'bio']
    labels = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    perfs = [perf_dict[m] for m in models]
    rewards = [reward_dict[m] for m in models]

    # Plot accuracy
    ax = axes[0]
    bars = ax.bar(labels, perfs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, perf in zip(bars, perfs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{perf:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (80%)')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random (50%)')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Bandit-v0: Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Plot average reward
    ax = axes[1]
    bars = ax.bar(labels, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Bandit-v0: Average Reward', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(rewards) * 1.2])
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_hidden_activity(trial_data, env, model_name, output_path_prefix='images/q2_bandit'):
    """Analyze hidden unit activity during Bandit trials."""

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


def plot_arm_preferences(trial_data_dict, models_dict, env, output_path='images/q2_bandit_arm_preferences.png'):
    """Analyze which arms each model prefers."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        trial_data = trial_data_dict[model_key]
        predictions = np.array(trial_data['predictions'])

        # Count arm selections (excluding fixation = 0)
        arm_choices = predictions[predictions > 0]

        if len(arm_choices) == 0:
            continue

        # Count how often each arm was chosen
        arm1_count = np.sum(arm_choices == 1)
        arm2_count = np.sum(arm_choices == 2)
        total_choices = arm1_count + arm2_count

        ax = axes[idx]

        # Bar plot
        bars = ax.bar(['Arm 1\n(80% reward)', 'Arm 2\n(20% reward)'],
                     [arm1_count, arm2_count],
                     color=['#2ca02c', '#d62728'],
                     alpha=0.7, edgecolor='black', linewidth=2)

        # Add counts on bars
        for bar, count in zip(bars, [arm1_count, arm2_count]):
            height = bar.get_height()
            pct = count / total_choices * 100 if total_choices > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add optimal line
        ax.axhline(y=total_choices * 0.8, color='green', linestyle='--',
                  linewidth=2, alpha=0.5, label='Optimal (80%)')

        ax.set_ylabel('Number of Selections', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, max(arm1_count, arm2_count) * 1.2])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)

    plt.suptitle('Arm Selection Preferences (Excluding Fixation)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_learning_behavior(trial_data_dict, output_path='images/q2_bandit_learning_over_time.png'):
    """Analyze how arm preferences evolve over trials."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    window_size = 50  # Rolling window for smoothing

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        trial_data = trial_data_dict[model_key]
        predictions = np.array(trial_data['predictions'])
        ground_truths = np.array(trial_data['ground_truths'])

        # Calculate rolling accuracy
        correct = predictions == ground_truths
        rolling_acc = np.convolve(correct, np.ones(window_size)/window_size, mode='valid')

        # Calculate rolling arm 1 preference (among decision trials)
        decision_trials = ground_truths > 0
        arm1_choices = (predictions == 1) & decision_trials
        rolling_arm1_pref = np.convolve(arm1_choices.astype(float),
                                       np.ones(window_size)/window_size, mode='valid')

        ax = axes[idx]

        trial_nums = np.arange(len(rolling_acc)) + window_size//2

        # Plot accuracy
        ax.plot(trial_nums, rolling_acc, linewidth=2, color='blue',
               label='Accuracy', alpha=0.8)

        # Plot arm 1 preference
        ax.plot(trial_nums, rolling_arm1_pref, linewidth=2, color='green',
               label='P(choose Arm 1)', alpha=0.8)

        # Add reference lines
        ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5,
                  alpha=0.4, label='Optimal (80%)')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5,
                  alpha=0.4, label='Random (50%)')

        ax.set_xlabel('Trial Number', fontsize=11)
        ax.set_ylabel('Proportion', fontsize=11)
        ax.set_title(f'{model_name}: Learning Over Time\n(Rolling window: {window_size} trials)',
                    fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Learning Dynamics: How Models Learn Arm Preferences', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_example_trials(trial_data_dict, models_dict, env, output_path='images/q2_bandit_example_predictions.png'):
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
        ax.plot(time, pred_probs[:, 0], 'gray', linewidth=2, label='P(Fixate)', alpha=0.6, linestyle='--')
        ax.plot(time, pred_probs[:, 1], 'g-', linewidth=2, label='P(Arm 1 - 80%)', alpha=0.8)
        ax.plot(time, pred_probs[:, 2], 'r-', linewidth=2, label='P(Arm 2 - 20%)', alpha=0.8)

        # Calculate accuracy on this trial
        final_pred = np.argmax(pred_probs[-1, :])
        final_true = gt[-1]
        correct = '✓' if final_pred == final_true else '✗'

        action_names = {0: 'Fixate', 1: 'Arm 1', 2: 'Arm 2'}

        ax.set_ylabel('Probability / Action', fontsize=11)
        ax.set_ylim([-0.1, 2.5])
        ax.set_title(f'{model_name} {correct} (Pred: {action_names[final_pred]}, True: {action_names[final_true]})',
                    fontsize=12, fontweight='bold')
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


if __name__ == '__main__':
    print("="*70)
    print("Question 2: Analysis of Bandit-v0 Results")
    print("="*70)
    print()

    print("[1] Loading saved models and data...")
    checkpoint = torch.load('checkpoints/question_2_bandit_models_and_data.pt', weights_only=False)

    env_config = checkpoint['env_config']
    task = env_config['task']
    kwargs_env = {'dt': env_config['dt'], 'n': env_config['n_arms'], 'p': env_config['probs']}
    seq_len = env_config['seq_len']

    dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=16, seq_len=seq_len)
    env = dataset.env

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # Infer hidden size from checkpoint
    vanilla_fc_weight_shape = checkpoint['vanilla_model']['fc.weight'].shape
    hidden_size = vanilla_fc_weight_shape[1]

    print(f"Task: {task}")
    print(f"Number of arms: {env_config['n_arms']}")
    print(f"Reward probabilities: {env_config['probs']}")
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
    reward_dict = checkpoint['reward_dict']

    print("[3] Performance Summary:")
    print("-"*70)
    print(f"Vanilla RNN:          Acc: {perf_dict['vanilla']:.3f}, Reward: {reward_dict['vanilla']:.3f}")
    print(f"Leaky RNN:            Acc: {perf_dict['leaky']:.3f}, Reward: {reward_dict['leaky']:.3f}")
    print(f"Leaky RNN + FA:       Acc: {perf_dict['leaky_fa']:.3f}, Reward: {reward_dict['leaky_fa']:.3f}")
    print(f"Bio-Realistic RNN:    Acc: {perf_dict['bio']:.3f}, Reward: {reward_dict['bio']:.3f}")
    print()

    print("[4] Visualizing task structure...")
    plot_task_structure(env)
    print()

    print("[5] Generating training curves...")
    plot_training_curves(loss_dict)
    print()

    print("[6] Generating performance comparison...")
    plot_performance_comparison(perf_dict, reward_dict)
    print()

    print("[7] Analyzing arm selection preferences...")
    plot_arm_preferences(trial_data_dict, models_dict, env)
    print()

    print("[8] Analyzing learning dynamics over time...")
    analyze_learning_behavior(trial_data_dict)
    print()

    print("[9] Analyzing hidden unit activity for each model...")
    analyze_hidden_activity(trial_data_dict['vanilla'], env, 'Vanilla RNN')
    analyze_hidden_activity(trial_data_dict['leaky'], env, 'Leaky RNN')
    analyze_hidden_activity(trial_data_dict['leaky_fa'], env, 'Leaky RNN + FA')
    analyze_hidden_activity(trial_data_dict['bio'], env, 'Bio-Realistic RNN')
    print()

    print("[10] Plotting example trial predictions...")
    plot_example_trials(trial_data_dict, models_dict, env)
    print()

    print("="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nGenerated plots:")
    print("  - images/q2_bandit_task_structure.png")
    print("  - images/q2_bandit_training_curves.png")
    print("  - images/q2_bandit_performance.png")
    print("  - images/q2_bandit_arm_preferences.png")
    print("  - images/q2_bandit_learning_over_time.png")
    print("  - images/q2_bandit_*_activity.png (4 plots)")
    print("  - images/q2_bandit_example_predictions.png")
    print("\nKey findings to discuss:")
    print("  - Do models learn to prefer the optimal arm (80% reward)?")
    print("  - How quickly do different architectures converge?")
    print("  - Compare exploration vs exploitation strategies")
    print("  - Does biological realism help or hurt in simple decision tasks?")
    print("="*70)
