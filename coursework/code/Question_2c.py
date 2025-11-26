import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import neurogym as ngym
import os

from Question_2a import Net, train_model, evaluate_model

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
    ax.set_ylim([0, 1.0])
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


def analyze_prediction_errors(all_trial_data, models_dict, env, output_path='images/q2c_prediction_errors.png'):
    """Analyze prediction confidence and errors across models."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ['Vanilla RNN', 'Leaky RNN', 'Leaky RNN + FA', 'Bio-Realistic RNN']
    model_keys = ['vanilla', 'leaky', 'leaky_fa', 'bio']

    device = torch.device('cpu')

    for idx, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        net = models_dict[model_key]
        net.eval()

        predictions = []
        ground_truths = []
        confidences = []

        with torch.no_grad():
            for trial_idx in range(min(200, len(all_trial_data[model_key]['trial_info']))):
                env.new_trial()
                ob, gt = env.ob, env.gt

                inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
                action_pred, _ = net(inputs)

                action_pred_np = action_pred.detach().cpu().numpy()
                pred_probs = np.exp(action_pred_np[-1, 0, :]) / np.sum(np.exp(action_pred_np[-1, 0, :]))

                pred_action = np.argmax(pred_probs)
                true_action = gt[-1]
                confidence = pred_probs[pred_action]

                predictions.append(pred_action)
                ground_truths.append(true_action)
                confidences.append(confidence)

        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        confidences = np.array(confidences)

        correct_mask = predictions == ground_truths
        incorrect_mask = ~correct_mask

        ax = axes[idx]

        if np.sum(correct_mask) > 0:
            ax.hist(confidences[correct_mask], bins=20, alpha=0.6,
                   label=f'Correct ({np.sum(correct_mask)})', color='green', edgecolor='black')
        if np.sum(incorrect_mask) > 0:
            ax.hist(confidences[incorrect_mask], bins=20, alpha=0.6,
                   label=f'Incorrect ({np.sum(incorrect_mask)})', color='red', edgecolor='black')

        ax.set_xlabel('Prediction Confidence', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{model_name}\nAccuracy: {np.mean(correct_mask):.3f}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])

    plt.suptitle('Prediction Confidence: Correct vs Incorrect Trials', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("Question 2c: DelayMatchSampleDistractor1D Task with Brain-Inspired RNNs")
    print("="*70)
    print()

    print("[1] Setting up NeuroGym DelayMatchSampleDistractor1D task...")
    task = 'DelayMatchSampleDistractor1D-v0'
    kwargs_env = {'dt': 20}
    seq_len = 400

    dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=16, seq_len=seq_len)
    env = dataset.env

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = 50

    print(f"Task: {task}")
    print(f"Description: Working memory with distractors - maintain stimulus despite interference")
    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}")
    print(f"Hidden size: {hidden_size}")
    print()

    print("[1b] Visualizing task structure...")
    plot_task_structure(env)
    print()

    common_lr = 0.001
    common_noise = 0.15
    num_steps = 2000

    loss_dict = {}
    perf_dict = {}
    trial_data_dict = {}
    models_dict = {}

    print("[2] Training Vanilla RNN...")
    print("-"*70)
    net_vanilla = Net(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        model_type='vanilla'
    ).to(device)

    loss_vanilla = train_model(net_vanilla, dataset, num_steps=num_steps, lr=common_lr)
    perf_vanilla, data_vanilla = evaluate_model(net_vanilla, env, num_trials=500)

    loss_dict['vanilla'] = loss_vanilla
    perf_dict['vanilla'] = perf_vanilla
    trial_data_dict['vanilla'] = data_vanilla
    models_dict['vanilla'] = net_vanilla

    print(f"Vanilla RNN Performance: {perf_vanilla:.3f}")
    print()

    print("[3] Training Leaky RNN...")
    print("-"*70)
    net_leaky = Net(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        model_type='leaky',
        dt=env.dt,
        tau=100,
        sigma_rec=common_noise
    ).to(device)

    loss_leaky = train_model(net_leaky, dataset, num_steps=num_steps, lr=common_lr)
    perf_leaky, data_leaky = evaluate_model(net_leaky, env, num_trials=500)

    loss_dict['leaky'] = loss_leaky
    perf_dict['leaky'] = perf_leaky
    trial_data_dict['leaky'] = data_leaky
    models_dict['leaky'] = net_leaky

    print(f"Leaky RNN Performance: {perf_leaky:.3f}")
    print()

    print("[4] Training Leaky RNN + Feedback Alignment...")
    print("-"*70)
    net_leaky_fa = Net(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        model_type='leaky_fa',
        dt=env.dt,
        tau=100,
        sigma_rec=common_noise
    ).to(device)

    loss_leaky_fa = train_model(net_leaky_fa, dataset, num_steps=num_steps, lr=common_lr)
    perf_leaky_fa, data_leaky_fa = evaluate_model(net_leaky_fa, env, num_trials=500)

    loss_dict['leaky_fa'] = loss_leaky_fa
    perf_dict['leaky_fa'] = perf_leaky_fa
    trial_data_dict['leaky_fa'] = data_leaky_fa
    models_dict['leaky_fa'] = net_leaky_fa

    print(f"Leaky RNN + FA Performance: {perf_leaky_fa:.3f}")
    print()

    print("[5] Training Biologically Realistic RNN...")
    print("-"*70)
    net_bio = Net(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        model_type='bio_realistic',
        dt=env.dt,
        tau=100,
        sigma_rec=common_noise,
        exc_ratio=0.8
    ).to(device)

    loss_bio = train_model(net_bio, dataset, num_steps=num_steps, lr=common_lr,
                          beta_L1=0.0005, beta_L2=0.01)
    perf_bio, data_bio = evaluate_model(net_bio, env, num_trials=500)

    loss_dict['bio'] = loss_bio
    perf_dict['bio'] = perf_bio
    trial_data_dict['bio'] = data_bio
    models_dict['bio'] = net_bio

    print(f"Bio-Realistic RNN Performance: {perf_bio:.3f}")
    print()

    print("[6] Performance Summary:")
    print("-"*70)
    print(f"Vanilla RNN:          {perf_vanilla:.3f}")
    print(f"Leaky RNN:            {perf_leaky:.3f}")
    print(f"Leaky RNN + FA:       {perf_leaky_fa:.3f}")
    print(f"Bio-Realistic RNN:    {perf_bio:.3f}")
    print()

    print("[7] Generating visualizations...")
    print("-"*70)
    plot_training_curves(loss_dict)
    plot_performance_comparison(perf_dict)

    print("\nAnalyzing hidden unit activity for each model...")
    analyze_hidden_activity(data_vanilla, env, 'Vanilla RNN')
    analyze_hidden_activity(data_leaky, env, 'Leaky RNN')
    analyze_hidden_activity(data_leaky_fa, env, 'Leaky RNN + FA')
    analyze_hidden_activity(data_bio, env, 'Bio-Realistic RNN')

    print("\nAnalyzing prediction confidence and errors...")
    analyze_prediction_errors(trial_data_dict, models_dict, env)
    print()

    print("[8] Saving results...")
    print("-"*70)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'vanilla_model': net_vanilla.state_dict(),
        'leaky_model': net_leaky.state_dict(),
        'leaky_fa_model': net_leaky_fa.state_dict(),
        'bio_model': net_bio.state_dict(),
        'vanilla_data': data_vanilla,
        'leaky_data': data_leaky,
        'leaky_fa_data': data_leaky_fa,
        'bio_data': data_bio,
        'loss_dict': loss_dict,
        'perf_dict': perf_dict,
        'env_config': {'dt': env.dt, 'task': task, 'seq_len': seq_len}
    }, 'checkpoints/question_2c_models_and_data.pt')
    print("Saved: checkpoints/question_2c_models_and_data.pt")
    print()

    print("="*70)
    print("Question 2c Complete!")
    print("="*70)
    print("\nTask: DelayMatchSampleDistractor1D-v0 (Working Memory + Distractors)")
    print("Key differences from ReadySetGo (timing task):")
    print("  - Requires maintaining stimulus information during delay")
    print("  - Multiple distractor stimuli presented - tests robustness")
    print("  - Must resist interference and maintain memory until matching test")
    print("  - Tests working memory + distractor resistance, not timing")
    print("\nDo conclusions from Q2b hold?")
    print("  - Check if brain-inspired features (time constants, noise, E/I balance)")
    print("    help with distractor resistance and robust memory maintenance")
    print("  - Noise might actually help: stochastic resonance for signal robustness")
    print("  - Time constants (tau) should help maintain persistent representations")
    print("="*70)
