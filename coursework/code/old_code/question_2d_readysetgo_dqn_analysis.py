import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import neurogym as ngym

from Question_2a import Net
from readysetgo import infer_go_action_index  # reuse helper
from readysetgo import detect_go_time


def load_models(checkpoint_path='checkpoints/question_2d_readysetgo_dqn.pt'):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    env_kwargs = ckpt['env_kwargs']
    task = ckpt['task']
    env = ngym.make(task, **env_kwargs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    hidden_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_types = {'vanilla': 'vanilla', 'leaky': 'leaky', 'leaky_fa': 'leaky_fa', 'bio': 'bio_realistic'}
    models = {}
    for key, mtype in model_types.items():
        net = Net(obs_dim, hidden_size, act_dim, model_type=mtype, dt=env_kwargs['dt'], tau=150, sigma_rec=0.1,
                  exc_ratio=0.8 if mtype == 'bio_realistic' else None, sparsity=0.2 if mtype == 'bio_realistic' else None).to(device)
        net.load_state_dict(ckpt['policy_states'][key])
        net.eval()
        models[key] = net
    return models, env


def collect_trial_data(models, env, num_trials=100):
    trial_data = {k: {'activities': [], 'outputs': [], 'stimuli': [], 'targets': [], 'correct': []} for k in models}
    go_idx = infer_go_action_index(env)
    device = next(iter(models.values())).fc.weight.device

    for _ in range(num_trials):
        env.new_trial()
        ob, gt = env.ob, env.gt
        inputs = torch.from_numpy(ob[:, np.newaxis, :]).float().to(device)  # [T,1,obs]
        for key, model in models.items():
            with torch.no_grad():
                out, act = model(inputs)
                probs = torch.softmax(out, dim=-1).cpu().numpy()[:, 0, :]
                pred = probs[-1].argmax()
                correct = (pred == gt[-1])
                trial_data[key]['activities'].append(act[:, 0, :].cpu().numpy())
                trial_data[key]['outputs'].append(probs)
                trial_data[key]['stimuli'].append(ob)
                trial_data[key]['targets'].append(gt)
                trial_data[key]['correct'].append(correct)
    return trial_data, go_idx


def evaluate(models, env, episodes=200):
    scores = {}
    for key, net in models.items():
        net.eval()
        total_reward = 0.0
        steps_list = []
        correct_final = 0
        total_final = 0
        timing_errors = []
        entropy_list = []
        q_means = []
        q_max = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            last_action = None
            target_go_idx = env.trial.get('go', None) if hasattr(env, 'trial') else None
            while not done and steps < 200:
                state = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(1)
                with torch.no_grad():
                    q_vals, _ = net(state)
                    logits = q_vals.squeeze(0).squeeze(0)
                    probs = torch.softmax(logits, dim=-1)
                    ent = (-probs * probs.log()).sum().item()
                    entropy_list.append(ent)
                    q_means.append(logits.mean().item())
                    q_max.append(logits.max().item())
                    action = logits.argmax().item()
                    last_action = action
                obs, reward, done, _, _ = env.step(action)
                ep_reward += reward
                steps += 1
            total_reward += ep_reward
            steps_list.append(steps)
            if last_action is not None and len(env.gt) > 0:
                total_final += 1
                correct_final += int(last_action == env.gt[-1])
            if target_go_idx is not None:
                produced_idx = steps
                timing_errors.append(produced_idx - target_go_idx)
        stats = {
            'avg_reward': total_reward / episodes,
            'avg_steps': np.mean(steps_list),
            'final_acc': (correct_final / total_final) if total_final > 0 else 0.0,
            'timing_err_mean': np.mean(timing_errors) if timing_errors else 0.0,
            'timing_err_std': np.std(timing_errors) if timing_errors else 0.0,
            'entropy_mean': np.mean(entropy_list) if entropy_list else 0.0,
            'q_mean': np.mean(q_means) if q_means else 0.0,
            'q_max_mean': np.mean(q_max) if q_max else 0.0,
        }
        scores[key] = stats
        print(f"{key}: reward={stats['avg_reward']:.3f}, acc={stats['final_acc']:.3f}, "
              f"steps/ep={stats['avg_steps']:.1f}, "
              f"timing_err={stats['timing_err_mean']:.1f}±{stats['timing_err_std']:.1f}, "
              f"ent={stats['entropy_mean']:.3f}, q_mean={stats['q_mean']:.3f}, q_max={stats['q_max_mean']:.3f}")
    return scores


def plot_task_structure(env, trial_data, output_path='images/question_2D_RSG_task_structure.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(trial_data.keys())
    num_trials = 3
    fig, axes = plt.subplots(len(models), num_trials, figsize=(4 * num_trials, 3 * len(models)), sharex=False)
    for r, key in enumerate(models):
        for c in range(num_trials):
            ob = trial_data[key]['stimuli'][c]
            gt = trial_data[key]['targets'][c]
            probs = trial_data[key]['outputs'][c]
            ax = axes[r, c] if len(models) > 1 else axes[c]
            t = np.arange(ob.shape[0]) * env.dt
            ax.plot(t, ob[:, 0], color='gray', label='Stim')
            ax.plot(t, probs[:, 0], color='black', linestyle='--', label='P(fix)')
            if probs.shape[1] > 1:
                ax.plot(t, probs[:, 1], color='blue', label='P(go)')
            ax.axhline(0.5, color='k', linestyle=':', linewidth=1)
            ax.set_ylim(0, 1.05)
            ax.set_title(f'{key} Trial {c+1}')
            if c == 0:
                ax.set_ylabel('Prob')
            ax.set_xlabel('Time (ms)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_timing_accuracy(trial_data, env, go_idx, output_path='images/question_2D_RSG_timing.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dt = env.dt
    model_names = ['vanilla', 'leaky', 'leaky_fa', 'bio']
    labels = ['Vanilla', 'Leaky', 'Leaky+FA', 'Bio-Realistic']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    all_target_times = []
    all_errors = []
    model_data = {}

    # First pass: collect timing data
    for name in model_names:
        produced_times = []
        target_times = []
        outputs = trial_data[name]['outputs']
        targets = trial_data[name]['targets']
        for out, gt in zip(outputs, targets):
            out_arr = np.asarray(out)
            gt_arr = np.asarray(gt, dtype=int)
            # Produced time: argmax of go prob over time
            produced_idx = int(np.argmax(out_arr[:, go_idx]))
            target_go = np.where(gt_arr == go_idx)[0]
            target_idx = int(target_go[0]) if len(target_go) > 0 else len(gt_arr)
            produced_times.append(produced_idx * dt)
            target_times.append(target_idx * dt)
        produced_times = np.array(produced_times)
        target_times = np.array(target_times)
        errors = produced_times - target_times
        model_data[name] = {'produced': produced_times, 'target': target_times, 'errors': errors}
        all_target_times.extend(target_times.tolist())
        all_errors.extend(errors.tolist())

    # Common bins for error histogram
    all_errors = np.array(all_errors)
    error_min, error_max = all_errors.min(), all_errors.max()
    bin_width = 50
    bins = np.arange(error_min, error_max + bin_width, bin_width)

    # Plot scatter and error hist
    for idx, name in enumerate(model_names):
        produced_times = model_data[name]['produced']
        target_times = model_data[name]['target']
        errors = model_data[name]['errors']
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax1.scatter(target_times, produced_times, alpha=0.5, s=30, color=colors[idx], label=labels[idx])
        ax2.hist(errors, bins=bins, alpha=0.5, color=colors[idx],
                 label=f'{labels[idx]} (μ={mean_error:.1f}ms, σ={std_error:.1f}ms)')

    max_time = max(all_target_times) * 1.1 if all_target_times else dt * 50
    ax1.plot([0, max_time], [0, max_time], 'k--', linewidth=2, label='Perfect timing')
    ax1.set_xlabel('Target Time (ms)', fontsize=12)
    ax1.set_ylabel('Produced Time (ms)', fontsize=12)
    ax1.set_title('Timing Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Timing Error (ms)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_heatmaps(trial_data, output_path='images/question_2D_RSG_heatmaps.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(trial_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    im = None
    for ax, key in zip(axes.flatten(), models):
        acts_list = [a for a, ok in zip(trial_data[key]['activities'], trial_data[key]['correct']) if ok]
        if len(acts_list) == 0:
            ax.axis('off'); continue
        min_T = min(a.shape[0] for a in acts_list)
        acts = np.stack([a[:min_T] for a in acts_list], axis=0)
        avg = acts.mean(axis=0)
        im = ax.imshow(avg.T, aspect='auto', cmap='viridis',
                       extent=[0, (avg.shape[0]-1), 0, avg.shape[1]])
        ax.set_title(key, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (bins)')
        ax.set_ylabel('Hidden unit')
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label='Activity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    models, env = load_models()
    eval_scores = evaluate(models, env, episodes=300)
    trial_data, go_idx = collect_trial_data(models, env, num_trials=150)
    plot_task_structure(env, trial_data, output_path='images/question_2D_RSG_task_structure.png')
    plot_timing_accuracy(trial_data, env, go_idx, output_path='images/question_2D_RSG_timing.png')
    plot_heatmaps(trial_data, output_path='images/question_2D_RSG_heatmaps.png')
