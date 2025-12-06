"""
Analysis for DQN trained on MultiSensoryIntegration-v0
Generates plots to analyze model performance and neural activity
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import neurogym as ngym
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from math import pi

from Question_2a import Net


def load_models(checkpoint_path='checkpoints/question_2d_MSI_dqn.pt'):
    """Load models from the MSI DQN checkpoint."""
    ckpt = torch.load(checkpoint_path, weights_only=False)
    
    # Setup environment
    task = ckpt.get('task', 'MultiSensoryIntegration-v0')
    env_kwargs = ckpt.get('env_kwargs', {'dt': 100})
    env = ngym.make(task, **env_kwargs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    hidden_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model types - matching the training code
    model_configs = {
        'Vanilla RNN': 'vanilla',
        'Leaky RNN': 'leaky',
        'Leaky + FA': 'leaky_fa',
        'Bio-Realistic': 'bio_realistic'
    }
    
    models = {}
    for name, mtype in model_configs.items():
        # Create network with parameters matching training
        model_kwargs = {'dt': env_kwargs['dt'], 'tau': 100, 'sigma_rec': 0.15}
        # Note: Don't add exc_ratio - the saved models don't have Dale's split
        
        net = Net(obs_dim, hidden_size, act_dim, model_type=mtype, **model_kwargs).to(device)
        
        # Load with strict=False to handle architecture mismatches
        try:
            net.load_state_dict(ckpt['models'][name], strict=False)
        except Exception as e:
            print(f"Warning loading {name}: {e}")
            print("Attempting to load with strict=False...")
            net.load_state_dict(ckpt['models'][name], strict=False)
        
        net.eval()
        models[name] = net
    
    return models, env


def collect_trial_data(models, env, num_trials=200):
    """Collect trial data for visualization."""
    trial_data = {name: {'activities': [], 'outputs': [], 'stimuli': [], 
                         'targets': [], 'correct': [], 'trial_info': [],
                         'predictions': []} 
                  for name in models}
    device = next(iter(models.values())).fc.weight.device

    for _ in range(num_trials):
        env.new_trial()
        ob, gt = env.ob, env.gt
        inputs = torch.from_numpy(ob[:, np.newaxis, :]).float().to(device)  # [T,1,obs]
        
        for name, model in models.items():
            with torch.no_grad():
                out, act, _ = model(inputs)
                probs = torch.softmax(out, dim=-1).cpu().numpy()[:, 0, :]
                pred = probs[-1].argmax()
                correct = (pred == gt[-1])
                trial_data[name]['activities'].append(act[:, 0, :].cpu().numpy())
                trial_data[name]['outputs'].append(probs)
                trial_data[name]['stimuli'].append(ob)
                trial_data[name]['targets'].append(gt)
                trial_data[name]['correct'].append(correct)
                trial_data[name]['trial_info'].append(env.trial.copy())
                trial_data[name]['predictions'].append(probs)
    
    return trial_data


def evaluate(models, env, episodes=200):
    """Evaluate trained models on the MSI task."""
    scores = {}
    device = next(iter(models.values())).fc.weight.device
    
    for name, net in models.items():
        net.eval()
        total_reward = 0.0
        steps_list = []
        correct_final = 0
        total_final = 0
        
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            last_action = None
            
            while not done and steps < 100:
                state = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals, _, _ = net(state)
                    action = int(q_vals[0, 0].argmax().item())
                    last_action = action
                    
                obs, reward, done, _, _ = env.step(action)
                ep_reward += reward
                steps += 1
                
            total_reward += ep_reward
            steps_list.append(steps)
            
            if last_action is not None and len(env.gt) > 0:
                total_final += 1
                correct_final += int(last_action == env.gt[-1])
        
        stats = {
            'avg_reward': total_reward / episodes,
            'avg_steps': np.mean(steps_list),
            'final_acc': (correct_final / total_final) if total_final > 0 else 0.0,
        }
        scores[name] = stats
        print(f"{name}: reward={stats['avg_reward']:.3f}, acc={stats['final_acc']:.3f}, "
              f"steps/ep={stats['avg_steps']:.1f}")
    return scores


def plot_performance_comparison(trial_data, output_path='images/question_2D_MSI_performance.png'):
    """Compare performance across models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_names = list(trial_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Calculate accuracies
    accuracies = []
    for name in model_names:
        acc = np.mean(trial_data[name]['correct'])
        accuracies.append(acc)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Chance (33%)')
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('DQN Performance on MultiSensoryIntegration-v0', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pca_trajectories(trial_data, output_path='images/question_2D_MSI_pca_trajectories.png'):
    """Plot PCA trajectories colored by choice."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_names = list(trial_data.keys())
    colors_choice = {0: '#7f7f7f', 1: '#1f77b4', 2: '#d62728'}
    labels_choice = {0: 'Fixate', 1: 'Left', 2: 'Right'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        
        # Get correct trials only
        activities = trial_data[name]['activities']
        targets = trial_data[name]['targets']
        correct = trial_data[name]['correct']
        
        # Concatenate all activities for PCA
        all_acts = []
        all_choices = []
        for act, tgt, corr in zip(activities, targets, correct):
            if corr:  # Only correct trials
                all_acts.append(act)
                all_choices.append(tgt[-1])
        
        if len(all_acts) == 0:
            ax.text(0.5, 0.5, 'No correct trials', ha='center', va='center')
            continue
        
        # Concatenate and do PCA
        concat_acts = np.concatenate(all_acts, axis=0)
        pca = PCA(n_components=2)
        pca.fit(concat_acts)
        
        # Plot trajectories
        for act, choice in zip(all_acts, all_choices):
            traj = pca.transform(act)
            ax.plot(traj[:, 0], traj[:, 1], color=colors_choice[choice], alpha=0.3, linewidth=1)
            # Mark start
            ax.scatter(traj[0, 0], traj[0, 1], color=colors_choice[choice], 
                      s=20, marker='o', alpha=0.5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=colors_choice[i], lw=2, label=labels_choice[i]) 
                             for i in [0, 1, 2]]
            ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.suptitle('Neural Trajectories (PCA) - Correct Trials Only', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_heatmaps(trial_data, output_path='images/question_2D_MSI_heatmaps.png'):
    """Plot activity heatmaps for correct trials."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_names = list(trial_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        
        # Get correct trials
        acts_list = [a for a, ok in zip(trial_data[name]['activities'], 
                                        trial_data[name]['correct']) if ok]
        
        if len(acts_list) == 0:
            ax.text(0.5, 0.5, 'No correct trials', ha='center', va='center')
            ax.set_title(name)
            continue
        
        # Pad to same length
        min_T = min(a.shape[0] for a in acts_list)
        acts = np.stack([a[:min_T] for a in acts_list], axis=0)
        avg = acts.mean(axis=0)
        
        im = ax.imshow(avg.T, aspect='auto', cmap='viridis',
                      extent=[0, avg.shape[0]-1, 0, avg.shape[1]])
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (steps)', fontsize=11)
        ax.set_ylabel('Hidden unit', fontsize=11)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Activity')
    
    plt.suptitle('Average Hidden Unit Activity (Correct Trials)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_task_structure(env, trial_data, output_path='images/question_2D_MSI_task_structure.png'):
    """Visualize task structure and model responses."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_names = list(trial_data.keys())
    num_trials = 3
    
    fig, axes = plt.subplots(len(model_names), num_trials, 
                            figsize=(4 * num_trials, 3 * len(model_names)))
    
    for r, name in enumerate(model_names):
        for c in range(num_trials):
            ob = trial_data[name]['stimuli'][c]
            probs = trial_data[name]['outputs'][c]
            ax = axes[r, c] if len(model_names) > 1 else axes[c]
            
            t = np.arange(ob.shape[0]) * env.dt
            
            # Plot fixation cue
            ax.plot(t, ob[:, 0], color='gray', label='Fixation', linewidth=1.5, linestyle=':')
            
            # Plot action probabilities
            ax.plot(t, probs[:, 0], color='black', linestyle='--', label='P(fix)', linewidth=1.5)
            ax.plot(t, probs[:, 1], color='blue', label='P(left)', linewidth=1.5)
            ax.plot(t, probs[:, 2], color='red', label='P(right)', linewidth=1.5)
            
            ax.axhline(0.5, color='k', linestyle=':', linewidth=1, alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.set_title(f'{name} Trial {c+1}', fontsize=11)
            if c == 0:
                ax.set_ylabel('Probability', fontsize=10)
            ax.set_xlabel('Time (ms)', fontsize=10)
            
            if r == 0 and c == 0:
                ax.legend(fontsize=9, loc='upper left')
    
    plt.suptitle('MultiSensoryIntegration Task Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_coherence_difficulty(trial_data, output_path='images/question_2D_MSI_coherence_analysis.png'):
    """Analyze accuracy as a function of coherence (task difficulty)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    model_names = list(trial_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        
        # Extract coherences and correctness
        coherences = []
        correct_list = []
        
        for i, trial_info in enumerate(trial_data[name]['trial_info']):
            coh = trial_info.get('coh', 0)
            coherences.append(coh)
            correct_list.append(trial_data[name]['correct'][i])
        
        coherences = np.array(coherences)
        correct_list = np.array(correct_list)
        
        # Bin by coherence
        unique_coh = np.unique(coherences)
        mean_acc = []
        std_acc = []
        counts = []
        
        for coh in unique_coh:
            mask = coherences == coh
            if np.sum(mask) > 0:
                acc = correct_list[mask].mean()
                mean_acc.append(acc)
                std_acc.append(correct_list[mask].std() / np.sqrt(np.sum(mask)))
                counts.append(np.sum(mask))
            else:
                mean_acc.append(0)
                std_acc.append(0)
                counts.append(0)
        
        mean_acc = np.array(mean_acc)
        std_acc = np.array(std_acc)
        
        # Plot with error bars
        ax.errorbar(unique_coh, mean_acc, yerr=std_acc, 
                    marker='o', markersize=8, linewidth=2.5,
                    capsize=5, capthick=2, color=colors[idx],
                    label=name)
        
        # Add count annotations
        for coh, acc, count in zip(unique_coh, mean_acc, counts):
            ax.text(coh, acc + 0.05, f'n={count}', ha='center', 
                   fontsize=8, alpha=0.7)
        
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Chance (50%)')
        ax.set_xlabel('Coherence (Higher = Easier)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.suptitle('Accuracy vs Coherence (Task Difficulty)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_modality_weighting_analysis(trial_data, output_path='images/question_2D_MSI_modality_weighting.png'):
    """Analyze how models perform across different modality weightings."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    model_names = list(trial_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        
        # Extract modality weights and correctness
        weights = []
        correct_list = []
        
        for i, trial_info in enumerate(trial_data[name]['trial_info']):
            weight = trial_info.get('coh_prop', 0.5)  # Proportion of coherence in modality 1
            weights.append(weight)
            correct_list.append(trial_data[name]['correct'][i])
        
        weights = np.array(weights)
        correct_list = np.array(correct_list)
        
        # Bin by weight
        bins = np.linspace(0, 1, 6)  # 5 bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_acc = []
        std_acc = []
        counts = []
        
        for i in range(len(bins) - 1):
            mask = (weights >= bins[i]) & (weights < bins[i+1])
            if np.sum(mask) > 0:
                acc = correct_list[mask].mean()
                mean_acc.append(acc)
                std_acc.append(correct_list[mask].std() / np.sqrt(np.sum(mask)))
                counts.append(np.sum(mask))
            else:
                mean_acc.append(np.nan)
                std_acc.append(0)
                counts.append(0)
        
        mean_acc = np.array(mean_acc)
        std_acc = np.array(std_acc)
        
        # Plot with error bars
        valid = ~np.isnan(mean_acc)
        ax.errorbar(bin_centers[valid], mean_acc[valid], yerr=std_acc[valid],
                    marker='o', markersize=8, linewidth=2.5,
                    capsize=5, capthick=2, color=colors[idx],
                    label=name)
        
        # Add count annotations
        for bc, acc, count in zip(bin_centers[valid], mean_acc[valid], np.array(counts)[valid]):
            ax.text(bc, acc + 0.05, f'n={count}', ha='center',
                   fontsize=8, alpha=0.7)
        
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(0.5, color='purple', linestyle=':', linewidth=1.5, alpha=0.5, label='Equal weighting')
        ax.set_xlabel('Modality 1 Weight (0=Mod2 only, 1=Mod1 only)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.suptitle('Accuracy vs Modality Weighting (Multi-Sensory Integration)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrices(trial_data, output_path='images/question_2D_MSI_confusion_matrices.png'):
    """Plot confusion matrices for all models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    model_names = list(trial_data.keys())
    action_names = ['Fixate (0)', 'Left (1)', 'Right (2)']
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        
        predictions = np.array(trial_data[name]['predictions'])
        ground_truths = np.array(trial_data[name]['targets'])
        
        # Get final actions
        pred_final = []
        true_final = []
        for p, t in zip(predictions, ground_truths):
            if len(p) > 0 and len(t) > 0:
                pred_final.append(np.argmax(p[-1]))
                true_final.append(t[-1])
        
        pred_final = np.array(pred_final)
        true_final = np.array(true_final)
        
        # Build confusion matrix
        conf_matrix = np.zeros((3, 3))
        for true_label in range(3):
            for pred_label in range(3):
                mask = (true_final == true_label) & (pred_final == pred_label)
                conf_matrix[true_label, pred_label] = np.sum(mask)
        
        # Normalize by row (true labels)
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        conf_matrix_norm = conf_matrix / row_sums
        
        # Plot
        im = ax.imshow(conf_matrix_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                count = int(conf_matrix[i, j])
                pct = conf_matrix_norm[i, j]
                color = 'white' if pct > 0.5 else 'black'
                ax.text(j, i, f'{pct:.2f}\n(n={count})',
                       ha='center', va='center', color=color,
                       fontsize=10, fontweight='bold')
        
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(action_names, fontsize=10)
        ax.set_yticklabels(action_names, fontsize=10)
        ax.set_xlabel('Predicted Action', fontsize=11)
        ax.set_ylabel('True Action', fontsize=11)
        ax.set_title(name, fontsize=13, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Proportion', fontsize=10)
    
    plt.suptitle('Confusion Matrices: What Do Models Actually Predict?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_decision_confidence(trial_data, output_path='images/question_2D_MSI_decision_confidence.png'):
    """Analyze decision confidence (max output probability)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    model_names = list(trial_data.keys())
    colors_correct = '#2ca02c'
    colors_incorrect = '#d62728'
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        
        outputs = trial_data[name]['outputs']
        correct_trials = trial_data[name]['correct']
        
        confidence_correct = []
        confidence_incorrect = []
        
        for output, corr in zip(outputs, correct_trials):
            # Get max probability at final timestep
            final_probs = output[-1]
            max_prob = np.max(final_probs)
            
            if corr:
                confidence_correct.append(max_prob)
            else:
                confidence_incorrect.append(max_prob)
        
        # Plot histograms
        bins = np.linspace(0, 1, 21)
        ax.hist(confidence_correct, bins=bins, alpha=0.6, color=colors_correct,
               label=f'Correct (n={len(confidence_correct)})', edgecolor='black', linewidth=1)
        ax.hist(confidence_incorrect, bins=bins, alpha=0.6, color=colors_incorrect,
               label=f'Incorrect (n={len(confidence_incorrect)})', edgecolor='black', linewidth=1)
        
        # Add mean lines
        if len(confidence_correct) > 0:
            mean_corr = np.mean(confidence_correct)
            ax.axvline(mean_corr, color=colors_correct, linestyle='--', linewidth=2.5,
                      label=f'Mean correct: {mean_corr:.3f}')
        
        if len(confidence_incorrect) > 0:
            mean_incorr = np.mean(confidence_incorrect)
            ax.axvline(mean_incorr, color=colors_incorrect, linestyle='--', linewidth=2.5,
                      label=f'Mean incorrect: {mean_incorr:.3f}')
        
        ax.set_xlabel('Max Probability (Confidence)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Decision Confidence: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_analysis(trial_data, output_path='images/question_2D_MSI_error_analysis.png'):
    """Analyze when and how models make errors."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    model_names = list(trial_data.keys())
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        
        # Categorize errors by coherence
        coherences_correct = []
        coherences_incorrect = []
        
        for i, trial_info in enumerate(trial_data[name]['trial_info']):
            coh = trial_info.get('coh', 0)
            if trial_data[name]['correct'][i]:
                coherences_correct.append(coh)
            else:
                coherences_incorrect.append(coh)
        
        # Plot as stacked histogram
        all_coh = np.concatenate([coherences_correct, coherences_incorrect])
        if len(all_coh) > 0:
            bins = np.linspace(all_coh.min(), all_coh.max(), 8)
            
            ax.hist(coherences_correct, bins=bins, alpha=0.7, color='#2ca02c',
                   label='Correct', edgecolor='black', linewidth=1)
            ax.hist(coherences_incorrect, bins=bins, alpha=0.7, color='#d62728',
                   label='Incorrect', edgecolor='black', linewidth=1, bottom=0)
            
            # Calculate error rate per bin
            ax2 = ax.twinx()
            bin_centers = (bins[:-1] + bins[1:]) / 2
            error_rates = []
            
            for i in range(len(bins) - 1):
                corr_in_bin = np.sum((np.array(coherences_correct) >= bins[i]) & 
                                    (np.array(coherences_correct) < bins[i+1]))
                incorr_in_bin = np.sum((np.array(coherences_incorrect) >= bins[i]) & 
                                      (np.array(coherences_incorrect) < bins[i+1]))
                total_in_bin = corr_in_bin + incorr_in_bin
                
                if total_in_bin > 0:
                    error_rates.append(incorr_in_bin / total_in_bin)
                else:
                    error_rates.append(0)
            
            ax2.plot(bin_centers, error_rates, color='black', marker='o', 
                    linewidth=2.5, markersize=8, label='Error Rate')
            ax2.set_ylabel('Error Rate', fontsize=11, color='black')
            ax2.set_ylim([0, 1])
            ax2.tick_params(axis='y', labelcolor='black')
        
        ax.set_xlabel('Coherence', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        if len(all_coh) > 0:
            ax2.legend(loc='upper right', fontsize=10)
    
    plt.suptitle('Error Analysis: Where Do Models Fail?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_choice_decoding_timecourse(trial_data, output_path='images/question_2D_MSI_choice_decoding.png'):
    """Train linear decoders per timestep to track when choice becomes linearly separable."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_names = list(trial_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, name in enumerate(model_names):
        activities = np.array(trial_data[name]['activities'])
        targets = np.array(trial_data[name]['targets'])
        
        # Get final target for each trial
        labels = np.array([t[-1] for t in targets])
        
        if activities.ndim == 2:
            # [N, H] -> expand to [N, 1, H]
            activities = activities[:, np.newaxis, :]
        
        num_trials, num_steps, _ = activities.shape
        
        # Split train/test
        split = int(0.7 * num_trials)
        idx_perm = np.arange(num_trials)
        np.random.seed(42)
        np.random.shuffle(idx_perm)
        train_idx, test_idx = idx_perm[:split], idx_perm[split:]
        
        accs = []
        for t in range(num_steps):
            X_train = activities[train_idx, t, :]
            y_train = labels[train_idx]
            X_test = activities[test_idx, t, :]
            y_test = labels[test_idx]
            
            # Train logistic regression
            clf = LogisticRegression(max_iter=200, random_state=42)
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            accs.append(acc)
        
        ax.plot(np.arange(num_steps), accs, color=colors[idx], 
               label=name, linewidth=2.5)
    
    ax.axhline(1/3, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Chance (33%)')
    ax.set_xlabel('Time step', fontsize=13)
    ax.set_ylabel('Choice decoding accuracy', fontsize=13)
    ax.set_title('Linear Choice Decoding Over Time', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_temporal_dynamics(trial_data, env, output_path='images/question_2D_MSI_temporal_dynamics.png'):
    """Show how decision probabilities evolve over time."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    model_names = list(trial_data.keys())
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        
        outputs = trial_data[name]['outputs']
        correct_trials = trial_data[name]['correct']
        targets = trial_data[name]['targets']
        
        # Separate by choice (left vs right) for correct trials only
        left_probs = []
        right_probs = []
        
        for output, corr, tgt in zip(outputs, correct_trials, targets):
            if corr and len(tgt) > 0:
                final_choice = tgt[-1]
                if final_choice == 1:  # Left
                    left_probs.append(output[:, 1])  # P(left)
                elif final_choice == 2:  # Right
                    right_probs.append(output[:, 2])  # P(right)
        
        if len(left_probs) > 0 and len(right_probs) > 0:
            # Pad to same length
            min_len = min(min(len(p) for p in left_probs), 
                         min(len(p) for p in right_probs))
            
            left_probs = np.array([p[:min_len] for p in left_probs])
            right_probs = np.array([p[:min_len] for p in right_probs])
            
            # Calculate means and SEM
            mean_left = left_probs.mean(axis=0)
            sem_left = left_probs.std(axis=0) / np.sqrt(len(left_probs))
            
            mean_right = right_probs.mean(axis=0)
            sem_right = right_probs.std(axis=0) / np.sqrt(len(right_probs))
            
            time = np.arange(min_len) * env.dt
            
            # Plot with error bands
            ax.plot(time, mean_left, color='blue', linewidth=2.5, label=f'P(Left) when Left is correct (n={len(left_probs)})')
            ax.fill_between(time, mean_left - sem_left, mean_left + sem_left, 
                           color='blue', alpha=0.2)
            
            ax.plot(time, mean_right, color='red', linewidth=2.5, label=f'P(Right) when Right is correct (n={len(right_probs)})')
            ax.fill_between(time, mean_right - sem_right, mean_right + sem_right,
                           color='red', alpha=0.2)
            
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_ylabel('Probability', fontsize=11)
            ax.set_title(name, fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient correct trials', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name, fontsize=13, fontweight='bold')
    
    plt.suptitle('Temporal Dynamics: How Decisions Emerge Over Time (Correct Trials)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_radar_comparison(trial_data, output_path='images/question_2D_MSI_radar_comparison.png'):
    """Radar plot comparing models across multiple metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    model_names = list(trial_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Define metrics
    categories = ['Accuracy', 'Balanced Acc', 'Left Recall', 'Right Recall', 'Consistency']
    N = len(categories)
    
    # Calculate metrics for each model
    for idx, name in enumerate(model_names):
        predictions = np.array(trial_data[name]['predictions'])
        ground_truths = np.array(trial_data[name]['targets'])
        correct = np.array(trial_data[name]['correct'])
        
        # Get final actions
        pred_final = []
        true_final = []
        for p, t in zip(predictions, ground_truths):
            if len(p) > 0 and len(t) > 0:
                pred_final.append(np.argmax(p[-1]))
                true_final.append(t[-1])
        
        pred_final = np.array(pred_final)
        true_final = np.array(true_final)
        
        # Calculate metrics
        accuracy = correct.mean()
        
        # Balanced accuracy (per-class recall)
        recalls = []
        for cls in [1, 2]:  # Left and Right only (skip fixate)
            mask = true_final == cls
            if np.sum(mask) > 0:
                recall = np.sum((pred_final == cls) & mask) / np.sum(mask)
                recalls.append(recall)
        
        balanced_acc = np.mean(recalls) if len(recalls) > 0 else 0
        left_recall = recalls[0] if len(recalls) > 0 else 0
        right_recall = recalls[1] if len(recalls) > 1 else 0
        
        # Consistency: std of correctness (lower is more consistent)
        consistency = 1 - correct.std()  # Invert so higher is better
        
        values = [accuracy, balanced_acc, left_recall, right_recall, consistency]
        
        # Plot
        angles = [n / float(N) * 2 * pi for n in range(N)]
        values += values[:1]  # Close the loop
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    ax.set_xticks(angles[:-1] if len(angles) > N else angles)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.title('Multi-Dimensional Model Comparison (DQN)', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_example_predictions(models, env, output_path='images/question_2D_MSI_example_predictions.png'):
    """Plot example trials showing model predictions over time."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    
    model_names = list(models.keys())
    device = next(iter(models.values())).fc.weight.device
    
    for row, name in enumerate(model_names):
        net = models[name]
        net.eval()
        
        for col in range(3):
            ax = axes[row, col]
            
            # Generate a trial
            env.new_trial()
            ob, gt = env.ob, env.gt
            trial = env.trial
            
            # Run model
            inputs = torch.from_numpy(ob[:, np.newaxis, :]).float().to(device)
            with torch.no_grad():
                out, _, _ = net(inputs)
                probs = torch.softmax(out, dim=-1).cpu().numpy()[:, 0, :]
            
            time = np.arange(len(ob)) * env.dt
            
            # Plot ground truth as background shading
            for t_idx, action in enumerate(gt):
                color_map = {0: 'gray', 1: 'lightblue', 2: 'lightcoral'}
                if action in color_map:
                    if t_idx < len(time) - 1:
                        ax.axvspan(time[t_idx], time[t_idx + 1], 
                                 color=color_map[action], alpha=0.2)
            
            # Plot predicted probabilities
            ax.plot(time, probs[:, 0], color='black', linestyle='--', 
                   label='P(fix)', linewidth=1.5, alpha=0.7)
            ax.plot(time, probs[:, 1], color='blue', label='P(left)', linewidth=2)
            ax.plot(time, probs[:, 2], color='red', label='P(right)', linewidth=2)
            
            # Check if correct
            final_pred = np.argmax(probs[-1, :])
            final_true = gt[-1]
            correct = '✓' if final_pred == final_true else '✗'
            
            coh = trial.get('coh', 0)
            coh_prop = trial.get('coh_prop', 0.5)
            
            ax.axhline(0.5, color='k', linestyle=':', linewidth=1, alpha=0.5)
            ax.set_ylim([0, 1.05])
            
            if col == 0:
                ax.set_ylabel(name, fontsize=11, fontweight='bold')
            
            if row == 0:
                ax.set_title(f'Trial {col+1}', fontsize=11, fontweight='bold')
            
            # Add info text
            ax.text(0.02, 0.98, f'{correct} coh={coh:.0f}, wt={coh_prop:.2f}',
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            if row == 3:
                ax.set_xlabel('Time (ms)', fontsize=10)
            
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Example Trials: Model Predictions Over Time (DQN)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    print("="*70)
    print("MultiSensoryIntegration DQN Analysis")
    print("="*70)
    
    models, env = load_models()
    print("\n[1] Evaluating models...")
    eval_scores = evaluate(models, env, episodes=300)
    
    print("\n[2] Collecting trial data...")
    trial_data = collect_trial_data(models, env, num_trials=200)
    
    print("\n[3] Generating performance comparison...")
    plot_performance_comparison(trial_data, output_path='images/question_2D_MSI_performance.png')
    
    print("\n[4] Generating PCA trajectories...")
    plot_pca_trajectories(trial_data, output_path='images/question_2D_MSI_pca_trajectories.png')
    
    print("\n[5] Generating activity heatmaps...")
    plot_heatmaps(trial_data, output_path='images/question_2D_MSI_heatmaps.png')
    
    print("\n[6] Generating task structure visualization...")
    plot_task_structure(env, trial_data, output_path='images/question_2D_MSI_task_structure.png')
    
    print("\n[7] Analyzing coherence-difficulty relationship...")
    analyze_coherence_difficulty(trial_data, output_path='images/question_2D_MSI_coherence_analysis.png')
    
    print("\n[8] Analyzing modality weighting effects...")
    plot_modality_weighting_analysis(trial_data, output_path='images/question_2D_MSI_modality_weighting.png')
    
    print("\n[9] Generating confusion matrices...")
    plot_confusion_matrices(trial_data, output_path='images/question_2D_MSI_confusion_matrices.png')
    
    print("\n[10] Analyzing decision confidence...")
    plot_decision_confidence(trial_data, output_path='images/question_2D_MSI_decision_confidence.png')
    
    print("\n[11] Analyzing error patterns...")
    plot_error_analysis(trial_data, output_path='images/question_2D_MSI_error_analysis.png')
    
    print("\n[12] Choice decoding timecourse...")
    plot_choice_decoding_timecourse(trial_data, output_path='images/question_2D_MSI_choice_decoding.png')
    
    print("\n[13] Analyzing temporal dynamics...")
    plot_temporal_dynamics(trial_data, env, output_path='images/question_2D_MSI_temporal_dynamics.png')
    
    print("\n[14] Creating radar comparison plot...")
    plot_radar_comparison(trial_data, output_path='images/question_2D_MSI_radar_comparison.png')
    
    print("\n[15] Plotting example predictions...")
    plot_example_predictions(models, env, output_path='images/question_2D_MSI_example_predictions.png')
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nGenerated plots:")
    print("  1. question_2D_MSI_performance.png - Model accuracy comparison")
    print("  2. question_2D_MSI_pca_trajectories.png - Neural trajectories (PCA)")
    print("  3. question_2D_MSI_heatmaps.png - Activity heatmaps")
    print("  4. question_2D_MSI_task_structure.png - Task structure visualization")
    print("  5. question_2D_MSI_coherence_analysis.png - Accuracy vs coherence (difficulty)")
    print("  6. question_2D_MSI_modality_weighting.png - Multi-sensory integration analysis")
    print("  7. question_2D_MSI_confusion_matrices.png - Prediction error patterns")
    print("  8. question_2D_MSI_decision_confidence.png - Confidence calibration")
    print("  9. question_2D_MSI_error_analysis.png - Error analysis by difficulty")
    print(" 10. question_2D_MSI_choice_decoding.png - Linear decoding timecourse")
    print(" 11. question_2D_MSI_temporal_dynamics.png - Decision emergence over time")
    print(" 12. question_2D_MSI_radar_comparison.png - Multi-dimensional comparison")
    print(" 13. question_2D_MSI_example_predictions.png - Example trial predictions")
    print("\nKey analyses (now matching supervised learning analysis):")
    print("  ✓ Coherence analysis - how difficulty affects performance")
    print("  ✓ Modality weighting - how models integrate multi-sensory information")
    print("  ✓ Confusion matrices - what errors do models make?")
    print("  ✓ Decision confidence - are models well-calibrated?")
    print("  ✓ Error patterns - where do models fail?")
    print("  ✓ Choice decoding - when does choice become separable?")
    print("  ✓ Temporal dynamics - how decisions emerge over time")
    print("  ✓ Radar comparison - multi-metric model comparison")
    print("  ✓ Example predictions - actual trial timecourses")
    print("\nComparison with supervised learning:")
    print("  - Run Question_2_multisensory_analysis.py for supervised results")
    print("  - Compare DQN vs supervised across all these metrics")
    print("  - Look for differences in how RL vs supervised learn the task")
    print("="*70)
