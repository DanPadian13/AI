import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import neurogym as ngym
import os

from Question_2a import Net
import torch.optim as optim


def evaluate_model_bandit(net, env, num_trials=500):
    """Evaluate model on Bandit task with detailed metrics."""
    net.eval()
    device = next(net.parameters()).device

    trial_data = {
        'activities': [],
        'trial_info': [],
        'correct': [],
        'predictions': [],
        'ground_truths': [],
        'rewards': []
    }

    total_reward = 0.0

    with torch.no_grad():
        for i in range(num_trials):
            env.unwrapped.new_trial()
            ob, gt = env.unwrapped.ob, env.unwrapped.gt

            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
            action_pred, rnn_activity = net(inputs)

            action_pred = action_pred.detach().cpu().numpy()
            choice = np.argmax(action_pred[-1, 0, :])
            correct = (choice == gt[-1])

            # Get reward for this choice
            trial_info = env.unwrapped.trial
            reward = trial_info.get('reward', 0.0) if correct else 0.0
            total_reward += reward

            trial_data['activities'].append(rnn_activity[:, 0, :].detach().cpu().numpy())
            trial_data['trial_info'].append(trial_info)
            trial_data['correct'].append(correct)
            trial_data['predictions'].append(choice)
            trial_data['ground_truths'].append(gt[-1])
            trial_data['rewards'].append(reward)

    performance = np.mean(trial_data['correct'])
    avg_reward = total_reward / num_trials

    return performance, avg_reward, trial_data


def train_model_with_lr_decay_bandit(net, dataset, num_steps=5000, lr=0.001, print_step=200,
                                      beta_L1=0.0, beta_L2=0.0, class_weights=None):
    """Training function for Bandit task with learning rate decay."""
    optimizer = optim.Adam(net.parameters(), lr=lr)
    device = next(net.parameters()).device

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    loss_history = []
    running_loss = 0.0
    running_task_loss = 0.0
    running_l1_loss = 0.0
    running_l2_loss = 0.0

    lr_decay_steps = [int(num_steps * 0.3), int(num_steps * 0.6), int(num_steps * 0.85)]
    lr_decays_done = [False, False, False]

    for i in range(num_steps):
        # Multiple learning rate drops at 30%, 60%, 85% of training
        for idx, step in enumerate(lr_decay_steps):
            if not lr_decays_done[idx] and i == step:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                print(f'\n>>> Learning rate dropped to {optimizer.param_groups[0]["lr"]:.6f} at step {i+1}\n')
                lr_decays_done[idx] = True

        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

        optimizer.zero_grad()
        output, activity = net(inputs)
        output = output.view(-1, output.size(-1))

        # Task loss
        task_loss = criterion(output, labels)
        loss = task_loss

        # L1 regularization on weights (sparse connectivity)
        l1_loss = torch.tensor(0.0, device=device)
        if beta_L1 > 0:
            l1_loss = beta_L1 * sum(torch.sum(torch.abs(p)) for p in net.parameters())
            loss = loss + l1_loss

        # L2 regularization on firing rates (low firing rates)
        l2_loss = torch.tensor(0.0, device=device)
        if beta_L2 > 0:
            l2_loss = beta_L2 * torch.mean(activity ** 2)
            loss = loss + l2_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

        running_loss += loss.item()
        running_task_loss += task_loss.item()
        running_l1_loss += l1_loss.item()
        running_l2_loss += l2_loss.item()

        if i % print_step == (print_step - 1):
            avg_loss = running_loss / print_step
            avg_task = running_task_loss / print_step
            avg_l1 = running_l1_loss / print_step
            avg_l2 = running_l2_loss / print_step
            if beta_L1 > 0 or beta_L2 > 0:
                print(f'Step {i+1:5d}, Loss: {avg_loss:.4f} (Task: {avg_task:.4f}, L1: {avg_l1:.4f}, L2: {avg_l2:.4f})')
            else:
                print(f'Step {i+1:5d}, Loss: {avg_loss:.4f}')
            # Save task loss only for fair comparison across models
            loss_history.append(avg_task)
            running_loss = 0.0
            running_task_loss = 0.0
            running_l1_loss = 0.0
            running_l2_loss = 0.0

    return loss_history


# Use GPU if available for faster training
USE_MPS = False

if torch.cuda.is_available():
    device = torch.device('cuda')
elif USE_MPS and torch.backends.mps.is_available():
    device = torch.device('mps')  # Apple Silicon GPU
else:
    device = torch.device('cpu')
print(f"Using device: {device}")


if __name__ == '__main__':
    # ============ TRAINING TOGGLES ============
    TRAIN_VANILLA = True
    TRAIN_LEAKY = True
    TRAIN_LEAKY_FA = True
    TRAIN_BIO = True

    # Quick test mode (faster but less accurate)
    QUICK_TEST = False  # Set to True for faster testing with fewer steps
    # ==========================================

    print("=" * 70)
    print("Question 2: Training Models on Bandit-v0 Task")
    print("=" * 70)
    print(f"\nTraining: Vanilla={TRAIN_VANILLA}, Leaky={TRAIN_LEAKY}, "
          f"Leaky+FA={TRAIN_LEAKY_FA}, Bio={TRAIN_BIO}\n")

    print("[1] Setting up NeuroGym Bandit-v0 task...")
    task = 'Bandit-v0'

    # Bandit task configuration
    # n=2 means 2-arm bandit (2 choices)
    # p=(0.8, 0.2) means arm 1 has 80% reward prob, arm 2 has 20%
    kwargs_env = {
        'dt': 100,
        'n': 2,  # 2-arm bandit
        'p': (0.8, 0.2),  # Different reward probabilities
    }

    seq_len = 100  # Bandit trials are short
    batch_size = 64 if not QUICK_TEST else 32

    dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=batch_size, seq_len=seq_len)
    env = dataset.env
    print(f"Batch size: {batch_size} (larger = faster training)")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = 64
    common_lr = 0.002
    common_noise = 0.1

    print(f"Task: {task}")
    print(f"Input size: {input_size}")
    print(f"Output size: {output_size} (fixation + {kwargs_env['n']} arms)")
    print(f"Hidden size: {hidden_size}")

    # Adjust training duration based on mode
    if QUICK_TEST:
        num_steps = 3000
        num_eval_trials = 200
        print(">>> QUICK TEST MODE: Using 3000 steps <<<\n")
    else:
        num_steps = 10000
        num_eval_trials = 500

    # For Bandit task, we want the model to learn to choose the better arm
    # Most timesteps are fixation, so we weight the decision timesteps higher
    class_weights = torch.tensor([1.0, 3.0, 3.0], dtype=torch.float32).to(device)
    print(f"Training strategy: Balanced class weights {class_weights.tolist()}")
    print(f"  Fixation weighted 1x, decisions weighted 3x")

    loss_dict = {}
    perf_dict = {}
    reward_dict = {}
    trial_data_dict = {}
    models_dict = {}

    # ---------------- Vanilla ----------------
    if TRAIN_VANILLA:
        print("\n[2] Training Vanilla RNN...")
        print("-" * 70)
        net_vanilla = Net(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            model_type='vanilla'
        ).to(device)

        loss_vanilla = train_model_with_lr_decay_bandit(
            net_vanilla, dataset, num_steps=num_steps, lr=common_lr,
            class_weights=class_weights
        )
        perf_vanilla, reward_vanilla, data_vanilla = evaluate_model_bandit(
            net_vanilla, env, num_trials=num_eval_trials
        )

        loss_dict['vanilla'] = loss_vanilla
        perf_dict['vanilla'] = perf_vanilla
        reward_dict['vanilla'] = reward_vanilla
        trial_data_dict['vanilla'] = data_vanilla
        models_dict['vanilla'] = net_vanilla

        print(f"Vanilla RNN - Accuracy: {perf_vanilla:.3f}, Avg Reward: {reward_vanilla:.3f}")
        print()
    else:
        print("\n[2] Skipping Vanilla RNN\n")

    # ---------------- Leaky ----------------
    if TRAIN_LEAKY:
        print("[3] Training Leaky RNN...")
        print("-" * 70)
        net_leaky = Net(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            model_type='leaky',
            dt=env.dt,
            tau=100,
            sigma_rec=common_noise
        ).to(device)

        loss_leaky = train_model_with_lr_decay_bandit(
            net_leaky, dataset, num_steps=num_steps, lr=common_lr,
            class_weights=class_weights
        )
        perf_leaky, reward_leaky, data_leaky = evaluate_model_bandit(
            net_leaky, env, num_trials=num_eval_trials
        )

        loss_dict['leaky'] = loss_leaky
        perf_dict['leaky'] = perf_leaky
        reward_dict['leaky'] = reward_leaky
        trial_data_dict['leaky'] = data_leaky
        models_dict['leaky'] = net_leaky

        print(f"Leaky RNN - Accuracy: {perf_leaky:.3f}, Avg Reward: {reward_leaky:.3f}")
        print()
    else:
        print("[3] Skipping Leaky RNN\n")

    # ---------------- Leaky + FA ----------------
    if TRAIN_LEAKY_FA:
        print("[4] Training Leaky RNN + Feedback Alignment...")
        print("-" * 70)
        net_leaky_fa = Net(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            model_type='leaky_fa',
            dt=env.dt,
            tau=100,
            sigma_rec=common_noise
        ).to(device)

        loss_leaky_fa = train_model_with_lr_decay_bandit(
            net_leaky_fa, dataset, num_steps=num_steps, lr=common_lr,
            class_weights=class_weights
        )
        perf_leaky_fa, reward_leaky_fa, data_leaky_fa = evaluate_model_bandit(
            net_leaky_fa, env, num_trials=num_eval_trials
        )

        loss_dict['leaky_fa'] = loss_leaky_fa
        perf_dict['leaky_fa'] = perf_leaky_fa
        reward_dict['leaky_fa'] = reward_leaky_fa
        trial_data_dict['leaky_fa'] = data_leaky_fa
        models_dict['leaky_fa'] = net_leaky_fa

        print(f"Leaky RNN + FA - Accuracy: {perf_leaky_fa:.3f}, Avg Reward: {reward_leaky_fa:.3f}")
        print()
    else:
        print("[4] Skipping Leaky RNN + FA\n")

    # ---------------- Bio-realistic ----------------
    if TRAIN_BIO:
        print("[5] Training Biologically Realistic RNN...")
        print("-" * 70)
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

        loss_bio = train_model_with_lr_decay_bandit(
            net_bio, dataset, num_steps=num_steps, lr=common_lr,
            beta_L1=0.0001, beta_L2=0.005, class_weights=class_weights
        )
        perf_bio, reward_bio, data_bio = evaluate_model_bandit(
            net_bio, env, num_trials=num_eval_trials
        )

        loss_dict['bio'] = loss_bio
        perf_dict['bio'] = perf_bio
        reward_dict['bio'] = reward_bio
        trial_data_dict['bio'] = data_bio
        models_dict['bio'] = net_bio

        print(f"Bio-Realistic RNN - Accuracy: {perf_bio:.3f}, Avg Reward: {reward_bio:.3f}")
        print()
    else:
        print("[5] Skipping Bio-Realistic RNN\n")

    # ---------------- Summary ----------------
    print("[6] Performance Summary:")
    print("-" * 70)
    label_map = {
        'vanilla': "Vanilla RNN",
        'leaky': "Leaky RNN",
        'leaky_fa': "Leaky RNN + FA",
        'bio': "Bio-Realistic RNN"
    }
    for key, label in label_map.items():
        if key in perf_dict:
            print(f"{label:20s} Acc: {perf_dict[key]:.3f}, Reward: {reward_dict[key]:.3f}")
        else:
            print(f"{label:20s} (not trained)")
    print()

    # ---------------- Save ----------------
    print("[7] Saving models and data...")
    print("-" * 70)
    os.makedirs('checkpoints', exist_ok=True)

    save_payload = {
        'loss_dict': loss_dict,
        'perf_dict': perf_dict,
        'reward_dict': reward_dict,
        'trial_data_dict': trial_data_dict,
        'env_config': {'dt': env.dt, 'task': task, 'seq_len': seq_len,
                      'n_arms': kwargs_env['n'], 'probs': kwargs_env['p']}
    }

    # Only save models that were actually trained
    if 'vanilla' in models_dict:
        save_payload['vanilla_model'] = models_dict['vanilla'].state_dict()
    if 'leaky' in models_dict:
        save_payload['leaky_model'] = models_dict['leaky'].state_dict()
    if 'leaky_fa' in models_dict:
        save_payload['leaky_fa_model'] = models_dict['leaky_fa'].state_dict()
    if 'bio' in models_dict:
        save_payload['bio_model'] = models_dict['bio'].state_dict()

    torch.save(save_payload, 'checkpoints/question_2_bandit_models_and_data.pt')
    print("Saved: checkpoints/question_2_bandit_models_and_data.pt")
    print()

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nTask: Bandit-v0 (Multi-Armed Bandit)")
    print("Key characteristics:")
    print("  - Decision-making task: choose between multiple arms")
    print("  - Reward probabilities: Arm 1 = 80%, Arm 2 = 20%")
    print("  - Tests exploration vs exploitation trade-off")
    print("  - Requires learning reward statistics over time")
    print("\nExpected Performance:")
    print("  - Optimal strategy: Always choose Arm 1 (80% reward)")
    print("  - Perfect learner: 80% accuracy (matching optimal arm probability)")
    print("  - Random baseline: 50% accuracy")
    print("  - All models should learn to prefer the higher reward arm")
    print("\nNext steps:")
    print("  Run Question_2_bandit_analysis.py to generate visualizations and analysis")
    print("=" * 70)
