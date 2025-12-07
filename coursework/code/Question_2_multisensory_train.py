import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import neurogym as ngym
import os

from Question_2a import Net
import torch.optim as optim


def evaluate_model_balanced(net, env, num_trials=500):
    """Evaluate model on MultiSensoryIntegration task with balanced metrics."""
    net.eval()
    device = next(net.parameters()).device

    trial_data = {
        'activities': [],
        'trial_info': [],
        'correct': [],
        'predictions': [],
        'ground_truths': []
    }

    with torch.no_grad():
        for i in range(num_trials):
            env.unwrapped.new_trial()
            ob, gt = env.unwrapped.ob, env.unwrapped.gt

            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
            action_pred, rnn_activity, _ = net(inputs)

            action_pred = action_pred.detach().cpu().numpy()
            choice = np.argmax(action_pred[-1, 0, :])
            correct = (choice == gt[-1])

            trial_data['activities'].append(rnn_activity[:, 0, :].detach().cpu().numpy())
            trial_data['trial_info'].append(env.unwrapped.trial)
            trial_data['correct'].append(correct)
            trial_data['predictions'].append(choice)
            trial_data['ground_truths'].append(gt[-1])

    predictions = np.array(trial_data['predictions'])
    ground_truths = np.array(trial_data['ground_truths'])

    # Calculate balanced accuracy (average per-class recall)
    unique_classes = np.unique(ground_truths)
    per_class_recalls = []

    for cls in unique_classes:
        mask = ground_truths == cls
        if np.sum(mask) > 0:
            recall = np.sum((predictions == cls) & mask) / np.sum(mask)
            per_class_recalls.append(recall)

    performance = np.mean(trial_data['correct'])
    balanced_acc = np.mean(per_class_recalls) if per_class_recalls else 0.0

    return performance, balanced_acc, trial_data


def train_model_with_lr_decay(net, dataset, num_steps=5000, lr=0.001, print_step=200,
                              beta_L1=0.0, beta_L2=0.0, class_weights=None):
    """Training function with learning rate decay."""
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
        # Learning rate decay at 30%, 60%, 85% of training
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
        output, activity, _ = net(inputs)
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
            loss_history.append(avg_task)
            running_loss = 0.0
            running_task_loss = 0.0
            running_l1_loss = 0.0
            running_l2_loss = 0.0

    return loss_history


# Device selection
USE_MPS = False

if torch.cuda.is_available():
    device = torch.device('cuda')
elif USE_MPS and torch.backends.mps.is_available():
    device = torch.device('mps')
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
    print("Question 2: Training Models on MultiSensoryIntegration-v0")
    print("=" * 70)
    print(f"\nTraining: Vanilla={TRAIN_VANILLA}, Leaky={TRAIN_LEAKY}, "
          f"Leaky+FA={TRAIN_LEAKY_FA}, Bio={TRAIN_BIO}\n")

    print("[1] Setting up NeuroGym MultiSensoryIntegration-v0 task...")
    task = 'MultiSensoryIntegration-v0'
    kwargs_env = {'dt': 100}
    seq_len = 50  # Short trials (11 timesteps)

    # Larger batch size for faster training
    batch_size = 64 if not QUICK_TEST else 32

    dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=batch_size, seq_len=seq_len)
    env = dataset.env
    print(f"Batch size: {batch_size}")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = 64
    common_lr = 0.001
    common_noise = 0.1

    print(f"Task: {task}")
    print(f"Description: Integrate information from multiple sensory modalities")
    print(f"Input size: {input_size} (fixation + 4 sensory channels)")
    print(f"Output size: {output_size} (fixate, left, right)")
    print(f"Hidden size: {hidden_size}")

    # Adjust training duration based on mode
    if QUICK_TEST:
        num_steps = 3000
        num_eval_trials = 200
        print("\n>>> QUICK TEST MODE: Using 3000 steps <<<\n")
    else:
        num_steps = 10000
        num_eval_trials = 200

    # Class weights: balance fixation vs decision classes
    # 90.9% fixation, so weight decisions higher
    class_weights = torch.tensor([1.0, 5.0, 5.0], dtype=torch.float32).to(device)
    print(f"Training strategy: Balanced class weights {class_weights.tolist()}")
    print(f"  Fixation weighted 1x, decisions weighted 5x (compensates for 91% fixation)\n")

    loss_dict = {}
    perf_dict = {}
    bal_acc_dict = {}
    trial_data_dict = {}
    models_dict = {}

    # ---------------- Vanilla RNN ----------------
    if TRAIN_VANILLA:
        print("[2] Training Vanilla RNN...")
        print("-" * 70)
        net_vanilla = Net(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            model_type='vanilla'
        ).to(device)

        loss_vanilla = train_model_with_lr_decay(
            net_vanilla, dataset, num_steps=num_steps, lr=common_lr,
            class_weights=class_weights
        )
        perf_vanilla, bal_acc_vanilla, data_vanilla = evaluate_model_balanced(
            net_vanilla, env, num_trials=num_eval_trials
        )

        loss_dict['vanilla'] = loss_vanilla
        perf_dict['vanilla'] = perf_vanilla
        bal_acc_dict['vanilla'] = bal_acc_vanilla
        trial_data_dict['vanilla'] = data_vanilla
        models_dict['vanilla'] = net_vanilla

        print(f"Vanilla RNN - Accuracy: {perf_vanilla:.3f}, Balanced Acc: {bal_acc_vanilla:.3f}")
        print()
    else:
        print("[2] Skipping Vanilla RNN\n")

    # ---------------- Leaky RNN ----------------
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

        loss_leaky = train_model_with_lr_decay(
            net_leaky, dataset, num_steps=num_steps, lr=common_lr,
            class_weights=class_weights
        )
        perf_leaky, bal_acc_leaky, data_leaky = evaluate_model_balanced(
            net_leaky, env, num_trials=num_eval_trials
        )

        loss_dict['leaky'] = loss_leaky
        perf_dict['leaky'] = perf_leaky
        bal_acc_dict['leaky'] = bal_acc_leaky
        trial_data_dict['leaky'] = data_leaky
        models_dict['leaky'] = net_leaky

        print(f"Leaky RNN - Accuracy: {perf_leaky:.3f}, Balanced Acc: {bal_acc_leaky:.3f}")
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

        loss_leaky_fa = train_model_with_lr_decay(
            net_leaky_fa, dataset, num_steps=num_steps, lr=common_lr,
            class_weights=class_weights
        )
        perf_leaky_fa, bal_acc_leaky_fa, data_leaky_fa = evaluate_model_balanced(
            net_leaky_fa, env, num_trials=num_eval_trials
        )

        loss_dict['leaky_fa'] = loss_leaky_fa
        perf_dict['leaky_fa'] = perf_leaky_fa
        bal_acc_dict['leaky_fa'] = bal_acc_leaky_fa
        trial_data_dict['leaky_fa'] = data_leaky_fa
        models_dict['leaky_fa'] = net_leaky_fa

        print(f"Leaky RNN + FA - Accuracy: {perf_leaky_fa:.3f}, Balanced Acc: {bal_acc_leaky_fa:.3f}")
        print()
    else:
        print("[4] Skipping Leaky RNN + FA\n")

    # ---------------- Bio-realistic RNN ----------------
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

        loss_bio = train_model_with_lr_decay(
            net_bio, dataset, num_steps=num_steps, lr=common_lr,
            beta_L1=0.00005, beta_L2=0.01, class_weights=class_weights
        )
        perf_bio, bal_acc_bio, data_bio = evaluate_model_balanced(
            net_bio, env, num_trials=num_eval_trials
        )

        loss_dict['bio'] = loss_bio
        perf_dict['bio'] = perf_bio
        bal_acc_dict['bio'] = bal_acc_bio
        trial_data_dict['bio'] = data_bio
        models_dict['bio'] = net_bio

        print(f"Bio-Realistic RNN - Accuracy: {perf_bio:.3f}, Balanced Acc: {bal_acc_bio:.3f}")
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
            print(f"{label:20s} Acc: {perf_dict[key]:.3f}, Bal Acc: {bal_acc_dict[key]:.3f}")
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
        'bal_acc_dict': bal_acc_dict,
        'trial_data_dict': trial_data_dict,
        'env_config': {'dt': env.dt, 'task': task, 'seq_len': seq_len}
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

    torch.save(save_payload, 'checkpoints/question_2_multisensory_models_and_data.pt')
    print("Saved: checkpoints/question_2_multisensory_models_and_data.pt")
    print()

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nTask: MultiSensoryIntegration-v0")
    print("Key characteristics:")
    print("  - Multi-modal sensory integration task")
    print("  - 5 input features: fixation + 4 sensory modalities")
    print("  - 3 output actions: fixate, left, right")
    print("  - Very short trials: only 11 timesteps!")
    print("  - Variable difficulty via coherence (5-50) and modality weighting")
    print("\nExpected Performance:")
    print("  - This is an EASY task - models should achieve 80-95% accuracy")
    print("  - Tests multi-modal evidence integration")
    print("  - All architectures should perform well")
    print("  - Differences show how architectures combine information")
    print("\nNext steps:")
    print("  Run Question_2_multisensory_analysis.py to generate visualizations")
    print("=" * 70)
