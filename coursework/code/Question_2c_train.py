import numpy as np
import torch
import torch.nn as nn
import neurogym as ngym
import os

from Question_2a import Net, evaluate_model
import torch.optim as optim


def train_model_with_lr_decay(net, dataset, num_steps=5000, lr=0.001, print_step=200,
                              beta_L1=0.0, beta_L2=0.0, class_weights=None):
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

    lr_dropped = False

    for i in range(num_steps):
        # Drop learning rate by 0.5 halfway through training
        if not lr_dropped and i == num_steps // 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            print(f'\n>>> Learning rate dropped to {optimizer.param_groups[0]["lr"]:.6f} at step {i+1}\n')
            lr_dropped = True

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


device = torch.device('cpu')
print(f"Using device: {device}")


if __name__ == '__main__':
    # ============ TRAINING TOGGLES ============
    TRAIN_VANILLA = True
    TRAIN_LEAKY = True
    TRAIN_LEAKY_FA = True
    TRAIN_BIO = True
    # ==========================================

    print("=" * 70)
    print("Question 2c: Training Models on DelayMatchSampleDistractor1D Task")
    print("=" * 70)
    print(f"\nTraining: Vanilla={TRAIN_VANILLA}, Leaky={TRAIN_LEAKY}, "
          f"Leaky+FA={TRAIN_LEAKY_FA}, Bio={TRAIN_BIO}\n")

    print("[1] Setting up NeuroGym DelayMatchSampleDistractor1D task...")
    task = 'DelayMatchSampleDistractor1D-v0'
    kwargs_env = {'dt': 20}
    seq_len = 300

    dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=16, seq_len=seq_len)
    env = dataset.env

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = 64
    common_lr = 0.002
    common_noise = 0.01
    num_steps = 2000
    class_weights = torch.tensor([0.2,1], dtype=torch.float32).to(device)

    loss_dict = {}
    perf_dict = {}
    trial_data_dict = {}
    models_dict = {}

    # ---------------- Vanilla ----------------
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
        perf_vanilla, data_vanilla = evaluate_model(net_vanilla, env, num_trials=500)

        loss_dict['vanilla'] = loss_vanilla
        perf_dict['vanilla'] = perf_vanilla
        trial_data_dict['vanilla'] = data_vanilla
        models_dict['vanilla'] = net_vanilla

        print(f"Vanilla RNN Performance: {perf_vanilla:.3f}")
        print()
    else:
        print("[2] Skipping Vanilla RNN\n")

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

        loss_leaky = train_model_with_lr_decay(
            net_leaky, dataset, num_steps=num_steps, lr=common_lr,
            class_weights=class_weights
        )
        perf_leaky, data_leaky = evaluate_model(net_leaky, env, num_trials=500)

        loss_dict['leaky'] = loss_leaky
        perf_dict['leaky'] = perf_leaky
        trial_data_dict['leaky'] = data_leaky
        models_dict['leaky'] = net_leaky

        print(f"Leaky RNN Performance: {perf_leaky:.3f}")
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
        perf_leaky_fa, data_leaky_fa = evaluate_model(net_leaky_fa, env, num_trials=500)

        loss_dict['leaky_fa'] = loss_leaky_fa
        perf_dict['leaky_fa'] = perf_leaky_fa
        trial_data_dict['leaky_fa'] = data_leaky_fa
        models_dict['leaky_fa'] = net_leaky_fa

        print(f"Leaky RNN + FA Performance: {perf_leaky_fa:.3f}")
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

        loss_bio = train_model_with_lr_decay(
            net_bio, dataset, num_steps=num_steps, lr=common_lr,
            beta_L1=0.0005, beta_L2=0.01,
            class_weights=class_weights
        )
        perf_bio, data_bio = evaluate_model(net_bio, env, num_trials=500)

        loss_dict['bio'] = loss_bio
        perf_dict['bio'] = perf_bio
        trial_data_dict['bio'] = data_bio
        models_dict['bio'] = net_bio

        print(f"Bio-Realistic RNN Performance: {perf_bio:.3f}")
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
            print(f"{label:20s} {perf_dict[key]:.3f}")
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

    torch.save(save_payload, 'checkpoints/question_2c_models_and_data.pt')
    print("Saved: checkpoints/question_2c_models_and_data.pt")
    print()

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nTask: DelayMatchSampleDistractor1D-v0 (Working Memory + Distractors)")
    print("Key differences from ReadySetGo (timing task):")
    print("  - Requires maintaining stimulus information during delay")
    print("  - Multiple distractor stimuli presented - tests robustness")
    print("  - Must resist interference and maintain memory until matching test")
    print("  - Tests working memory + distractor resistance, not timing")
    print("\nNext steps:")
    print("  Run Question_2c_analysis.py to generate visualizations and analysis")
    print("=" * 70)
