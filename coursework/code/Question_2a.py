import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import neurogym as ngym
from plots import plot_question_2a_results

logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)


class FeedbackAlignmentLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight_feedback = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        output = nn.functional.linear(input, self.weight, self.bias)

        if self.training and input.requires_grad:
            def backward_hook(grad):
                return grad.mm(self.weight_feedback)
            input.register_hook(backward_hook)

        return output


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        
    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size).to(input.device)
    
    def recurrence(self, input, hidden):
        total_input = self.input2h(input) + self.h2h(hidden)
        output = torch.tanh(total_input)
        return output
    
    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input)
        
        output = []
        for i in range(input.size(0)):
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
        
        output = torch.stack(output, dim=0)
        return output, hidden


class LeakyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dt=None, tau=100, sigma_rec=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        self._sigma_rec = np.sqrt(2 * alpha) * sigma_rec
        
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        
    def init_hidden(self, input):
        batch_size = input.shape[1]
        return (torch.zeros(batch_size, self.hidden_size).to(input.device),
                torch.zeros(batch_size, self.hidden_size).to(input.device))
    
    def recurrence(self, input, hidden):
        state, output = hidden
        total_input = self.input2h(input) + self.h2h(output)
        state = state * self.oneminusalpha + total_input * self.alpha
        
        if self._sigma_rec > 0:
            state += self._sigma_rec * torch.randn_like(state)
        
        output = torch.relu(state)
        return state, output
    
    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input)
        
        output = []
        for i in range(input.size(0)):
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden[1])
        
        output = torch.stack(output, dim=0)
        return output, hidden


class LeakyRNNFeedbackAlignment(nn.Module):
    def __init__(self, input_size, hidden_size, dt=None, tau=100, sigma_rec=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        self._sigma_rec = np.sqrt(2 * alpha) * sigma_rec
        
        self.input2h = FeedbackAlignmentLinear(input_size, hidden_size)
        self.h2h = FeedbackAlignmentLinear(hidden_size, hidden_size)
        
    def init_hidden(self, input):
        batch_size = input.shape[1]
        return (torch.zeros(batch_size, self.hidden_size).to(input.device),
                torch.zeros(batch_size, self.hidden_size).to(input.device))
    
    def recurrence(self, input, hidden):
        state, output = hidden
        total_input = self.input2h(input) + self.h2h(output)
        state = state * self.oneminusalpha + total_input * self.alpha
        
        if self._sigma_rec > 0:
            state += self._sigma_rec * torch.randn_like(state)
        
        output = torch.relu(state)
        return state, output
    
    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input)
        
        output = []
        for i in range(input.size(0)):
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden[1])
        
        output = torch.stack(output, dim=0)
        return output, hidden


class BiologicallyRealisticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dt=None, tau=100, sigma_rec=0,
                 exc_ratio=0.8, sparsity=0.2, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        self.exc_ratio = exc_ratio
        self.sparsity = sparsity

        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau

        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        self._sigma_rec = np.sqrt(2 * alpha) * sigma_rec

        self.input2h = FeedbackAlignmentLinear(input_size, hidden_size)
        self.h2h = FeedbackAlignmentLinear(hidden_size, hidden_size)

        # Dale's principle: excitatory/inhibitory neurons
        n_exc = int(hidden_size * exc_ratio)
        self.dale_mask = torch.ones(hidden_size)
        self.dale_mask[n_exc:] = -1
        self.dale_mask = nn.Parameter(self.dale_mask.unsqueeze(0), requires_grad=False)

        # Sparse connectivity mask
        sparse_mask = torch.rand(hidden_size, hidden_size) < sparsity
        sparse_mask.fill_diagonal_(True)
        self.sparse_mask = nn.Parameter(sparse_mask.float(), requires_grad=False)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return (torch.zeros(batch_size, self.hidden_size).to(input.device),
                torch.zeros(batch_size, self.hidden_size).to(input.device))

    def recurrence(self, input, hidden):
        state, output = hidden

        # Apply Dale's principle: rectify then apply sign
        h2h_weight = torch.relu(self.h2h.weight) * self.dale_mask
        # Apply sparsity mask
        h2h_weight = h2h_weight * self.sparse_mask

        total_input = self.input2h(input) + nn.functional.linear(output, h2h_weight, self.h2h.bias)
        state = state * self.oneminusalpha + total_input * self.alpha

        if self._sigma_rec > 0:
            state += self._sigma_rec * torch.randn_like(state)

        output = torch.relu(state)
        return state, output

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input)

        output = []
        for i in range(input.size(0)):
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden[1])

        output = torch.stack(output, dim=0)
        return output, hidden


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 model_type='vanilla', **kwargs):
        super().__init__()
        self.model_type = model_type

        if model_type == 'vanilla':
            self.rnn = VanillaRNN(input_size, hidden_size, **kwargs)
        elif model_type == 'leaky':
            self.rnn = LeakyRNN(input_size, hidden_size, **kwargs)
        elif model_type == 'leaky_fa':
            self.rnn = LeakyRNNFeedbackAlignment(input_size, hidden_size, **kwargs)
        elif model_type == 'bio_realistic':
            self.rnn = BiologicallyRealisticRNN(input_size, hidden_size, **kwargs)
        else:
            raise ValueError("model_type must be 'vanilla', 'leaky', 'leaky_fa', or 'bio_realistic'")

        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        out = self.fc(rnn_activity)
        return out, rnn_activity


def train_model(net, dataset, num_steps=5000, lr=0.001, print_step=200, beta_L2=0.0):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = next(net.parameters()).device

    loss_history = []
    running_loss = 0.0
    running_task_loss = 0.0
    running_reg_loss = 0.0

    for i in range(num_steps):
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

        optimizer.zero_grad()
        output, activity = net(inputs)
        output = output.view(-1, output.size(-1))

        # Task loss
        task_loss = criterion(output, labels)

        # L2 regularization on firing rates (biological realism)
        if beta_L2 > 0:
            firing_rate_loss = beta_L2 * torch.mean(activity ** 2)
            loss = task_loss + firing_rate_loss
        else:
            firing_rate_loss = torch.tensor(0.0)
            loss = task_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_task_loss += task_loss.item()
        running_reg_loss += firing_rate_loss.item()

        if i % print_step == (print_step - 1):
            avg_loss = running_loss / print_step
            avg_task = running_task_loss / print_step
            avg_reg = running_reg_loss / print_step
            if beta_L2 > 0:
                print(f'Step {i+1:5d}, Loss: {avg_loss:.4f} (Task: {avg_task:.4f}, L2: {avg_reg:.4f})')
            else:
                print(f'Step {i+1:5d}, Loss: {avg_loss:.4f}')
            loss_history.append(avg_loss)
            running_loss = 0.0
            running_task_loss = 0.0
            running_reg_loss = 0.0

    return loss_history


def evaluate_model(net, env, num_trials=500):
    net.eval()
    device = next(net.parameters()).device
    
    trial_data = {
        'activities': [],
        'trial_info': [],
        'correct': []
    }
    
    with torch.no_grad():
        for i in range(num_trials):
            env.new_trial()
            ob, gt = env.ob, env.gt
            
            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
            action_pred, rnn_activity = net(inputs)
            
            action_pred = action_pred.detach().cpu().numpy()
            choice = np.argmax(action_pred[-1, 0, :])
            correct = (choice == gt[-1])
            
            trial_data['activities'].append(rnn_activity[:, 0, :].detach().cpu().numpy())
            trial_data['trial_info'].append(env.unwrapped.trial)
            trial_data['correct'].append(correct)
    
    performance = np.mean(trial_data['correct'])
    return performance, trial_data

if __name__ == "__main__":
    print("="*70)
    print("Question 2a: Brain-Inspired RNNs on ReadySetGo Timing Task")
    print("="*70)
    
    device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    print("\n[1] Setting up NeuroGym task...")
    task = 'ReadySetGo-v0'
    kwargs = {'dt': 20}
    seq_len = 100
    
    dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16, seq_len=seq_len)
    env = dataset.env
    
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = 50
    
    print(f"Task: {task}")
    print(f"Description: Timing task - measure and produce time intervals")
    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}")
    print(f"Hidden size: {hidden_size}")
    
    print("\n[2] Training Vanilla RNN...")
    print("-"*70)
    net_vanilla = Net(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        model_type='vanilla'
    ).to(device)
    
    loss_vanilla = train_model(net_vanilla, dataset, num_steps=5000)
    
    print("\n[3] Training Leaky RNN (τ=100, decay, noise)...")
    print("-"*70)
    net_leaky = Net(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        model_type='leaky',
        dt=env.dt,
        tau=100,
        sigma_rec=0.15
    ).to(device)
    
    loss_leaky = train_model(net_leaky, dataset, num_steps=5000)

    print("\n[3b] Training Leaky RNN with Feedback Alignment (biologically realistic)...")
    print("-"*70)
    net_leaky_fa = Net(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        model_type='leaky_fa',
        dt=env.dt,
        tau=100,
        sigma_rec=0.15
    ).to(device)

    loss_leaky_fa = train_model(net_leaky_fa, dataset, num_steps=5000)

    print("\n[3c] Training Biologically Realistic RNN (FA + Dale + Sparse + L2)...")
    print("-"*70)
    net_bio = Net(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        model_type='bio_realistic',
        dt=env.dt,
        tau=100,
        sigma_rec=0.15,
        exc_ratio=0.8,
        sparsity=0.2
    ).to(device)

    loss_bio = train_model(net_bio, dataset, num_steps=5000, beta_L2=0.01)

    print("\n[4] Evaluating models...")
    print("-"*70)
    
    perf_vanilla, data_vanilla = evaluate_model(net_vanilla, env, num_trials=500)
    perf_leaky, data_leaky = evaluate_model(net_leaky, env, num_trials=500)
    perf_leaky_fa, data_leaky_fa = evaluate_model(net_leaky_fa, env, num_trials=500)
    perf_bio, data_bio = evaluate_model(net_bio, env, num_trials=500)

    print(f"\nVanilla RNN: {perf_vanilla:.3f}")
    print(f"Leaky RNN: {perf_leaky:.3f}")
    print(f"Leaky RNN + Feedback Alignment: {perf_leaky_fa:.3f}")
    print(f"Biologically Realistic RNN: {perf_bio:.3f}")
    
    print("\n[5] Saving training data...")
    import os
    os.makedirs('data', exist_ok=True)
    torch.save({
        'loss_vanilla': loss_vanilla,
        'loss_leaky': loss_leaky,
        'loss_leaky_fa': loss_leaky_fa,
        'loss_bio': loss_bio,
        'perf_vanilla': perf_vanilla,
        'perf_leaky': perf_leaky,
        'perf_leaky_fa': perf_leaky_fa,
        'perf_bio': perf_bio,
        'task': task
    }, 'data/question_2a_training_data.pt')
    print("Saved: data/question_2a_training_data.pt")

    print("\n[5b] Plotting results...")
    plot_question_2a_results()
    
    print("\n[6] Saving models and data...")
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
        'env_config': {'dt': env.dt, 'task': task}
    }, 'checkpoints/question_2a_models_and_data.pt')
    print("Saved: checkpoints/question_2a_models_and_data.pt")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    print("\nTask: ReadySetGo-v0 (Timing Task)")
    print("Why this task is perfect for testing brain-inspired models:")
    print("  - Requires temporal processing and timing")
    print("  - Leaky RNN's time constant (τ) directly helps with timing")
    print("  - Tests ability to measure and reproduce intervals")
    print("\nProgression of Biological Realism:")
    print("1. Vanilla RNN: Standard backprop, no time constants")
    print("2. Leaky RNN: τ=100, decay, noise + standard backprop")
    print("3. Leaky RNN + FA: τ=100, decay, noise + random backprop weights")
    print("   → Feedback Alignment: biologically realistic learning")
    print("4. Biologically Realistic RNN: FA + Dale's principle + Sparse + L2")
    print("   → Dale's principle: 80% excitatory, 20% inhibitory neurons")
    print("   → Sparse connectivity: 20% connection probability")
    print("   → L2 regularization: low firing rates (β=0.01)")
    print("="*70)
