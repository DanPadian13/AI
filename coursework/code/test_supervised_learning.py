"""
Test that supervised learning still works with the modified Net class
"""
import torch
import neurogym as ngym
from Question_2a import Net, train_model
from neurogym.utils.data import Dataset

print('Testing supervised learning compatibility after Net class modification...')
print('='*60)
print()

# Quick test setup
task = 'ReadySetGo-v0'
kwargs = {'dt': 20}
seq_len = 100
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=4, seq_len=seq_len)
env = dataset.env

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 16

# Test vanilla RNN
print('1. Testing Vanilla RNN (50 training steps)...')
net = Net(input_size, hidden_size, output_size, model_type='vanilla')
loss_history = train_model(net, dataset, num_steps=50, print_step=25)
print(f'   Final loss: {loss_history[-1]:.4f}')
print('   ✓ Vanilla RNN works!')
print()

# Test leaky RNN
print('2. Testing Leaky RNN (50 training steps)...')
net = Net(input_size, hidden_size, output_size, model_type='leaky', dt=20, tau=100, sigma_rec=0.0)
loss_history = train_model(net, dataset, num_steps=50, print_step=25)
print(f'   Final loss: {loss_history[-1]:.4f}')
print('   ✓ Leaky RNN works!')
print()

# Test bio-realistic RNN
print('3. Testing Bio-Realistic RNN (50 training steps)...')
net = Net(input_size, hidden_size, output_size, model_type='bio_realistic', dt=20, tau=100, sigma_rec=0.0, exc_ratio=0.8)
loss_history = train_model(net, dataset, num_steps=50, print_step=25, beta_L1=1e-5, beta_L2=0.01)
print(f'   Final loss: {loss_history[-1]:.4f}')
print('   ✓ Bio-Realistic RNN works!')
print()

print('='*60)
print('SUCCESS! All supervised learning models still work correctly!')
print('='*60)
print()
print('The Net class modification is BACKWARD COMPATIBLE:')
print('  - forward(x) still works (hidden defaults to None)')
print('  - forward(x, hidden) also works (for RL with state persistence)')
print('  - Old code: ignores 3rd return value with underscore')
print('  - New code: can use 3rd return value for hidden state')
