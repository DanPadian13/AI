import numpy as np
import neurogym as ngym
from neurogym.utils.data import Dataset

task = 'DelayMatchSampleDistractor1D-v0'
kwargs_env = {'dt': 20}
dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=1, seq_len=300)
env = dataset.env

print('Analyzing DelayMatchSampleDistractor1D task...')
print('='*60)

# Analyze class distribution
all_labels = []
for i in range(100):
    env.new_trial()
    gt = env.gt
    all_labels.extend(gt)

all_labels = np.array(all_labels)
total = len(all_labels)
n_fixate = np.sum(all_labels == 0)
n_match = np.sum(all_labels == 1)

print(f'\nClass distribution over 100 trials ({total} timesteps):')
print(f'  Action 0 (fixate): {n_fixate:5d} ({n_fixate/total*100:.2f}%)')
print(f'  Action 1 (match):  {n_match:5d} ({n_match/total*100:.2f}%)')
print()

ratio = n_fixate / n_match if n_match > 0 else 0
print(f'Imbalance ratio (fixate:match): {ratio:.1f}:1')
print()
print(f'Current class weights: [0.1, 1.0]')
print(f'  → Effective weight on fixate: 0.1 × {n_fixate} = {0.1 * n_fixate:.0f}')
print(f'  → Effective weight on match:  1.0 × {n_match} = {1.0 * n_match:.0f}')
print()
print(f'Recommended balanced weights: [{1.0/ratio:.4f}, 1.0]')
print(f'  → This makes both classes contribute equally')
