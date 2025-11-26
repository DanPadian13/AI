import numpy as np
import neurogym as ngym

task = 'DelayMatchSampleDistractor1D-v0'
kwargs_env = {'dt': 20}
dataset = ngym.Dataset(task, env_kwargs=kwargs_env, batch_size=1, seq_len=300)
env = dataset.env

correct_if_always_0 = []
correct_if_always_1 = []
correct_if_random = []
final_actions = []

for i in range(500):
    env.new_trial()
    gt = env.gt
    final_action = gt[-1]
    final_actions.append(final_action)

    correct_if_always_0.append(1 if final_action == 0 else 0)
    correct_if_always_1.append(1 if final_action == 1 else 0)

    random_pred = np.random.randint(0, 2)
    correct_if_random.append(1 if random_pred == final_action else 0)

print('Expected performance for different strategies (500 trials):')
print('='*60)
print(f'Always predict 0 (fixate):  {np.mean(correct_if_always_0):.3f}')
print(f'Always predict 1 (match):   {np.mean(correct_if_always_1):.3f}')
print(f'Random 50/50 guessing:       {np.mean(correct_if_random):.3f}')
print()

final_actions = np.array(final_actions)
print('Distribution of correct final actions:')
print(f'Action 0 (fixate): {np.sum(final_actions == 0):3d} ({np.mean(final_actions == 0)*100:.1f}%)')
print(f'Action 1 (match):  {np.sum(final_actions == 1):3d} ({np.mean(final_actions == 1)*100:.1f}%)')
print()
print('INSIGHT:')
print('If the model learns to always predict the majority class,')
print(f'it would get {max(np.mean(correct_if_always_0), np.mean(correct_if_always_1)):.1%} accuracy')
