import numpy as np
import torch
import neurogym as ngym

CHECKPOINT = 'checkpoints/question_2a_models_and_data.pt'


def main():
    checkpoint = torch.load(CHECKPOINT, weights_only=False)
    cfg = checkpoint['env_config']
    env = ngym.Dataset(cfg['task'], env_kwargs={'dt': cfg['dt']},
                       batch_size=1, seq_len=100).env

    print(f"Task: {cfg['task']}")
    print(f"Action space: {env.action_space.n}")
    actions = getattr(env, 'actions', None)
    if actions is not None:
        print("Action labels:", actions)

    for trial in range(3):
        env.new_trial()
        gt = env.gt
        print(f"\nTrial {trial+1}")
        print("gt shape:", gt.shape)
        unique_vals = np.unique(gt)
        print("Unique target values:", unique_vals)
        pos_idx = np.where(gt > 0)[0]
        print("Positive target indices (first 10):", pos_idx[:10])
        if len(unique_vals) < 10:
            print("gt sequence (first 20):", gt[:20])


if __name__ == "__main__":
    main()
