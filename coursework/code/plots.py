import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import torch
import os
import logging

logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)


def plot_question_2a_results(data_path='data/question_2a_training_data.pt',
                              output_path='images/question_2a_results.png',
                              figsize=(14, 4),
                              dpi=150):
    """
    Plot learning curves and test performance for Question 2a.

    Args:
        data_path: Path to saved training data
        output_path: Path to save the plot
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure
    """
    print(f"Loading data from: {data_path}")
    data = torch.load(data_path, weights_only=False)

    loss_vanilla = data['loss_vanilla']
    loss_leaky = data['loss_leaky']
    loss_leaky_fa = data['loss_leaky_fa']
    loss_bio = data['loss_bio']
    perf_vanilla = data['perf_vanilla']
    perf_leaky = data['perf_leaky']
    perf_leaky_fa = data['perf_leaky_fa']
    perf_bio = data['perf_bio']

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Learning curves - create x-axis with actual step numbers
    # Losses are recorded every 50 steps
    # Skip first value (index 0) as it starts too high
    steps = np.arange(2, len(loss_vanilla) + 1) * 50

    ax.plot(steps, loss_vanilla[1:], label='Vanilla', linewidth=1.5, color='#1f77b4')
    ax.plot(steps, loss_leaky[1:], label='Leaky', linewidth=1.5, color='#ff7f0e')
    ax.plot(steps, loss_leaky_fa[1:], label='Leaky+FA', linewidth=1.5, color='#2ca02c')
    ax.plot(steps, loss_bio[1:], label='Bio-Realistic', linewidth=1.5, color='#d62728')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Task Loss', fontsize=12)

    ax.set_title('Learning Curves: Task Loss Across Models', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return fig


if __name__ == "__main__":
    print("="*70)
    print("Plotting Question 2a Results")
    print("="*70)

    if not os.path.exists('data/question_2a_training_data.pt'):
        print("\nError: Training data not found!")
        print("Please run Question_2a.py first to generate training data.")
        print("="*70)
    else:
        plot_question_2a_results()
        print("\n" + "="*70)
        print("Plotting complete!")
        print("="*70)
