import numpy as np
import matplotlib.pyplot as plt
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

    # Learning curves
    ax.plot(loss_vanilla, label='Vanilla', linewidth=2, color='#1f77b4')
    ax.plot(loss_leaky, label='Leaky', linewidth=2, color='#ff7f0e')
    ax.plot(loss_leaky_fa, label='Leaky+FA', linewidth=2, color='#2ca02c')
    ax.plot(loss_bio, label='Bio-Realistic', linewidth=2, color='#d62728')
    ax.set_xlabel('Training Step (×200)', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Learning Curves: Progression of Biological Realism', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

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
