"""
Attention Visualizer

Provides visualization tools for attention weights in GNN models.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Tuple


class AttentionVisualizer:
    """Visualizer for attention weights."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 100
    ):
        """
        Initialize attention visualizer.

        Args:
            figsize: Figure size
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style('white')

    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        node_labels: Optional[torch.Tensor] = None,
        title: str = 'Attention Weights',
        save_path: Optional[str] = None
    ):
        """
        Plot attention weights as heatmap.

        Args:
            attention_weights: Attention weights matrix [num_nodes, num_nodes]
            node_labels: Node labels for annotation
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Convert to numpy
        attention_np = attention_weights.cpu().detach().numpy()

        # Plot heatmap
        sns.heatmap(
            attention_np,
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'},
            square=True
        )

        ax.set_xlabel('Target Nodes', fontsize=12)
        ax.set_ylabel('Source Nodes', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_attention_distribution(
        self,
        attention_weights: torch.Tensor,
        num_heads: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of attention weights.

        Args:
            attention_weights: Attention weights
            num_heads: Number of attention heads
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Flatten attention weights
        attention_flat = attention_weights.cpu().detach().numpy().flatten()

        # Plot histogram
        ax.hist(attention_flat, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Attention Weight', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Attention Weight Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_attn = np.mean(attention_flat)
        std_attn = np.std(attention_flat)
        ax.axvline(mean_attn, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_attn:.4f}')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_multihead_attention(
        self,
        attention_weights: torch.Tensor,
        head_labels: Optional[list] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot attention weights for multiple heads.

        Args:
            attention_weights: Multi-head attention weights [num_heads, num_nodes, num_nodes]
            head_labels: Labels for attention heads
            save_path: Path to save figure
        """
        num_heads = attention_weights.size(0)
        num_cols = min(4, num_heads)
        num_rows = (num_heads + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

        if num_heads == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)

        for head_idx in range(num_heads):
            row = head_idx // num_cols
            col = head_idx % num_cols

            attention_np = attention_weights[head_idx].cpu().detach().numpy()

            sns.heatmap(
                attention_np,
                cmap='YlOrRd',
                ax=axes[row, col],
                cbar=True,
                square=True,
                cbar_kws={'shrink': 0.8}
            )

            head_label = head_labels[head_idx] if head_labels else f'Head {head_idx + 1}'
            axes[row, col].set_title(head_label, fontsize=10)
            axes[row, col].set_xlabel('')
            axes[row, col].set_ylabel('')

        # Hide unused subplots
        for idx in range(num_heads, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].axis('off')

        plt.suptitle('Multi-Head Attention Weights', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()


def plot_attention_weights(
    attention_weights: torch.Tensor,
    **kwargs
):
    """
    Convenience function for plotting attention weights.

    Args:
        attention_weights: Attention weights
        **kwargs: Additional arguments
    """
    visualizer = AttentionVisualizer()
    visualizer.plot_attention_heatmap(attention_weights, **kwargs)
