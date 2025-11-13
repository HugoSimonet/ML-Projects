"""
Attention visualization utilities
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from PIL import Image


class AttentionVisualizer:
    """
    Visualizer for attention weights
    """

    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer: Text tokenizer for token labels
        """
        self.tokenizer = tokenizer

    def visualize_cross_attention(
        self,
        image: Image.Image,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        token_ids: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        layer_idx: int = -1,
        head_idx: int = 0
    ):
        """
        Visualize cross-modal attention

        Args:
            image: Input image
            attention_weights: Attention weights [layers, batch, heads, query_len, key_len]
            tokens: Token strings
            token_ids: Token IDs (if tokens not provided)
            save_path: Path to save visualization
            layer_idx: Layer to visualize (-1 for last)
            head_idx: Attention head to visualize
        """
        # Get attention for specific layer and head
        attn = attention_weights[layer_idx, 0, head_idx].cpu().numpy()

        # Decode tokens if needed
        if tokens is None and token_ids is not None and self.tokenizer is not None:
            tokens = []
            for token_id in token_ids[0]:
                token = self.tokenizer.decode([token_id.item()], skip_special_tokens=False)
                tokens.append(token)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot image
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Input Image')

        # Plot attention heatmap
        im = axes[1].imshow(attn, cmap='viridis', aspect='auto')
        axes[1].set_xlabel('Image Regions')
        axes[1].set_ylabel('Text Tokens')
        axes[1].set_title(f'Cross-Attention (Layer {layer_idx}, Head {head_idx})')

        if tokens:
            axes[1].set_yticks(range(len(tokens)))
            axes[1].set_yticklabels(tokens, fontsize=8)

        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

    def visualize_attention_map(
        self,
        image: Image.Image,
        attention_weights: torch.Tensor,
        token_idx: int,
        save_path: Optional[str] = None,
        resize_to_image: bool = True
    ):
        """
        Visualize attention map overlaid on image

        Args:
            image: Input image
            attention_weights: Attention weights [query_len, key_len]
            token_idx: Token index to visualize
            save_path: Path to save visualization
            resize_to_image: Resize attention to image size
        """
        # Get attention for specific token
        attn = attention_weights[token_idx].cpu().numpy()

        # Reshape to spatial grid (assuming 7x7 spatial features)
        grid_size = int(np.sqrt(len(attn)))
        attn_map = attn.reshape(grid_size, grid_size)

        # Resize to image size if needed
        if resize_to_image:
            from scipy.ndimage import zoom
            scale = np.array(image.size) / grid_size
            attn_map = zoom(attn_map, scale[::-1], order=1)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Original Image')

        # Attention heatmap
        axes[1].imshow(attn_map, cmap='hot')
        axes[1].axis('off')
        axes[1].set_title('Attention Heatmap')

        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(attn_map, cmap='hot', alpha=0.5)
        axes[2].axis('off')
        axes[2].set_title('Attention Overlay')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

    def visualize_multi_head_attention(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        num_heads: int = 12
    ):
        """
        Visualize all attention heads

        Args:
            attention_weights: Attention weights [batch, heads, query_len, key_len]
            tokens: Token strings
            save_path: Path to save visualization
            num_heads: Number of heads to visualize
        """
        attn = attention_weights[0].cpu().numpy()  # [heads, query_len, key_len]

        # Create subplots
        rows = int(np.ceil(num_heads / 4))
        cols = min(4, num_heads)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = axes.flatten() if num_heads > 1 else [axes]

        for head_idx in range(min(num_heads, len(attn))):
            im = axes[head_idx].imshow(attn[head_idx], cmap='viridis', aspect='auto')
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].set_xlabel('Key')
            axes[head_idx].set_ylabel('Query')

            if tokens and len(tokens) == attn[head_idx].shape[0]:
                axes[head_idx].set_yticks(range(len(tokens)))
                axes[head_idx].set_yticklabels(tokens, fontsize=6)

            plt.colorbar(im, ax=axes[head_idx])

        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot attention heatmap

    Args:
        attention_weights: Attention weights [query_len, key_len]
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    attn = attention_weights.cpu().numpy() if isinstance(attention_weights, torch.Tensor) else attention_weights

    sns.heatmap(
        attn,
        xticklabels=x_labels if x_labels else [],
        yticklabels=y_labels if y_labels else [],
        cmap='viridis',
        cbar=True,
        square=False
    )

    plt.title(title)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")

    plt.show()


def plot_cross_modal_attention(
    v2l_attention: torch.Tensor,
    l2v_attention: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Plot vision-to-language and language-to-vision attention

    Args:
        v2l_attention: Vision-to-language attention [query_len, key_len]
        l2v_attention: Language-to-vision attention [query_len, key_len]
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # V2L attention
    im1 = axes[0].imshow(v2l_attention.cpu().numpy(), cmap='viridis', aspect='auto')
    axes[0].set_title('Vision-to-Language Attention')
    axes[0].set_xlabel('Vision Features')
    axes[0].set_ylabel('Language Tokens')
    plt.colorbar(im1, ax=axes[0])

    # L2V attention
    im2 = axes[1].imshow(l2v_attention.cpu().numpy(), cmap='viridis', aspect='auto')
    axes[1].set_title('Language-to-Vision Attention')
    axes[1].set_xlabel('Language Tokens')
    axes[1].set_ylabel('Vision Features')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Cross-modal attention saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test visualization
    print("Testing attention visualization...")

    # Create dummy attention weights
    attention = torch.randn(8, 49)  # 8 tokens, 49 image regions
    attention = torch.softmax(attention, dim=-1)

    tokens = ['a', 'cat', 'sitting', 'on', 'a', 'chair', '.', '[SEP]']

    plot_attention_heatmap(
        attention,
        y_labels=tokens,
        title="Test Attention Heatmap"
    )

    print("Visualization test complete")
