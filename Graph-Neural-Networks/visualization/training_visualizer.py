"""
Training Visualizer

Provides visualization tools for training metrics and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import numpy as np


class TrainingVisualizer:
    """Visualizer for training metrics and curves."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 100
    ):
        """
        Initialize training visualizer.

        Args:
            figsize: Figure size
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style('whitegrid')

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        train_metrics: Optional[Dict[str, List[float]]] = None,
        val_metrics: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot training and validation curves.

        Args:
            train_losses: Training losses
            val_losses: Validation losses
            train_metrics: Training metrics
            val_metrics: Validation metrics
            save_path: Path to save figure
        """
        num_plots = 1 + (1 if train_metrics else 0)
        fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 6, 5))

        if num_plots == 1:
            axes = [axes]

        # Plot losses
        epochs = list(range(1, len(train_losses) + 1))

        axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2)
        if val_losses:
            axes[0].plot(epochs, val_losses, label='Val Loss', linewidth=2)

        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot metrics
        if train_metrics and num_plots > 1:
            for metric_name, values in train_metrics.items():
                axes[1].plot(epochs, values, label=f'Train {metric_name}', linewidth=2)

            if val_metrics:
                for metric_name, values in val_metrics.items():
                    axes[1].plot(epochs, values, label=f'Val {metric_name}',
                               linewidth=2, linestyle='--')

            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Metric Value', fontsize=12)
            axes[1].set_title('Training Metrics', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_metric_comparison(
        self,
        metrics: Dict[str, float],
        title: str = 'Model Performance',
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart comparing different metrics.

        Args:
            metrics: Dictionary of metric names to values
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        colors = sns.color_palette('husl', len(metric_names))
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'accuracy',
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of multiple models.

        Args:
            results: Dictionary of model name to metrics
            metric: Metric to compare
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        model_names = list(results.keys())
        metric_values = [results[name].get(metric, 0) for name in model_names]

        colors = sns.color_palette('Set2', len(model_names))
        bars = ax.bar(model_names, metric_values, color=colors, alpha=0.7)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=11
            )

        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'Model Comparison - {metric.capitalize()}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    **kwargs
):
    """
    Convenience function for plotting training curves.

    Args:
        train_losses: Training losses
        val_losses: Validation losses
        **kwargs: Additional arguments
    """
    visualizer = TrainingVisualizer()
    visualizer.plot_training_curves(train_losses, val_losses, **kwargs)
