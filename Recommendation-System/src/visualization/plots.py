"""Visualization utilities for recommendation systems."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger


class Plotter:
    """Visualization tools for recommendation system evaluation."""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize plotter.

        Args:
            style: Matplotlib style to use.
        """
        try:
            plt.style.use(style)
        except:
            pass
        sns.set_palette("husl")
        self.figures = []
        logger.info("Initialized Plotter")

    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        save_path: str = None
    ) -> None:
        """
        Plot comparison of metrics across models.

        Args:
            results: Dictionary of {model_name: {metric_name: value}}.
            metrics: List of metrics to plot.
            save_path: Path to save the figure.
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))

        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            models = []
            values = []

            for model_name, model_results in results.items():
                if metric in model_results:
                    models.append(model_name)
                    values.append(model_results[metric])

            ax.bar(models, values)
            ax.set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        self.figures.append(fig)
        plt.show()

    def plot_ranking_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        metric_name: str = "precision",
        save_path: str = None
    ) -> None:
        """
        Plot ranking metrics across different K values.

        Args:
            results: Dictionary of {model_name: {metric@k: value}}.
            k_values: List of K values.
            metric_name: Name of the metric ('precision', 'recall', 'ndcg').
            save_path: Path to save the figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for model_name, model_results in results.items():
            metric_values = []
            for k in k_values:
                metric_key = f"{metric_name}@{k}"
                if metric_key in model_results:
                    metric_values.append(model_results[metric_key])
                else:
                    metric_values.append(0)

            ax.plot(k_values, metric_values, marker='o', label=model_name, linewidth=2)

        ax.set_xlabel('K', fontsize=12)
        ax.set_ylabel(f'{metric_name.capitalize()}@K', fontsize=12)
        ax.set_title(f'{metric_name.capitalize()}@K Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        self.figures.append(fig)
        plt.show()

    def plot_all_ranking_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        save_path: str = None
    ) -> None:
        """
        Plot all ranking metrics in subplots.

        Args:
            results: Dictionary of {model_name: {metric@k: value}}.
            k_values: List of K values.
            save_path: Path to save the figure.
        """
        metrics = ['precision', 'recall', 'ndcg', 'map']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for ax, metric in zip(axes, metrics):
            for model_name, model_results in results.items():
                metric_values = []
                for k in k_values:
                    metric_key = f"{metric}@{k}"
                    if metric_key in model_results:
                        metric_values.append(model_results[metric_key])
                    else:
                        metric_values.append(0)

                ax.plot(k_values, metric_values, marker='o', label=model_name, linewidth=2)

            ax.set_xlabel('K', fontsize=11)
            ax.set_ylabel(f'{metric.capitalize()}@K', fontsize=11)
            ax.set_title(f'{metric.capitalize()}@K', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        self.figures.append(fig)
        plt.show()

    def plot_diversity_coverage(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: str = None
    ) -> None:
        """
        Plot diversity and coverage metrics.

        Args:
            results: Dictionary of {model_name: {'diversity': value, 'coverage': value}}.
            save_path: Path to save the figure.
        """
        models = list(results.keys())
        diversity = [results[m].get('diversity', 0) for m in models]
        coverage = [results[m].get('coverage', 0) for m in models]
        novelty = [results[m].get('novelty', 0) for m in models]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Diversity
        axes[0].bar(models, diversity, color='skyblue')
        axes[0].set_title('Diversity', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)

        # Coverage
        axes[1].bar(models, coverage, color='lightcoral')
        axes[1].set_title('Coverage', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        # Novelty
        axes[2].bar(models, novelty, color='lightgreen')
        axes[2].set_title('Novelty', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Score', fontsize=12)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        self.figures.append(fig)
        plt.show()

    def plot_dataset_statistics(
        self,
        data: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Plot dataset statistics.

        Args:
            data: DataFrame with columns [user_id, item_id, rating].
            save_path: Path to save the figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Rating distribution
        axes[0, 0].hist(data['rating'], bins=20, edgecolor='black')
        axes[0, 0].set_title('Rating Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Rating', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # User rating count distribution
        user_counts = data.groupby('user_id').size()
        axes[0, 1].hist(user_counts, bins=50, edgecolor='black')
        axes[0, 1].set_title('User Activity Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Number of Ratings per User', fontsize=11)
        axes[0, 1].set_ylabel('Number of Users', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

        # Item rating count distribution
        item_counts = data.groupby('item_id').size()
        axes[1, 0].hist(item_counts, bins=50, edgecolor='black')
        axes[1, 0].set_title('Item Popularity Distribution', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Ratings per Item', fontsize=11)
        axes[1, 0].set_ylabel('Number of Items', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

        # Sparsity
        n_users = data['user_id'].nunique()
        n_items = data['item_id'].nunique()
        n_ratings = len(data)
        sparsity = 1 - (n_ratings / (n_users * n_items))

        stats_text = f"""
Dataset Statistics:
------------------
Users: {n_users:,}
Items: {n_items:,}
Ratings: {n_ratings:,}
Sparsity: {sparsity:.2%}

Avg ratings/user: {n_ratings/n_users:.1f}
Avg ratings/item: {n_ratings/n_items:.1f}

Rating range: {data['rating'].min():.1f} - {data['rating'].max():.1f}
Rating mean: {data['rating'].mean():.2f}
Rating std: {data['rating'].std():.2f}
        """

        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        self.figures.append(fig)
        plt.show()

    def plot_model_comparison_heatmap(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        save_path: str = None
    ) -> None:
        """
        Plot heatmap comparing models across metrics.

        Args:
            results: Dictionary of {model_name: {metric_name: value}}.
            metrics: List of metrics to include.
            save_path: Path to save the figure.
        """
        # Create matrix
        models = list(results.keys())
        matrix = []

        for model in models:
            row = []
            for metric in metrics:
                if metric in results[model]:
                    row.append(results[model][metric])
                else:
                    row.append(0)
            matrix.append(row)

        # Create DataFrame
        df = pd.DataFrame(matrix, index=models, columns=metrics)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': 'Score'})
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Models', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        self.figures.append(fig)
        plt.show()

    def close_all(self):
        """Close all figures."""
        for fig in self.figures:
            plt.close(fig)
        self.figures = []
