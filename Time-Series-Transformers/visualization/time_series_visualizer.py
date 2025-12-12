"""
Time Series Visualization
Comprehensive visualization tools for time series forecasting
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class TimeSeriesVisualizer:
    """
    Visualization tools for time series forecasting
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (12, 6)):
        """
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use('default')  # Use default if seaborn style not available
        self.figsize = figsize
        sns.set_palette("husl")

    def plot_forecasts(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = 'Time Series Forecast',
        save_path: Optional[str] = None
    ):
        """
        Plot actual vs predicted values

        Args:
            actual: Actual values [N]
            predicted: Predicted values [N]
            timestamps: Time indices
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if timestamps is None:
            timestamps = np.arange(len(actual))

        ax.plot(timestamps, actual, label='Actual', color='blue', linewidth=2)
        ax.plot(timestamps, predicted, label='Predicted', color='red', linewidth=2, linestyle='--')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_probabilistic_forecast(
        self,
        actual: np.ndarray,
        mean_pred: np.ndarray,
        quantiles: Dict[float, np.ndarray],
        timestamps: Optional[np.ndarray] = None,
        title: str = 'Probabilistic Forecast',
        save_path: Optional[str] = None
    ):
        """
        Plot probabilistic forecast with prediction intervals

        Args:
            actual: Actual values [N]
            mean_pred: Mean predictions [N]
            quantiles: Dictionary of quantile predictions {quantile: values}
            timestamps: Time indices
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if timestamps is None:
            timestamps = np.arange(len(actual))

        # Plot actual
        ax.plot(timestamps, actual, label='Actual', color='black', linewidth=2)

        # Plot mean prediction
        ax.plot(timestamps, mean_pred, label='Mean Prediction', color='red', linewidth=2)

        # Plot prediction intervals
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        alphas = [0.3, 0.4, 0.5]

        # Sort quantile pairs
        quantile_pairs = []
        sorted_q = sorted(quantiles.keys())

        for i, q in enumerate(sorted_q):
            if q < 0.5:
                complement = 1 - q
                if complement in sorted_q:
                    quantile_pairs.append((q, complement))

        # Plot intervals
        for i, (lower_q, upper_q) in enumerate(quantile_pairs):
            color_idx = i % len(colors)
            ax.fill_between(
                timestamps,
                quantiles[lower_q],
                quantiles[upper_q],
                alpha=alphas[color_idx],
                color=colors[color_idx],
                label=f'{int((upper_q - lower_q) * 100)}% Interval'
            )

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_multivariate_forecast(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        variable_names: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
        title: str = 'Multivariate Forecast',
        save_path: Optional[str] = None
    ):
        """
        Plot multivariate time series forecast

        Args:
            actual: Actual values [N, num_variables]
            predicted: Predicted values [N, num_variables]
            variable_names: Names of variables
            timestamps: Time indices
            title: Plot title
            save_path: Path to save figure
        """
        num_vars = actual.shape[1]

        if variable_names is None:
            variable_names = [f'Variable {i+1}' for i in range(num_vars)]

        if timestamps is None:
            timestamps = np.arange(len(actual))

        # Create subplots
        fig, axes = plt.subplots(num_vars, 1, figsize=(self.figsize[0], self.figsize[1] * num_vars / 2))

        if num_vars == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(timestamps, actual[:, i], label='Actual', color='blue', linewidth=2)
            ax.plot(timestamps, predicted[:, i], label='Predicted', color='red', linewidth=2, linestyle='--')

            ax.set_ylabel(variable_names[i], fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time', fontsize=12)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_residuals(
        self,
        residuals: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = 'Forecast Residuals',
        save_path: Optional[str] = None
    ):
        """
        Plot residual analysis

        Args:
            residuals: Forecast residuals [N]
            timestamps: Time indices
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if timestamps is None:
            timestamps = np.arange(len(residuals))

        # Time series plot
        axes[0, 0].plot(timestamps, residuals, color='purple', linewidth=1)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram
        axes[0, 1].hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residual Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # Autocorrelation
        from matplotlib.pyplot import acorr
        axes[1, 1].acorr(residuals - np.mean(residuals), maxlags=50, usevlines=True, normed=True)
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].set_title('Residual Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_anomalies(
        self,
        data: np.ndarray,
        anomalies: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = 'Anomaly Detection',
        save_path: Optional[str] = None
    ):
        """
        Plot time series with detected anomalies

        Args:
            data: Time series data [N]
            anomalies: Binary anomaly labels [N]
            timestamps: Time indices
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if timestamps is None:
            timestamps = np.arange(len(data))

        # Plot normal data
        normal_mask = anomalies == 0
        ax.plot(timestamps[normal_mask], data[normal_mask], 'b-', label='Normal', linewidth=2)

        # Plot anomalies
        anomaly_mask = anomalies == 1
        if np.any(anomaly_mask):
            ax.scatter(
                timestamps[anomaly_mask],
                data[anomaly_mask],
                color='red',
                s=100,
                marker='o',
                label='Anomaly',
                zorder=5
            )

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        title: str = 'Attention Weights',
        save_path: Optional[str] = None
    ):
        """
        Plot attention weight heatmap

        Args:
            attention_weights: Attention weights [seq_len, seq_len] or [heads, seq_len, seq_len]
            row_labels: Labels for rows
            col_labels: Labels for columns
            title: Plot title
            save_path: Path to save figure
        """
        # Average over heads if 3D
        if attention_weights.ndim == 3:
            attention_weights = attention_weights.mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            attention_weights,
            cmap='YlOrRd',
            xticklabels=col_labels if col_labels else False,
            yticklabels=row_labels if row_labels else False,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = 'Training History',
        save_path: Optional[str] = None
    ):
        """
        Plot training and validation loss

        Args:
            history: Dictionary with 'train_loss' and 'val_loss'
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = np.arange(1, len(history['train_loss']) + 1)

        # Loss plot
        axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            axes[0].plot(epochs, history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Learning rate plot
        if 'learning_rate' in history:
            axes[1].plot(epochs, history['learning_rate'], color='green', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Learning Rate', fontsize=12)
            axes[1].set_title('Learning Rate Schedule', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_metric_comparison(
        self,
        metrics: Dict[str, float],
        title: str = 'Model Performance Metrics',
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart of metrics

        Args:
            metrics: Dictionary of metric names and values
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        bars = ax.bar(metric_names, metric_values, color='skyblue', edgecolor='black', linewidth=1.5)

        # Color bars based on values (lower is better for most metrics)
        for i, bar in enumerate(bars):
            if metric_values[i] > np.median(metric_values):
                bar.set_color('lightcoral')
            else:
                bar.set_color('lightgreen')

        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(metric_values):
            ax.text(i, v + 0.01 * max(metric_values), f'{v:.4f}', ha='center', fontsize=9)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_interactive_forecast(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        quantiles: Optional[Dict[float, np.ndarray]] = None,
        title: str = 'Interactive Forecast'
    ) -> go.Figure:
        """
        Create interactive forecast plot using Plotly

        Args:
            actual: Actual values [N]
            predicted: Predicted values [N]
            timestamps: Time indices
            quantiles: Optional quantile predictions
            title: Plot title

        Returns:
            Plotly figure object
        """
        if timestamps is None:
            timestamps = np.arange(len(actual))

        fig = go.Figure()

        # Add actual values
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))

        # Add predictions
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Add prediction intervals if provided
        if quantiles:
            sorted_q = sorted(quantiles.keys())
            for i, q in enumerate(sorted_q):
                if q < 0.5:
                    complement = 1 - q
                    if complement in sorted_q:
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([timestamps, timestamps[::-1]]),
                            y=np.concatenate([quantiles[q], quantiles[complement][::-1]]),
                            fill='toself',
                            fillcolor=f'rgba(0, 100, 200, {0.2 + i * 0.1})',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{int((complement - q) * 100)}% Interval',
                            showlegend=True
                        ))

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def plot_causal_graph(
        self,
        adjacency_matrix: np.ndarray,
        variable_names: Optional[List[str]] = None,
        title: str = 'Causal Graph',
        save_path: Optional[str] = None
    ):
        """
        Plot causal graph from adjacency matrix

        Args:
            adjacency_matrix: Causal adjacency matrix [num_vars, num_vars]
            variable_names: Names of variables
            title: Plot title
            save_path: Path to save figure
        """
        import networkx as nx

        num_vars = adjacency_matrix.shape[0]

        if variable_names is None:
            variable_names = [f'Var_{i}' for i in range(num_vars)]

        # Create directed graph
        G = nx.DiGraph()
        G.add_nodes_from(variable_names)

        # Add edges
        for i in range(num_vars):
            for j in range(num_vars):
                if adjacency_matrix[i, j] > 0:
                    G.add_edge(variable_names[i], variable_names[j],
                              weight=adjacency_matrix[i, j])

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        pos = nx.spring_layout(G, k=2, iterations=50)

        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=3000, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                              arrows=True, arrowsize=20, ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
