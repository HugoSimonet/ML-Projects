"""
Visualization tools for federated learning
Plots training curves, client distributions, and privacy budgets
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import json
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class FLVisualizer:
    """
    Federated Learning Visualizer

    Comprehensive visualization of FL training progress
    """

    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize visualizer

        Args:
            log_path: Path to training log JSON file
        """
        self.log_data = None

        if log_path:
            self.load_log(log_path)

    def load_log(self, log_path: str):
        """Load training log from JSON"""
        with open(log_path, 'r') as f:
            self.log_data = json.load(f)

    def plot_accuracy_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot accuracy curves over training rounds"""
        if not self.log_data:
            raise ValueError("No log data loaded")

        round_history = self.log_data['round_history']

        rounds = [r['round'] for r in round_history]
        train_acc = [r['avg_train_accuracy'] for r in round_history]
        test_acc = [r['global_test_accuracy'] for r in round_history]

        plt.figure(figsize=(12, 6))
        plt.plot(rounds, train_acc, label='Train Accuracy', marker='o',
                linewidth=2, markersize=4)
        plt.plot(rounds, test_acc, label='Test Accuracy', marker='s',
                linewidth=2, markersize=4)

        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Federated Learning: Accuracy over Rounds', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_loss_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot loss curves over training rounds"""
        if not self.log_data:
            raise ValueError("No log data loaded")

        round_history = self.log_data['round_history']

        rounds = [r['round'] for r in round_history]
        train_loss = [r['avg_train_loss'] for r in round_history]
        test_loss = [r['global_test_loss'] for r in round_history]

        plt.figure(figsize=(12, 6))
        plt.plot(rounds, train_loss, label='Train Loss', marker='o',
                linewidth=2, markersize=4)
        plt.plot(rounds, test_loss, label='Test Loss', marker='s',
                linewidth=2, markersize=4)

        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Federated Learning: Loss over Rounds', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_client_participation(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot client participation distribution"""
        if not self.log_data:
            raise ValueError("No log data loaded")

        participation = self.log_data['summary']['client_participation']
        clients = list(participation.keys())
        counts = [participation[c] for c in clients]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(clients)), counts, color='steelblue', alpha=0.7)
        plt.xlabel('Client ID', fontsize=12)
        plt.ylabel('Participation Count', fontsize=12)
        plt.title('Client Participation Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_privacy_budget(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot privacy budget consumption"""
        if not self.log_data:
            raise ValueError("No log data loaded")

        round_history = self.log_data['round_history']

        # Check if privacy data exists
        if 'privacy' not in round_history[0]:
            print("No privacy data in logs")
            return

        rounds = [r['round'] for r in round_history]
        epsilons = [r['privacy']['current_epsilon'] for r in round_history]

        plt.figure(figsize=(12, 6))
        plt.plot(rounds, epsilons, label='Privacy Budget (ε)', marker='o',
                linewidth=2, markersize=4, color='red')

        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Privacy Budget (ε)', fontsize=12)
        plt.title('Privacy Budget Consumption', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_all(self, output_dir: str = './plots'):
        """Generate and save all plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Generating all visualizations...")

        self.plot_accuracy_curves(
            save_path=output_path / 'accuracy_curves.png',
            show=False
        )
        print("✓ Accuracy curves saved")

        self.plot_loss_curves(
            save_path=output_path / 'loss_curves.png',
            show=False
        )
        print("✓ Loss curves saved")

        self.plot_client_participation(
            save_path=output_path / 'client_participation.png',
            show=False
        )
        print("✓ Client participation saved")

        try:
            self.plot_privacy_budget(
                save_path=output_path / 'privacy_budget.png',
                show=False
            )
            print("✓ Privacy budget saved")
        except:
            print("✗ No privacy data available")

        print(f"\nAll plots saved to {output_path}")


def plot_training_curves(
    rounds: List[int],
    train_acc: List[float],
    test_acc: List[float],
    save_path: Optional[str] = None
):
    """
    Quick function to plot training curves

    Args:
        rounds: List of round numbers
        train_acc: List of training accuracies
        test_acc: List of test accuracies
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, train_acc, label='Train Accuracy', marker='o')
    plt.plot(rounds, test_acc, label='Test Accuracy', marker='s')

    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_client_distribution(
    client_data_counts: Dict[int, int],
    save_path: Optional[str] = None
):
    """
    Plot data distribution across clients

    Args:
        client_data_counts: Dictionary mapping client ID to data count
        save_path: Path to save plot
    """
    clients = list(client_data_counts.keys())
    counts = [client_data_counts[c] for c in clients]

    plt.figure(figsize=(12, 6))
    plt.bar(clients, counts, color='steelblue', alpha=0.7)
    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.title('Data Distribution Across Clients')
    plt.grid(True, axis='y', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Federated Learning Visualizer")
    print("Usage: python -m visualization.fl_visualizer [log_path]")

    import sys

    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        print(f"Loading log from: {log_path}")

        visualizer = FLVisualizer(log_path)
        visualizer.plot_all(output_dir='./plots')
    else:
        print("No log path provided. Creating example plots...")

        # Example data
        rounds = list(range(50))
        train_acc = [60 + i * 0.5 + np.random.randn() * 2 for i in rounds]
        test_acc = [58 + i * 0.48 + np.random.randn() * 2 for i in rounds]

        plot_training_curves(rounds, train_acc, test_acc)
