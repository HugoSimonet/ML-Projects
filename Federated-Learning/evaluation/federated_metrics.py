"""
Federated Learning Evaluation Metrics
Comprehensive metrics for evaluating FL systems
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class FederatedMetrics:
    """
    Comprehensive metrics tracker for federated learning

    Tracks convergence, communication cost, fairness, and privacy metrics
    """

    def __init__(self):
        """Initialize metrics tracker"""
        self.round_metrics = []
        self.client_metrics = defaultdict(list)
        self.communication_costs = []
        self.privacy_budgets = []

    def update(
        self,
        round_num: int,
        global_metrics: Dict[str, float],
        client_metrics: Optional[List[Dict[str, float]]] = None,
        communication_cost: Optional[float] = None,
        privacy_budget: Optional[Tuple[float, float]] = None
    ):
        """
        Update metrics for current round

        Args:
            round_num: Current round number
            global_metrics: Global model metrics
            client_metrics: List of client-specific metrics
            communication_cost: Communication cost in bytes
            privacy_budget: (epsilon, delta) privacy budget
        """
        # Store global metrics
        metrics_dict = {'round': round_num, **global_metrics}
        self.round_metrics.append(metrics_dict)

        # Store client metrics
        if client_metrics:
            for client_id, metrics in enumerate(client_metrics):
                self.client_metrics[client_id].append({
                    'round': round_num,
                    **metrics
                })

        # Store communication cost
        if communication_cost is not None:
            self.communication_costs.append({
                'round': round_num,
                'cost': communication_cost
            })

        # Store privacy budget
        if privacy_budget is not None:
            self.privacy_budgets.append({
                'round': round_num,
                'epsilon': privacy_budget[0],
                'delta': privacy_budget[1]
            })

    def get_convergence_metrics(self) -> Dict[str, float]:
        """
        Compute convergence metrics

        Returns:
            Dictionary of convergence metrics
        """
        if not self.round_metrics:
            return {}

        # Extract accuracies
        accuracies = [m.get('test_accuracy', 0) for m in self.round_metrics]

        # Find best accuracy
        best_accuracy = max(accuracies) if accuracies else 0
        best_round = accuracies.index(best_accuracy) if accuracies else 0

        # Compute convergence speed (rounds to reach 90% of best)
        target_acc = 0.9 * best_accuracy
        convergence_round = next(
            (i for i, acc in enumerate(accuracies) if acc >= target_acc),
            len(accuracies)
        )

        # Compute stability (variance in last 10 rounds)
        recent_acc = accuracies[-10:] if len(accuracies) >= 10 else accuracies
        stability = float(np.std(recent_acc)) if recent_acc else 0

        return {
            'best_accuracy': best_accuracy,
            'best_round': best_round,
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'convergence_round': convergence_round,
            'stability_std': stability,
            'total_rounds': len(accuracies)
        }

    def get_communication_metrics(self) -> Dict[str, float]:
        """
        Compute communication cost metrics

        Returns:
            Dictionary of communication metrics
        """
        if not self.communication_costs:
            return {}

        total_cost = sum(c['cost'] for c in self.communication_costs)
        avg_cost = total_cost / len(self.communication_costs)
        max_cost = max(c['cost'] for c in self.communication_costs)

        return {
            'total_communication_bytes': total_cost,
            'avg_communication_bytes': avg_cost,
            'max_communication_bytes': max_cost,
            'total_communication_mb': total_cost / (1024 * 1024),
            'num_rounds': len(self.communication_costs)
        }

    def get_fairness_metrics(self) -> Dict[str, float]:
        """
        Compute fairness metrics across clients

        Returns:
            Dictionary of fairness metrics
        """
        if not self.client_metrics:
            return {}

        # Get final round metrics for each client
        final_accuracies = []
        for client_id, metrics_list in self.client_metrics.items():
            if metrics_list:
                final_accuracies.append(metrics_list[-1].get('accuracy', 0))

        if not final_accuracies:
            return {}

        # Compute fairness metrics
        mean_acc = np.mean(final_accuracies)
        std_acc = np.std(final_accuracies)
        min_acc = np.min(final_accuracies)
        max_acc = np.max(final_accuracies)

        # Coefficient of variation (lower is fairer)
        cv = std_acc / mean_acc if mean_acc > 0 else float('inf')

        # Jain's fairness index (0 to 1, 1 is perfectly fair)
        jain_index = (np.sum(final_accuracies) ** 2) / (
            len(final_accuracies) * np.sum(np.array(final_accuracies) ** 2)
        )

        return {
            'mean_client_accuracy': float(mean_acc),
            'std_client_accuracy': float(std_acc),
            'min_client_accuracy': float(min_acc),
            'max_client_accuracy': float(max_acc),
            'accuracy_range': float(max_acc - min_acc),
            'coefficient_of_variation': float(cv),
            'jain_fairness_index': float(jain_index)
        }

    def get_privacy_metrics(self) -> Dict[str, float]:
        """
        Compute privacy metrics

        Returns:
            Dictionary of privacy metrics
        """
        if not self.privacy_budgets:
            return {}

        final_budget = self.privacy_budgets[-1]

        return {
            'final_epsilon': final_budget['epsilon'],
            'final_delta': final_budget['delta'],
            'total_rounds': len(self.privacy_budgets),
            'epsilon_per_round': final_budget['epsilon'] / len(self.privacy_budgets)
        }

    def get_summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of all metrics

        Returns:
            Dictionary with all metric categories
        """
        return {
            'convergence': self.get_convergence_metrics(),
            'communication': self.get_communication_metrics(),
            'fairness': self.get_fairness_metrics(),
            'privacy': self.get_privacy_metrics()
        }

    def print_summary(self):
        """Print formatted summary of metrics"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("FEDERATED LEARNING EVALUATION SUMMARY")
        print("=" * 60)

        if summary['convergence']:
            print("\nConvergence Metrics:")
            for key, value in summary['convergence'].items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        if summary['communication']:
            print("\nCommunication Metrics:")
            for key, value in summary['communication'].items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        if summary['fairness']:
            print("\nFairness Metrics:")
            for key, value in summary['fairness'].items():
                print(f"  {key}: {value:.4f}")

        if summary['privacy']:
            print("\nPrivacy Metrics:")
            for key, value in summary['privacy'].items():
                print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

        print("=" * 60)


def compute_communication_cost(
    model_state: Dict[str, torch.Tensor],
    num_clients: int = 1,
    compression_ratio: float = 1.0
) -> float:
    """
    Compute communication cost in bytes

    Args:
        model_state: Model state dictionary
        num_clients: Number of clients (multiplier)
        compression_ratio: Compression ratio (0 to 1)

    Returns:
        Total communication cost in bytes
    """
    total_bytes = 0

    for param in model_state.values():
        # Each parameter in bytes (assuming float32 = 4 bytes)
        param_bytes = param.numel() * 4
        total_bytes += param_bytes

    # Account for bidirectional communication (download + upload)
    total_bytes *= 2

    # Account for multiple clients
    total_bytes *= num_clients

    # Apply compression
    total_bytes *= compression_ratio

    return total_bytes


def compute_convergence_metrics(
    accuracy_history: List[float],
    target_accuracy: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute convergence metrics from accuracy history

    Args:
        accuracy_history: List of accuracies per round
        target_accuracy: Target accuracy (if None, uses 95% of best)

    Returns:
        Convergence metrics
    """
    if not accuracy_history:
        return {}

    best_accuracy = max(accuracy_history)

    if target_accuracy is None:
        target_accuracy = 0.95 * best_accuracy

    # Find convergence round
    convergence_round = next(
        (i for i, acc in enumerate(accuracy_history) if acc >= target_accuracy),
        len(accuracy_history)
    )

    # Compute rate of improvement
    if len(accuracy_history) > 1:
        improvements = [
            accuracy_history[i] - accuracy_history[i-1]
            for i in range(1, len(accuracy_history))
        ]
        avg_improvement = np.mean(improvements)
    else:
        avg_improvement = 0

    return {
        'best_accuracy': best_accuracy,
        'convergence_round': convergence_round,
        'average_improvement_per_round': avg_improvement,
        'total_improvement': accuracy_history[-1] - accuracy_history[0]
        if len(accuracy_history) > 1 else 0
    }


def compute_fairness_metrics(
    client_accuracies: List[float]
) -> Dict[str, float]:
    """
    Compute fairness metrics across clients

    Args:
        client_accuracies: List of client accuracies

    Returns:
        Fairness metrics
    """
    if not client_accuracies:
        return {}

    accuracies = np.array(client_accuracies)

    # Basic statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)

    # Fairness indices
    # Jain's fairness index
    jain_index = (np.sum(accuracies) ** 2) / (
        len(accuracies) * np.sum(accuracies ** 2)
    )

    # Coefficient of variation
    cv = std_acc / mean_acc if mean_acc > 0 else float('inf')

    # Gini coefficient
    sorted_acc = np.sort(accuracies)
    n = len(sorted_acc)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_acc)) / (n * np.sum(sorted_acc)) - (n + 1) / n

    return {
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'min_accuracy': float(min_acc),
        'max_accuracy': float(max_acc),
        'jain_fairness_index': float(jain_index),
        'coefficient_of_variation': float(cv),
        'gini_coefficient': float(gini)
    }


def compute_model_divergence(
    client_models: List[Dict[str, torch.Tensor]],
    global_model: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute divergence between client and global models

    Args:
        client_models: List of client model states
        global_model: Global model state

    Returns:
        Divergence metrics
    """
    divergences = []

    for client_model in client_models:
        divergence = 0.0
        for key in global_model.keys():
            if key in client_model:
                diff = client_model[key] - global_model[key]
                divergence += torch.sum(diff ** 2).item()
        divergences.append(divergence ** 0.5)  # L2 norm

    return {
        'mean_divergence': float(np.mean(divergences)),
        'std_divergence': float(np.std(divergences)),
        'max_divergence': float(np.max(divergences)),
        'min_divergence': float(np.min(divergences))
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create metrics tracker
    metrics = FederatedMetrics()

    # Simulate 10 rounds of training
    for round_num in range(10):
        # Global metrics
        global_metrics = {
            'test_accuracy': 70 + round_num * 2 + np.random.randn(),
            'test_loss': 1.0 - round_num * 0.08
        }

        # Client metrics
        client_metrics = [
            {
                'accuracy': 65 + round_num * 2 + np.random.randn() * 5,
                'loss': 1.1 - round_num * 0.08,
                'num_samples': 100
            }
            for _ in range(5)
        ]

        # Communication cost (example: 10MB per round)
        comm_cost = 10 * 1024 * 1024

        # Privacy budget (example)
        privacy_budget = (0.1 * (round_num + 1), 1e-5)

        # Update metrics
        metrics.update(
            round_num=round_num,
            global_metrics=global_metrics,
            client_metrics=client_metrics,
            communication_cost=comm_cost,
            privacy_budget=privacy_budget
        )

    # Print summary
    metrics.print_summary()

    # Test individual metric functions
    print("\n" + "=" * 60)
    print("Testing Individual Metric Functions")
    print("=" * 60)

    accuracy_history = [60, 65, 72, 78, 82, 85, 87, 88, 89, 89.5]
    conv_metrics = compute_convergence_metrics(accuracy_history, target_accuracy=85)
    print("\nConvergence Metrics:")
    for key, value in conv_metrics.items():
        print(f"  {key}: {value:.2f}")

    client_accs = [85, 88, 82, 90, 86, 84, 87, 89, 83, 86]
    fair_metrics = compute_fairness_metrics(client_accs)
    print("\nFairness Metrics:")
    for key, value in fair_metrics.items():
        print(f"  {key}: {value:.4f}")
