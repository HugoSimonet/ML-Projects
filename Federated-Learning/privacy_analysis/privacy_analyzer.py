"""
Privacy Analysis Tools for Federated Learning
Analyzes privacy budget, utility-privacy tradeoffs, and privacy guarantees
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class PrivacyAnalyzer:
    """
    Privacy analyzer for federated learning

    Analyzes privacy budgets and privacy-utility tradeoffs
    """

    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize privacy analyzer

        Args:
            log_path: Path to training log JSON
        """
        self.log_data = None
        self.privacy_history = []

        if log_path:
            self.load_log(log_path)

    def load_log(self, log_path: str):
        """Load training log"""
        with open(log_path, 'r') as f:
            self.log_data = json.load(f)

        # Extract privacy history
        if 'round_history' in self.log_data:
            for round_data in self.log_data['round_history']:
                if 'privacy' in round_data:
                    self.privacy_history.append({
                        'round': round_data['round'],
                        'epsilon': round_data['privacy']['current_epsilon'],
                        'delta': round_data['privacy']['current_delta'],
                        'accuracy': round_data.get('global_test_accuracy', 0)
                    })

    def plot_privacy_budget(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot privacy budget over rounds"""
        if not self.privacy_history:
            print("No privacy data available")
            return

        rounds = [p['round'] for p in self.privacy_history]
        epsilons = [p['epsilon'] for p in self.privacy_history]

        plt.figure(figsize=(12, 6))
        plt.plot(rounds, epsilons, marker='o', linewidth=2, markersize=5)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Privacy Budget (ε)', fontsize=12)
        plt.title('Privacy Budget Consumption Over Training', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_utility_privacy_tradeoff(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot utility (accuracy) vs privacy budget"""
        if not self.privacy_history:
            print("No privacy data available")
            return

        epsilons = [p['epsilon'] for p in self.privacy_history]
        accuracies = [p['accuracy'] for p in self.privacy_history]

        plt.figure(figsize=(12, 6))
        plt.scatter(epsilons, accuracies, alpha=0.6, s=100, c=range(len(epsilons)),
                   cmap='viridis')
        plt.colorbar(label='Round')
        plt.xlabel('Privacy Budget (ε)', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.title('Privacy-Utility Tradeoff', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def analyze_privacy_cost(self) -> Dict[str, float]:
        """
        Analyze privacy cost

        Returns:
            Privacy cost metrics
        """
        if not self.privacy_history:
            return {}

        final_epsilon = self.privacy_history[-1]['epsilon']
        final_accuracy = self.privacy_history[-1]['accuracy']
        total_rounds = len(self.privacy_history)

        # Compute epsilon per round
        epsilon_per_round = final_epsilon / total_rounds

        # Compute accuracy improvement per epsilon
        if final_epsilon > 0:
            acc_per_epsilon = final_accuracy / final_epsilon
        else:
            acc_per_epsilon = 0

        return {
            'final_epsilon': final_epsilon,
            'final_delta': self.privacy_history[-1]['delta'],
            'total_rounds': total_rounds,
            'epsilon_per_round': epsilon_per_round,
            'final_accuracy': final_accuracy,
            'accuracy_per_epsilon': acc_per_epsilon
        }

    def generate_report(self, output_path: Optional[str] = None):
        """Generate comprehensive privacy report"""
        if not self.privacy_history:
            print("No privacy data available for report")
            return

        # Analyze privacy cost
        cost_metrics = self.analyze_privacy_cost()

        report = [
            "=" * 80,
            "PRIVACY ANALYSIS REPORT",
            "=" * 80,
            "",
            "Privacy Budget:",
            f"  Final Epsilon: {cost_metrics['final_epsilon']:.6f}",
            f"  Final Delta: {cost_metrics['final_delta']:.2e}",
            f"  Epsilon per Round: {cost_metrics['epsilon_per_round']:.6f}",
            "",
            "Utility Metrics:",
            f"  Final Accuracy: {cost_metrics['final_accuracy']:.2f}%",
            f"  Accuracy per Epsilon: {cost_metrics['accuracy_per_epsilon']:.2f}%",
            "",
            "Training Statistics:",
            f"  Total Rounds: {cost_metrics['total_rounds']}",
            "",
            "Privacy Guarantees:",
            f"  This model satisfies ({cost_metrics['final_epsilon']:.4f}, "
            f"{cost_metrics['final_delta']:.2e})-differential privacy.",
            "",
            "Interpretation:",
        ]

        # Add interpretation based on epsilon value
        epsilon = cost_metrics['final_epsilon']
        if epsilon < 1.0:
            report.append("  Strong privacy protection (ε < 1.0)")
        elif epsilon < 5.0:
            report.append("  Moderate privacy protection (1.0 ≤ ε < 5.0)")
        elif epsilon < 10.0:
            report.append("  Weak privacy protection (5.0 ≤ ε < 10.0)")
        else:
            report.append("  Minimal privacy protection (ε ≥ 10.0)")

        report.append("=" * 80)

        report_text = "\n".join(report)
        print(report_text)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to {output_path}")

    def plot_all(self, output_dir: str = './privacy_analysis'):
        """Generate all privacy plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Generating privacy analysis plots...")

        self.plot_privacy_budget(
            save_path=output_path / 'privacy_budget.png',
            show=False
        )
        print("✓ Privacy budget plot saved")

        self.plot_utility_privacy_tradeoff(
            save_path=output_path / 'utility_privacy_tradeoff.png',
            show=False
        )
        print("✓ Utility-privacy tradeoff plot saved")

        self.generate_report(output_path / 'privacy_report.txt')

        print(f"\nAll privacy analysis saved to {output_path}")


def analyze_privacy_budget(
    epsilon: float,
    delta: float,
    num_rounds: int,
    sample_rate: float
) -> Dict[str, float]:
    """
    Quick privacy budget analysis

    Args:
        epsilon: Privacy parameter
        delta: Privacy parameter
        num_rounds: Number of training rounds
        sample_rate: Client sampling rate per round

    Returns:
        Privacy analysis metrics
    """
    epsilon_per_round = epsilon / num_rounds
    total_privacy_loss = epsilon

    # Estimate composition
    # Using basic composition: ε_total = T * ε_per_round
    # For advanced composition, use RDP accounting

    analysis = {
        'epsilon': epsilon,
        'delta': delta,
        'num_rounds': num_rounds,
        'epsilon_per_round': epsilon_per_round,
        'sample_rate': sample_rate,
        'total_privacy_loss': total_privacy_loss
    }

    return analysis


if __name__ == "__main__":
    import sys

    print("Privacy Analyzer for Federated Learning")
    print("=" * 80)

    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        print(f"Loading log from: {log_path}\n")

        analyzer = PrivacyAnalyzer(log_path)
        analyzer.plot_all()
    else:
        print("No log path provided. Running example analysis...\n")

        # Example privacy analysis
        result = analyze_privacy_budget(
            epsilon=1.0,
            delta=1e-5,
            num_rounds=100,
            sample_rate=0.1
        )

        print("Privacy Budget Analysis:")
        for key, value in result.items():
            if isinstance(value, float):
                if value < 0.01:
                    print(f"  {key}: {value:.2e}")
                else:
                    print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
