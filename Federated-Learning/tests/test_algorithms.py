"""
Unit tests for federated learning algorithms
Tests FedAvg, FedProx, and FedNova aggregation
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import FedAvgAggregator, FedProxAggregator, FedNovaAggregator


class TestFedAvg:
    """Test FedAvg aggregator"""

    def test_initialization(self):
        """Test aggregator initialization"""
        aggregator = FedAvgAggregator(device='cpu')
        assert aggregator.device == 'cpu'
        assert aggregator.aggregation_count == 0

    def test_simple_aggregation(self):
        """Test basic aggregation of two models"""
        aggregator = FedAvgAggregator()

        # Create two simple model updates
        update1 = {'weight': torch.tensor([1.0, 2.0, 3.0])}
        update2 = {'weight': torch.tensor([3.0, 4.0, 5.0])}

        # Aggregate with equal weights
        result = aggregator.aggregate([update1, update2])

        # Expected: average of [1,2,3] and [3,4,5] = [2,3,4]
        expected = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(result['weight'], expected)

    def test_weighted_aggregation(self):
        """Test weighted aggregation"""
        aggregator = FedAvgAggregator()

        update1 = {'weight': torch.tensor([1.0, 2.0])}
        update2 = {'weight': torch.tensor([3.0, 4.0])}

        # Weights: 3:1 ratio
        result = aggregator.aggregate([update1, update2], client_weights=[3.0, 1.0])

        # Expected: (3*[1,2] + 1*[3,4]) / 4 = [6/4, 10/4] = [1.5, 2.5]
        expected = torch.tensor([1.5, 2.5])
        assert torch.allclose(result['weight'], expected)

    def test_single_client(self):
        """Test aggregation with single client"""
        aggregator = FedAvgAggregator()

        update = {'weight': torch.tensor([5.0, 6.0, 7.0])}
        result = aggregator.aggregate([update])

        # Should return same update
        assert torch.allclose(result['weight'], update['weight'])

    def test_multiple_parameters(self):
        """Test aggregation with multiple model parameters"""
        aggregator = FedAvgAggregator()

        update1 = {
            'weight': torch.tensor([1.0, 2.0]),
            'bias': torch.tensor([0.5])
        }
        update2 = {
            'weight': torch.tensor([3.0, 4.0]),
            'bias': torch.tensor([1.5])
        }

        result = aggregator.aggregate([update1, update2])

        assert torch.allclose(result['weight'], torch.tensor([2.0, 3.0]))
        assert torch.allclose(result['bias'], torch.tensor([1.0]))

    def test_empty_updates_raises_error(self):
        """Test that empty updates raise ValueError"""
        aggregator = FedAvgAggregator()

        with pytest.raises(ValueError):
            aggregator.aggregate([])

    def test_aggregation_counter(self):
        """Test that aggregation counter increments"""
        aggregator = FedAvgAggregator()
        update = {'weight': torch.tensor([1.0])}

        assert aggregator.aggregation_count == 0
        aggregator.aggregate([update])
        assert aggregator.aggregation_count == 1
        aggregator.aggregate([update])
        assert aggregator.aggregation_count == 2


class TestFedProx:
    """Test FedProx aggregator"""

    def test_initialization(self):
        """Test FedProx initialization with mu"""
        aggregator = FedProxAggregator(mu=0.01)
        assert aggregator.mu == 0.01
        assert aggregator.device == 'cpu'

    def test_aggregation_same_as_fedavg(self):
        """Test that aggregation logic is same as FedAvg"""
        fedavg = FedAvgAggregator()
        fedprox = FedProxAggregator(mu=0.01)

        update1 = {'weight': torch.tensor([1.0, 2.0])}
        update2 = {'weight': torch.tensor([3.0, 4.0])}

        result_fedavg = fedavg.aggregate([update1, update2])
        result_fedprox = fedprox.aggregate([update1, update2])

        assert torch.allclose(result_fedavg['weight'], result_fedprox['weight'])

    def test_proximal_loss_computation(self):
        """Test proximal loss computation"""
        aggregator = FedProxAggregator(mu=0.5)

        local_model = {'weight': torch.tensor([2.0, 3.0])}
        global_model = {'weight': torch.tensor([1.0, 1.0])}

        # ||[2,3] - [1,1]||^2 = ||[1,2]||^2 = 1 + 4 = 5
        # proximal_loss = (0.5/2) * 5 = 1.25
        loss = aggregator.compute_proximal_loss(local_model, global_model)
        assert abs(loss - 1.25) < 1e-5

    def test_mu_update(self):
        """Test updating mu value"""
        aggregator = FedProxAggregator(mu=0.01)
        assert aggregator.get_proximal_term() == 0.01

        aggregator.set_proximal_term(0.05)
        assert aggregator.get_proximal_term() == 0.05

    def test_negative_mu_raises_error(self):
        """Test that negative mu raises error"""
        aggregator = FedProxAggregator(mu=0.01)

        with pytest.raises(ValueError):
            aggregator.set_proximal_term(-0.1)


class TestFedNova:
    """Test FedNova aggregator"""

    def test_initialization(self):
        """Test FedNova initialization"""
        aggregator = FedNovaAggregator(tau_eff_default=1.0)
        assert aggregator.tau_eff_default == 1.0

    def test_aggregation_with_tau_eff(self):
        """Test aggregation with effective local steps"""
        aggregator = FedNovaAggregator()

        update1 = {'weight': torch.tensor([2.0, 4.0])}
        update2 = {'weight': torch.tensor([6.0, 8.0])}

        # With tau_eff [2.0, 1.0] and equal data weights
        result = aggregator.aggregate(
            [update1, update2],
            client_weights=[1.0, 1.0],
            tau_eff=[2.0, 1.0]
        )

        # Normalized updates: [2/2, 4/2] = [1,2] and [6/1, 8/1] = [6,8]
        # Weights: 1*2=2 and 1*1=1, total=3, normalized: [2/3, 1/3]
        # Weighted avg: 2/3*[1,2] + 1/3*[6,8] = [2/3+2, 4/3+8/3] = [8/3, 12/3] = [2.67, 4]
        # Coefficient: 3/2 = 1.5
        # Final: 1.5 * [2.67, 4] = [4, 6]
        expected_approx = torch.tensor([4.0, 6.0])
        assert torch.allclose(result['weight'], expected_approx, atol=0.1)

    def test_tau_eff_computation_no_momentum(self):
        """Test tau_eff computation without momentum"""
        aggregator = FedNovaAggregator()

        tau_eff = aggregator.compute_tau_eff(
            local_steps=10,
            learning_rate=0.01,
            momentum=0.0
        )

        # Without momentum, tau_eff = local_steps
        assert tau_eff == 10.0

    def test_tau_eff_computation_with_momentum(self):
        """Test tau_eff computation with momentum"""
        aggregator = FedNovaAggregator()

        tau_eff = aggregator.compute_tau_eff(
            local_steps=5,
            learning_rate=0.01,
            momentum=0.9
        )

        # With momentum 0.9: (1 - 0.9^5) / (1 - 0.9) = (1 - 0.59049) / 0.1 = 4.0951
        expected = (1 - 0.9**5) / (1 - 0.9)
        assert abs(tau_eff - expected) < 1e-3


class TestBaseAggregator:
    """Test base aggregator utilities"""

    def test_compute_update_norm(self):
        """Test L2 norm computation"""
        aggregator = FedAvgAggregator()

        update = {'weight': torch.tensor([3.0, 4.0])}
        norm = aggregator.compute_update_norm(update)

        # ||[3,4]|| = sqrt(9 + 16) = 5
        assert abs(norm - 5.0) < 1e-5

    def test_clip_updates(self):
        """Test update clipping"""
        aggregator = FedAvgAggregator()

        # Update with norm = 5
        update = {'weight': torch.tensor([3.0, 4.0])}

        # Clip to max norm = 2.5
        clipped = aggregator.clip_updates([update], max_norm=2.5)

        # Should be scaled by 2.5/5 = 0.5
        expected = torch.tensor([1.5, 2.0])
        assert torch.allclose(clipped[0]['weight'], expected)

    def test_get_statistics(self):
        """Test statistics computation"""
        aggregator = FedAvgAggregator()

        updates = [
            {'weight': torch.tensor([3.0, 4.0])},  # norm = 5
            {'weight': torch.tensor([6.0, 8.0])}   # norm = 10
        ]

        stats = aggregator.get_statistics(updates)

        assert stats['num_updates'] == 2
        assert abs(stats['mean_norm'] - 7.5) < 1e-5
        assert abs(stats['max_norm'] - 10.0) < 1e-5
        assert abs(stats['min_norm'] - 5.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
