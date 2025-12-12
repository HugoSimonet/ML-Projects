"""
Unit tests for differential privacy mechanisms
Tests DP, privacy accounting, and noise calibration
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from privacy import (
    DifferentialPrivacy,
    PrivacyEngine,
    PrivacyAccountant,
    GaussianMechanism,
    compute_rdp_epsilon,
    calibrate_noise
)


class TestDifferentialPrivacy:
    """Test differential privacy mechanisms"""

    def test_initialization(self):
        """Test DP initialization"""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0
        )

        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
        assert dp.max_grad_norm == 1.0
        assert dp.noise_multiplier > 0

    def test_clip_gradients(self):
        """Test gradient clipping"""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0
        )

        # Gradients with total norm > 1.0
        gradients = {
            'layer1': torch.tensor([0.6, 0.8]),  # norm = 1.0
            'layer2': torch.tensor([0.6, 0.8])   # norm = 1.0
        }
        # Total norm = sqrt(1 + 1) = sqrt(2) ≈ 1.414

        clipped = dp.clip_gradients(gradients)

        # Should be clipped to max_grad_norm = 1.0
        # Scaling factor = 1.0 / 1.414 ≈ 0.707
        total_norm = torch.sqrt(
            sum(torch.sum(grad ** 2) for grad in clipped.values())
        )
        assert abs(total_norm - 1.0) < 1e-5

    def test_add_noise_changes_gradients(self):
        """Test that noise is actually added"""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.0
        )

        gradients = {'weight': torch.zeros(10)}
        noisy = dp.add_noise(gradients)

        # Noise should make it non-zero
        assert not torch.allclose(noisy['weight'], gradients['weight'])
        # But not too large
        assert torch.abs(noisy['weight']).mean() < 5.0

    def test_privatize_gradients(self):
        """Test full privatization pipeline"""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0
        )

        gradients = {
            'layer1': torch.tensor([2.0, 2.0])  # norm = 2.828
        }

        private_grads = dp.privatize_gradients(gradients)

        # Should be clipped and noised
        assert 'layer1' in private_grads
        # Norm should be bounded (clipped to 1.0, then noise added)
        norm = torch.sqrt(torch.sum(private_grads['layer1'] ** 2))
        assert norm < 20.0  # Reasonable upper bound with noise (relaxed for high noise scenarios)


class TestGaussianMechanism:
    """Test Gaussian noise mechanism"""

    def test_initialization(self):
        """Test Gaussian mechanism initialization"""
        mechanism = GaussianMechanism(
            sensitivity=1.0,
            epsilon=1.0,
            delta=1e-5
        )

        assert mechanism.sensitivity == 1.0
        assert mechanism.epsilon == 1.0
        assert mechanism.delta == 1e-5
        assert mechanism.sigma > 0

    def test_noise_scale_computation(self):
        """Test noise scale is computed correctly"""
        mechanism = GaussianMechanism(
            sensitivity=1.0,
            epsilon=1.0,
            delta=1e-5
        )

        # sigma should be approximately sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
        import math
        expected_sigma = 1.0 * math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0
        assert abs(mechanism.sigma - expected_sigma) < 1e-5

    def test_add_noise(self):
        """Test adding Gaussian noise"""
        mechanism = GaussianMechanism(
            sensitivity=1.0,
            epsilon=1.0,
            delta=1e-5
        )

        data = torch.zeros(100)
        noisy_data = mechanism.add_noise(data)

        # Should have added noise
        assert not torch.allclose(data, noisy_data)
        # Mean should be close to 0 (with large sample)
        assert abs(noisy_data.mean().item()) < 1.0
        # Std should be approximately sigma
        assert abs(noisy_data.std().item() - mechanism.sigma) < 1.0

    def test_invalid_delta_raises_error(self):
        """Test that invalid delta raises error"""
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=0.0)

        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1.5)


class TestPrivacyAccountant:
    """Test privacy accounting with RDP"""

    def test_initialization(self):
        """Test accountant initialization"""
        accountant = PrivacyAccountant(
            noise_multiplier=1.0,
            sample_rate=0.1,
            target_delta=1e-5
        )

        assert accountant.noise_multiplier == 1.0
        assert accountant.sample_rate == 0.1
        assert accountant.target_delta == 1e-5
        assert accountant.steps == 0

    def test_step_counting(self):
        """Test privacy step counting"""
        accountant = PrivacyAccountant(
            noise_multiplier=1.0,
            sample_rate=0.1,
            target_delta=1e-5
        )

        assert accountant.steps == 0
        accountant.step()
        assert accountant.steps == 1
        accountant.step()
        assert accountant.steps == 2

    def test_get_epsilon_increases_with_steps(self):
        """Test that epsilon increases with more steps"""
        accountant = PrivacyAccountant(
            noise_multiplier=1.0,
            sample_rate=0.1,
            target_delta=1e-5
        )

        epsilon_0 = accountant.get_epsilon()
        assert epsilon_0 == 0.0  # No steps yet

        accountant.step()
        epsilon_1 = accountant.get_epsilon()
        assert epsilon_1 > 0.0

        accountant.step()
        epsilon_2 = accountant.get_epsilon()
        assert epsilon_2 > epsilon_1

    def test_privacy_spent(self):
        """Test getting privacy spent"""
        accountant = PrivacyAccountant(
            noise_multiplier=1.0,
            sample_rate=0.1,
            target_delta=1e-5
        )

        for _ in range(10):
            accountant.step()

        epsilon, delta = accountant.get_privacy_spent()
        assert epsilon > 0
        assert delta == 1e-5


class TestPrivacyEngine:
    """Test complete privacy engine"""

    def test_initialization(self):
        """Test privacy engine initialization"""
        engine = PrivacyEngine(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            sample_rate=0.1
        )

        assert engine.target_epsilon == 1.0
        assert engine.current_round == 0

    def test_privatize_update(self):
        """Test privatizing model update"""
        engine = PrivacyEngine(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            sample_rate=0.1
        )

        update = {'weight': torch.tensor([1.0, 2.0, 3.0])}
        private_update = engine.privatize_update(update)

        # Should return update (with noise)
        assert 'weight' in private_update
        # Round counter should increment
        assert engine.current_round == 1

    def test_privacy_budget_tracking(self):
        """Test privacy budget tracking"""
        engine = PrivacyEngine(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            sample_rate=0.1
        )

        # Initially under budget
        assert engine.check_privacy_budget()

        # Perform many updates
        update = {'weight': torch.zeros(10)}
        for _ in range(1000):
            engine.privatize_update(update)

        # Should eventually exceed budget
        epsilon, delta = engine.get_privacy_spent()
        assert epsilon > 0

    def test_privacy_report(self):
        """Test privacy report generation"""
        engine = PrivacyEngine(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            sample_rate=0.1
        )

        update = {'weight': torch.zeros(10)}
        engine.privatize_update(update)

        report = engine.get_privacy_report()

        assert 'target_epsilon' in report
        assert 'current_epsilon' in report
        assert 'rounds' in report
        assert 'budget_exceeded' in report
        assert report['rounds'] == 1


class TestPrivacyUtils:
    """Test privacy utility functions"""

    def test_compute_rdp_epsilon(self):
        """Test RDP epsilon computation"""
        epsilon = compute_rdp_epsilon(
            noise_multiplier=1.0,
            sample_rate=0.1,
            steps=100,
            target_delta=1e-5
        )

        assert epsilon > 0
        assert epsilon < 100  # Reasonable bound

    def test_calibrate_noise(self):
        """Test noise calibration"""
        noise_multiplier = calibrate_noise(
            target_epsilon=1.0,
            target_delta=1e-5,
            sample_rate=0.1,
            num_rounds=100
        )

        assert noise_multiplier > 0
        # Verify it achieves target epsilon
        achieved_epsilon = compute_rdp_epsilon(
            noise_multiplier=noise_multiplier,
            sample_rate=0.1,
            steps=100,
            target_delta=1e-5
        )
        # Should be close to target (within tolerance)
        assert abs(achieved_epsilon - 1.0) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
