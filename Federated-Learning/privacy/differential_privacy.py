"""
Differential Privacy Implementation for Federated Learning
Implements DP-SGD, privacy accounting, and noise calibration
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Track privacy budget consumption"""
    epsilon: float
    delta: float
    steps: int = 0
    composition_type: str = "basic"  # basic, advanced, rdp


class GaussianMechanism:
    """Gaussian noise mechanism for differential privacy"""

    def __init__(self, sensitivity: float, epsilon: float, delta: float):
        """
        Initialize Gaussian mechanism

        Args:
            sensitivity: L2 sensitivity of the query
            epsilon: Privacy parameter
            delta: Privacy parameter
        """
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = self._compute_noise_scale()

    def _compute_noise_scale(self) -> float:
        """Compute noise scale (standard deviation) for Gaussian mechanism"""
        if self.delta == 0 or self.delta >= 1:
            raise ValueError("Delta must be in (0, 1) for Gaussian mechanism")

        # Gaussian mechanism: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        sigma = self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return sigma

    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Add calibrated Gaussian noise to data"""
        noise = torch.normal(mean=0, std=self.sigma, size=data.shape, device=data.device)
        return data + noise


class LaplaceMechanism:
    """Laplace noise mechanism for differential privacy"""

    def __init__(self, sensitivity: float, epsilon: float):
        """
        Initialize Laplace mechanism

        Args:
            sensitivity: L1 sensitivity of the query
            epsilon: Privacy parameter
        """
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.scale = sensitivity / epsilon

    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Add calibrated Laplace noise to data"""
        noise = torch.distributions.Laplace(0, self.scale).sample(data.shape).to(data.device)
        return data + noise


class PrivacyAccountant:
    """
    Privacy accounting using Rényi Differential Privacy (RDP)
    Based on: "Rényi Differential Privacy" (Mironov, 2017)
    """

    def __init__(self, noise_multiplier: float, sample_rate: float, target_delta: float):
        """
        Initialize privacy accountant

        Args:
            noise_multiplier: Noise multiplier (sigma / sensitivity)
            sample_rate: Sampling rate of clients per round
            target_delta: Target delta for privacy guarantee
        """
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.target_delta = target_delta
        self.steps = 0
        self.rdp_orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def compute_rdp(self, alpha: float) -> float:
        """
        Compute Rényi Differential Privacy for a given order alpha

        Args:
            alpha: Order of RDP

        Returns:
            RDP value
        """
        if self.sample_rate == 0:
            return 0

        if alpha <= 1:
            raise ValueError("Alpha must be greater than 1")

        # RDP for Gaussian mechanism with subsampling
        q = self.sample_rate
        sigma = self.noise_multiplier

        if q == 1:
            # No subsampling
            rdp = alpha / (2 * sigma ** 2)
        else:
            # Simplified RDP bound for subsampled Gaussian mechanism
            # Based on: Privacy Amplification by Subsampling
            # rdp ≈ q^2 * alpha / (2 * sigma^2) for small q
            # More accurate formula for general case:
            rdp = min(
                alpha / (2 * sigma ** 2),  # No subsampling bound
                q**2 * alpha / (2 * sigma ** 2) + q * alpha * (alpha - 1) / (2 * sigma ** 2)
            )

        return rdp

    def get_epsilon(self) -> float:
        """
        Convert RDP to (ε, δ)-DP

        Returns:
            Privacy parameter epsilon
        """
        if self.steps == 0:
            return 0.0

        rdp_values = [self.compute_rdp(alpha) for alpha in self.rdp_orders]

        # Convert RDP to (ε, δ)-DP
        epsilon = min([
            rdp * self.steps + math.log(1 / self.target_delta) / (alpha - 1)
            for alpha, rdp in zip(self.rdp_orders, rdp_values)
        ])

        return epsilon

    def step(self):
        """Record a privacy-consuming step"""
        self.steps += 1

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get total privacy spent (epsilon, delta)"""
        epsilon = self.get_epsilon()
        return epsilon, self.target_delta


class DifferentialPrivacy:
    """Main differential privacy interface for federated learning"""

    def __init__(
        self,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        noise_multiplier: Optional[float] = None,
        mechanism: str = "gaussian"
    ):
        """
        Initialize differential privacy mechanism

        Args:
            epsilon: Privacy parameter
            delta: Privacy parameter
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise multiplier (if None, computed from epsilon/delta)
            mechanism: Noise mechanism ('gaussian' or 'laplace')
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.mechanism_type = mechanism

        if noise_multiplier is None:
            # Compute noise multiplier from epsilon and delta
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier

        logger.info(f"DP initialized: ε={epsilon}, δ={delta}, "
                   f"noise_multiplier={self.noise_multiplier}, "
                   f"max_grad_norm={max_grad_norm}")

    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier from epsilon and delta"""
        if self.mechanism_type == "gaussian":
            # Using standard composition
            sigma = math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
            return sigma
        else:
            return 1.0 / self.epsilon

    def clip_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clip gradients to bounded sensitivity

        Args:
            gradients: Dictionary of gradients

        Returns:
            Clipped gradients
        """
        # Compute L2 norm of all gradients
        total_norm = torch.sqrt(sum(
            torch.sum(grad ** 2) for grad in gradients.values()
        ))

        # Compute clipping factor
        clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-6))

        # Clip gradients
        clipped_gradients = {
            name: grad * clip_factor
            for name, grad in gradients.items()
        }

        return clipped_gradients

    def add_noise(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add calibrated noise to gradients

        Args:
            gradients: Dictionary of gradients

        Returns:
            Noisy gradients
        """
        noisy_gradients = {}

        for name, grad in gradients.items():
            if self.mechanism_type == "gaussian":
                # Gaussian noise
                noise_std = self.noise_multiplier * self.max_grad_norm
                noise = torch.normal(mean=0, std=noise_std, size=grad.shape, device=grad.device)
            else:
                # Laplace noise
                noise_scale = self.noise_multiplier * self.max_grad_norm
                noise = torch.distributions.Laplace(0, noise_scale).sample(grad.shape).to(grad.device)

            noisy_gradients[name] = grad + noise

        return noisy_gradients

    def privatize_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy to gradients (clip + add noise)

        Args:
            gradients: Dictionary of gradients

        Returns:
            Privatized gradients
        """
        # Step 1: Clip gradients
        clipped_gradients = self.clip_gradients(gradients)

        # Step 2: Add noise
        noisy_gradients = self.add_noise(clipped_gradients)

        return noisy_gradients


class PrivacyEngine:
    """
    Complete privacy engine for federated learning
    Manages DP application and privacy accounting
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        noise_multiplier: Optional[float] = None,
        sample_rate: float = 0.1,
        mechanism: str = "gaussian"
    ):
        """
        Initialize privacy engine

        Args:
            epsilon: Target privacy parameter
            delta: Target privacy parameter
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise multiplier (if None, computed)
            sample_rate: Client sampling rate per round
            mechanism: Noise mechanism
        """
        self.dp = DifferentialPrivacy(
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            mechanism=mechanism
        )

        self.accountant = PrivacyAccountant(
            noise_multiplier=self.dp.noise_multiplier,
            sample_rate=sample_rate,
            target_delta=delta
        )

        self.target_epsilon = epsilon
        self.current_round = 0

    def privatize_update(
        self,
        model_update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy to model update

        Args:
            model_update: Model update (gradients or parameters)

        Returns:
            Privatized model update
        """
        privatized_update = self.dp.privatize_gradients(model_update)
        self.accountant.step()
        self.current_round += 1

        return privatized_update

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy spent"""
        return self.accountant.get_privacy_spent()

    def check_privacy_budget(self) -> bool:
        """Check if privacy budget is exceeded"""
        current_epsilon, _ = self.get_privacy_spent()
        return current_epsilon <= self.target_epsilon

    def get_privacy_report(self) -> Dict[str, any]:
        """Generate privacy report"""
        epsilon, delta = self.get_privacy_spent()

        return {
            'target_epsilon': self.target_epsilon,
            'target_delta': self.accountant.target_delta,
            'current_epsilon': epsilon,
            'current_delta': delta,
            'rounds': self.current_round,
            'noise_multiplier': self.dp.noise_multiplier,
            'max_grad_norm': self.dp.max_grad_norm,
            'budget_remaining': max(0, self.target_epsilon - epsilon),
            'budget_exceeded': epsilon > self.target_epsilon
        }


def compute_rdp_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    target_delta: float
) -> float:
    """
    Compute epsilon using RDP accounting

    Args:
        noise_multiplier: Noise multiplier
        sample_rate: Sampling rate
        steps: Number of steps
        target_delta: Target delta

    Returns:
        Computed epsilon
    """
    accountant = PrivacyAccountant(noise_multiplier, sample_rate, target_delta)
    accountant.steps = steps
    return accountant.get_epsilon()


def calibrate_noise(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    num_rounds: int,
    sensitivity: float = 1.0
) -> float:
    """
    Calibrate noise multiplier to achieve target privacy budget

    Args:
        target_epsilon: Target epsilon
        target_delta: Target delta
        sample_rate: Sampling rate per round
        num_rounds: Total number of rounds
        sensitivity: Sensitivity (max_grad_norm)

    Returns:
        Calibrated noise multiplier
    """
    # Binary search for noise multiplier
    lower, upper = 0.1, 100.0
    tolerance = 0.01

    while upper - lower > tolerance:
        mid = (lower + upper) / 2
        epsilon = compute_rdp_epsilon(mid, sample_rate, num_rounds, target_delta)

        if epsilon < target_epsilon:
            upper = mid
        else:
            lower = mid

    noise_multiplier = (lower + upper) / 2
    logger.info(f"Calibrated noise multiplier: {noise_multiplier} "
               f"for target (ε={target_epsilon}, δ={target_delta})")

    return noise_multiplier


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize privacy engine
    privacy_engine = PrivacyEngine(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        sample_rate=0.1
    )

    # Simulate gradient privatization
    dummy_gradients = {
        'layer1': torch.randn(100, 50),
        'layer2': torch.randn(50, 10)
    }

    for round_num in range(10):
        private_grads = privacy_engine.privatize_update(dummy_gradients)
        report = privacy_engine.get_privacy_report()
        print(f"Round {round_num}: ε = {report['current_epsilon']:.4f}")

    # Final privacy report
    final_report = privacy_engine.get_privacy_report()
    print("\nFinal Privacy Report:")
    for key, value in final_report.items():
        print(f"  {key}: {value}")
