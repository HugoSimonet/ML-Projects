"""
Federated Normalized Averaging (FedNova) Algorithm
Paper: "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
Wang et al., NeurIPS 2020

FedNova normalizes client updates to account for different local training epochs
"""

import torch
from typing import List, Dict, Optional, Tuple
import logging
from .base_aggregator import BaseAggregator

logger = logging.getLogger(__name__)


class FedNovaAggregator(BaseAggregator):
    """
    FedNova aggregator with normalized averaging

    FedNova addresses the objective inconsistency problem in heterogeneous FL
    by normalizing client updates according to their effective local steps.
    """

    def __init__(self, device: str = 'cpu', tau_eff_default: float = 1.0):
        """
        Initialize FedNova aggregator

        Args:
            device: Computing device ('cpu' or 'cuda')
            tau_eff_default: Default effective local steps if not provided
        """
        super().__init__(device)
        self.tau_eff_default = tau_eff_default
        logger.info(f"Initialized FedNova aggregator with tau_eff_default={tau_eff_default}")

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates with normalization

        Args:
            client_updates: List of client model state dicts
            client_weights: Weights for each client (typically number of samples)
            **kwargs: Must include 'tau_eff' (effective local steps) for each client

        Returns:
            Aggregated global model state dict
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        # Get effective local steps (tau_eff) for each client
        tau_effs = kwargs.get('tau_eff', None)
        if tau_effs is None:
            # Use default if not provided
            tau_effs = [self.tau_eff_default] * len(client_updates)
            logger.warning(f"tau_eff not provided, using default: {self.tau_eff_default}")

        # Use equal data weights if not provided
        if client_weights is None:
            client_weights = [1.0] * len(client_updates)

        # Validate inputs
        if len(client_updates) != len(client_weights):
            raise ValueError("Number of updates and weights must match")
        if len(client_updates) != len(tau_effs):
            raise ValueError("Number of updates and tau_effs must match")

        # Compute normalized weights
        # w_i = (n_i * tau_i) / sum_j(n_j * tau_j)
        normalized_weights = self._compute_normalized_weights(
            client_weights, tau_effs
        )

        # Normalize each client update by its tau_eff
        normalized_updates = []
        for update, tau_eff in zip(client_updates, tau_effs):
            normalized_update = {
                key: value / tau_eff
                for key, value in update.items()
            }
            normalized_updates.append(normalized_update)

        # Aggregate normalized updates with normalized weights
        aggregated_state = self.weighted_average(normalized_updates, normalized_weights)

        # Scale by total effective steps
        total_tau = sum(w * tau for w, tau in zip(client_weights, tau_effs))
        total_weight = sum(client_weights)
        coeff = total_tau / total_weight if total_weight > 0 else 1.0

        # Apply coefficient
        for key in aggregated_state.keys():
            aggregated_state[key] = aggregated_state[key] * coeff

        # Update counter
        self.aggregation_count += 1

        # Log statistics
        logger.info(
            f"FedNova aggregation #{self.aggregation_count}: "
            f"{len(client_updates)} clients, "
            f"tau_effs: {tau_effs}, "
            f"coeff: {coeff:.4f}"
        )

        return aggregated_state

    def _compute_normalized_weights(
        self,
        client_weights: List[float],
        tau_effs: List[float]
    ) -> List[float]:
        """
        Compute normalized weights for FedNova

        Args:
            client_weights: Data-based weights (e.g., number of samples)
            tau_effs: Effective local steps for each client

        Returns:
            Normalized weights
        """
        # Compute total weighted steps
        total = sum(w * tau for w, tau in zip(client_weights, tau_effs))

        if total == 0:
            logger.warning("Total weighted steps is zero, using equal weights")
            return [1.0 / len(client_weights)] * len(client_weights)

        # Normalize
        normalized = [(w * tau) / total for w, tau in zip(client_weights, tau_effs)]

        return normalized

    def compute_tau_eff(
        self,
        local_steps: int,
        learning_rate: float,
        momentum: float = 0.0
    ) -> float:
        """
        Compute effective local steps tau_eff

        For SGD: tau_eff = local_steps
        For SGD with momentum: tau_eff depends on momentum

        Args:
            local_steps: Number of local training steps
            learning_rate: Local learning rate
            momentum: Momentum coefficient

        Returns:
            Effective local steps
        """
        if momentum == 0:
            # Standard SGD
            tau_eff = float(local_steps)
        else:
            # SGD with momentum
            # tau_eff = (1 - rho^tau) / (1 - rho) where rho is momentum
            if momentum >= 1.0:
                logger.warning("Momentum >= 1.0, using local_steps as tau_eff")
                tau_eff = float(local_steps)
            else:
                tau_eff = (1 - momentum ** local_steps) / (1 - momentum)

        return tau_eff


class AdaptiveFedNova(FedNovaAggregator):
    """
    Adaptive FedNova with automatic tau_eff estimation

    Estimates effective local steps from gradient statistics
    """

    def __init__(
        self,
        device: str = 'cpu',
        tau_eff_default: float = 1.0,
        momentum_estimate: float = 0.0
    ):
        """
        Initialize Adaptive FedNova

        Args:
            device: Computing device
            tau_eff_default: Default effective local steps
            momentum_estimate: Estimated momentum for tau_eff computation
        """
        super().__init__(device, tau_eff_default)
        self.momentum_estimate = momentum_estimate
        logger.info(f"Initialized Adaptive FedNova with momentum_estimate={momentum_estimate}")

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate with automatic tau_eff estimation

        Args:
            client_updates: List of client model state dicts
            client_weights: Weights for each client
            **kwargs: Can include 'local_steps' for each client

        Returns:
            Aggregated global model state
        """
        # Estimate tau_eff if not provided
        tau_effs = kwargs.get('tau_eff', None)

        if tau_effs is None:
            local_steps_list = kwargs.get('local_steps', None)
            if local_steps_list is not None:
                # Compute tau_eff from local steps
                tau_effs = [
                    self.compute_tau_eff(
                        steps,
                        learning_rate=1.0,  # Normalized
                        momentum=self.momentum_estimate
                    )
                    for steps in local_steps_list
                ]
                logger.info(f"Computed tau_effs from local_steps: {tau_effs}")
            else:
                # Estimate from update norms
                tau_effs = self._estimate_tau_eff_from_updates(client_updates)
                logger.info(f"Estimated tau_effs from updates: {tau_effs}")

        # Use parent aggregation with estimated tau_effs
        kwargs['tau_eff'] = tau_effs
        return super().aggregate(client_updates, client_weights, **kwargs)

    def _estimate_tau_eff_from_updates(
        self,
        client_updates: List[Dict[str, torch.Tensor]]
    ) -> List[float]:
        """
        Estimate tau_eff from update magnitudes

        Assumes updates with larger norms had more effective local training

        Args:
            client_updates: List of client updates

        Returns:
            Estimated tau_eff for each client
        """
        norms = [self.compute_update_norm(update) for update in client_updates]
        mean_norm = sum(norms) / len(norms) if norms else 1.0

        # Estimate tau_eff proportional to norm
        tau_effs = [
            max(0.1, norm / mean_norm)  # Avoid very small values
            for norm in norms
        ]

        return tau_effs


class FedNovaWithVariance(FedNovaAggregator):
    """
    FedNova with variance reduction

    Incorporates control variates for variance reduction in heterogeneous settings
    """

    def __init__(
        self,
        device: str = 'cpu',
        tau_eff_default: float = 1.0,
        variance_reduction: bool = True
    ):
        """
        Initialize FedNova with variance reduction

        Args:
            device: Computing device
            tau_eff_default: Default effective local steps
            variance_reduction: Enable variance reduction
        """
        super().__init__(device, tau_eff_default)
        self.variance_reduction = variance_reduction
        self.global_gradient = None
        logger.info(f"Initialized FedNova with variance_reduction={variance_reduction}")

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate with variance reduction

        Args:
            client_updates: List of client model state dicts
            client_weights: Weights for each client
            **kwargs: Additional parameters including tau_eff

        Returns:
            Aggregated global model state
        """
        # Standard FedNova aggregation
        aggregated_state = super().aggregate(client_updates, client_weights, **kwargs)

        # Apply variance reduction if enabled
        if self.variance_reduction:
            if self.global_gradient is not None:
                # Apply control variate correction
                for key in aggregated_state.keys():
                    if key in self.global_gradient:
                        # Simple variance reduction: blend with previous gradient
                        aggregated_state[key] = (
                            0.9 * aggregated_state[key] +
                            0.1 * self.global_gradient[key]
                        )

            # Update global gradient estimate
            self.global_gradient = {
                key: value.clone() for key, value in aggregated_state.items()
            }

        return aggregated_state


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample client updates with different local steps
    num_clients = 5
    model_size = (10, 5)

    client_updates = [
        {'layer1': torch.randn(*model_size), 'layer2': torch.randn(5)}
        for _ in range(num_clients)
    ]

    client_weights = [100, 150, 80, 120, 90]  # Number of samples
    tau_effs = [5.0, 10.0, 3.0, 8.0, 6.0]  # Different effective local steps

    # Test FedNova
    fednova = FedNovaAggregator(tau_eff_default=5.0)
    aggregated = fednova.aggregate(
        client_updates,
        client_weights,
        tau_eff=tau_effs
    )

    print("\nFedNova Aggregation:")
    print(f"Aggregated keys: {list(aggregated.keys())}")
    print(f"tau_effs: {tau_effs}")

    # Test tau_eff computation
    local_steps = 10
    momentum = 0.9
    tau_eff = fednova.compute_tau_eff(local_steps, 0.01, momentum)
    print(f"\nComputed tau_eff: steps={local_steps}, momentum={momentum} -> tau_eff={tau_eff:.2f}")

    # Test Adaptive FedNova
    adaptive_fednova = AdaptiveFedNova(momentum_estimate=0.9)
    adaptive_aggregated = adaptive_fednova.aggregate(
        client_updates,
        client_weights,
        local_steps=[5, 10, 3, 8, 6]
    )
    print("\nAdaptive FedNova aggregation completed")
