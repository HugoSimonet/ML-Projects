"""
Federated Proximal (FedProx) Algorithm
Paper: "Federated Optimization in Heterogeneous Networks"
Li et al., 2020

FedProx addresses system heterogeneity by adding a proximal term to the local objective
"""

import torch
from typing import List, Dict, Optional
import logging
from .base_aggregator import BaseAggregator

logger = logging.getLogger(__name__)


class FedProxAggregator(BaseAggregator):
    """
    FedProx aggregator with proximal term

    FedProx handles heterogeneous systems by adding a proximal term (μ/2)||w - w_t||^2
    to the local objective, where w_t is the global model at round t and μ controls
    the strength of the proximal term.
    """

    def __init__(self, device: str = 'cpu', mu: float = 0.01):
        """
        Initialize FedProx aggregator

        Args:
            device: Computing device ('cpu' or 'cuda')
            mu: Proximal term coefficient (controls heterogeneity handling)
        """
        super().__init__(device)
        self.mu = mu
        logger.info(f"Initialized FedProx aggregator with mu={mu}")

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using weighted averaging

        Note: The proximal term is applied during client training, not aggregation.
        The aggregation itself is the same as FedAvg.

        Args:
            client_updates: List of client model state dicts
            client_weights: Weights for each client (typically number of samples)
            **kwargs: Additional parameters

        Returns:
            Aggregated global model state dict
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        # Use equal weights if not provided
        if client_weights is None:
            client_weights = [1.0] * len(client_updates)
            logger.debug("Using equal weights for FedProx aggregation")

        # Validate inputs
        if len(client_updates) != len(client_weights):
            raise ValueError(
                f"Number of updates ({len(client_updates)}) must match "
                f"number of weights ({len(client_weights)})"
            )

        # FedProx aggregation is identical to FedAvg
        # The difference is in the client-side optimization
        aggregated_state = self.weighted_average(client_updates, client_weights)

        # Update counter
        self.aggregation_count += 1

        # Log statistics
        stats = self.get_statistics(client_updates)
        logger.info(
            f"FedProx aggregation #{self.aggregation_count}: "
            f"{stats['num_updates']} clients, "
            f"mean norm: {stats['mean_norm']:.4f}, "
            f"std norm: {stats['std_norm']:.4f}, "
            f"mu: {self.mu}"
        )

        return aggregated_state

    def compute_proximal_loss(
        self,
        local_model: Dict[str, torch.Tensor],
        global_model: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute proximal term: (mu/2) * ||w_local - w_global||^2

        This is typically computed on the client side during training,
        but provided here for analysis purposes.

        Args:
            local_model: Local model state
            global_model: Global model state

        Returns:
            Proximal loss value
        """
        proximal_loss = 0.0

        for key in local_model.keys():
            if key in global_model:
                diff = local_model[key] - global_model[key]
                proximal_loss += torch.sum(diff ** 2).item()

        proximal_loss = (self.mu / 2) * proximal_loss

        return proximal_loss

    def get_proximal_term(self) -> float:
        """Get the proximal coefficient mu"""
        return self.mu

    def set_proximal_term(self, mu: float):
        """
        Update the proximal coefficient

        Args:
            mu: New proximal coefficient
        """
        if mu < 0:
            raise ValueError("Proximal coefficient mu must be non-negative")

        old_mu = self.mu
        self.mu = mu
        logger.info(f"Updated proximal term: {old_mu} -> {mu}")


class AdaptiveFedProx(FedProxAggregator):
    """
    Adaptive FedProx with dynamic mu adjustment

    Automatically adjusts the proximal term based on convergence behavior
    """

    def __init__(
        self,
        device: str = 'cpu',
        mu_init: float = 0.01,
        mu_min: float = 0.001,
        mu_max: float = 1.0,
        adapt_frequency: int = 10
    ):
        """
        Initialize Adaptive FedProx

        Args:
            device: Computing device
            mu_init: Initial proximal coefficient
            mu_min: Minimum allowed mu
            mu_max: Maximum allowed mu
            adapt_frequency: Rounds between mu adjustments
        """
        super().__init__(device, mu=mu_init)
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.adapt_frequency = adapt_frequency
        self.previous_loss = None
        self.loss_history = []

        logger.info(
            f"Initialized Adaptive FedProx: "
            f"mu_init={mu_init}, range=[{mu_min}, {mu_max}], "
            f"adapt_freq={adapt_frequency}"
        )

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate with adaptive mu adjustment

        Args:
            client_updates: List of client model state dicts
            client_weights: Weights for each client
            **kwargs: Must include 'current_loss' for adaptation

        Returns:
            Aggregated global model state
        """
        # Standard aggregation
        aggregated_state = super().aggregate(client_updates, client_weights, **kwargs)

        # Adapt mu if loss is provided
        current_loss = kwargs.get('current_loss', None)
        if current_loss is not None:
            self.loss_history.append(current_loss)

            # Adapt mu periodically
            if len(self.loss_history) >= self.adapt_frequency:
                self._adapt_mu()

        return aggregated_state

    def _adapt_mu(self):
        """Adapt proximal coefficient based on loss history"""
        if len(self.loss_history) < 2:
            return

        # Compute loss trend
        recent_losses = self.loss_history[-self.adapt_frequency:]
        loss_decrease = recent_losses[0] - recent_losses[-1]
        avg_loss = sum(recent_losses) / len(recent_losses)

        # Adjust mu based on convergence speed
        if loss_decrease < 0:
            # Loss increasing -> increase mu (more regularization)
            new_mu = min(self.mu * 1.2, self.mu_max)
            logger.info(f"Loss increasing, increasing mu: {self.mu:.4f} -> {new_mu:.4f}")
        elif loss_decrease > avg_loss * 0.1:
            # Fast convergence -> decrease mu (less regularization)
            new_mu = max(self.mu * 0.9, self.mu_min)
            logger.info(f"Fast convergence, decreasing mu: {self.mu:.4f} -> {new_mu:.4f}")
        else:
            # Stable convergence -> keep mu
            new_mu = self.mu

        self.mu = new_mu


class RobustFedProx(FedProxAggregator):
    """
    Robust FedProx with outlier detection and handling

    Filters out potentially malicious or anomalous client updates
    """

    def __init__(
        self,
        device: str = 'cpu',
        mu: float = 0.01,
        outlier_threshold: float = 2.0
    ):
        """
        Initialize Robust FedProx

        Args:
            device: Computing device
            mu: Proximal coefficient
            outlier_threshold: Threshold for outlier detection (in std devs)
        """
        super().__init__(device, mu)
        self.outlier_threshold = outlier_threshold
        logger.info(f"Initialized Robust FedProx with outlier threshold={outlier_threshold}")

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate with outlier detection

        Args:
            client_updates: List of client model state dicts
            client_weights: Weights for each client
            **kwargs: Additional parameters

        Returns:
            Aggregated global model state (without outliers)
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        # Detect outliers based on update norms
        norms = [self.compute_update_norm(update) for update in client_updates]
        mean_norm = sum(norms) / len(norms)
        std_norm = (sum((n - mean_norm)**2 for n in norms) / len(norms))**0.5

        # Filter outliers
        filtered_updates = []
        filtered_weights = []

        for i, (update, norm) in enumerate(zip(client_updates, norms)):
            if abs(norm - mean_norm) <= self.outlier_threshold * std_norm:
                filtered_updates.append(update)
                if client_weights:
                    filtered_weights.append(client_weights[i])
            else:
                logger.warning(
                    f"Detected outlier update {i}: norm={norm:.4f}, "
                    f"mean={mean_norm:.4f}, std={std_norm:.4f}"
                )

        if len(filtered_updates) == 0:
            logger.error("All updates were filtered as outliers, using original updates")
            filtered_updates = client_updates
            filtered_weights = client_weights

        logger.info(
            f"Filtered {len(client_updates) - len(filtered_updates)} outliers, "
            f"using {len(filtered_updates)} updates"
        )

        # Use parent aggregation on filtered updates
        return super().aggregate(
            filtered_updates,
            filtered_weights if filtered_weights else None,
            **kwargs
        )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample client updates
    num_clients = 5
    model_size = (10, 5)

    client_updates = [
        {'layer1': torch.randn(*model_size), 'layer2': torch.randn(5)}
        for _ in range(num_clients)
    ]

    client_weights = [100, 150, 80, 120, 90]

    # Test FedProx
    fedprox = FedProxAggregator(mu=0.01)
    aggregated = fedprox.aggregate(client_updates, client_weights)

    print("\nFedProx Aggregation:")
    print(f"Aggregated keys: {list(aggregated.keys())}")
    print(f"Proximal term mu: {fedprox.get_proximal_term()}")

    # Test proximal loss computation
    global_model = {'layer1': torch.zeros(*model_size), 'layer2': torch.zeros(5)}
    prox_loss = fedprox.compute_proximal_loss(client_updates[0], global_model)
    print(f"Example proximal loss: {prox_loss:.4f}")

    # Test Robust FedProx with outlier
    client_updates_with_outlier = client_updates + [
        {'layer1': torch.randn(*model_size) * 10, 'layer2': torch.randn(5) * 10}
    ]
    robust_fedprox = RobustFedProx(mu=0.01, outlier_threshold=2.0)
    robust_aggregated = robust_fedprox.aggregate(client_updates_with_outlier)
    print("\nRobust FedProx filtered outliers successfully")
