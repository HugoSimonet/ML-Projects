"""
Federated Averaging (FedAvg) Algorithm
Paper: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
McMahan et al., 2017
"""

import torch
from typing import List, Dict, Optional
import logging
from .base_aggregator import BaseAggregator

logger = logging.getLogger(__name__)


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging aggregator

    The classic FedAvg algorithm that aggregates client models by weighted averaging
    based on the number of data samples at each client.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize FedAvg aggregator

        Args:
            device: Computing device ('cpu' or 'cuda')
        """
        super().__init__(device)
        logger.info("Initialized FedAvg aggregator")

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using weighted averaging

        Args:
            client_updates: List of client model state dicts
            client_weights: Weights for each client (typically number of samples)
                          If None, uses equal weights
            **kwargs: Additional parameters (unused in FedAvg)

        Returns:
            Aggregated global model state dict
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        # Use equal weights if not provided
        if client_weights is None:
            client_weights = [1.0] * len(client_updates)
            logger.debug("Using equal weights for FedAvg aggregation")
        else:
            logger.debug(f"Using provided weights for FedAvg aggregation: {client_weights}")

        # Validate inputs
        if len(client_updates) != len(client_weights):
            raise ValueError(
                f"Number of updates ({len(client_updates)}) must match "
                f"number of weights ({len(client_weights)})"
            )

        # Perform weighted averaging
        aggregated_state = self.weighted_average(client_updates, client_weights)

        # Update counter
        self.aggregation_count += 1

        # Log statistics
        stats = self.get_statistics(client_updates)
        logger.info(
            f"FedAvg aggregation #{self.aggregation_count}: "
            f"{stats['num_updates']} clients, "
            f"mean norm: {stats['mean_norm']:.4f}, "
            f"std norm: {stats['std_norm']:.4f}"
        )

        return aggregated_state

    def aggregate_gradients(
        self,
        client_gradients: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client gradients (alternative to model averaging)

        Args:
            client_gradients: List of client gradient dicts
            client_weights: Weights for each client

        Returns:
            Aggregated gradients
        """
        # FedAvg treats gradients the same as model parameters
        return self.aggregate(client_gradients, client_weights)

    def aggregate_delta(
        self,
        global_model: Dict[str, torch.Tensor],
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model deltas (difference from global model)

        Args:
            global_model: Current global model state
            client_models: List of client model states
            client_weights: Weights for each client

        Returns:
            New global model state
        """
        # Compute deltas
        client_deltas = []
        for client_model in client_models:
            delta = {}
            for key in global_model.keys():
                delta[key] = client_model[key] - global_model[key]
            client_deltas.append(delta)

        # Aggregate deltas
        aggregated_delta = self.aggregate(client_deltas, client_weights)

        # Apply to global model
        new_global_model = {}
        for key in global_model.keys():
            new_global_model[key] = global_model[key] + aggregated_delta[key]

        return new_global_model


class AdaptiveFedAvg(FedAvgAggregator):
    """
    Adaptive FedAvg with learning rate scheduling and momentum

    Extends FedAvg with server-side optimization techniques
    """

    def __init__(
        self,
        device: str = 'cpu',
        server_lr: float = 1.0,
        server_momentum: float = 0.0
    ):
        """
        Initialize Adaptive FedAvg

        Args:
            device: Computing device
            server_lr: Server-side learning rate
            server_momentum: Server-side momentum
        """
        super().__init__(device)
        self.server_lr = server_lr
        self.server_momentum = server_momentum
        self.velocity = None

        logger.info(
            f"Initialized Adaptive FedAvg: "
            f"lr={server_lr}, momentum={server_momentum}"
        )

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate with server-side optimization

        Args:
            client_updates: List of client model state dicts
            client_weights: Weights for each client
            **kwargs: Must include 'global_model' for delta computation

        Returns:
            New global model state
        """
        global_model = kwargs.get('global_model', None)

        # Standard FedAvg aggregation
        aggregated_state = super().aggregate(client_updates, client_weights)

        # Apply server-side optimization if global model is provided
        if global_model is not None:
            # Compute pseudo-gradient (delta)
            pseudo_gradient = {}
            for key in global_model.keys():
                pseudo_gradient[key] = global_model[key] - aggregated_state[key]

            # Initialize velocity
            if self.velocity is None:
                self.velocity = {
                    key: torch.zeros_like(value)
                    for key, value in pseudo_gradient.items()
                }

            # Update with momentum
            updated_state = {}
            for key in global_model.keys():
                # Momentum update
                self.velocity[key] = (
                    self.server_momentum * self.velocity[key] +
                    pseudo_gradient[key]
                )

                # Apply learning rate
                updated_state[key] = (
                    global_model[key] -
                    self.server_lr * self.velocity[key]
                )

            return updated_state
        else:
            return aggregated_state


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

    client_weights = [100, 150, 80, 120, 90]  # Number of samples per client

    # Test FedAvg
    fedavg = FedAvgAggregator()
    aggregated = fedavg.aggregate(client_updates, client_weights)

    print("\nFedAvg Aggregation:")
    print(f"Aggregated keys: {list(aggregated.keys())}")
    print(f"Layer1 shape: {aggregated['layer1'].shape}")
    print(f"Layer1 mean: {aggregated['layer1'].mean().item():.4f}")

    # Test Adaptive FedAvg
    adaptive_fedavg = AdaptiveFedAvg(server_lr=0.9, server_momentum=0.5)
    global_model = {'layer1': torch.zeros(*model_size), 'layer2': torch.zeros(5)}

    updated = adaptive_fedavg.aggregate(
        client_updates,
        client_weights,
        global_model=global_model
    )

    print("\nAdaptive FedAvg Aggregation:")
    print(f"Updated keys: {list(updated.keys())}")
    print(f"Layer1 mean: {updated['layer1'].mean().item():.4f}")
