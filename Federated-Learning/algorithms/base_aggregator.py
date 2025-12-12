"""
Base aggregator class for federated learning algorithms
"""

import torch
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseAggregator(ABC):
    """Abstract base class for federated aggregation algorithms"""

    def __init__(self, device: str = 'cpu'):
        """
        Initialize base aggregator

        Args:
            device: Computing device ('cpu' or 'cuda')
        """
        self.device = device
        self.aggregation_count = 0

    @abstractmethod
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates into global model

        Args:
            client_updates: List of client model updates
            client_weights: Optional weights for each client
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Aggregated global model state
        """
        pass

    def weighted_average(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform weighted averaging of client updates

        Args:
            client_updates: List of client model updates
            weights: Weight for each client

        Returns:
            Weighted average model state
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        if len(client_updates) != len(weights):
            raise ValueError("Number of updates and weights must match")

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Initialize aggregated state with zeros
        aggregated_state = {}
        for key in client_updates[0].keys():
            aggregated_state[key] = torch.zeros_like(
                client_updates[0][key],
                device=self.device
            )

        # Weighted sum
        for update, weight in zip(client_updates, normalized_weights):
            for key in update.keys():
                aggregated_state[key] += weight * update[key].to(self.device)

        return aggregated_state

    def simple_average(
        self,
        client_updates: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform simple averaging (equal weights)

        Args:
            client_updates: List of client model updates

        Returns:
            Average model state
        """
        num_clients = len(client_updates)
        weights = [1.0] * num_clients
        return self.weighted_average(client_updates, weights)

    def compute_update_norm(self, update: Dict[str, torch.Tensor]) -> float:
        """
        Compute L2 norm of model update

        Args:
            update: Model update

        Returns:
            L2 norm
        """
        total_norm = 0.0
        for param in update.values():
            total_norm += torch.sum(param ** 2).item()
        return total_norm ** 0.5

    def clip_updates(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        max_norm: float
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Clip client updates to maximum norm

        Args:
            client_updates: List of client updates
            max_norm: Maximum allowed norm

        Returns:
            Clipped updates
        """
        clipped_updates = []

        for update in client_updates:
            norm = self.compute_update_norm(update)
            if norm > max_norm:
                scale = max_norm / norm
                clipped_update = {
                    key: value * scale
                    for key, value in update.items()
                }
                clipped_updates.append(clipped_update)
                logger.debug(f"Clipped update: norm {norm:.4f} -> {max_norm:.4f}")
            else:
                clipped_updates.append(update)

        return clipped_updates

    def get_statistics(
        self,
        client_updates: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Compute statistics about client updates

        Args:
            client_updates: List of client updates

        Returns:
            Dictionary of statistics
        """
        norms = [self.compute_update_norm(update) for update in client_updates]

        stats = {
            'num_updates': len(client_updates),
            'mean_norm': sum(norms) / len(norms) if norms else 0.0,
            'max_norm': max(norms) if norms else 0.0,
            'min_norm': min(norms) if norms else 0.0,
            'std_norm': (sum((n - sum(norms)/len(norms))**2 for n in norms) / len(norms))**0.5 if len(norms) > 1 else 0.0
        }

        return stats
