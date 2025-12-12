"""Federated learning algorithms"""

from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator
from .fednova import FedNovaAggregator
from .base_aggregator import BaseAggregator

__all__ = [
    'BaseAggregator',
    'FedAvgAggregator',
    'FedProxAggregator',
    'FedNovaAggregator'
]
