"""
Base Graph Neural Network Module

Provides a base class for all GNN models with common functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseGNN(nn.Module, ABC):
    """Base class for Graph Neural Network models."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Initialize base GNN model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            activation: Activation function name
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Activation function
        self.activation = self._get_activation(activation)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])

    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'gelu': F.gelu,
            'elu': F.elu,
            'leaky_relu': F.leaky_relu,
            'selu': F.selu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        return activations.get(name.lower(), F.relu)

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the GNN.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        pass

    def reset_parameters(self):
        """Reset model parameters."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings from the model."""
        return self.forward(x, edge_index)

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
