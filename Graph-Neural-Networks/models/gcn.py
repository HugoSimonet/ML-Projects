"""
Graph Convolutional Network (GCN)

Implementation of the Graph Convolutional Network from:
"Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree
from typing import Optional

from .base_gnn import BaseGNN


class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        improved: bool = False
    ):
        """
        Initialize GCN layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
            improved: Whether to use improved GCN
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.improved = improved

        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCN layer.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Linear transformation
        x = torch.mm(x, self.weight)

        # Aggregate neighbors
        out = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            out[dst] += norm[i] * x[src]

        if self.bias is not None:
            out += self.bias

        return out


class GCN(BaseGNN):
    """Graph Convolutional Network model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        improved: bool = False,
        cached: bool = False
    ):
        """
        Initialize GCN model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            activation: Activation function name
            improved: Whether to use improved GCN
            cached: Whether to cache normalized edge weights
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, activation)

        self.improved = improved
        self.cached = cached

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, improved=improved, cached=cached))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, improved=improved, cached=cached))

        self.convs.append(GCNConv(hidden_dim, output_dim, improved=improved, cached=cached))

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of GCN.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout_layer(x)

        # Final layer without activation
        x = self.convs[-1](x, edge_index)

        return x

    def reset_parameters(self):
        """Reset model parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
