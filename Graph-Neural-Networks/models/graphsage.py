"""
GraphSAGE (Graph Sample and Aggregate)

Implementation of GraphSAGE from:
"Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv as PyGSAGEConv
from typing import Optional

from .base_gnn import BaseGNN


class SAGEConv(nn.Module):
    """GraphSAGE Convolution Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = 'mean',
        normalize: bool = True,
        bias: bool = True
    ):
        """
        Initialize GraphSAGE layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            aggregator: Aggregation method ('mean', 'max', 'lstm')
            normalize: Whether to normalize embeddings
            bias: Whether to use bias
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        self.normalize = normalize

        # Linear transformations
        self.lin_self = nn.Linear(in_features, out_features, bias=bias)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=False)

        # LSTM aggregator
        if aggregator == 'lstm':
            self.lstm = nn.LSTM(in_features, out_features, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        self.lin_self.reset_parameters()
        self.lin_neigh.reset_parameters()
        if hasattr(self, 'lstm'):
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GraphSAGE layer.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        edge_src, edge_dst = edge_index

        # Aggregate neighbor features
        if self.aggregator == 'mean':
            neigh_feats = self._aggregate_mean(x, edge_index)
        elif self.aggregator == 'max':
            neigh_feats = self._aggregate_max(x, edge_index)
        elif self.aggregator == 'lstm':
            neigh_feats = self._aggregate_lstm(x, edge_index)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        # Combine self and neighbor features
        out = self.lin_self(x) + self.lin_neigh(neigh_feats)

        # Normalize
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def _aggregate_mean(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Mean aggregation of neighbor features."""
        num_nodes = x.size(0)
        edge_src, edge_dst = edge_index

        # Initialize aggregated features
        agg_feats = torch.zeros_like(x)
        degree = torch.zeros(num_nodes, device=x.device)

        # Aggregate
        for i in range(edge_index.size(1)):
            src, dst = edge_src[i], edge_dst[i]
            agg_feats[dst] += x[src]
            degree[dst] += 1

        # Average
        degree[degree == 0] = 1  # Avoid division by zero
        agg_feats = agg_feats / degree.unsqueeze(-1)

        return agg_feats

    def _aggregate_max(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Max pooling aggregation of neighbor features."""
        num_nodes = x.size(0)
        edge_src, edge_dst = edge_index

        # Initialize with negative infinity
        agg_feats = torch.full_like(x, float('-inf'))

        # Max pooling
        for i in range(edge_index.size(1)):
            src, dst = edge_src[i], edge_dst[i]
            agg_feats[dst] = torch.max(agg_feats[dst], x[src])

        # Replace -inf with zeros
        agg_feats[agg_feats == float('-inf')] = 0

        return agg_feats

    def _aggregate_lstm(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """LSTM aggregation of neighbor features."""
        num_nodes = x.size(0)
        edge_src, edge_dst = edge_index

        # Group neighbors by destination node
        agg_feats = torch.zeros(num_nodes, self.out_features, device=x.device)

        for node in range(num_nodes):
            mask = edge_dst == node
            if mask.any():
                neighbors = x[edge_src[mask]]
                # LSTM expects [batch_size, seq_len, features]
                neighbors = neighbors.unsqueeze(0)
                _, (h_n, _) = self.lstm(neighbors)
                agg_feats[node] = h_n.squeeze(0)

        return agg_feats


class GraphSAGE(BaseGNN):
    """GraphSAGE model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        aggregator: str = 'mean',
        dropout: float = 0.1,
        activation: str = 'relu',
        normalize: bool = True
    ):
        """
        Initialize GraphSAGE model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GraphSAGE layers
            aggregator: Aggregation method ('mean', 'max', 'lstm')
            dropout: Dropout rate
            activation: Activation function name
            normalize: Whether to normalize embeddings
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, activation)

        self.aggregator = aggregator
        self.normalize = normalize

        # GraphSAGE layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            PyGSAGEConv(
                input_dim,
                hidden_dim,
                normalize=normalize,
                aggr=aggregator
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                PyGSAGEConv(
                    hidden_dim,
                    hidden_dim,
                    normalize=normalize,
                    aggr=aggregator
                )
            )

        # Output layer
        self.convs.append(
            PyGSAGEConv(
                hidden_dim,
                output_dim,
                normalize=normalize,
                aggr=aggregator
            )
        )

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
        Forward pass of GraphSAGE.

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

        # Final layer
        x = self.convs[-1](x, edge_index)

        return x

    def inference(self, x_all: torch.Tensor, edge_index: torch.Tensor,
                  batch_size: int = 1024) -> torch.Tensor:
        """
        Inference on large graphs using mini-batching.

        Args:
            x_all: All node features
            edge_index: All edge indices
            batch_size: Batch size for inference

        Returns:
            Node embeddings for all nodes
        """
        # This is a simplified version - full implementation would use NeighborSampler
        return self.forward(x_all, edge_index)

    def reset_parameters(self):
        """Reset model parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
