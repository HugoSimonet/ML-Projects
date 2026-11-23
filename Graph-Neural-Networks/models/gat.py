"""
Graph Attention Network (GAT)

Implementation of the Graph Attention Network from:
"Graph Attention Networks" (Veličković et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple

from .base_gnn import BaseGNN


class GATLayer(nn.Module):
    """Single Graph Attention Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        concat: bool = True,
        dropout: float = 0.1,
        alpha: float = 0.2,
        bias: bool = True
    ):
        """
        Initialize GAT layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_heads: Number of attention heads
            concat: Whether to concatenate or average heads
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
            bias: Whether to use bias
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha

        # Linear transformations
        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention parameters
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, num_heads)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if bias and concat:
            self.bias = nn.Parameter(torch.zeros(num_heads * out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of GAT layer.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features and attention weights
        """
        num_nodes = x.size(0)

        # Linear transformation
        h = torch.mm(x, self.W)
        h = h.view(-1, self.num_heads, self.out_features)

        # Compute attention scores
        edge_src, edge_dst = edge_index
        h_src = h[edge_src]  # [num_edges, num_heads, out_features]
        h_dst = h[edge_dst]  # [num_edges, num_heads, out_features]

        # Attention mechanism
        h_cat = torch.cat([h_src, h_dst], dim=-1)  # [num_edges, num_heads, 2*out_features]
        e = torch.einsum('ehf,fh->eh', h_cat, self.a)
        e = self.leakyrelu(e)

        # Softmax per node
        attention = torch.zeros(num_nodes, self.num_heads, device=x.device)
        attention_weights = torch.zeros_like(e)

        for node in range(num_nodes):
            mask = edge_dst == node
            if mask.any():
                node_attention = F.softmax(e[mask], dim=0)
                attention_weights[mask] = node_attention

        attention = self.dropout_layer(attention_weights)

        # Apply attention
        h_prime = torch.zeros(num_nodes, self.num_heads, self.out_features, device=x.device)
        for i in range(edge_index.size(1)):
            src, dst = edge_src[i], edge_dst[i]
            h_prime[dst] += attention[i].unsqueeze(-1) * h[src]

        if self.concat:
            h_prime = h_prime.view(-1, self.num_heads * self.out_features)
        else:
            h_prime = h_prime.mean(dim=1)

        if self.bias is not None:
            h_prime += self.bias

        return h_prime, attention


class GAT(BaseGNN):
    """Graph Attention Network model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = 'elu',
        alpha: float = 0.2,
        concat_heads: bool = True
    ):
        """
        Initialize GAT model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            activation: Activation function name
            alpha: LeakyReLU negative slope
            concat_heads: Whether to concatenate attention heads
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, activation)

        self.num_heads = num_heads
        self.alpha = alpha
        self.concat_heads = concat_heads

        # GAT layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=concat_heads
            )
        )

        # Hidden layers
        in_dim = hidden_dim * num_heads if concat_heads else hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=concat_heads
                )
            )

        # Output layer (average heads)
        self.convs.append(
            GATConv(
                in_dim,
                output_dim,
                heads=1,
                dropout=dropout,
                concat=False
            )
        )

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(in_dim) for _ in range(num_layers - 1)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of GAT.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes]
            return_attention_weights: Whether to return attention weights

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        attention_weights = []

        for i, conv in enumerate(self.convs[:-1]):
            if return_attention_weights:
                x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_idx, attn))
            else:
                x = conv(x, edge_index)

            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout_layer(x)

        # Final layer
        if return_attention_weights:
            x, (edge_idx, attn) = self.convs[-1](x, edge_index, return_attention_weights=True)
            attention_weights.append((edge_idx, attn))
            return x, attention_weights
        else:
            x = self.convs[-1](x, edge_index)
            return x

    def reset_parameters(self):
        """Reset model parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
