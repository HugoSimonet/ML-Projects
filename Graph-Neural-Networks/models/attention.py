"""
Attention Mechanisms for Graph Neural Networks

Implements various attention mechanisms for GNNs including:
- Multi-head attention
- Graph attention
- Self-attention for nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as scatter_softmax
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for graph nodes."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor [num_nodes, embed_dim]
            key: Key tensor [num_nodes, embed_dim]
            value: Value tensor [num_nodes, embed_dim]
            attn_mask: Attention mask [num_nodes, num_nodes]

        Returns:
            Output tensor [num_nodes, embed_dim] and attention weights
        """
        batch_size = query.size(0)

        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        q = q * self.scaling
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.reshape(batch_size, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        return output, attn_weights


class GraphAttention(nn.Module):
    """Graph attention mechanism for node features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Initialize graph attention layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate or average heads
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Linear transformation per head
        self.W = nn.Parameter(torch.zeros(size=(num_heads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(num_heads, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of graph attention.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features and attention coefficients
        """
        num_nodes = x.size(0)
        edge_src, edge_dst = edge_index

        # Linear transformation for each head
        h = torch.matmul(x.unsqueeze(0).repeat(self.num_heads, 1, 1), self.W)
        # h shape: [num_heads, num_nodes, out_features]

        # Compute attention coefficients
        h_src = h[:, edge_src, :]  # [num_heads, num_edges, out_features]
        h_dst = h[:, edge_dst, :]  # [num_heads, num_edges, out_features]

        # Concatenate source and destination features
        h_cat = torch.cat([h_src, h_dst], dim=-1)  # [num_heads, num_edges, 2*out_features]

        # Compute attention scores
        e = torch.matmul(h_cat, self.a).squeeze(-1)  # [num_heads, num_edges]
        e = self.leakyrelu(e)

        # Normalize attention coefficients using softmax
        attention = torch.zeros(self.num_heads, num_nodes, num_nodes, device=x.device)
        attention[:, edge_src, edge_dst] = e

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)

        # Apply attention to features
        h_prime = torch.matmul(attention, h)  # [num_heads, num_nodes, out_features]

        if self.concat:
            # Concatenate heads
            h_prime = h_prime.transpose(0, 1).reshape(num_nodes, -1)
        else:
            # Average heads
            h_prime = h_prime.mean(dim=0)

        return h_prime, attention


class SelfAttention(nn.Module):
    """Self-attention mechanism for graph nodes."""

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize self-attention layer.

        Args:
            embed_dim: Embedding dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of self-attention.

        Args:
            x: Input features [num_nodes, embed_dim]

        Returns:
            Output features and attention weights
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        return out, attn
