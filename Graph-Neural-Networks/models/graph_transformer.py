"""
Graph Transformer

Implementation of Graph Transformer for capturing long-range dependencies in graphs.
Combines graph structure with transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_gnn import BaseGNN
from .attention import MultiHeadAttention


class TransformerLayer(nn.Module):
    """Single Graph Transformer Layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Initialize Graph Transformer layer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            dropout: Dropout rate
            activation: Activation function name
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Feed-forward network
        self.ff_net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of transformer layer.

        Args:
            x: Node features [num_nodes, embed_dim]
            edge_index: Edge indices (optional, for graph bias)
            attn_mask: Attention mask

        Returns:
            Updated node features [num_nodes, embed_dim]
        """
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask)
        x = residual + self.dropout(attn_out)

        # Feed-forward network with residual connection
        residual = x
        x = self.norm2(x)
        ff_out = self.ff_net(x)
        x = residual + ff_out

        return x


class GraphTransformer(BaseGNN):
    """Graph Transformer model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_graph_bias: bool = True
    ):
        """
        Initialize Graph Transformer model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            dropout: Dropout rate
            activation: Activation function name
            use_graph_bias: Whether to use graph structure bias in attention
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, activation)

        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.use_graph_bias = use_graph_bias

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ff_dim, dropout, activation)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        # Graph structure bias (optional)
        if use_graph_bias:
            self.graph_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Graph Transformer.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        num_nodes = x.size(0)

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        if num_nodes <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :num_nodes, :].squeeze(0)
        else:
            # Interpolate positional encoding for larger graphs
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=num_nodes,
                mode='linear'
            ).transpose(1, 2).squeeze(0)
            x = x + pos_enc

        # Create attention mask from graph structure
        attn_mask = None
        if self.use_graph_bias and edge_index is not None:
            attn_mask = self._create_attention_mask(edge_index, num_nodes)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, edge_index, attn_mask)

        # Output projection
        x = self.output_proj(x)

        return x

    def _create_attention_mask(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Create attention mask based on graph structure.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes

        Returns:
            Attention mask [num_nodes, num_nodes]
        """
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1

        # Add self-loops
        adj = adj + torch.eye(num_nodes, device=edge_index.device)

        # Convert to attention mask (0 for allowed, -inf for masked)
        mask = torch.where(adj > 0, torch.zeros_like(adj), torch.full_like(adj, float('-inf')))

        return mask

    def reset_parameters(self):
        """Reset model parameters."""
        self.input_proj.reset_parameters()
        nn.init.normal_(self.pos_encoding, std=0.02)
        for layer in self.transformer_layers:
            for module in layer.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        for module in self.output_proj.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        if self.use_graph_bias:
            nn.init.zeros_(self.graph_bias)
