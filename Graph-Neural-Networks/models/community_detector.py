"""
Community Detection Model

GNN-based model for detecting communities in social networks.
Uses modularity optimization and graph clustering techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base_gnn import BaseGNN


class CommunityDetector(nn.Module):
    """GNN-based community detection model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_communities: int,
        gnn_type: str = 'gcn',
        num_layers: int = 3,
        dropout: float = 0.1,
        modularity_weight: float = 1.0
    ):
        """
        Initialize community detector.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_communities: Number of communities to detect
            gnn_type: Type of GNN backbone ('gcn', 'gat', 'graphsage')
            num_layers: Number of GNN layers
            dropout: Dropout rate
            modularity_weight: Weight for modularity loss
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_communities = num_communities
        self.modularity_weight = modularity_weight

        # Create GNN backbone
        from . import GCN, GAT, GraphSAGE

        if gnn_type == 'gcn':
            self.gnn = GCN(input_dim, hidden_dim, hidden_dim, num_layers, dropout)
        elif gnn_type == 'gat':
            self.gnn = GAT(input_dim, hidden_dim, hidden_dim, num_layers, dropout=dropout)
        elif gnn_type == 'graphsage':
            self.gnn = GraphSAGE(input_dim, hidden_dim, hidden_dim, num_layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Community assignment layer
        self.community_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_communities)
        )

        # Community embeddings
        self.community_embeddings = nn.Parameter(
            torch.randn(num_communities, hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of community detector.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes]

        Returns:
            Community logits [num_nodes, num_communities] and node embeddings
        """
        # Get node embeddings from GNN
        node_embeddings = self.gnn(x, edge_index, batch)

        # Predict community assignments
        community_logits = self.community_classifier(node_embeddings)

        return community_logits, node_embeddings

    def compute_modularity(
        self,
        community_assignments: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute modularity of community structure.

        Modularity Q = (1/2m) * sum_ij [A_ij - (k_i * k_j / 2m)] * delta(c_i, c_j)

        Args:
            community_assignments: Community assignments [num_nodes]
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes

        Returns:
            Modularity score
        """
        num_edges = edge_index.size(1)
        m = num_edges / 2  # Total number of edges (undirected)

        # Compute degree of each node
        degree = torch.zeros(num_nodes, device=edge_index.device)
        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            degree[src] += 1

        # Compute modularity
        modularity = 0.0
        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]

            # Check if nodes are in same community
            same_community = (community_assignments[src] == community_assignments[dst]).float()

            # A_ij = 1 (edge exists)
            # Expected edges: k_i * k_j / 2m
            expected = degree[src] * degree[dst] / (2 * m)

            modularity += (1 - expected) * same_community

        modularity /= (2 * m)

        return modularity

    def modularity_loss(
        self,
        community_logits: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute modularity-based loss.

        Args:
            community_logits: Community logits [num_nodes, num_communities]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Modularity loss
        """
        # Soft community assignments
        community_probs = F.softmax(community_logits, dim=-1)

        # Compute soft modularity
        num_nodes = community_logits.size(0)
        num_edges = edge_index.size(1)
        m = num_edges / 2

        # Compute degree
        degree = torch.zeros(num_nodes, device=edge_index.device)
        for i in range(num_edges):
            src = edge_index[0, i]
            degree[src] += 1

        # Soft modularity
        modularity = 0.0
        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]

            # Probability of being in same community
            same_community_prob = (community_probs[src] * community_probs[dst]).sum()

            # Expected edges
            expected = degree[src] * degree[dst] / (2 * m)

            modularity += (1 - expected) * same_community_prob

        modularity /= (2 * m)

        # Return negative modularity as loss (we want to maximize modularity)
        return -modularity

    def get_communities(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get community assignments for nodes.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Community assignments [num_nodes]
        """
        self.eval()
        with torch.no_grad():
            community_logits, _ = self.forward(x, edge_index)
            community_assignments = torch.argmax(community_logits, dim=-1)
        return community_assignments
