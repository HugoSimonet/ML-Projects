"""
Influence Maximization Model

GNN-based model for finding influential nodes in social networks.
Implements influence spread prediction and seed selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .base_gnn import BaseGNN


class InfluenceMaximizer(nn.Module):
    """GNN-based influence maximization model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        gnn_type: str = 'gat',
        num_layers: int = 3,
        dropout: float = 0.1,
        influence_threshold: float = 0.1
    ):
        """
        Initialize influence maximizer.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            gnn_type: Type of GNN backbone ('gcn', 'gat', 'graphsage')
            num_layers: Number of GNN layers
            dropout: Dropout rate
            influence_threshold: Threshold for influence propagation
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.influence_threshold = influence_threshold

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

        # Influence prediction head
        self.influence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Influence spread predictor
        self.spread_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of influence maximizer.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes]

        Returns:
            Influence scores [num_nodes, 1] and node embeddings
        """
        # Get node embeddings from GNN
        node_embeddings = self.gnn(x, edge_index, batch)

        # Predict influence scores
        influence_scores = self.influence_predictor(node_embeddings)

        return influence_scores, node_embeddings

    def predict_influence_spread(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        seed_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict influence spread from seed nodes.

        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            seed_nodes: Seed node indices [num_seeds]

        Returns:
            Influence probabilities for all nodes [num_nodes]
        """
        num_nodes = node_embeddings.size(0)
        influence_probs = torch.zeros(num_nodes, device=node_embeddings.device)

        # Initialize seed nodes with influence probability 1
        influence_probs[seed_nodes] = 1.0

        # Propagate influence through edges
        edge_src, edge_dst = edge_index

        for i in range(edge_index.size(1)):
            src, dst = edge_src[i], edge_dst[i]

            # Concatenate source and destination embeddings
            edge_feat = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=-1)

            # Predict influence spread probability
            spread_prob = self.spread_predictor(edge_feat.unsqueeze(0)).squeeze()

            # Update influence probability
            influence_probs[dst] = max(
                influence_probs[dst],
                influence_probs[src] * spread_prob
            )

        return influence_probs

    def select_influential_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        k: int,
        method: str = 'greedy'
    ) -> List[int]:
        """
        Select top-k influential nodes.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            k: Number of influential nodes to select
            method: Selection method ('greedy', 'degree', 'pagerank')

        Returns:
            List of influential node indices
        """
        self.eval()
        with torch.no_grad():
            influence_scores, node_embeddings = self.forward(x, edge_index)

            if method == 'greedy':
                # Greedy selection based on marginal influence spread
                selected = []
                for _ in range(k):
                    best_node = None
                    best_spread = -1

                    for node in range(x.size(0)):
                        if node in selected:
                            continue

                        # Compute marginal influence spread
                        seeds = torch.tensor(selected + [node], device=x.device)
                        spread = self.predict_influence_spread(
                            node_embeddings, edge_index, seeds
                        ).sum().item()

                        if spread > best_spread:
                            best_spread = spread
                            best_node = node

                    if best_node is not None:
                        selected.append(best_node)

                return selected

            elif method == 'degree':
                # Select nodes with highest degree
                degree = torch.zeros(x.size(0), device=x.device)
                for i in range(edge_index.size(1)):
                    src = edge_index[0, i]
                    degree[src] += 1

                _, top_nodes = torch.topk(degree, k)
                return top_nodes.tolist()

            elif method == 'pagerank':
                # Select nodes with highest influence scores
                _, top_nodes = torch.topk(influence_scores.squeeze(), k)
                return top_nodes.tolist()

            else:
                raise ValueError(f"Unknown selection method: {method}")

    def compute_influence_loss(
        self,
        influence_scores: torch.Tensor,
        true_influence: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute influence prediction loss.

        Args:
            influence_scores: Predicted influence scores [num_nodes, 1]
            true_influence: Ground truth influence [num_nodes, 1]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Influence loss
        """
        # MSE loss for influence scores
        mse_loss = F.mse_loss(influence_scores, true_influence)

        # Ranking loss to preserve order of influence
        num_nodes = influence_scores.size(0)
        ranking_loss = 0.0

        for i in range(num_nodes):
            for j in range(num_nodes):
                if true_influence[i] > true_influence[j]:
                    # Node i should have higher influence than node j
                    ranking_loss += F.relu(
                        influence_scores[j] - influence_scores[i] + 0.1
                    )

        ranking_loss /= (num_nodes * num_nodes)

        # Total loss
        total_loss = mse_loss + 0.1 * ranking_loss

        return total_loss
