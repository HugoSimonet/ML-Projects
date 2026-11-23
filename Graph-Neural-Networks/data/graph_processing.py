"""
Graph Processing Module

Provides tools for processing and transforming graph data.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from typing import List, Tuple, Optional
import networkx as nx


class GraphProcessor:
    """Graph data processor."""

    def __init__(self):
        """Initialize graph processor."""
        pass

    @staticmethod
    def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Add self-loops to graph.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes

        Returns:
            Edge indices with self-loops
        """
        self_loop_edges = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loop_edges], dim=1)
        return edge_index

    @staticmethod
    def normalize_features(x: torch.Tensor) -> torch.Tensor:
        """
        Normalize node features.

        Args:
            x: Node features [num_nodes, feature_dim]

        Returns:
            Normalized features
        """
        row_sum = x.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1  # Avoid division by zero
        return x / row_sum

    @staticmethod
    def compute_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute node degrees.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes

        Returns:
            Node degrees [num_nodes]
        """
        degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        for i in range(edge_index.size(1)):
            src = edge_index[0, i]
            degree[src] += 1
        return degree

    @staticmethod
    def extract_subgraph(
        subset: torch.Tensor,
        edge_index: torch.Tensor,
        relabel_nodes: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract subgraph induced by node subset.

        Args:
            subset: Node subset indices
            edge_index: Edge indices [2, num_edges]
            relabel_nodes: Whether to relabel nodes

        Returns:
            Subgraph edge indices and node mapping
        """
        return subgraph(subset, edge_index, relabel_nodes=relabel_nodes)

    @staticmethod
    def k_hop_neighbors(
        node_idx: int,
        edge_index: torch.Tensor,
        num_hops: int,
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract k-hop neighborhood.

        Args:
            node_idx: Central node index
            edge_index: Edge indices [2, num_edges]
            num_hops: Number of hops
            num_nodes: Total number of nodes

        Returns:
            Subset of nodes and subgraph edge indices
        """
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, num_hops, edge_index, num_nodes=num_nodes
        )
        return subset, sub_edge_index

    @staticmethod
    def create_negative_edges(
        edge_index: torch.Tensor,
        num_nodes: int,
        num_neg_samples: int
    ) -> torch.Tensor:
        """
        Create negative edge samples for link prediction.

        Args:
            edge_index: Positive edge indices [2, num_edges]
            num_nodes: Number of nodes
            num_neg_samples: Number of negative samples

        Returns:
            Negative edge indices [2, num_neg_samples]
        """
        # Create set of existing edges
        pos_edges = set()
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            pos_edges.add((src, dst))

        # Sample negative edges
        neg_edges = []
        while len(neg_edges) < num_neg_samples:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)

            if src != dst and (src, dst) not in pos_edges:
                neg_edges.append([src, dst])

        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t()
        return neg_edge_index

    @staticmethod
    def compute_pagerank(
        edge_index: torch.Tensor,
        num_nodes: int,
        alpha: float = 0.85,
        max_iter: int = 100
    ) -> torch.Tensor:
        """
        Compute PageRank scores.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            alpha: Damping factor
            max_iter: Maximum iterations

        Returns:
            PageRank scores [num_nodes]
        """
        # Convert to NetworkX graph
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            G.add_edge(src, dst)

        # Compute PageRank
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter)

        # Convert to tensor
        pagerank_tensor = torch.zeros(num_nodes)
        for node, score in pagerank.items():
            pagerank_tensor[node] = score

        return pagerank_tensor


class GraphSampler:
    """Graph sampling for mini-batch training."""

    def __init__(
        self,
        num_neighbors: List[int] = [25, 10],
        replace: bool = False
    ):
        """
        Initialize graph sampler.

        Args:
            num_neighbors: Number of neighbors to sample per layer
            replace: Whether to sample with replacement
        """
        self.num_neighbors = num_neighbors
        self.replace = replace

    def sample_neighbors(
        self,
        node_idx: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Sample neighbors for given nodes.

        Args:
            node_idx: Node indices to sample neighbors for
            edge_index: Edge indices [2, num_edges]
            num_nodes: Total number of nodes

        Returns:
            Sampled nodes, sampled edges, and batch sizes per layer
        """
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[dst].append(src)

        # Sample layer by layer
        current_nodes = node_idx.tolist()
        sampled_nodes = [current_nodes]
        batch_sizes = [len(current_nodes)]

        for num_sample in self.num_neighbors:
            next_nodes = []

            for node in current_nodes:
                neighbors = adj_list[node]

                if len(neighbors) <= num_sample:
                    sampled = neighbors
                else:
                    indices = np.random.choice(
                        len(neighbors),
                        num_sample,
                        replace=self.replace
                    )
                    sampled = [neighbors[i] for i in indices]

                next_nodes.extend(sampled)

            current_nodes = list(set(next_nodes))
            sampled_nodes.append(current_nodes)
            batch_sizes.append(len(current_nodes))

        # Flatten sampled nodes
        all_nodes = []
        for nodes in sampled_nodes:
            all_nodes.extend(nodes)
        all_nodes = list(set(all_nodes))

        # Extract subgraph
        node_mapping = {node: i for i, node in enumerate(all_nodes)}
        sampled_edges = []

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in node_mapping and dst in node_mapping:
                sampled_edges.append([node_mapping[src], node_mapping[dst]])

        if len(sampled_edges) == 0:
            sampled_edges = [[0, 0]]

        sampled_node_idx = torch.tensor(all_nodes, dtype=torch.long)
        sampled_edge_index = torch.tensor(sampled_edges, dtype=torch.long).t()

        return sampled_node_idx, sampled_edge_index, batch_sizes

    def random_walk_sample(
        self,
        start_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        walk_length: int = 10,
        num_walks: int = 5
    ) -> torch.Tensor:
        """
        Sample nodes using random walks.

        Args:
            start_nodes: Starting node indices
            edge_index: Edge indices [2, num_edges]
            num_nodes: Total number of nodes
            walk_length: Length of each random walk
            num_walks: Number of walks per start node

        Returns:
            Sampled node indices
        """
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)

        sampled_nodes = set()

        for start_node in start_nodes.tolist():
            for _ in range(num_walks):
                current = start_node
                sampled_nodes.add(current)

                for _ in range(walk_length):
                    neighbors = adj_list[current]
                    if len(neighbors) == 0:
                        break

                    # Random walk step
                    current = np.random.choice(neighbors)
                    sampled_nodes.add(current)

        return torch.tensor(list(sampled_nodes), dtype=torch.long)
