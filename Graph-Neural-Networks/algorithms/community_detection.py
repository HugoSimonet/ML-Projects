"""
Community Detection Algorithms

Implements various community detection algorithms for social networks.
"""

import torch
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set
import community as community_louvain


class LouvainCommunityDetection:
    """Louvain method for community detection."""

    def __init__(self, resolution: float = 1.0):
        """
        Initialize Louvain community detection.

        Args:
            resolution: Resolution parameter for modularity
        """
        self.resolution = resolution

    def detect(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_weights: torch.Tensor = None
    ) -> Dict[int, int]:
        """
        Detect communities using Louvain method.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            node_weights: Optional node weights

        Returns:
            Dictionary mapping node ID to community ID
        """
        # Convert to NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            G.add_edge(src, dst)

        # Run Louvain algorithm
        communities = community_louvain.best_partition(
            G,
            resolution=self.resolution
        )

        return communities

    def compute_modularity(
        self,
        edge_index: torch.Tensor,
        communities: Dict[int, int],
        num_nodes: int
    ) -> float:
        """
        Compute modularity of detected communities.

        Args:
            edge_index: Edge indices [2, num_edges]
            communities: Community assignments
            num_nodes: Number of nodes

        Returns:
            Modularity score
        """
        # Convert to NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            G.add_edge(src, dst)

        # Compute modularity
        community_sets = {}
        for node, comm_id in communities.items():
            if comm_id not in community_sets:
                community_sets[comm_id] = set()
            community_sets[comm_id].add(node)

        return nx.algorithms.community.modularity(G, community_sets.values())


class ModularityOptimizer:
    """Modularity optimization for community detection."""

    def __init__(self, max_iterations: int = 100):
        """
        Initialize modularity optimizer.

        Args:
            max_iterations: Maximum number of iterations
        """
        self.max_iterations = max_iterations

    def optimize(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        initial_communities: Dict[int, int] = None
    ) -> Tuple[Dict[int, int], float]:
        """
        Optimize community structure for maximum modularity.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            initial_communities: Initial community assignments

        Returns:
            Optimized communities and modularity score
        """
        # Initialize communities
        if initial_communities is None:
            communities = {i: i for i in range(num_nodes)}
        else:
            communities = initial_communities.copy()

        # Build adjacency list and compute degrees
        adj_list = [set() for _ in range(num_nodes)]
        degree = np.zeros(num_nodes)

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].add(dst)
            degree[src] += 1

        m = edge_index.size(1) / 2  # Number of edges (undirected)

        # Iterative optimization
        best_modularity = self._compute_modularity(communities, adj_list, degree, m)
        improved = True
        iterations = 0

        while improved and iterations < self.max_iterations:
            improved = False
            iterations += 1

            for node in range(num_nodes):
                current_comm = communities[node]
                best_comm = current_comm
                best_delta = 0

                # Try moving node to neighbor communities
                neighbor_comms = set()
                for neighbor in adj_list[node]:
                    neighbor_comms.add(communities[neighbor])

                for target_comm in neighbor_comms:
                    if target_comm == current_comm:
                        continue

                    # Compute modularity change
                    delta = self._compute_modularity_delta(
                        node, current_comm, target_comm,
                        communities, adj_list, degree, m
                    )

                    if delta > best_delta:
                        best_delta = delta
                        best_comm = target_comm

                # Move node if improvement found
                if best_comm != current_comm:
                    communities[node] = best_comm
                    improved = True

            current_modularity = self._compute_modularity(communities, adj_list, degree, m)
            best_modularity = current_modularity

        return communities, best_modularity

    def _compute_modularity(
        self,
        communities: Dict[int, int],
        adj_list: List[Set[int]],
        degree: np.ndarray,
        m: float
    ) -> float:
        """Compute modularity of community structure."""
        modularity = 0.0

        for node in range(len(adj_list)):
            comm_i = communities[node]

            for neighbor in adj_list[node]:
                comm_j = communities[neighbor]

                if comm_i == comm_j:
                    expected = degree[node] * degree[neighbor] / (2 * m)
                    modularity += 1 - expected

        return modularity / (2 * m)

    def _compute_modularity_delta(
        self,
        node: int,
        old_comm: int,
        new_comm: int,
        communities: Dict[int, int],
        adj_list: List[Set[int]],
        degree: np.ndarray,
        m: float
    ) -> float:
        """Compute change in modularity when moving node."""
        delta = 0.0

        for neighbor in adj_list[node]:
            neighbor_comm = communities[neighbor]

            # Remove contribution from old community
            if neighbor_comm == old_comm:
                expected = degree[node] * degree[neighbor] / (2 * m)
                delta -= (1 - expected)

            # Add contribution to new community
            if neighbor_comm == new_comm:
                expected = degree[node] * degree[neighbor] / (2 * m)
                delta += (1 - expected)

        return delta / (2 * m)


class HierarchicalClustering:
    """Hierarchical clustering for community detection."""

    def __init__(self, method: str = 'average'):
        """
        Initialize hierarchical clustering.

        Args:
            method: Linkage method ('single', 'average', 'complete')
        """
        self.method = method

    def detect(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_communities: int
    ) -> Dict[int, int]:
        """
        Detect communities using hierarchical clustering.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            num_communities: Desired number of communities

        Returns:
            Community assignments
        """
        # Convert to NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            G.add_edge(src, dst)

        # Compute shortest path distances
        distances = dict(nx.all_pairs_shortest_path_length(G))

        # Build distance matrix
        dist_matrix = np.full((num_nodes, num_nodes), num_nodes)
        for i in range(num_nodes):
            dist_matrix[i, i] = 0
            if i in distances:
                for j, dist in distances[i].items():
                    dist_matrix[i, j] = dist

        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster

        linkage_matrix = linkage(dist_matrix, method=self.method)
        clusters = fcluster(linkage_matrix, num_communities, criterion='maxclust')

        # Convert to dictionary
        communities = {i: int(clusters[i]) - 1 for i in range(num_nodes)}

        return communities


def detect_communities(
    edge_index: torch.Tensor,
    num_nodes: int,
    method: str = 'louvain',
    **kwargs
) -> Dict[int, int]:
    """
    Detect communities in graph.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        method: Detection method ('louvain', 'modularity', 'hierarchical')
        **kwargs: Additional arguments for detection method

    Returns:
        Community assignments
    """
    if method == 'louvain':
        detector = LouvainCommunityDetection(**kwargs)
        return detector.detect(edge_index, num_nodes)
    elif method == 'modularity':
        optimizer = ModularityOptimizer(**kwargs)
        communities, _ = optimizer.optimize(edge_index, num_nodes)
        return communities
    elif method == 'hierarchical':
        clusterer = HierarchicalClustering(**kwargs)
        num_communities = kwargs.get('num_communities', 10)
        return clusterer.detect(edge_index, num_nodes, num_communities)
    else:
        raise ValueError(f"Unknown detection method: {method}")
