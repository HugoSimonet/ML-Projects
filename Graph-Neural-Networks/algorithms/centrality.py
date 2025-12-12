"""
Centrality Measures

Implements various centrality measures for graph analysis.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict


def compute_pagerank(
    edge_index: torch.Tensor,
    num_nodes: int,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> torch.Tensor:
    """
    Compute PageRank centrality.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        alpha: Damping factor
        max_iter: Maximum iterations
        tol: Convergence tolerance

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
    pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)

    # Convert to tensor
    pagerank_tensor = torch.zeros(num_nodes)
    for node, score in pagerank.items():
        pagerank_tensor[node] = score

    return pagerank_tensor


def compute_betweenness(
    edge_index: torch.Tensor,
    num_nodes: int,
    normalized: bool = True
) -> torch.Tensor:
    """
    Compute betweenness centrality.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        normalized: Whether to normalize scores

    Returns:
        Betweenness centrality scores [num_nodes]
    """
    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)

    # Compute betweenness
    betweenness = nx.betweenness_centrality(G, normalized=normalized)

    # Convert to tensor
    betweenness_tensor = torch.zeros(num_nodes)
    for node, score in betweenness.items():
        betweenness_tensor[node] = score

    return betweenness_tensor


def compute_closeness(
    edge_index: torch.Tensor,
    num_nodes: int
) -> torch.Tensor:
    """
    Compute closeness centrality.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes

    Returns:
        Closeness centrality scores [num_nodes]
    """
    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)

    # Compute closeness
    closeness = nx.closeness_centrality(G)

    # Convert to tensor
    closeness_tensor = torch.zeros(num_nodes)
    for node, score in closeness.items():
        closeness_tensor[node] = score

    return closeness_tensor


def compute_degree_centrality(
    edge_index: torch.Tensor,
    num_nodes: int,
    normalized: bool = True
) -> torch.Tensor:
    """
    Compute degree centrality.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        normalized: Whether to normalize by (num_nodes - 1)

    Returns:
        Degree centrality scores [num_nodes]
    """
    # Compute degree
    degree = torch.zeros(num_nodes, dtype=torch.float)

    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        degree[src] += 1

    # Normalize
    if normalized and num_nodes > 1:
        degree = degree / (num_nodes - 1)

    return degree


def compute_eigenvector_centrality(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_iter: int = 100,
    tol: float = 1e-6
) -> torch.Tensor:
    """
    Compute eigenvector centrality.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Eigenvector centrality scores [num_nodes]
    """
    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)

    try:
        # Compute eigenvector centrality
        eigenvector = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)

        # Convert to tensor
        eigenvector_tensor = torch.zeros(num_nodes)
        for node, score in eigenvector.items():
            eigenvector_tensor[node] = score

        return eigenvector_tensor
    except:
        # Return uniform scores if computation fails
        return torch.ones(num_nodes) / num_nodes
