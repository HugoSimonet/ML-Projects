"""
Graph Metrics

Implements various evaluation metrics for graph tasks.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Dict, List, Tuple
import networkx as nx


def compute_node_classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_prob: torch.Tensor = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute node classification metrics.

    Args:
        y_true: True labels [num_nodes]
        y_pred: Predicted labels [num_nodes]
        y_prob: Prediction probabilities [num_nodes, num_classes]
        average: Averaging method for multi-class metrics

    Returns:
        Dictionary of metrics
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    metrics = {
        'accuracy': accuracy_score(y_true_np, y_pred_np),
        'f1_score': f1_score(y_true_np, y_pred_np, average=average, zero_division=0),
        'precision': precision_score(y_true_np, y_pred_np, average=average, zero_division=0),
        'recall': recall_score(y_true_np, y_pred_np, average=average, zero_division=0)
    }

    # Macro and micro F1
    metrics['f1_macro'] = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true_np, y_pred_np, average='micro', zero_division=0)

    # AUC-ROC if probabilities provided
    if y_prob is not None:
        y_prob_np = y_prob.cpu().numpy()
        num_classes = y_prob.size(1)

        if num_classes == 2:
            # Binary classification
            metrics['auc_roc'] = roc_auc_score(y_true_np, y_prob_np[:, 1])
            metrics['auc_pr'] = average_precision_score(y_true_np, y_prob_np[:, 1])
        else:
            # Multi-class classification
            try:
                metrics['auc_roc'] = roc_auc_score(
                    y_true_np, y_prob_np,
                    multi_class='ovr', average=average
                )
            except:
                metrics['auc_roc'] = 0.0

    return metrics


def compute_link_prediction_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_score: torch.Tensor = None,
    k: int = 10
) -> Dict[str, float]:
    """
    Compute link prediction metrics.

    Args:
        y_true: True edge labels [num_edges]
        y_pred: Predicted edge labels [num_edges]
        y_score: Edge scores [num_edges]
        k: Top-k for hit rate calculation

    Returns:
        Dictionary of metrics
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    metrics = {
        'accuracy': accuracy_score(y_true_np, y_pred_np),
        'f1_score': f1_score(y_true_np, y_pred_np, zero_division=0),
        'precision': precision_score(y_true_np, y_pred_np, zero_division=0),
        'recall': recall_score(y_true_np, y_pred_np, zero_division=0)
    }

    # AUC metrics if scores provided
    if y_score is not None:
        y_score_np = y_score.cpu().numpy()
        metrics['auc_roc'] = roc_auc_score(y_true_np, y_score_np)
        metrics['auc_pr'] = average_precision_score(y_true_np, y_score_np)

        # Hit Rate @ K
        metrics[f'hit_rate@{k}'] = compute_hit_rate(y_true_np, y_score_np, k)

        # Mean Reciprocal Rank
        metrics['mrr'] = compute_mrr(y_true_np, y_score_np)

    return metrics


def compute_community_detection_metrics(
    communities_pred: Dict[int, int],
    communities_true: Dict[int, int] = None,
    edge_index: torch.Tensor = None,
    num_nodes: int = None
) -> Dict[str, float]:
    """
    Compute community detection metrics.

    Args:
        communities_pred: Predicted community assignments
        communities_true: True community assignments (optional)
        edge_index: Edge indices for modularity computation
        num_nodes: Number of nodes

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Modularity
    if edge_index is not None and num_nodes is not None:
        metrics['modularity'] = compute_modularity(
            communities_pred, edge_index, num_nodes
        )

    # Coverage
    if edge_index is not None:
        metrics['coverage'] = compute_coverage(communities_pred, edge_index)

    # Conductance
    if edge_index is not None and num_nodes is not None:
        metrics['conductance'] = compute_conductance(
            communities_pred, edge_index, num_nodes
        )

    # NMI if ground truth available
    if communities_true is not None:
        metrics['nmi'] = compute_nmi(communities_pred, communities_true)

    # Number of communities
    metrics['num_communities'] = len(set(communities_pred.values()))

    # Community size statistics
    community_sizes = {}
    for node, comm in communities_pred.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1

    sizes = list(community_sizes.values())
    metrics['avg_community_size'] = np.mean(sizes)
    metrics['max_community_size'] = np.max(sizes)
    metrics['min_community_size'] = np.min(sizes)

    return metrics


def compute_influence_metrics(
    seed_set: List[int],
    influenced_nodes: List[int],
    edge_index: torch.Tensor,
    num_nodes: int
) -> Dict[str, float]:
    """
    Compute influence maximization metrics.

    Args:
        seed_set: Selected seed nodes
        influenced_nodes: Nodes influenced by seed set
        edge_index: Edge indices
        num_nodes: Total number of nodes

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'influence_spread': len(influenced_nodes),
        'coverage': len(influenced_nodes) / num_nodes,
        'influence_per_seed': len(influenced_nodes) / len(seed_set) if len(seed_set) > 0 else 0,
        'num_seeds': len(seed_set)
    }

    # Seed set quality (average degree of seed nodes)
    degree = torch.zeros(num_nodes)
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        degree[src] += 1

    seed_degrees = [degree[node].item() for node in seed_set]
    metrics['avg_seed_degree'] = np.mean(seed_degrees) if seed_degrees else 0
    metrics['seed_degree_std'] = np.std(seed_degrees) if seed_degrees else 0

    return metrics


def compute_modularity(
    communities: Dict[int, int],
    edge_index: torch.Tensor,
    num_nodes: int
) -> float:
    """
    Compute modularity of community structure.

    Args:
        communities: Community assignments
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes

    Returns:
        Modularity score
    """
    m = edge_index.size(1) / 2  # Number of edges (undirected)

    # Compute degree
    degree = torch.zeros(num_nodes)
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        degree[src] += 1

    # Compute modularity
    modularity = 0.0
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()

        if src in communities and dst in communities:
            if communities[src] == communities[dst]:
                expected = degree[src].item() * degree[dst].item() / (2 * m)
                modularity += 1 - expected

    modularity /= (2 * m)

    return modularity


def compute_coverage(
    communities: Dict[int, int],
    edge_index: torch.Tensor
) -> float:
    """
    Compute coverage (fraction of edges within communities).

    Args:
        communities: Community assignments
        edge_index: Edge indices [2, num_edges]

    Returns:
        Coverage score
    """
    intra_edges = 0
    total_edges = edge_index.size(1)

    for i in range(total_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()

        if src in communities and dst in communities:
            if communities[src] == communities[dst]:
                intra_edges += 1

    return intra_edges / total_edges if total_edges > 0 else 0.0


def compute_conductance(
    communities: Dict[int, int],
    edge_index: torch.Tensor,
    num_nodes: int
) -> float:
    """
    Compute average conductance of communities.

    Args:
        communities: Community assignments
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes

    Returns:
        Average conductance
    """
    # Group nodes by community
    community_nodes = {}
    for node, comm in communities.items():
        if comm not in community_nodes:
            community_nodes[comm] = []
        community_nodes[comm].append(node)

    # Compute conductance for each community
    conductances = []

    for comm, nodes in community_nodes.items():
        node_set = set(nodes)
        cut_edges = 0
        volume = 0

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()

            if src in node_set:
                volume += 1
                if dst not in node_set:
                    cut_edges += 1

        if volume > 0:
            conductance = cut_edges / volume
            conductances.append(conductance)

    return np.mean(conductances) if conductances else 0.0


def compute_nmi(
    communities_pred: Dict[int, int],
    communities_true: Dict[int, int]
) -> float:
    """
    Compute Normalized Mutual Information between predicted and true communities.

    Args:
        communities_pred: Predicted community assignments
        communities_true: True community assignments

    Returns:
        NMI score
    """
    from sklearn.metrics import normalized_mutual_info_score

    # Align node indices
    nodes = sorted(set(communities_pred.keys()) & set(communities_true.keys()))

    pred_labels = [communities_pred[node] for node in nodes]
    true_labels = [communities_true[node] for node in nodes]

    return normalized_mutual_info_score(true_labels, pred_labels)


def compute_hit_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int
) -> float:
    """
    Compute Hit Rate @ K.

    Args:
        y_true: True labels
        y_score: Prediction scores
        k: Top-k threshold

    Returns:
        Hit rate
    """
    # Get top-k predictions
    top_k_indices = np.argsort(y_score)[-k:]

    # Check if any positive sample is in top-k
    hits = np.any(y_true[top_k_indices] == 1)

    return float(hits)


def compute_mrr(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        y_true: True labels
        y_score: Prediction scores

    Returns:
        MRR score
    """
    # Sort by score
    sorted_indices = np.argsort(y_score)[::-1]

    # Find rank of first positive sample
    for rank, idx in enumerate(sorted_indices, 1):
        if y_true[idx] == 1:
            return 1.0 / rank

    return 0.0
