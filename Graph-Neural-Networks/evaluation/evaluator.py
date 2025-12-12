"""
Graph Evaluator

Provides comprehensive evaluation for graph neural network models.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Dict, Optional, List
from .graph_metrics import (
    compute_node_classification_metrics,
    compute_link_prediction_metrics,
    compute_community_detection_metrics,
    compute_influence_metrics
)


class GraphEvaluator:
    """Comprehensive evaluator for GNN models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu'
    ):
        """
        Initialize graph evaluator.

        Args:
            model: GNN model to evaluate
            device: Device to use for evaluation
        """
        self.model = model
        self.device = device
        self.model.to(device)

    def evaluate_node_classification(
        self,
        data: Data,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Evaluate node classification performance.

        Args:
            data: Graph data
            mask: Mask for evaluation nodes

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Forward pass
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)

            # Apply mask if provided
            if mask is None:
                mask = torch.ones(data.num_nodes, dtype=torch.bool)

            # Get predictions
            pred = out[mask].argmax(dim=-1)
            true = data.y[mask]

            # Compute probabilities
            prob = torch.softmax(out[mask], dim=-1)

            # Compute metrics
            metrics = compute_node_classification_metrics(
                true, pred, prob
            )

        return metrics

    def evaluate_link_prediction(
        self,
        data: Data,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate link prediction performance.

        Args:
            data: Graph data
            pos_edges: Positive edge samples [2, num_pos]
            neg_edges: Negative edge samples [2, num_neg]

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Forward pass
            data = data.to(self.device)
            pos_edges = pos_edges.to(self.device)
            neg_edges = neg_edges.to(self.device)

            embeddings = self.model(data.x, data.edge_index)

            # Compute edge scores
            pos_scores = self._compute_edge_scores(embeddings, pos_edges)
            neg_scores = self._compute_edge_scores(embeddings, neg_edges)

            # Combine scores and labels
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(pos_scores.size(0)),
                torch.zeros(neg_scores.size(0))
            ])

            # Threshold predictions
            threshold = 0.5
            predictions = (scores > threshold).float()

            # Compute metrics
            metrics = compute_link_prediction_metrics(
                labels, predictions, scores
            )

        return metrics

    def evaluate_community_detection(
        self,
        data: Data,
        community_detector: nn.Module,
        true_communities: Optional[Dict[int, int]] = None
    ) -> Dict[str, float]:
        """
        Evaluate community detection performance.

        Args:
            data: Graph data
            community_detector: Community detection model
            true_communities: Ground truth communities (optional)

        Returns:
            Dictionary of metrics
        """
        community_detector.eval()

        with torch.no_grad():
            # Forward pass
            data = data.to(self.device)
            community_logits, _ = community_detector(data.x, data.edge_index)

            # Get community assignments
            community_assignments = community_logits.argmax(dim=-1)

            # Convert to dictionary
            communities_pred = {
                i: community_assignments[i].item()
                for i in range(data.num_nodes)
            }

            # Compute metrics
            metrics = compute_community_detection_metrics(
                communities_pred,
                true_communities,
                data.edge_index,
                data.num_nodes
            )

        return metrics

    def evaluate_influence_maximization(
        self,
        data: Data,
        influence_model: nn.Module,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate influence maximization performance.

        Args:
            data: Graph data
            influence_model: Influence maximization model
            k: Number of seed nodes to select

        Returns:
            Dictionary of metrics
        """
        influence_model.eval()

        with torch.no_grad():
            # Forward pass
            data = data.to(self.device)
            seed_nodes = influence_model.select_influential_nodes(
                data.x, data.edge_index, k
            )

            # Simulate influence spread
            from ..algorithms import IndependentCascadeModel

            cascade_model = IndependentCascadeModel(
                data.edge_index, data.num_nodes
            )

            avg_spread = cascade_model.simulate(seed_nodes, num_simulations=100)

            # Find influenced nodes (single simulation for metric computation)
            influenced = cascade_model._single_simulation(seed_nodes)

            # Compute metrics
            metrics = compute_influence_metrics(
                seed_nodes,
                list(influenced),
                data.edge_index,
                data.num_nodes
            )

            metrics['simulated_spread'] = avg_spread

        return metrics

    def _compute_edge_scores(
        self,
        embeddings: torch.Tensor,
        edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge scores from node embeddings.

        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            edges: Edge indices [2, num_edges]

        Returns:
            Edge scores [num_edges]
        """
        # Get source and destination embeddings
        src_embeddings = embeddings[edges[0]]
        dst_embeddings = embeddings[edges[1]]

        # Compute similarity (dot product)
        scores = (src_embeddings * dst_embeddings).sum(dim=-1)

        # Apply sigmoid to get probabilities
        scores = torch.sigmoid(scores)

        return scores

    def evaluate_all_tasks(
        self,
        data: Data,
        tasks: List[str] = None,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on multiple tasks.

        Args:
            data: Graph data
            tasks: List of tasks to evaluate
            **kwargs: Additional arguments for specific tasks

        Returns:
            Dictionary of metrics for each task
        """
        if tasks is None:
            tasks = ['node_classification']

        results = {}

        if 'node_classification' in tasks and hasattr(data, 'test_mask'):
            results['node_classification'] = self.evaluate_node_classification(
                data, data.test_mask
            )

        if 'link_prediction' in tasks:
            pos_edges = kwargs.get('pos_edges')
            neg_edges = kwargs.get('neg_edges')
            if pos_edges is not None and neg_edges is not None:
                results['link_prediction'] = self.evaluate_link_prediction(
                    data, pos_edges, neg_edges
                )

        if 'community_detection' in tasks:
            community_detector = kwargs.get('community_detector')
            if community_detector is not None:
                results['community_detection'] = self.evaluate_community_detection(
                    data, community_detector,
                    kwargs.get('true_communities')
                )

        if 'influence_maximization' in tasks:
            influence_model = kwargs.get('influence_model')
            if influence_model is not None:
                results['influence_maximization'] = self.evaluate_influence_maximization(
                    data, influence_model,
                    kwargs.get('k', 10)
                )

        return results

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data: Data,
        task: str = 'node_classification',
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on a task.

        Args:
            models: Dictionary of model name to model
            data: Graph data
            task: Task to evaluate
            **kwargs: Additional arguments for evaluation

        Returns:
            Dictionary of metrics for each model
        """
        results = {}

        for model_name, model in models.items():
            # Temporarily set model
            original_model = self.model
            self.model = model

            # Evaluate
            if task == 'node_classification':
                mask = kwargs.get('mask', data.test_mask if hasattr(data, 'test_mask') else None)
                results[model_name] = self.evaluate_node_classification(data, mask)
            elif task == 'link_prediction':
                results[model_name] = self.evaluate_link_prediction(
                    data,
                    kwargs['pos_edges'],
                    kwargs['neg_edges']
                )

            # Restore original model
            self.model = original_model

        return results
