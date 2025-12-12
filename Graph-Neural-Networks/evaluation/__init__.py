"""
Evaluation Module

Provides evaluation metrics and tools for graph neural networks.
"""

from .graph_metrics import (
    compute_node_classification_metrics,
    compute_link_prediction_metrics,
    compute_community_detection_metrics,
    compute_influence_metrics
)
from .evaluator import GraphEvaluator

__all__ = [
    'compute_node_classification_metrics',
    'compute_link_prediction_metrics',
    'compute_community_detection_metrics',
    'compute_influence_metrics',
    'GraphEvaluator'
]
