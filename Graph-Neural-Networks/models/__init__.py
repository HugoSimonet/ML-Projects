"""
Graph Neural Network Models

This module contains implementations of various GNN architectures:
- GCN (Graph Convolutional Networks)
- GAT (Graph Attention Networks)
- GraphSAGE (Inductive representation learning)
- Graph Transformer (Long-range dependencies)
"""

from .gcn import GCN, GCNLayer
from .gat import GAT, GATLayer
from .graphsage import GraphSAGE, SAGEConv
from .graph_transformer import GraphTransformer, TransformerLayer
from .attention import MultiHeadAttention, GraphAttention
from .base_gnn import BaseGNN
from .community_detector import CommunityDetector
from .influence_maximizer import InfluenceMaximizer

__all__ = [
    'GCN', 'GCNLayer',
    'GAT', 'GATLayer',
    'GraphSAGE', 'SAGEConv',
    'GraphTransformer', 'TransformerLayer',
    'MultiHeadAttention', 'GraphAttention',
    'BaseGNN',
    'CommunityDetector',
    'InfluenceMaximizer'
]
