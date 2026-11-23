"""
Graph Data Transforms

Provides transformation functions for graph data preprocessing.
"""

import torch
from torch_geometric.data import Data
from typing import Optional


class AddSelfLoops:
    """Add self-loops to graph."""

    def __call__(self, data: Data) -> Data:
        """Apply transform."""
        num_nodes = data.num_nodes
        self_loops = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
        data.edge_index = torch.cat([data.edge_index, self_loops], dim=1)
        return data


class NormalizeFeatures:
    """Normalize node features."""

    def __call__(self, data: Data) -> Data:
        """Apply transform."""
        row_sum = data.x.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1
        data.x = data.x / row_sum
        return data


class ToUndirected:
    """Convert graph to undirected."""

    def __call__(self, data: Data) -> Data:
        """Apply transform."""
        edge_index = data.edge_index
        reverse_edges = edge_index.flip(0)
        data.edge_index = torch.cat([edge_index, reverse_edges], dim=1)
        return data
