"""
Graph Data Loader

Provides data loaders for efficient graph batching and sampling.
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader, DataLoader
from typing import List, Optional


class GraphDataLoader:
    """Graph data loader with neighbor sampling."""

    def __init__(
        self,
        data: Data,
        batch_size: int = 32,
        num_neighbors: List[int] = [25, 10],
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """
        Initialize graph data loader.

        Args:
            data: Graph data object
            batch_size: Batch size
            num_neighbors: Number of neighbors to sample per layer
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
        """
        self.data = data
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Create neighbor sampler
        self.loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)

    def __len__(self):
        """Return number of batches."""
        return len(self.loader)


class MultiGraphDataLoader:
    """Data loader for multiple graphs."""

    def __init__(
        self,
        graphs: List[Data],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """
        Initialize multi-graph data loader.

        Args:
            graphs: List of graph data objects
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
        """
        self.graphs = graphs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Create PyG data loader
        self.loader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)

    def __len__(self):
        """Return number of batches."""
        return len(self.loader)


class LinkPredictionDataLoader:
    """Data loader for link prediction task."""

    def __init__(
        self,
        data: Data,
        batch_size: int = 32,
        neg_sampling_ratio: float = 1.0,
        shuffle: bool = True
    ):
        """
        Initialize link prediction data loader.

        Args:
            data: Graph data object
            batch_size: Batch size
            neg_sampling_ratio: Ratio of negative to positive samples
            shuffle: Whether to shuffle data
        """
        self.data = data
        self.batch_size = batch_size
        self.neg_sampling_ratio = neg_sampling_ratio
        self.shuffle = shuffle

        # Split edges into training and test
        num_edges = data.edge_index.size(1) // 2  # Undirected edges
        perm = torch.randperm(num_edges)

        num_test = int(0.1 * num_edges)
        num_val = int(0.05 * num_edges)

        test_indices = perm[:num_test]
        val_indices = perm[num_test:num_test + num_val]
        train_indices = perm[num_test + num_val:]

        # Extract edge subsets
        self.train_edges = data.edge_index[:, train_indices]
        self.val_edges = data.edge_index[:, val_indices]
        self.test_edges = data.edge_index[:, test_indices]

    def get_train_loader(self):
        """Get training edge loader."""
        return self._create_edge_loader(self.train_edges)

    def get_val_loader(self):
        """Get validation edge loader."""
        return self._create_edge_loader(self.val_edges)

    def get_test_loader(self):
        """Get test edge loader."""
        return self._create_edge_loader(self.test_edges)

    def _create_edge_loader(self, edges: torch.Tensor):
        """Create edge batch loader."""
        num_edges = edges.size(1)
        indices = torch.randperm(num_edges) if self.shuffle else torch.arange(num_edges)

        for i in range(0, num_edges, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_edges = edges[:, batch_indices]

            # Sample negative edges
            num_neg = int(batch_indices.size(0) * self.neg_sampling_ratio)
            neg_edges = self._sample_negative_edges(num_neg)

            yield batch_edges, neg_edges

    def _sample_negative_edges(self, num_samples: int) -> torch.Tensor:
        """Sample negative edges."""
        num_nodes = self.data.num_nodes

        # Create set of existing edges
        edge_set = set()
        for i in range(self.data.edge_index.size(1)):
            src, dst = self.data.edge_index[0, i].item(), self.data.edge_index[1, i].item()
            edge_set.add((src, dst))

        # Sample negative edges
        neg_edges = []
        while len(neg_edges) < num_samples:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()

            if src != dst and (src, dst) not in edge_set:
                neg_edges.append([src, dst])

        return torch.tensor(neg_edges, dtype=torch.long).t()
