"""
Graph Dataset Module

Provides dataset classes for loading and managing graph data.
"""

import os
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, KarateClub, CitationFull
from typing import Optional, List, Tuple


class GraphDataset(Dataset):
    """Base class for graph datasets."""

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[object] = None,
        pre_transform: Optional[object] = None
    ):
        """
        Initialize graph dataset.

        Args:
            root: Root directory for dataset
            name: Dataset name
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply once
        """
        self.name = name
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        return [f'{self.name}.pt']

    def download(self):
        """Download dataset."""
        pass

    def process(self):
        """Process dataset."""
        pass

    def len(self) -> int:
        """Return dataset size."""
        return 1

    def get(self, idx: int) -> Data:
        """Get graph data at index."""
        data = torch.load(os.path.join(self.processed_dir, f'{self.name}.pt'), weights_only=False)
        return data


class SocialNetworkDataset(Dataset):
    """Social network dataset class."""

    def __init__(
        self,
        name: str,
        root: str = 'data/',
        transform: Optional[object] = None,
        pre_transform: Optional[object] = None
    ):
        """
        Initialize social network dataset.

        Args:
            name: Dataset name ('facebook', 'twitter', 'karate', 'cora', 'citeseer', 'pubmed')
            root: Root directory for dataset
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply once
        """
        self.name = name
        self.dataset_name = name

        # Map dataset names to PyG datasets
        self.dataset_map = {
            'cora': ('Planetoid', 'Cora'),
            'citeseer': ('Planetoid', 'CiteSeer'),
            'pubmed': ('Planetoid', 'PubMed'),
            'karate': ('KarateClub', None)
        }

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        return [f'{self.name}.pt']

    def download(self):
        """Download dataset if needed."""
        if self.name in self.dataset_map:
            dataset_type, dataset_name = self.dataset_map[self.name]

            if dataset_type == 'Planetoid':
                dataset = Planetoid(root=self.raw_dir, name=dataset_name)
            elif dataset_type == 'KarateClub':
                dataset = KarateClub()
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            # Save to processed directory
            self._save_processed_data(dataset[0])
        else:
            # Create synthetic social network
            self._create_synthetic_network()

    def _create_synthetic_network(self):
        """Create synthetic social network."""
        np.random.seed(42)

        # Create a scale-free network using Barabási-Albert model
        num_nodes = 1000
        num_edges_attach = 3

        G = nx.barabasi_albert_graph(num_nodes, num_edges_attach)

        # Extract edge list
        edge_list = list(G.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Add reverse edges for undirected graph
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Generate node features (random)
        x = torch.randn(num_nodes, 128)

        # Generate labels (community structure)
        communities = nx.community.greedy_modularity_communities(G)
        y = torch.zeros(num_nodes, dtype=torch.long)
        for comm_id, community in enumerate(communities):
            for node in community:
                y[node] = comm_id

        # Create train/val/test masks
        num_train = int(0.6 * num_nodes)
        num_val = int(0.2 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[:num_train] = True
        val_mask[num_train:num_train + num_val] = True
        test_mask[num_train + num_val:] = True

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=len(communities)
        )

        self._save_processed_data(data)

    def _save_processed_data(self, data: Data):
        """Save processed data."""
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(data, os.path.join(self.processed_dir, f'{self.name}.pt'))

    def process(self):
        """Process dataset."""
        self.download()

    def len(self) -> int:
        """Return dataset size."""
        return 1

    def get(self, idx: int) -> Data:
        """Get graph data at index."""
        data = torch.load(os.path.join(self.processed_dir, f'{self.name}.pt'), weights_only=False)
        return data

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        data = self.get(0)
        if hasattr(data, 'num_classes'):
            return data.num_classes
        return len(torch.unique(data.y))

    @property
    def num_features(self) -> int:
        """Return number of node features."""
        data = self.get(0)
        return data.x.size(1)


def create_synthetic_social_network(
    num_nodes: int = 1000,
    num_communities: int = 10,
    inter_community_prob: float = 0.01,
    intra_community_prob: float = 0.3
) -> Data:
    """
    Create synthetic social network with community structure.

    Args:
        num_nodes: Number of nodes
        num_communities: Number of communities
        inter_community_prob: Probability of edges between communities
        intra_community_prob: Probability of edges within communities

    Returns:
        Graph data object
    """
    # Assign nodes to communities
    community_sizes = np.random.multinomial(
        num_nodes,
        [1 / num_communities] * num_communities
    )

    node_communities = []
    for comm_id, size in enumerate(community_sizes):
        node_communities.extend([comm_id] * size)

    np.random.shuffle(node_communities)
    node_communities = torch.tensor(node_communities, dtype=torch.long)

    # Generate edges based on community structure
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if node_communities[i] == node_communities[j]:
                # Same community
                if np.random.rand() < intra_community_prob:
                    edges.append([i, j])
            else:
                # Different communities
                if np.random.rand() < inter_community_prob:
                    edges.append([i, j])

    if len(edges) == 0:
        edges = [[0, 1]]  # Ensure at least one edge

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Add reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Generate node features
    x = torch.randn(num_nodes, 128)

    # Create masks
    num_train = int(0.6 * num_nodes)
    num_val = int(0.2 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:num_train] = True
    val_mask[num_train:num_train + num_val] = True
    test_mask[num_train + num_val:] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=node_communities,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_communities
    )

    return data
