"""
Data splitting utilities for federated learning
Implements IID and various non-IID data distribution strategies
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class IIDDataSplitter:
    """
    IID (Independent and Identically Distributed) data splitter

    Splits data evenly and randomly across clients
    """

    def __init__(self, dataset: Dataset, num_clients: int, seed: int = 42):
        """
        Initialize IID data splitter

        Args:
            dataset: PyTorch dataset to split
            num_clients: Number of clients
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed
        np.random.seed(seed)

        logger.info(f"Initialized IID splitter: {num_clients} clients, "
                   f"dataset size: {len(dataset)}")

    def split(self) -> List[Subset]:
        """
        Split dataset into IID subsets

        Returns:
            List of dataset subsets, one per client
        """
        # Shuffle indices
        indices = np.random.permutation(len(self.dataset))

        # Split evenly
        client_indices = np.array_split(indices, self.num_clients)

        # Create subsets
        client_datasets = [
            Subset(self.dataset, indices.tolist())
            for indices in client_indices
        ]

        # Log statistics
        sizes = [len(subset) for subset in client_datasets]
        logger.info(f"IID split: mean={np.mean(sizes):.1f}, "
                   f"std={np.std(sizes):.1f}, "
                   f"min={min(sizes)}, max={max(sizes)}")

        return client_datasets


class DirichletDataSplitter:
    """
    Dirichlet distribution-based non-IID data splitter

    Uses Dirichlet distribution to create realistic non-IID partitions
    Lower alpha = more heterogeneous
    """

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        alpha: float = 0.5,
        min_size: int = 10,
        seed: int = 42
    ):
        """
        Initialize Dirichlet data splitter

        Args:
            dataset: PyTorch dataset to split
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter (lower = more heterogeneous)
            min_size: Minimum samples per client
            seed: Random seed
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.min_size = min_size
        self.seed = seed
        np.random.seed(seed)

        # Get labels
        if hasattr(dataset, 'targets'):
            self.labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            self.labels = np.array(dataset.labels)
        else:
            # Try to extract labels from dataset
            try:
                self.labels = np.array([dataset[i][1] for i in range(len(dataset))])
            except:
                raise ValueError("Cannot extract labels from dataset")

        self.num_classes = len(np.unique(self.labels))

        logger.info(f"Initialized Dirichlet splitter: {num_clients} clients, "
                   f"alpha={alpha}, {self.num_classes} classes")

    def split(self) -> List[Subset]:
        """
        Split dataset using Dirichlet distribution

        Returns:
            List of dataset subsets, one per client
        """
        # Group indices by label
        label_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            label_indices[label].append(idx)

        # Initialize client indices
        client_indices = [[] for _ in range(self.num_clients)]

        # For each class, distribute samples according to Dirichlet
        for label, indices in label_indices.items():
            np.random.shuffle(indices)
            num_samples = len(indices)

            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(
                [self.alpha] * self.num_clients
            )

            # Convert proportions to sample counts
            proportions = np.array([
                p * (len(idx_list) < self.min_size) +
                p * (1 - self.min_size / num_samples)
                for p, idx_list in zip(proportions, client_indices)
            ])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * num_samples).astype(int)[:-1]

            # Split indices according to proportions
            split_indices = np.split(indices, proportions)

            # Assign to clients
            for client_id, client_idx in enumerate(split_indices):
                client_indices[client_id].extend(client_idx.tolist())

        # Create subsets
        client_datasets = [
            Subset(self.dataset, indices)
            for indices in client_indices
        ]

        # Log statistics
        self._log_statistics(client_indices)

        return client_datasets

    def _log_statistics(self, client_indices: List[List[int]]):
        """Log distribution statistics"""
        sizes = [len(indices) for indices in client_indices]
        logger.info(f"Dirichlet split: mean={np.mean(sizes):.1f}, "
                   f"std={np.std(sizes):.1f}, "
                   f"min={min(sizes)}, max={max(sizes)}")

        # Class distribution per client
        for client_id, indices in enumerate(client_indices[:3]):  # Log first 3 clients
            labels = self.labels[indices]
            unique, counts = np.unique(labels, return_counts=True)
            dist = dict(zip(unique.astype(int), counts.astype(int)))
            logger.debug(f"Client {client_id} class distribution: {dist}")


class PathologicalSplitter:
    """
    Pathological non-IID splitter

    Each client only has samples from a small number of classes
    (e.g., 2 classes per client)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        shards_per_client: int = 2,
        seed: int = 42
    ):
        """
        Initialize pathological splitter

        Args:
            dataset: PyTorch dataset to split
            num_clients: Number of clients
            shards_per_client: Number of class shards per client
            seed: Random seed
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.shards_per_client = shards_per_client
        self.seed = seed
        np.random.seed(seed)

        # Get labels
        if hasattr(dataset, 'targets'):
            self.labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            self.labels = np.array(dataset.labels)
        else:
            self.labels = np.array([dataset[i][1] for i in range(len(dataset))])

        self.num_classes = len(np.unique(self.labels))

        logger.info(f"Initialized Pathological splitter: {num_clients} clients, "
                   f"{shards_per_client} shards per client, "
                   f"{self.num_classes} classes")

    def split(self) -> List[Subset]:
        """
        Split dataset pathologically

        Returns:
            List of dataset subsets, one per client
        """
        # Sort by label
        sorted_indices = np.argsort(self.labels)

        # Divide into shards
        total_shards = self.num_clients * self.shards_per_client
        shard_size = len(self.dataset) // total_shards
        shards = []

        for i in range(total_shards):
            start = i * shard_size
            end = start + shard_size if i < total_shards - 1 else len(self.dataset)
            shards.append(sorted_indices[start:end].tolist())

        # Randomly assign shards to clients
        np.random.shuffle(shards)
        client_indices = [[] for _ in range(self.num_clients)]

        for client_id in range(self.num_clients):
            for shard_id in range(self.shards_per_client):
                shard_index = client_id * self.shards_per_client + shard_id
                if shard_index < len(shards):
                    client_indices[client_id].extend(shards[shard_index])

        # Create subsets
        client_datasets = [
            Subset(self.dataset, indices)
            for indices in client_indices
        ]

        # Log statistics
        self._log_statistics(client_indices)

        return client_datasets

    def _log_statistics(self, client_indices: List[List[int]]):
        """Log distribution statistics"""
        sizes = [len(indices) for indices in client_indices]
        logger.info(f"Pathological split: mean={np.mean(sizes):.1f}, "
                   f"std={np.std(sizes):.1f}")

        # Class distribution per client
        for client_id, indices in enumerate(client_indices[:3]):
            labels = self.labels[indices]
            unique, counts = np.unique(labels, return_counts=True)
            dist = dict(zip(unique.astype(int), counts.astype(int)))
            logger.debug(f"Client {client_id} class distribution: {dist}")


class NonIIDDataSplitter:
    """
    General non-IID data splitter

    Provides multiple strategies for creating non-IID data distributions
    """

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        strategy: str = "dirichlet",
        alpha: float = 0.5,
        shards_per_client: int = 2,
        seed: int = 42
    ):
        """
        Initialize non-IID data splitter

        Args:
            dataset: PyTorch dataset to split
            num_clients: Number of clients
            strategy: Splitting strategy ('iid', 'dirichlet', 'pathological')
            alpha: Dirichlet concentration parameter
            shards_per_client: Shards per client for pathological split
            seed: Random seed
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.strategy = strategy
        self.seed = seed

        # Create appropriate splitter
        if strategy == "iid":
            self.splitter = IIDDataSplitter(dataset, num_clients, seed)
        elif strategy == "dirichlet":
            self.splitter = DirichletDataSplitter(dataset, num_clients, alpha, seed=seed)
        elif strategy == "pathological":
            self.splitter = PathologicalSplitter(
                dataset, num_clients, shards_per_client, seed
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(f"Initialized NonIID splitter with strategy: {strategy}")

    def split(self) -> List[Subset]:
        """Split dataset according to chosen strategy"""
        return self.splitter.split()

    def get_client_statistics(
        self,
        client_datasets: List[Subset]
    ) -> Dict[int, Dict]:
        """
        Compute statistics for each client dataset

        Args:
            client_datasets: List of client datasets

        Returns:
            Dictionary of statistics per client
        """
        stats = {}

        for client_id, client_data in enumerate(client_datasets):
            # Get labels
            if hasattr(self.dataset, 'targets'):
                labels = np.array([
                    self.dataset.targets[idx]
                    for idx in client_data.indices
                ])
            else:
                labels = np.array([
                    self.dataset[idx][1]
                    for idx in client_data.indices
                ])

            # Compute statistics
            unique, counts = np.unique(labels, return_counts=True)
            class_dist = dict(zip(unique.astype(int), counts.astype(int)))

            stats[client_id] = {
                'size': len(client_data),
                'num_classes': len(unique),
                'class_distribution': class_dist,
                'class_balance': float(np.std(counts)) / (np.mean(counts) + 1e-6)
            }

        return stats


def visualize_data_distribution(
    client_datasets: List[Subset],
    dataset: Dataset,
    num_classes: int,
    save_path: Optional[str] = None
):
    """
    Visualize data distribution across clients

    Args:
        client_datasets: List of client datasets
        dataset: Original dataset
        num_classes: Number of classes
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available for visualization")
        return

    # Get labels
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    else:
        all_labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # Create distribution matrix
    num_clients = len(client_datasets)
    dist_matrix = np.zeros((num_clients, num_classes))

    for client_id, client_data in enumerate(client_datasets):
        labels = all_labels[client_data.indices]
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            dist_matrix[client_id, int(label)] = count

    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        dist_matrix,
        annot=True if num_clients <= 10 else False,
        fmt='g',
        cmap='YlOrRd',
        xticklabels=range(num_classes),
        yticklabels=range(num_clients)
    )
    plt.xlabel('Class')
    plt.ylabel('Client')
    plt.title('Data Distribution Across Clients')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved distribution plot to {save_path}")

    plt.close()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create dummy dataset
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        dataset = datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=transform
        )

        # Test IID split
        print("\n=== IID Split ===")
        iid_splitter = NonIIDDataSplitter(
            dataset,
            num_clients=10,
            strategy="iid"
        )
        iid_clients = iid_splitter.split()
        iid_stats = iid_splitter.get_client_statistics(iid_clients)
        print(f"Client 0: {iid_stats[0]}")

        # Test Dirichlet split
        print("\n=== Dirichlet Split (alpha=0.5) ===")
        dir_splitter = NonIIDDataSplitter(
            dataset,
            num_clients=10,
            strategy="dirichlet",
            alpha=0.5
        )
        dir_clients = dir_splitter.split()
        dir_stats = dir_splitter.get_client_statistics(dir_clients)
        print(f"Client 0: {dir_stats[0]}")

        # Test Pathological split
        print("\n=== Pathological Split (2 shards) ===")
        path_splitter = NonIIDDataSplitter(
            dataset,
            num_clients=10,
            strategy="pathological",
            shards_per_client=2
        )
        path_clients = path_splitter.split()
        path_stats = path_splitter.get_client_statistics(path_clients)
        print(f"Client 0: {path_stats[0]}")

    except Exception as e:
        print(f"Example execution failed (expected if MNIST not downloaded): {e}")
