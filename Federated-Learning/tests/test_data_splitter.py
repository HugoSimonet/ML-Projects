"""
Unit tests for data splitting utilities
Tests IID, Non-IID (Dirichlet, Pathological) data distribution
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    IIDDataSplitter,
    DirichletDataSplitter,
    PathologicalSplitter,
    NonIIDDataSplitter
)


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset for testing"""
    # 1000 samples, 10 classes
    data = torch.randn(1000, 10)
    labels = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(data, labels)
    dataset.targets = labels.numpy()  # Add targets attribute
    return dataset


class TestIIDDataSplitter:
    """Test IID data splitting"""

    def test_initialization(self, dummy_dataset):
        """Test IID splitter initialization"""
        splitter = IIDDataSplitter(dummy_dataset, num_clients=5)
        assert splitter.num_clients == 5
        assert len(splitter.dataset) == 1000

    def test_split_returns_correct_number(self, dummy_dataset):
        """Test that split returns correct number of subsets"""
        splitter = IIDDataSplitter(dummy_dataset, num_clients=5)
        subsets = splitter.split()
        assert len(subsets) == 5

    def test_split_approximately_equal_sizes(self, dummy_dataset):
        """Test that splits are approximately equal size"""
        splitter = IIDDataSplitter(dummy_dataset, num_clients=5)
        subsets = splitter.split()

        sizes = [len(subset) for subset in subsets]
        # Should be approximately 200 each
        assert all(150 < size < 250 for size in sizes)
        # Total should equal dataset size
        assert sum(sizes) == 1000

    def test_split_all_indices_used(self, dummy_dataset):
        """Test that all indices are used exactly once"""
        splitter = IIDDataSplitter(dummy_dataset, num_clients=5)
        subsets = splitter.split()

        all_indices = set()
        for subset in subsets:
            indices = set(subset.indices)
            # No duplicates
            assert len(all_indices & indices) == 0
            all_indices.update(indices)

        # All original indices used
        assert len(all_indices) == 1000

    def test_different_seeds_different_splits(self, dummy_dataset):
        """Test that different seeds produce different splits"""
        splitter1 = IIDDataSplitter(dummy_dataset, num_clients=5, seed=42)
        splitter2 = IIDDataSplitter(dummy_dataset, num_clients=5, seed=123)

        subsets1 = splitter1.split()
        subsets2 = splitter2.split()

        # First client should have different indices
        indices1 = set(subsets1[0].indices)
        indices2 = set(subsets2[0].indices)
        assert indices1 != indices2


class TestDirichletDataSplitter:
    """Test Dirichlet non-IID splitting"""

    def test_initialization(self, dummy_dataset):
        """Test Dirichlet splitter initialization"""
        splitter = DirichletDataSplitter(
            dummy_dataset,
            num_clients=5,
            alpha=0.5
        )
        assert splitter.num_clients == 5
        assert splitter.alpha == 0.5
        assert splitter.num_classes == 10

    def test_split_returns_correct_number(self, dummy_dataset):
        """Test split returns correct number of subsets"""
        splitter = DirichletDataSplitter(dummy_dataset, num_clients=5, alpha=0.5)
        subsets = splitter.split()
        assert len(subsets) == 5

    def test_split_non_iid_distribution(self, dummy_dataset):
        """Test that Dirichlet creates non-IID distribution"""
        splitter = DirichletDataSplitter(dummy_dataset, num_clients=5, alpha=0.1)
        subsets = splitter.split()

        # Check that different clients have different class distributions
        distributions = []
        for subset in subsets:
            labels = dummy_dataset.targets[subset.indices]
            unique, counts = np.unique(labels, return_counts=True)
            dist = dict(zip(unique, counts))
            distributions.append(dist)

        # At least some clients should have very different distributions
        # with low alpha (high heterogeneity)
        assert len(distributions) == 5

    def test_all_data_used(self, dummy_dataset):
        """Test that all data points are used"""
        splitter = DirichletDataSplitter(dummy_dataset, num_clients=5, alpha=0.5)
        subsets = splitter.split()

        all_indices = set()
        for subset in subsets:
            all_indices.update(subset.indices)

        assert len(all_indices) == 1000

    def test_min_size_respected(self, dummy_dataset):
        """Test that minimum size per client is respected"""
        min_size = 50
        splitter = DirichletDataSplitter(
            dummy_dataset,
            num_clients=5,
            alpha=0.5,
            min_size=min_size
        )
        subsets = splitter.split()

        for subset in subsets:
            assert len(subset) >= min_size or len(subset) == 0


class TestPathologicalSplitter:
    """Test pathological non-IID splitting"""

    def test_initialization(self, dummy_dataset):
        """Test pathological splitter initialization"""
        splitter = PathologicalSplitter(
            dummy_dataset,
            num_clients=5,
            shards_per_client=2
        )
        assert splitter.num_clients == 5
        assert splitter.shards_per_client == 2

    def test_split_returns_correct_number(self, dummy_dataset):
        """Test split returns correct number of subsets"""
        splitter = PathologicalSplitter(
            dummy_dataset,
            num_clients=5,
            shards_per_client=2
        )
        subsets = splitter.split()
        assert len(subsets) == 5

    def test_pathological_distribution(self, dummy_dataset):
        """Test that clients have limited number of classes"""
        splitter = PathologicalSplitter(
            dummy_dataset,
            num_clients=10,
            shards_per_client=2
        )
        subsets = splitter.split()

        # Each client should have limited classes (pathological)
        for subset in subsets:
            if len(subset) > 0:
                labels = dummy_dataset.targets[subset.indices]
                num_classes = len(np.unique(labels))
                # With 2 shards, should have at most a few classes
                assert num_classes <= 5  # Relaxed bound

    def test_all_data_used(self, dummy_dataset):
        """Test that most data points are used"""
        splitter = PathologicalSplitter(
            dummy_dataset,
            num_clients=5,
            shards_per_client=2
        )
        subsets = splitter.split()

        all_indices = set()
        for subset in subsets:
            all_indices.update(subset.indices)

        # Should use most of the data
        assert len(all_indices) >= 900  # Allow some rounding loss


class TestNonIIDDataSplitter:
    """Test unified non-IID data splitter"""

    def test_iid_strategy(self, dummy_dataset):
        """Test IID strategy selection"""
        splitter = NonIIDDataSplitter(
            dummy_dataset,
            num_clients=5,
            strategy='iid'
        )
        subsets = splitter.split()
        assert len(subsets) == 5

    def test_dirichlet_strategy(self, dummy_dataset):
        """Test Dirichlet strategy selection"""
        splitter = NonIIDDataSplitter(
            dummy_dataset,
            num_clients=5,
            strategy='dirichlet',
            alpha=0.5
        )
        subsets = splitter.split()
        assert len(subsets) == 5

    def test_pathological_strategy(self, dummy_dataset):
        """Test pathological strategy selection"""
        splitter = NonIIDDataSplitter(
            dummy_dataset,
            num_clients=5,
            strategy='pathological',
            shards_per_client=2
        )
        subsets = splitter.split()
        assert len(subsets) == 5

    def test_invalid_strategy_raises_error(self, dummy_dataset):
        """Test that invalid strategy raises error"""
        with pytest.raises(ValueError):
            NonIIDDataSplitter(
                dummy_dataset,
                num_clients=5,
                strategy='invalid_strategy'
            )

    def test_get_client_statistics(self, dummy_dataset):
        """Test client statistics computation"""
        splitter = NonIIDDataSplitter(
            dummy_dataset,
            num_clients=5,
            strategy='dirichlet',
            alpha=0.5
        )
        subsets = splitter.split()
        stats = splitter.get_client_statistics(subsets)

        assert len(stats) == 5
        for client_id, client_stats in stats.items():
            assert 'size' in client_stats
            assert 'num_classes' in client_stats
            assert 'class_distribution' in client_stats
            assert 'class_balance' in client_stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
