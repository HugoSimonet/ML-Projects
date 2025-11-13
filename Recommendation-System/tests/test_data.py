"""Tests for data handling."""

import pandas as pd
import pytest

from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def sample_ratings():
    """Create sample ratings data."""
    data = {
        'user_id': list(range(1, 101)) * 10,
        'item_id': [i % 50 + 1 for i in range(1000)],
        'rating': [3 + (i % 3) for i in range(1000)],
        'timestamp': list(range(1000))
    }
    return pd.DataFrame(data)


class TestDatasetLoader:
    """Tests for dataset loader."""

    def test_synthetic_data_creation(self):
        """Test creating synthetic data."""
        loader = DatasetLoader("synthetic")
        ratings, items = loader.create_synthetic_data(
            n_users=100,
            n_items=50,
            n_ratings=500
        )

        assert len(ratings) <= 500  # May be less due to deduplication
        assert ratings['user_id'].nunique() <= 100
        assert ratings['item_id'].nunique() <= 50


class TestDataPreprocessor:
    """Tests for data preprocessor."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.random_seed == 42

    def test_encode_ids(self, sample_ratings):
        """Test ID encoding."""
        preprocessor = DataPreprocessor()
        encoded = preprocessor.encode_ids(sample_ratings)

        assert encoded['user_id'].min() >= 0
        assert encoded['item_id'].min() >= 0

    def test_train_test_split(self, sample_ratings):
        """Test train/test splitting."""
        preprocessor = DataPreprocessor()
        train, test = preprocessor.train_test_split(sample_ratings, test_size=0.2)

        assert len(train) + len(test) <= len(sample_ratings)
        assert len(train) > len(test)

    def test_handle_cold_start(self, sample_ratings):
        """Test cold start filtering."""
        preprocessor = DataPreprocessor()
        filtered = preprocessor.handle_cold_start(
            sample_ratings,
            min_user_ratings=5,
            min_item_ratings=5
        )

        # Check that all users have at least 5 ratings
        user_counts = filtered.groupby('user_id').size()
        assert (user_counts >= 5).all()

        # Check that all items have at least 5 ratings
        item_counts = filtered.groupby('item_id').size()
        assert (item_counts >= 5).all()

    def test_get_statistics(self, sample_ratings):
        """Test statistics calculation."""
        preprocessor = DataPreprocessor()
        stats = preprocessor.get_statistics(sample_ratings)

        assert 'n_ratings' in stats
        assert 'n_users' in stats
        assert 'n_items' in stats
        assert 'sparsity' in stats
        assert stats['n_ratings'] == len(sample_ratings)
