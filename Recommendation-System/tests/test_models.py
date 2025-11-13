"""Tests for recommendation models."""

import numpy as np
import pandas as pd
import pytest

from src.models.collaborative_filtering import ItemBasedCF, UserBasedCF
from src.models.content_based import ContentBasedRecommender
from src.models.matrix_factorization import SVDRecommender


@pytest.fixture
def sample_data():
    """Create sample ratings data for testing."""
    data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'item_id': [1, 2, 3, 1, 2, 4, 2, 3, 4, 1, 3, 4],
        'rating': [5, 4, 3, 4, 5, 3, 5, 4, 2, 3, 4, 5]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_items():
    """Create sample item features for testing."""
    data = {
        'item_id': [1, 2, 3, 4],
        'title': ['Item 1', 'Item 2', 'Item 3', 'Item 4'],
        'genre': ['Action', 'Comedy', 'Action', 'Drama']
    }
    return pd.DataFrame(data)


class TestUserBasedCF:
    """Tests for User-based Collaborative Filtering."""

    def test_initialization(self):
        """Test model initialization."""
        model = UserBasedCF()
        assert model.name == "UserBasedCF"
        assert not model.is_trained

    def test_fit(self, sample_data):
        """Test model training."""
        model = UserBasedCF()
        model.fit(sample_data)
        assert model.is_trained

    def test_predict(self, sample_data):
        """Test rating prediction."""
        model = UserBasedCF()
        model.fit(sample_data)

        prediction = model.predict(user_id=1, item_id=1)
        assert isinstance(prediction, (int, float))
        assert 1 <= prediction <= 5

    def test_recommend(self, sample_data):
        """Test generating recommendations."""
        model = UserBasedCF()
        model.fit(sample_data)

        recommendations = model.recommend(user_id=1, n_items=2)
        assert len(recommendations) <= 2
        assert all(isinstance(item_id, (int, np.integer)) for item_id, _ in recommendations)


class TestItemBasedCF:
    """Tests for Item-based Collaborative Filtering."""

    def test_initialization(self):
        """Test model initialization."""
        model = ItemBasedCF()
        assert model.name == "ItemBasedCF"

    def test_fit(self, sample_data):
        """Test model training."""
        model = ItemBasedCF()
        model.fit(sample_data)
        assert model.is_trained


class TestSVDRecommender:
    """Tests for SVD-based recommender."""

    def test_initialization(self):
        """Test model initialization."""
        model = SVDRecommender()
        assert model.name == "SVD"

    def test_fit(self, sample_data):
        """Test model training."""
        model = SVDRecommender(n_epochs=5)
        model.fit(sample_data)
        assert model.is_trained

    def test_predict(self, sample_data):
        """Test rating prediction."""
        model = SVDRecommender(n_epochs=5)
        model.fit(sample_data)

        prediction = model.predict(user_id=1, item_id=1)
        assert isinstance(prediction, (int, float))


class TestContentBasedRecommender:
    """Tests for Content-based recommender."""

    def test_initialization(self):
        """Test model initialization."""
        model = ContentBasedRecommender()
        assert model.name == "ContentBased"

    def test_fit_with_features(self, sample_data, sample_items):
        """Test model training with item features."""
        model = ContentBasedRecommender()
        model.fit(sample_data, sample_items, ['title', 'genre'])
        assert model.is_trained

    def test_fit_without_features(self, sample_data):
        """Test model training without item features."""
        model = ContentBasedRecommender()
        model.fit(sample_data)
        assert model.is_trained


def test_model_save_load(sample_data, tmp_path):
    """Test saving and loading models."""
    model = SVDRecommender(n_epochs=5)
    model.fit(sample_data)

    # Save model
    save_path = tmp_path / "model.pkl"
    model.save(str(save_path))
    assert save_path.exists()

    # Load model
    loaded_model = SVDRecommender.load(str(save_path))
    assert loaded_model.is_trained

    # Test prediction with loaded model
    prediction = loaded_model.predict(user_id=1, item_id=1)
    assert isinstance(prediction, (int, float))
