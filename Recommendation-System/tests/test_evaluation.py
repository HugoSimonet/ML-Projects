"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import Evaluator


class TestEvaluator:
    """Tests for evaluation metrics."""

    def test_rmse(self):
        """Test RMSE calculation."""
        y_true = np.array([3, 4, 5, 2, 1])
        y_pred = np.array([3.1, 3.9, 4.8, 2.2, 1.1])

        rmse = Evaluator.rmse(y_true, y_pred)
        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_mae(self):
        """Test MAE calculation."""
        y_true = np.array([3, 4, 5, 2, 1])
        y_pred = np.array([3.1, 3.9, 4.8, 2.2, 1.1])

        mae = Evaluator.mae(y_true, y_pred)
        assert isinstance(mae, float)
        assert mae >= 0

    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        relevant_items = {1, 2, 3, 4, 5}
        recommended_items = [1, 2, 6, 7, 8, 9, 10]

        precision = Evaluator.precision_at_k(relevant_items, recommended_items, k=5)
        assert 0 <= precision <= 1
        assert precision == 2/5  # 2 relevant items in top 5

    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        relevant_items = {1, 2, 3, 4, 5}
        recommended_items = [1, 2, 6, 7, 8, 9, 10]

        recall = Evaluator.recall_at_k(relevant_items, recommended_items, k=5)
        assert 0 <= recall <= 1
        assert recall == 2/5  # 2 out of 5 relevant items found

    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        relevant_items = {1: 5.0, 2: 4.0, 3: 3.0}
        recommended_items = [1, 5, 2, 6, 3]

        ndcg = Evaluator.ndcg_at_k(relevant_items, recommended_items, k=5)
        assert 0 <= ndcg <= 1

    def test_diversity(self):
        """Test diversity calculation."""
        recommendations = [
            [1, 2, 3, 4, 5],
            [2, 3, 6, 7, 8],
            [1, 4, 9, 10, 11]
        ]

        diversity = Evaluator.diversity(recommendations)
        assert 0 <= diversity <= 1

    def test_coverage(self):
        """Test coverage calculation."""
        recommendations = [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ]
        all_items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

        coverage = Evaluator.coverage(recommendations, all_items)
        assert 0 <= coverage <= 1
        assert coverage == 0.5  # 5 out of 10 items recommended
