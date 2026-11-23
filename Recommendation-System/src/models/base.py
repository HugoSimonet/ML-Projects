"""Base recommender class."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for all recommender models."""

    def __init__(self, name: str = "BaseRecommender"):
        """
        Initialize recommender.

        Args:
            name: Name of the recommender model.
        """
        self.name = name
        self.is_trained = False

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train the recommender model.

        Args:
            train_data: Training data with columns [user_id, item_id, rating].
        """
        pass

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User identifier.
            item_id: Item identifier.

        Returns:
            Predicted rating.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a user.

        Args:
            user_id: User identifier.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items user has already interacted with.

        Returns:
            List of (item_id, score) tuples sorted by score.
        """
        pass

    def batch_recommend(
        self,
        user_ids: List[int],
        n_items: int = 10,
        exclude_seen: bool = True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users.

        Args:
            user_ids: List of user identifiers.
            n_items: Number of items to recommend per user.
            exclude_seen: Whether to exclude items users have already interacted with.

        Returns:
            Dictionary mapping user_id to list of (item_id, score) tuples.
        """
        recommendations = {}
        for user_id in user_ids:
            try:
                recommendations[user_id] = self.recommend(user_id, n_items, exclude_seen)
            except Exception as e:
                print(f"Error generating recommendations for user {user_id}: {e}")
                recommendations[user_id] = []

        return recommendations

    def save(self, filepath: str) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save the model.
        """
        import joblib
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'BaseRecommender':
        """
        Load model from disk.

        Args:
            filepath: Path to the saved model.

        Returns:
            Loaded model instance.
        """
        import joblib
        return joblib.load(filepath)
