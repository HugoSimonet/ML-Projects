"""Collaborative filtering recommenders."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from surprise import Dataset, KNNBasic, Reader
from surprise.model_selection import cross_validate

from .base import BaseRecommender
from ..utils.config_loader import load_config


class CollaborativeFilteringBase(BaseRecommender):
    """Base class for collaborative filtering methods."""

    def __init__(
        self,
        name: str,
        n_neighbors: int = 40,
        min_k: int = 1,
        similarity: str = "cosine"
    ):
        """
        Initialize collaborative filtering recommender.

        Args:
            name: Name of the recommender.
            n_neighbors: Maximum number of neighbors to consider.
            min_k: Minimum number of neighbors required for prediction.
            similarity: Similarity measure ('cosine', 'pearson', 'msd').
        """
        super().__init__(name)
        self.n_neighbors = n_neighbors
        self.min_k = min_k
        self.similarity = similarity

        self.model = None
        self.trainset = None
        self.train_data = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train the collaborative filtering model.

        Args:
            train_data: Training data with columns [user_id, item_id, rating].
        """
        logger.info(f"Training {self.name} with {len(train_data)} ratings...")

        self.train_data = train_data

        # Create Surprise dataset
        reader = Reader(rating_scale=(
            train_data['rating'].min(),
            train_data['rating'].max()
        ))

        data = Dataset.load_from_df(
            train_data[['user_id', 'item_id', 'rating']],
            reader
        )

        # Build full trainset
        self.trainset = data.build_full_trainset()

        # Train model
        self.model.fit(self.trainset)
        self.is_trained = True

        logger.info(f"{self.name} training completed")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User identifier.
            item_id: Item identifier.

        Returns:
            Predicted rating.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        prediction = self.model.predict(user_id, item_id)
        return prediction.est

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
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")

        # Get all items
        all_items = self.train_data['item_id'].unique()

        # Get items user has already rated
        if exclude_seen:
            user_items = self.train_data[
                self.train_data['user_id'] == user_id
            ]['item_id'].values
            candidate_items = [item for item in all_items if item not in user_items]
        else:
            candidate_items = all_items

        # Predict ratings for all candidate items
        predictions = []
        for item_id in candidate_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class UserBasedCF(CollaborativeFilteringBase):
    """User-based collaborative filtering recommender."""

    def __init__(
        self,
        n_neighbors: int = None,
        min_k: int = None,
        similarity: str = None
    ):
        """
        Initialize user-based collaborative filtering.

        Args:
            n_neighbors: Maximum number of neighbors to consider.
            min_k: Minimum number of neighbors required for prediction.
            similarity: Similarity measure ('cosine', 'pearson', 'msd').
        """
        config = load_config()
        cf_config = config["models"]["collaborative_filtering"]["user_based"]

        n_neighbors = n_neighbors or cf_config["n_neighbors"]
        min_k = min_k or cf_config["min_k"]
        similarity = similarity or cf_config["similarity"]

        super().__init__("UserBasedCF", n_neighbors, min_k, similarity)

        # Initialize user-based KNN model
        sim_options = {
            'name': similarity,
            'user_based': True
        }

        self.model = KNNBasic(
            k=n_neighbors,
            min_k=min_k,
            sim_options=sim_options,
            verbose=False
        )

        logger.info(f"Initialized UserBasedCF (k={n_neighbors}, similarity={similarity})")


class ItemBasedCF(CollaborativeFilteringBase):
    """Item-based collaborative filtering recommender."""

    def __init__(
        self,
        n_neighbors: int = None,
        min_k: int = None,
        similarity: str = None
    ):
        """
        Initialize item-based collaborative filtering.

        Args:
            n_neighbors: Maximum number of neighbors to consider.
            min_k: Minimum number of neighbors required for prediction.
            similarity: Similarity measure ('cosine', 'pearson', 'msd').
        """
        config = load_config()
        cf_config = config["models"]["collaborative_filtering"]["item_based"]

        n_neighbors = n_neighbors or cf_config["n_neighbors"]
        min_k = min_k or cf_config["min_k"]
        similarity = similarity or cf_config["similarity"]

        super().__init__("ItemBasedCF", n_neighbors, min_k, similarity)

        # Initialize item-based KNN model
        sim_options = {
            'name': similarity,
            'user_based': False
        }

        self.model = KNNBasic(
            k=n_neighbors,
            min_k=min_k,
            sim_options=sim_options,
            verbose=False
        )

        logger.info(f"Initialized ItemBasedCF (k={n_neighbors}, similarity={similarity})")
