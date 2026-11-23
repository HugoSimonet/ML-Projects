"""Matrix factorization recommenders."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from loguru import logger
from scipy.sparse import coo_matrix
from surprise import Dataset, Reader, SVD

from .base import BaseRecommender
from ..utils.config_loader import load_config


class SVDRecommender(BaseRecommender):
    """
    Singular Value Decomposition (SVD) recommender using Surprise library.

    Uses matrix factorization to decompose the user-item rating matrix.
    """

    def __init__(
        self,
        n_factors: int = None,
        n_epochs: int = None,
        lr_all: float = None,
        reg_all: float = None
    ):
        """
        Initialize SVD recommender.

        Args:
            n_factors: Number of latent factors.
            n_epochs: Number of training epochs.
            lr_all: Learning rate for all parameters.
            reg_all: Regularization for all parameters.
        """
        super().__init__("SVD")

        config = load_config()
        svd_config = config["models"]["matrix_factorization"]["svd"]

        self.n_factors = n_factors or svd_config["n_factors"]
        self.n_epochs = n_epochs or svd_config["n_epochs"]
        self.lr_all = lr_all or svd_config["lr_all"]
        self.reg_all = reg_all or svd_config["reg_all"]

        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            verbose=False
        )

        self.trainset = None
        self.train_data = None

        logger.info(f"Initialized SVD (factors={self.n_factors}, epochs={self.n_epochs})")

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train the SVD model.

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


class ALSRecommender(BaseRecommender):
    """
    Alternating Least Squares (ALS) recommender using implicit library.

    Optimized for implicit feedback data but can work with explicit ratings.
    """

    def __init__(
        self,
        factors: int = None,
        regularization: float = None,
        iterations: int = None,
        alpha: float = None
    ):
        """
        Initialize ALS recommender.

        Args:
            factors: Number of latent factors.
            regularization: Regularization parameter.
            iterations: Number of iterations.
            alpha: Confidence parameter for implicit feedback.
        """
        super().__init__("ALS")

        config = load_config()
        als_config = config["models"]["matrix_factorization"]["als"]

        self.factors = factors or als_config["factors"]
        self.regularization = regularization or als_config["regularization"]
        self.iterations = iterations or als_config["iterations"]
        self.alpha = alpha or als_config["alpha"]

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
            random_state=42
        )

        self.train_data = None
        self.user_item_matrix = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}

        logger.info(f"Initialized ALS (factors={self.factors}, iterations={self.iterations})")

    def create_sparse_matrix(self, train_data: pd.DataFrame) -> coo_matrix:
        """
        Create sparse user-item matrix.

        Args:
            train_data: Training data with columns [user_id, item_id, rating].

        Returns:
            Sparse COO matrix.
        """
        # Create mappings
        unique_users = train_data['user_id'].unique()
        unique_items = train_data['item_id'].unique()

        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        # Map user_id and item_id to indices
        user_indices = train_data['user_id'].map(self.user_to_idx).values
        item_indices = train_data['item_id'].map(self.item_to_idx).values
        ratings = train_data['rating'].values

        # Create sparse matrix (items x users for implicit library)
        sparse_matrix = coo_matrix(
            (ratings, (item_indices, user_indices)),
            shape=(len(unique_items), len(unique_users))
        )

        return sparse_matrix.tocsr()

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train the ALS model.

        Args:
            train_data: Training data with columns [user_id, item_id, rating].
        """
        logger.info(f"Training {self.name} with {len(train_data)} ratings...")

        self.train_data = train_data

        # Create sparse matrix
        self.user_item_matrix = self.create_sparse_matrix(train_data)

        # Train model
        self.model.fit(self.user_item_matrix)
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

        # Check if user and item are in the training data
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            # Cold start: return mean rating
            return self.train_data['rating'].mean()

        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]

        # Get user and item factors
        user_factors = self.model.user_factors[user_idx]
        item_factors = self.model.item_factors[item_idx]

        # Compute dot product
        prediction = np.dot(user_factors, item_factors)

        # Scale to rating range
        rating_min = self.train_data['rating'].min()
        rating_max = self.train_data['rating'].max()
        prediction = np.clip(prediction, rating_min, rating_max)

        return float(prediction)

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

        # Check if user is in training data
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not found in training data. Returning popular items.")
            # Return most popular items
            popular_items = self.train_data.groupby('item_id')['rating'].agg(['count', 'mean'])
            popular_items = popular_items.sort_values(['count', 'mean'], ascending=False)
            return [(item_id, score) for item_id, score in
                    zip(popular_items.head(n_items).index, popular_items.head(n_items)['mean'])]

        user_idx = self.user_to_idx[user_id]

        # Get recommendations from the model
        if exclude_seen:
            # Get user's interaction vector
            user_items = self.user_item_matrix[:, user_idx].toarray().flatten()
        else:
            user_items = None

        # Get top-N recommendations
        item_indices, scores = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=n_items,
            filter_already_liked_items=exclude_seen
        )

        # Convert indices back to item IDs
        recommendations = []
        for item_idx, score in zip(item_indices, scores):
            item_id = self.idx_to_item[item_idx]
            recommendations.append((item_id, float(score)))

        return recommendations

    def get_similar_items(self, item_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        """
        Get items similar to a given item.

        Args:
            item_id: Item identifier.
            n_items: Number of similar items to return.

        Returns:
            List of (item_id, similarity_score) tuples.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting similar items")

        if item_id not in self.item_to_idx:
            logger.warning(f"Item {item_id} not found in training data")
            return []

        item_idx = self.item_to_idx[item_id]

        # Get similar items
        similar_indices, scores = self.model.similar_items(item_idx, N=n_items+1)

        # Convert indices back to item IDs (excluding the item itself)
        similar_items = []
        for idx, score in zip(similar_indices[1:], scores[1:]):
            similar_item_id = self.idx_to_item[idx]
            similar_items.append((similar_item_id, float(score)))

        return similar_items
