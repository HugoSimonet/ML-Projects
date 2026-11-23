"""Content-based filtering recommender."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender
from ..utils.config_loader import load_config


class ContentBasedRecommender(BaseRecommender):
    """
    Content-based filtering recommender using TF-IDF.

    Recommends items similar to those the user has liked based on item features.
    """

    def __init__(
        self,
        max_features: int = None,
        ngram_range: Tuple[int, int] = None
    ):
        """
        Initialize content-based recommender.

        Args:
            max_features: Maximum number of features for TF-IDF.
            ngram_range: N-gram range for TF-IDF.
        """
        super().__init__("ContentBased")

        config = load_config()
        cb_config = config["models"]["content_based"]["tfidf"]

        self.max_features = max_features or cb_config["max_features"]
        self.ngram_range = tuple(ngram_range) if ngram_range else tuple(cb_config["ngram_range"])

        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english'
        )

        self.train_data = None
        self.item_profiles = None
        self.item_similarity = None
        self.item_features = None

        logger.info(f"Initialized ContentBased (max_features={self.max_features}, ngram_range={self.ngram_range})")

    def create_item_profiles(self, items_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Create item profiles from features.

        Args:
            items_df: DataFrame with item information.
            feature_columns: List of columns to use for creating profiles.

        Returns:
            DataFrame with item profiles.
        """
        # Combine feature columns into a single text field
        items_df = items_df.copy()
        items_df['profile'] = items_df[feature_columns].astype(str).agg(' '.join, axis=1)

        return items_df

    def fit(
        self,
        train_data: pd.DataFrame,
        items_df: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None
    ) -> None:
        """
        Train the content-based model.

        Args:
            train_data: Training data with columns [user_id, item_id, rating].
            items_df: DataFrame with item features.
            feature_columns: List of feature columns to use.
        """
        logger.info(f"Training {self.name} with {len(train_data)} ratings...")

        self.train_data = train_data

        if items_df is None or feature_columns is None:
            logger.warning("No item features provided. Using item IDs as features.")
            # Create dummy features based on item IDs
            unique_items = train_data['item_id'].unique()
            items_df = pd.DataFrame({
                'item_id': unique_items,
                'features': [f"item_{i}" for i in unique_items]
            })
            feature_columns = ['features']

        # Create item profiles
        self.item_features = self.create_item_profiles(items_df, feature_columns)

        # Ensure all items in train_data are in item_features
        train_items = set(train_data['item_id'].unique())
        feature_items = set(self.item_features['item_id'].unique())

        if not train_items.issubset(feature_items):
            missing_items = train_items - feature_items
            logger.warning(f"Missing features for {len(missing_items)} items. Creating dummy features.")

            dummy_features = pd.DataFrame({
                'item_id': list(missing_items),
                'profile': [f"item_{i}" for i in missing_items]
            })
            self.item_features = pd.concat([self.item_features, dummy_features], ignore_index=True)

        # Compute TF-IDF matrix
        logger.info("Computing TF-IDF matrix...")
        self.item_profiles = self.tfidf.fit_transform(self.item_features['profile'])

        # Compute item-item similarity matrix
        logger.info("Computing item similarity matrix...")
        self.item_similarity = cosine_similarity(self.item_profiles)

        self.is_trained = True
        logger.info(f"{self.name} training completed")

    def get_item_index(self, item_id: int) -> Optional[int]:
        """
        Get index of item in the similarity matrix.

        Args:
            item_id: Item identifier.

        Returns:
            Index in the matrix, or None if not found.
        """
        try:
            return self.item_features[self.item_features['item_id'] == item_id].index[0]
        except IndexError:
            return None

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

        # Get items user has rated
        user_items = self.train_data[self.train_data['user_id'] == user_id]

        if len(user_items) == 0:
            # Cold start: return mean rating
            return self.train_data['rating'].mean()

        # Get item index
        item_idx = self.get_item_index(item_id)
        if item_idx is None:
            return self.train_data['rating'].mean()

        # Calculate weighted average of user's ratings based on similarity
        total_similarity = 0
        weighted_ratings = 0

        for _, row in user_items.iterrows():
            rated_item_idx = self.get_item_index(row['item_id'])
            if rated_item_idx is not None:
                similarity = self.item_similarity[item_idx, rated_item_idx]
                weighted_ratings += similarity * row['rating']
                total_similarity += similarity

        if total_similarity == 0:
            return self.train_data['rating'].mean()

        return weighted_ratings / total_similarity

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

        # Get user's rated items
        user_items = self.train_data[self.train_data['user_id'] == user_id]

        if len(user_items) == 0:
            logger.warning(f"User {user_id} has no ratings. Returning popular items.")
            # Return most popular items
            popular_items = self.train_data.groupby('item_id')['rating'].agg(['count', 'mean'])
            popular_items = popular_items.sort_values(['count', 'mean'], ascending=False)
            return [(item_id, score) for item_id, score in
                    zip(popular_items.head(n_items).index, popular_items.head(n_items)['mean'])]

        # Get all items
        all_items = self.item_features['item_id'].unique()

        # Exclude seen items if requested
        if exclude_seen:
            seen_items = set(user_items['item_id'].values)
            candidate_items = [item for item in all_items if item not in seen_items]
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

        item_idx = self.get_item_index(item_id)
        if item_idx is None:
            logger.warning(f"Item {item_id} not found in trained data")
            return []

        # Get similarity scores for this item
        similarities = self.item_similarity[item_idx]

        # Get top-N similar items (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_items+1]

        similar_items = []
        for idx in similar_indices:
            similar_item_id = self.item_features.iloc[idx]['item_id']
            similarity_score = similarities[idx]
            similar_items.append((similar_item_id, similarity_score))

        return similar_items
