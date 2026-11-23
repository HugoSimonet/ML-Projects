"""Hybrid recommendation system."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseRecommender
from ..utils.config_loader import load_config


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender that combines multiple recommendation strategies.

    Combines:
        - Collaborative filtering
        - Content-based filtering
        - Matrix factorization
        - Deep learning (optional)

    Uses weighted combination or rank-based fusion.
    """

    def __init__(
        self,
        models: Dict[str, BaseRecommender],
        weights: Optional[Dict[str, float]] = None,
        fusion_method: str = "weighted"
    ):
        """
        Initialize hybrid recommender.

        Args:
            models: Dictionary of {model_name: model_instance}.
            weights: Dictionary of {model_name: weight}. If None, uses equal weights.
            fusion_method: Method for combining predictions ('weighted' or 'rank_based').
        """
        super().__init__("Hybrid")

        self.models = models
        self.fusion_method = fusion_method

        # Set weights
        if weights is None:
            # Equal weights
            n_models = len(models)
            self.weights = {name: 1.0 / n_models for name in models.keys()}
        else:
            # Normalize weights to sum to 1
            total = sum(weights.values())
            self.weights = {name: w / total for name, w in weights.items()}

        self.train_data = None

        logger.info(f"Initialized Hybrid with {len(models)} models: {list(models.keys())}")
        logger.info(f"Weights: {self.weights}")
        logger.info(f"Fusion method: {fusion_method}")

    def fit(
        self,
        train_data: pd.DataFrame,
        items_df: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None
    ) -> None:
        """
        Train all models in the hybrid system.

        Args:
            train_data: Training data with columns [user_id, item_id, rating].
            items_df: Optional item features for content-based filtering.
            feature_columns: Optional feature columns for content-based filtering.
        """
        logger.info(f"Training {self.name} with {len(train_data)} ratings...")

        self.train_data = train_data

        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")

            try:
                # Content-based model needs item features
                if hasattr(model, 'fit') and 'content' in model_name.lower():
                    if items_df is not None and feature_columns is not None:
                        model.fit(train_data, items_df, feature_columns)
                    else:
                        model.fit(train_data)
                else:
                    model.fit(train_data)

                logger.info(f"{model_name} training completed")
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                # Remove failed model
                del self.models[model_name]
                del self.weights[model_name]

        # Renormalize weights after removing failed models
        if self.models:
            total = sum(self.weights.values())
            self.weights = {name: w / total for name, w in self.weights.items()}

        self.is_trained = True
        logger.info(f"{self.name} training completed with {len(self.models)} models")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair using weighted combination.

        Args:
            user_id: User identifier.
            item_id: Item identifier.

        Returns:
            Predicted rating.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = {}

        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                pred = model.predict(user_id, item_id)
                predictions[model_name] = pred
            except Exception as e:
                logger.warning(f"Error getting prediction from {model_name}: {e}")
                continue

        if not predictions:
            # Fallback to mean rating
            return self.train_data['rating'].mean()

        # Weighted average
        weighted_sum = sum(
            predictions[name] * self.weights[name]
            for name in predictions.keys()
            if name in self.weights
        )

        weight_sum = sum(
            self.weights[name]
            for name in predictions.keys()
            if name in self.weights
        )

        return weighted_sum / weight_sum if weight_sum > 0 else self.train_data['rating'].mean()

    def recommend_weighted(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations using weighted score combination.

        Args:
            user_id: User identifier.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items user has already interacted with.

        Returns:
            List of (item_id, score) tuples sorted by score.
        """
        # Get recommendations from each model
        all_recommendations = {}

        for model_name, model in self.models.items():
            try:
                # Get more recommendations than needed to have better fusion
                model_recs = model.recommend(user_id, n_items * 3, exclude_seen)

                # Store scores for each item
                for item_id, score in model_recs:
                    if item_id not in all_recommendations:
                        all_recommendations[item_id] = {}
                    all_recommendations[item_id][model_name] = score

            except Exception as e:
                logger.warning(f"Error getting recommendations from {model_name}: {e}")
                continue

        if not all_recommendations:
            logger.warning(f"No recommendations generated for user {user_id}")
            return []

        # Compute weighted scores
        final_scores = []

        for item_id, model_scores in all_recommendations.items():
            # Weighted average of available scores
            weighted_sum = sum(
                model_scores[name] * self.weights[name]
                for name in model_scores.keys()
                if name in self.weights
            )

            weight_sum = sum(
                self.weights[name]
                for name in model_scores.keys()
                if name in self.weights
            )

            if weight_sum > 0:
                final_score = weighted_sum / weight_sum
                final_scores.append((item_id, final_score))

        # Sort by score
        final_scores.sort(key=lambda x: x[1], reverse=True)

        return final_scores[:n_items]

    def recommend_rank_based(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations using rank-based fusion.

        Args:
            user_id: User identifier.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items user has already interacted with.

        Returns:
            List of (item_id, score) tuples sorted by score.
        """
        # Get recommendations from each model
        all_rankings = {}

        for model_name, model in self.models.items():
            try:
                # Get more recommendations than needed
                model_recs = model.recommend(user_id, n_items * 3, exclude_seen)

                # Convert to ranks (lower rank = better)
                for rank, (item_id, score) in enumerate(model_recs):
                    if item_id not in all_rankings:
                        all_rankings[item_id] = {}
                    all_rankings[item_id][model_name] = rank + 1  # 1-indexed ranks

            except Exception as e:
                logger.warning(f"Error getting recommendations from {model_name}: {e}")
                continue

        if not all_rankings:
            logger.warning(f"No recommendations generated for user {user_id}")
            return []

        # Compute weighted reciprocal ranks (Borda count variant)
        final_scores = []

        for item_id, model_ranks in all_rankings.items():
            # Weighted reciprocal rank
            weighted_rr = sum(
                (1.0 / model_ranks[name]) * self.weights[name]
                for name in model_ranks.keys()
                if name in self.weights
            )

            final_scores.append((item_id, weighted_rr))

        # Sort by score
        final_scores.sort(key=lambda x: x[1], reverse=True)

        return final_scores[:n_items]

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

        if self.fusion_method == "weighted":
            return self.recommend_weighted(user_id, n_items, exclude_seen)
        elif self.fusion_method == "rank_based":
            return self.recommend_rank_based(user_id, n_items, exclude_seen)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def get_model_contributions(
        self,
        user_id: int,
        item_id: int
    ) -> Dict[str, float]:
        """
        Get individual model contributions to a prediction.

        Args:
            user_id: User identifier.
            item_id: Item identifier.

        Returns:
            Dictionary of {model_name: prediction}.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing contributions")

        contributions = {}

        for model_name, model in self.models.items():
            try:
                pred = model.predict(user_id, item_id)
                contributions[model_name] = {
                    'prediction': pred,
                    'weight': self.weights[model_name],
                    'contribution': pred * self.weights[model_name]
                }
            except Exception as e:
                logger.warning(f"Error getting contribution from {model_name}: {e}")
                continue

        return contributions

    def explain_recommendation(
        self,
        user_id: int,
        item_id: int
    ) -> Dict[str, any]:
        """
        Explain why an item was recommended to a user.

        Args:
            user_id: User identifier.
            item_id: Item identifier.

        Returns:
            Dictionary with explanation details.
        """
        contributions = self.get_model_contributions(user_id, item_id)

        final_prediction = self.predict(user_id, item_id)

        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'final_prediction': final_prediction,
            'model_contributions': contributions,
            'fusion_method': self.fusion_method,
        }

        return explanation
