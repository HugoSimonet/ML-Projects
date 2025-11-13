"""Evaluation metrics for recommendation systems."""

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..models.base import BaseRecommender
from ..utils.config_loader import load_config


class Evaluator:
    """
    Comprehensive evaluator for recommendation systems.

    Implements:
        - Rating prediction metrics (RMSE, MAE)
        - Ranking metrics (Precision@K, Recall@K, NDCG@K, MAP@K)
        - Diversity and coverage metrics
        - Business metrics
    """

    def __init__(self):
        """Initialize evaluator."""
        self.config = load_config()
        logger.info("Initialized Evaluator")

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.

        Args:
            y_true: True ratings.
            y_pred: Predicted ratings.

        Returns:
            RMSE value.
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.

        Args:
            y_true: True ratings.
            y_pred: Predicted ratings.

        Returns:
            MAE value.
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def precision_at_k(
        relevant_items: Set[int],
        recommended_items: List[int],
        k: int
    ) -> float:
        """
        Calculate Precision@K.

        Args:
            relevant_items: Set of relevant item IDs.
            recommended_items: List of recommended item IDs.
            k: Number of recommendations to consider.

        Returns:
            Precision@K value.
        """
        if k == 0:
            return 0.0

        recommended_at_k = set(recommended_items[:k])
        n_relevant_and_recommended = len(relevant_items & recommended_at_k)

        return n_relevant_and_recommended / k

    @staticmethod
    def recall_at_k(
        relevant_items: Set[int],
        recommended_items: List[int],
        k: int
    ) -> float:
        """
        Calculate Recall@K.

        Args:
            relevant_items: Set of relevant item IDs.
            recommended_items: List of recommended item IDs.
            k: Number of recommendations to consider.

        Returns:
            Recall@K value.
        """
        if len(relevant_items) == 0:
            return 0.0

        recommended_at_k = set(recommended_items[:k])
        n_relevant_and_recommended = len(relevant_items & recommended_at_k)

        return n_relevant_and_recommended / len(relevant_items)

    @staticmethod
    def f1_at_k(
        relevant_items: Set[int],
        recommended_items: List[int],
        k: int
    ) -> float:
        """
        Calculate F1-Score@K.

        Args:
            relevant_items: Set of relevant item IDs.
            recommended_items: List of recommended item IDs.
            k: Number of recommendations to consider.

        Returns:
            F1@K value.
        """
        precision = Evaluator.precision_at_k(relevant_items, recommended_items, k)
        recall = Evaluator.recall_at_k(relevant_items, recommended_items, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def ndcg_at_k(
        relevant_items: Dict[int, float],
        recommended_items: List[int],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.

        Args:
            relevant_items: Dictionary of {item_id: relevance_score}.
            recommended_items: List of recommended item IDs.
            k: Number of recommendations to consider.

        Returns:
            NDCG@K value.
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0

        # DCG
        dcg = 0.0
        for i, item_id in enumerate(recommended_items[:k]):
            if item_id in relevant_items:
                relevance = relevant_items[item_id]
                dcg += relevance / np.log2(i + 2)  # i+2 because i starts at 0

        # IDCG
        ideal_relevances = sorted(relevant_items.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def map_at_k(
        relevant_items: Set[int],
        recommended_items: List[int],
        k: int
    ) -> float:
        """
        Calculate Mean Average Precision@K.

        Args:
            relevant_items: Set of relevant item IDs.
            recommended_items: List of recommended item IDs.
            k: Number of recommendations to consider.

        Returns:
            MAP@K value.
        """
        if len(relevant_items) == 0:
            return 0.0

        score = 0.0
        n_relevant = 0

        for i, item_id in enumerate(recommended_items[:k]):
            if item_id in relevant_items:
                n_relevant += 1
                score += n_relevant / (i + 1)

        if n_relevant == 0:
            return 0.0

        return score / min(len(relevant_items), k)

    @staticmethod
    def diversity(recommended_items: List[List[int]]) -> float:
        """
        Calculate diversity of recommendations across users.

        Measures how many unique items are recommended overall.

        Args:
            recommended_items: List of recommendation lists for different users.

        Returns:
            Diversity score (ratio of unique items to total recommendations).
        """
        all_items = [item for user_recs in recommended_items for item in user_recs]

        if len(all_items) == 0:
            return 0.0

        unique_items = len(set(all_items))
        total_items = len(all_items)

        return unique_items / total_items

    @staticmethod
    def coverage(
        recommended_items: List[List[int]],
        all_items: Set[int]
    ) -> float:
        """
        Calculate catalog coverage.

        Measures what fraction of items can be recommended.

        Args:
            recommended_items: List of recommendation lists for different users.
            all_items: Set of all available item IDs.

        Returns:
            Coverage score (ratio of recommended items to all items).
        """
        if len(all_items) == 0:
            return 0.0

        recommended_unique = set(item for user_recs in recommended_items for item in user_recs)

        return len(recommended_unique) / len(all_items)

    @staticmethod
    def novelty(
        recommended_items: List[List[int]],
        item_popularity: Dict[int, int]
    ) -> float:
        """
        Calculate novelty of recommendations.

        Higher novelty means recommending less popular items.

        Args:
            recommended_items: List of recommendation lists for different users.
            item_popularity: Dictionary of {item_id: popularity_count}.

        Returns:
            Average novelty score.
        """
        total_popularity = sum(item_popularity.values())

        if total_popularity == 0:
            return 0.0

        novelty_scores = []

        for user_recs in recommended_items:
            for item_id in user_recs:
                if item_id in item_popularity:
                    # Novelty = -log2(popularity / total)
                    prob = item_popularity[item_id] / total_popularity
                    novelty = -np.log2(prob) if prob > 0 else 0
                    novelty_scores.append(novelty)

        return np.mean(novelty_scores) if novelty_scores else 0.0

    def evaluate_rating_prediction(
        self,
        model: BaseRecommender,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate rating prediction performance.

        Args:
            model: Trained recommender model.
            test_data: Test data with columns [user_id, item_id, rating].

        Returns:
            Dictionary with RMSE and MAE.
        """
        logger.info("Evaluating rating prediction...")

        y_true = []
        y_pred = []

        for _, row in test_data.iterrows():
            try:
                pred = model.predict(row['user_id'], row['item_id'])
                y_true.append(row['rating'])
                y_pred.append(pred)
            except Exception as e:
                continue

        if len(y_true) == 0:
            logger.warning("No predictions could be made")
            return {'rmse': float('inf'), 'mae': float('inf')}

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        results = {
            'rmse': self.rmse(y_true, y_pred),
            'mae': self.mae(y_true, y_pred),
            'n_predictions': len(y_true)
        }

        logger.info(f"RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}")
        return results

    def evaluate_ranking(
        self,
        model: BaseRecommender,
        test_data: pd.DataFrame,
        k_values: List[int] = None,
        rating_threshold: float = 4.0
    ) -> Dict[str, any]:
        """
        Evaluate ranking performance.

        Args:
            model: Trained recommender model.
            test_data: Test data with columns [user_id, item_id, rating].
            k_values: List of K values to evaluate.
            rating_threshold: Threshold for considering items as relevant.

        Returns:
            Dictionary with ranking metrics for each K.
        """
        logger.info("Evaluating ranking performance...")

        if k_values is None:
            k_values = self.config["evaluation"]["k_values"]

        # Group test data by user
        user_relevant_items = {}
        for user_id, group in test_data.groupby('user_id'):
            # Items with rating >= threshold are considered relevant
            relevant = group[group['rating'] >= rating_threshold]
            user_relevant_items[user_id] = {
                row['item_id']: row['rating']
                for _, row in relevant.iterrows()
            }

        results = {k: {
            'precision': [],
            'recall': [],
            'f1': [],
            'ndcg': [],
            'map': []
        } for k in k_values}

        n_users = 0

        for user_id, relevant_items in user_relevant_items.items():
            if len(relevant_items) == 0:
                continue

            try:
                # Get recommendations
                recommendations = model.recommend(user_id, max(k_values), exclude_seen=True)
                recommended_ids = [item_id for item_id, _ in recommendations]

                # Evaluate for each K
                for k in k_values:
                    relevant_set = set(relevant_items.keys())

                    precision = self.precision_at_k(relevant_set, recommended_ids, k)
                    recall = self.recall_at_k(relevant_set, recommended_ids, k)
                    f1 = self.f1_at_k(relevant_set, recommended_ids, k)
                    ndcg = self.ndcg_at_k(relevant_items, recommended_ids, k)
                    map_score = self.map_at_k(relevant_set, recommended_ids, k)

                    results[k]['precision'].append(precision)
                    results[k]['recall'].append(recall)
                    results[k]['f1'].append(f1)
                    results[k]['ndcg'].append(ndcg)
                    results[k]['map'].append(map_score)

                n_users += 1

            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue

        # Aggregate results
        aggregated_results = {}
        for k in k_values:
            aggregated_results[f'precision@{k}'] = np.mean(results[k]['precision']) if results[k]['precision'] else 0.0
            aggregated_results[f'recall@{k}'] = np.mean(results[k]['recall']) if results[k]['recall'] else 0.0
            aggregated_results[f'f1@{k}'] = np.mean(results[k]['f1']) if results[k]['f1'] else 0.0
            aggregated_results[f'ndcg@{k}'] = np.mean(results[k]['ndcg']) if results[k]['ndcg'] else 0.0
            aggregated_results[f'map@{k}'] = np.mean(results[k]['map']) if results[k]['map'] else 0.0

        aggregated_results['n_users'] = n_users

        logger.info(f"Evaluated {n_users} users")
        for k in k_values:
            logger.info(f"K={k}: Precision={aggregated_results[f'precision@{k}']:.4f}, "
                       f"Recall={aggregated_results[f'recall@{k}']:.4f}, "
                       f"NDCG={aggregated_results[f'ndcg@{k}']:.4f}")

        return aggregated_results

    def evaluate_diversity_coverage(
        self,
        model: BaseRecommender,
        test_data: pd.DataFrame,
        all_items: Set[int],
        n_recommendations: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate diversity and coverage metrics.

        Args:
            model: Trained recommender model.
            test_data: Test data with columns [user_id, item_id, rating].
            all_items: Set of all available item IDs.
            n_recommendations: Number of recommendations per user.

        Returns:
            Dictionary with diversity and coverage metrics.
        """
        logger.info("Evaluating diversity and coverage...")

        users = test_data['user_id'].unique()
        all_recommendations = []

        for user_id in users:
            try:
                recommendations = model.recommend(user_id, n_recommendations, exclude_seen=True)
                recommended_ids = [item_id for item_id, _ in recommendations]
                all_recommendations.append(recommended_ids)
            except Exception as e:
                continue

        # Calculate item popularity from training data
        item_popularity = test_data['item_id'].value_counts().to_dict()

        results = {
            'diversity': self.diversity(all_recommendations),
            'coverage': self.coverage(all_recommendations, all_items),
            'novelty': self.novelty(all_recommendations, item_popularity),
            'n_users': len(all_recommendations)
        }

        logger.info(f"Diversity: {results['diversity']:.4f}, "
                   f"Coverage: {results['coverage']:.4f}, "
                   f"Novelty: {results['novelty']:.4f}")

        return results

    def evaluate_all(
        self,
        model: BaseRecommender,
        test_data: pd.DataFrame,
        all_items: Set[int],
        k_values: List[int] = None
    ) -> Dict[str, any]:
        """
        Run all evaluation metrics.

        Args:
            model: Trained recommender model.
            test_data: Test data with columns [user_id, item_id, rating].
            all_items: Set of all available item IDs.
            k_values: List of K values for ranking metrics.

        Returns:
            Dictionary with all evaluation results.
        """
        logger.info(f"Running comprehensive evaluation for {model.name}...")

        results = {
            'model_name': model.name,
            'rating_prediction': self.evaluate_rating_prediction(model, test_data),
            'ranking': self.evaluate_ranking(model, test_data, k_values),
            'diversity_coverage': self.evaluate_diversity_coverage(model, test_data, all_items)
        }

        logger.info(f"Evaluation completed for {model.name}")
        return results
