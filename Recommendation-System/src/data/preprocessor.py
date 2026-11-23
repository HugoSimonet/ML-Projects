"""Data preprocessing utilities."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ..utils.config_loader import get_data_path, load_config


class DataPreprocessor:
    """
    Preprocessor for recommendation system data.

    Handles:
        - Train/test splitting
        - User and item encoding
        - Rating normalization
        - Sparse matrix creation
        - Cold start handling
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize preprocessor.

        Args:
            random_seed: Random seed for reproducibility.
        """
        self.config = load_config()
        self.random_seed = random_seed or self.config["training"]["random_seed"]

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.rating_scaler = MinMaxScaler()

        self.is_fitted = False

        logger.info("Initialized DataPreprocessor")

    def encode_ids(
        self,
        ratings_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'item_id'
    ) -> pd.DataFrame:
        """
        Encode user and item IDs to continuous integers.

        Args:
            ratings_df: DataFrame with ratings.
            user_col: Name of user ID column.
            item_col: Name of item ID column.

        Returns:
            DataFrame with encoded IDs.
        """
        df = ratings_df.copy()

        if not self.is_fitted:
            df[user_col] = self.user_encoder.fit_transform(df[user_col])
            df[item_col] = self.item_encoder.fit_transform(df[item_col])
            self.is_fitted = True
        else:
            # Handle new users/items not seen during training
            known_users = set(self.user_encoder.classes_)
            known_items = set(self.item_encoder.classes_)

            df = df[
                df[user_col].isin(known_users) &
                df[item_col].isin(known_items)
            ]

            df[user_col] = self.user_encoder.transform(df[user_col])
            df[item_col] = self.item_encoder.transform(df[item_col])

        logger.info(f"Encoded {len(df)} ratings with {df[user_col].nunique()} users and {df[item_col].nunique()} items")
        return df

    def normalize_ratings(
        self,
        ratings_df: pd.DataFrame,
        rating_col: str = 'rating'
    ) -> pd.DataFrame:
        """
        Normalize ratings to [0, 1] scale.

        Args:
            ratings_df: DataFrame with ratings.
            rating_col: Name of rating column.

        Returns:
            DataFrame with normalized ratings.
        """
        df = ratings_df.copy()

        if not self.is_fitted:
            df[rating_col] = self.rating_scaler.fit_transform(
                df[[rating_col]]
            ).flatten()
        else:
            df[rating_col] = self.rating_scaler.transform(
                df[[rating_col]]
            ).flatten()

        return df

    def create_interaction_matrix(
        self,
        ratings_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        rating_col: str = 'rating'
    ) -> np.ndarray:
        """
        Create user-item interaction matrix.

        Args:
            ratings_df: DataFrame with ratings.
            user_col: Name of user ID column.
            item_col: Name of item ID column.
            rating_col: Name of rating column.

        Returns:
            User-item matrix (users x items).
        """
        n_users = ratings_df[user_col].max() + 1
        n_items = ratings_df[item_col].max() + 1

        matrix = np.zeros((n_users, n_items))

        for _, row in ratings_df.iterrows():
            matrix[int(row[user_col]), int(row[item_col])] = row[rating_col]

        logger.info(f"Created interaction matrix: {matrix.shape}")
        return matrix

    def train_test_split(
        self,
        ratings_df: pd.DataFrame,
        test_size: Optional[float] = None,
        stratify_by_user: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.

        Args:
            ratings_df: DataFrame with ratings.
            test_size: Proportion of data for testing.
            stratify_by_user: Whether to ensure all users are in train set.

        Returns:
            Tuple of (train_df, test_df).
        """
        test_size = test_size or self.config["training"]["test_size"]

        if stratify_by_user:
            # Ensure each user has at least one rating in train set
            train_list = []
            test_list = []

            for user_id, user_data in ratings_df.groupby('user_id'):
                if len(user_data) == 1:
                    # If user has only one rating, put it in train
                    train_list.append(user_data)
                else:
                    # Split user's ratings
                    user_train, user_test = train_test_split(
                        user_data,
                        test_size=test_size,
                        random_state=self.random_seed
                    )
                    train_list.append(user_train)
                    test_list.append(user_test)

            train_df = pd.concat(train_list, ignore_index=True)
            test_df = pd.concat(test_list, ignore_index=True)

        else:
            train_df, test_df = train_test_split(
                ratings_df,
                test_size=test_size,
                random_state=self.random_seed
            )

        logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df

    def create_cv_folds(
        self,
        ratings_df: pd.DataFrame,
        n_folds: Optional[int] = None
    ) -> list:
        """
        Create cross-validation folds.

        Args:
            ratings_df: DataFrame with ratings.
            n_folds: Number of folds.

        Returns:
            List of (train_df, test_df) tuples.
        """
        n_folds = n_folds or self.config["training"]["cv_folds"]

        folds = []
        ratings_shuffled = ratings_df.sample(frac=1, random_state=self.random_seed)

        fold_size = len(ratings_shuffled) // n_folds

        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(ratings_shuffled)

            test_df = ratings_shuffled.iloc[start_idx:end_idx]
            train_df = ratings_shuffled.drop(test_df.index)

            folds.append((train_df, test_df))

        logger.info(f"Created {n_folds} cross-validation folds")
        return folds

    def handle_cold_start(
        self,
        ratings_df: pd.DataFrame,
        min_user_ratings: int = 5,
        min_item_ratings: int = 5
    ) -> pd.DataFrame:
        """
        Filter out users and items with too few ratings.

        Args:
            ratings_df: DataFrame with ratings.
            min_user_ratings: Minimum number of ratings per user.
            min_item_ratings: Minimum number of ratings per item.

        Returns:
            Filtered DataFrame.
        """
        original_size = len(ratings_df)

        # Filter users
        user_counts = ratings_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = ratings_df[ratings_df['user_id'].isin(valid_users)]

        # Filter items
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        df = df[df['item_id'].isin(valid_items)]

        logger.info(f"Filtered cold start: {original_size} -> {len(df)} ratings")
        return df

    def get_statistics(self, ratings_df: pd.DataFrame) -> Dict[str, any]:
        """
        Get dataset statistics.

        Args:
            ratings_df: DataFrame with ratings.

        Returns:
            Dictionary with statistics.
        """
        stats = {
            'n_ratings': len(ratings_df),
            'n_users': ratings_df['user_id'].nunique(),
            'n_items': ratings_df['item_id'].nunique(),
            'sparsity': 1 - (len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['item_id'].nunique())),
            'rating_mean': ratings_df['rating'].mean(),
            'rating_std': ratings_df['rating'].std(),
            'rating_min': ratings_df['rating'].min(),
            'rating_max': ratings_df['rating'].max(),
            'avg_ratings_per_user': len(ratings_df) / ratings_df['user_id'].nunique(),
            'avg_ratings_per_item': len(ratings_df) / ratings_df['item_id'].nunique(),
        }

        return stats

    def preprocess_pipeline(
        self,
        ratings_df: pd.DataFrame,
        handle_cold_start: bool = True,
        encode: bool = True,
        normalize: bool = False,
        create_matrix: bool = False
    ) -> Dict[str, any]:
        """
        Run complete preprocessing pipeline.

        Args:
            ratings_df: DataFrame with ratings.
            handle_cold_start: Whether to filter cold start users/items.
            encode: Whether to encode user/item IDs.
            normalize: Whether to normalize ratings.
            create_matrix: Whether to create interaction matrix.

        Returns:
            Dictionary with preprocessed data and metadata.
        """
        logger.info("Starting preprocessing pipeline...")

        df = ratings_df.copy()

        # Get initial statistics
        stats_before = self.get_statistics(df)
        logger.info(f"Initial stats: {stats_before['n_users']} users, {stats_before['n_items']} items, {stats_before['n_ratings']} ratings")

        # Handle cold start
        if handle_cold_start:
            df = self.handle_cold_start(df)

        # Encode IDs
        if encode:
            df = self.encode_ids(df)

        # Normalize ratings
        if normalize:
            df = self.normalize_ratings(df)

        # Split data
        train_df, test_df = self.train_test_split(df)

        # Get final statistics
        stats_after = self.get_statistics(df)

        result = {
            'train': train_df,
            'test': test_df,
            'full': df,
            'stats_before': stats_before,
            'stats_after': stats_after,
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
        }

        # Create interaction matrix if requested
        if create_matrix:
            result['train_matrix'] = self.create_interaction_matrix(train_df)
            result['test_matrix'] = self.create_interaction_matrix(test_df)

        logger.info("Preprocessing pipeline completed")
        return result

    def save_preprocessed_data(
        self,
        data: Dict[str, any],
        dataset_name: str
    ) -> None:
        """
        Save preprocessed data to disk.

        Args:
            data: Dictionary with preprocessed data.
            dataset_name: Name of the dataset.
        """
        processed_path = get_data_path("processed") / dataset_name
        processed_path.mkdir(parents=True, exist_ok=True)

        # Save dataframes
        data['train'].to_csv(processed_path / "train.csv", index=False)
        data['test'].to_csv(processed_path / "test.csv", index=False)
        data['full'].to_csv(processed_path / "full.csv", index=False)

        # Save encoders and statistics
        import joblib
        joblib.dump(data['user_encoder'], processed_path / "user_encoder.pkl")
        joblib.dump(data['item_encoder'], processed_path / "item_encoder.pkl")

        # Save statistics
        import json
        with open(processed_path / "stats.json", 'w') as f:
            json.dump({
                'stats_before': data['stats_before'],
                'stats_after': data['stats_after']
            }, f, indent=2)

        logger.info(f"Saved preprocessed data to {processed_path}")
