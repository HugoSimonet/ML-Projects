"""Feature engineering module for transforming raw data into ML features."""

import logging
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import pickle


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering and transformation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize feature engineer.

        Args:
            config: Configuration dictionary containing:
                - numeric_features: List of numeric feature names
                - categorical_features: List of categorical feature names
                - text_features: List of text feature names
                - datetime_features: List of datetime feature names
                - scaling_method: Method for scaling (standard, minmax, none)
                - custom_transformations: Custom transformation functions
        """
        self.config = config
        self.numeric_features = config.get('numeric_features', [])
        self.categorical_features = config.get('categorical_features', [])
        self.text_features = config.get('text_features', [])
        self.datetime_features = config.get('datetime_features', [])
        self.scaling_method = config.get('scaling_method', 'standard')

        # Initialize transformers
        self.scaler = None
        self.encoders = {}
        self.vectorizers = {}
        self.is_fitted = False

        logger.info("Initialized FeatureEngineer")

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit transformers and transform data.

        Args:
            data: Raw data to transform

        Returns:
            Transformed data with engineered features
        """
        logger.info(f"Fitting and transforming data with {len(data)} rows")

        # Create a copy to avoid modifying original
        transformed_data = data.copy()

        # Handle numeric features
        if self.numeric_features:
            transformed_data = self._transform_numeric(transformed_data, fit=True)

        # Handle categorical features
        if self.categorical_features:
            transformed_data = self._transform_categorical(transformed_data, fit=True)

        # Handle text features
        if self.text_features:
            transformed_data = self._transform_text(transformed_data, fit=True)

        # Handle datetime features
        if self.datetime_features:
            transformed_data = self._transform_datetime(transformed_data)

        # Create interaction features
        transformed_data = self._create_interactions(transformed_data)

        self.is_fitted = True
        logger.info(f"Transformation complete: {len(transformed_data.columns)} features")

        return transformed_data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted transformers.

        Args:
            data: Raw data to transform

        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        logger.info(f"Transforming data with {len(data)} rows")

        transformed_data = data.copy()

        if self.numeric_features:
            transformed_data = self._transform_numeric(transformed_data, fit=False)

        if self.categorical_features:
            transformed_data = self._transform_categorical(transformed_data, fit=False)

        if self.text_features:
            transformed_data = self._transform_text(transformed_data, fit=False)

        if self.datetime_features:
            transformed_data = self._transform_datetime(transformed_data)

        transformed_data = self._create_interactions(transformed_data)

        return transformed_data

    def _transform_numeric(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Transform numeric features."""
        available_features = [f for f in self.numeric_features if f in data.columns]

        if not available_features:
            return data

        if fit:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                logger.info("No scaling applied to numeric features")
                return data

            data[available_features] = self.scaler.fit_transform(
                data[available_features]
            )
        else:
            if self.scaler is not None:
                data[available_features] = self.scaler.transform(
                    data[available_features]
                )

        return data

    def _transform_categorical(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Transform categorical features."""
        available_features = [f for f in self.categorical_features if f in data.columns]

        if not available_features:
            return data

        for feature in available_features:
            if fit:
                encoder = LabelEncoder()
                # Handle missing values
                data[feature] = data[feature].fillna('missing')
                data[f'{feature}_encoded'] = encoder.fit_transform(data[feature])
                self.encoders[feature] = encoder
            else:
                encoder = self.encoders.get(feature)
                if encoder:
                    data[feature] = data[feature].fillna('missing')
                    # Handle unseen categories
                    known_classes = set(encoder.classes_)
                    data[f'{feature}_encoded'] = data[feature].apply(
                        lambda x: encoder.transform([x])[0]
                        if x in known_classes
                        else -1
                    )

        return data

    def _transform_text(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Transform text features."""
        available_features = [f for f in self.text_features if f in data.columns]

        if not available_features:
            return data

        for feature in available_features:
            if fit:
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                # Handle missing values
                text_data = data[feature].fillna('').astype(str)
                tfidf_matrix = vectorizer.fit_transform(text_data)

                # Add TF-IDF features to dataframe
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f'{feature}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
                    index=data.index
                )
                data = pd.concat([data, tfidf_df], axis=1)
                self.vectorizers[feature] = vectorizer
            else:
                vectorizer = self.vectorizers.get(feature)
                if vectorizer:
                    text_data = data[feature].fillna('').astype(str)
                    tfidf_matrix = vectorizer.transform(text_data)

                    tfidf_df = pd.DataFrame(
                        tfidf_matrix.toarray(),
                        columns=[f'{feature}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
                        index=data.index
                    )
                    data = pd.concat([data, tfidf_df], axis=1)

        return data

    def _transform_datetime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform datetime features into useful components."""
        available_features = [f for f in self.datetime_features if f in data.columns]

        if not available_features:
            return data

        for feature in available_features:
            # Ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(data[feature]):
                data[feature] = pd.to_datetime(data[feature], errors='coerce')

            # Extract datetime components
            data[f'{feature}_year'] = data[feature].dt.year
            data[f'{feature}_month'] = data[feature].dt.month
            data[f'{feature}_day'] = data[feature].dt.day
            data[f'{feature}_dayofweek'] = data[feature].dt.dayofweek
            data[f'{feature}_hour'] = data[feature].dt.hour
            data[f'{feature}_is_weekend'] = (data[feature].dt.dayofweek >= 5).astype(int)

            # Cyclical encoding for periodic features
            data[f'{feature}_month_sin'] = np.sin(2 * np.pi * data[f'{feature}_month'] / 12)
            data[f'{feature}_month_cos'] = np.cos(2 * np.pi * data[f'{feature}_month'] / 12)
            data[f'{feature}_hour_sin'] = np.sin(2 * np.pi * data[f'{feature}_hour'] / 24)
            data[f'{feature}_hour_cos'] = np.cos(2 * np.pi * data[f'{feature}_hour'] / 24)

        return data

    def _create_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numeric features."""
        # Get available numeric features
        available_numeric = [f for f in self.numeric_features if f in data.columns]

        if len(available_numeric) < 2:
            return data

        # Create polynomial features for first few numeric features
        # (to avoid feature explosion)
        features_to_interact = available_numeric[:min(3, len(available_numeric))]

        for i, feat1 in enumerate(features_to_interact):
            for feat2 in features_to_interact[i+1:]:
                # Multiplication interaction
                data[f'{feat1}_x_{feat2}'] = data[feat1] * data[feat2]

        return data

    def save(self, path: str):
        """Save fitted transformers to disk."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before saving")

        state = {
            'config': self.config,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'vectorizers': self.vectorizers,
            'is_fitted': self.is_fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved FeatureEngineer to {path}")

    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """Load fitted transformers from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        engineer = cls(state['config'])
        engineer.scaler = state['scaler']
        engineer.encoders = state['encoders']
        engineer.vectorizers = state['vectorizers']
        engineer.is_fitted = state['is_fitted']

        logger.info(f"Loaded FeatureEngineer from {path}")
        return engineer
