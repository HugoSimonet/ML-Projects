"""Deep learning recommender using Neural Collaborative Filtering."""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from .base import BaseRecommender
from ..utils.config_loader import load_config


class RatingsDataset(Dataset):
    """PyTorch Dataset for ratings data."""

    def __init__(self, ratings_df: pd.DataFrame):
        """
        Initialize dataset.

        Args:
            ratings_df: DataFrame with columns [user_id, item_id, rating].
        """
        self.users = ratings_df['user_id'].values
        self.items = ratings_df['item_id'].values
        self.ratings = ratings_df['rating'].values

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32)
        )


class NCFModel(nn.Module):
    """Neural Collaborative Filtering model."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        """
        Initialize NCF model.

        Args:
            n_users: Number of users.
            n_items: Number of items.
            embedding_dim: Dimension of embeddings.
            hidden_layers: List of hidden layer sizes.
            dropout: Dropout rate.
        """
        super(NCFModel, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        layers = []
        input_size = embedding_dim * 2

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            user_ids: Tensor of user IDs.
            item_ids: Tensor of item IDs.

        Returns:
            Predicted ratings.
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)

        # Pass through MLP
        output = self.mlp(x)

        return output.squeeze()


class NCFRecommender(BaseRecommender):
    """
    Neural Collaborative Filtering recommender.

    Uses deep learning to learn user and item representations.
    """

    def __init__(
        self,
        embedding_dim: int = None,
        hidden_layers: List[int] = None,
        dropout: float = None,
        learning_rate: float = None,
        batch_size: int = None,
        epochs: int = None
    ):
        """
        Initialize NCF recommender.

        Args:
            embedding_dim: Dimension of embeddings.
            hidden_layers: List of hidden layer sizes.
            dropout: Dropout rate.
            learning_rate: Learning rate for optimizer.
            batch_size: Batch size for training.
            epochs: Number of training epochs.
        """
        super().__init__("NCF")

        config = load_config()
        ncf_config = config["models"]["deep_learning"]["ncf"]

        self.embedding_dim = embedding_dim or ncf_config["embedding_dim"]
        self.hidden_layers = hidden_layers or ncf_config["hidden_layers"]
        self.dropout = dropout or ncf_config["dropout"]
        self.learning_rate = learning_rate or ncf_config["learning_rate"]
        self.batch_size = batch_size or ncf_config["batch_size"]
        self.epochs = epochs or ncf_config["epochs"]

        self.model = None
        self.train_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}

        logger.info(f"Initialized NCF (embedding_dim={self.embedding_dim}, device={self.device})")

    def create_mappings(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create user and item ID mappings.

        Args:
            train_data: Training data with columns [user_id, item_id, rating].

        Returns:
            DataFrame with mapped IDs.
        """
        unique_users = train_data['user_id'].unique()
        unique_items = train_data['item_id'].unique()

        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        # Map IDs in dataframe
        df = train_data.copy()
        df['user_id'] = df['user_id'].map(self.user_to_idx)
        df['item_id'] = df['item_id'].map(self.item_to_idx)

        return df

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame = None) -> None:
        """
        Train the NCF model.

        Args:
            train_data: Training data with columns [user_id, item_id, rating].
            validation_data: Optional validation data.
        """
        logger.info(f"Training {self.name} with {len(train_data)} ratings...")

        # Store original data
        self.train_data = train_data.copy()

        # Create mappings and map IDs
        train_mapped = self.create_mappings(train_data)

        # Initialize model
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        self.model = NCFModel(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(self.device)

        # Create dataset and dataloader
        train_dataset = RatingsDataset(train_mapped)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            n_batches = 0

            for user_ids, item_ids, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)

                # Forward pass
                predictions = self.model(user_ids, item_ids)

                # Compute loss
                loss = criterion(predictions, ratings)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

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

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)

            prediction = self.model(user_tensor, item_tensor)

        return float(prediction.cpu().numpy()[0])

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

        # Get all items
        all_items = list(self.item_to_idx.keys())

        # Exclude seen items if requested
        if exclude_seen:
            seen_items = set(self.train_data[
                self.train_data['user_id'] == user_id
            ]['item_id'].values)
            candidate_items = [item for item in all_items if item not in seen_items]
        else:
            candidate_items = all_items

        # Predict ratings for all candidate items
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for item_id in candidate_items:
                item_idx = self.item_to_idx[item_id]

                user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)

                prediction = self.model(user_tensor, item_tensor)
                predictions.append((item_id, float(prediction.cpu().numpy()[0])))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]

    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save the model.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item,
            'config': {
                'embedding_dim': self.embedding_dim,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout,
            }
        }, filepath)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.

        Args:
            filepath: Path to the saved model.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore mappings
        self.user_to_idx = checkpoint['user_to_idx']
        self.item_to_idx = checkpoint['item_to_idx']
        self.idx_to_user = checkpoint['idx_to_user']
        self.idx_to_item = checkpoint['idx_to_item']

        # Restore config
        config = checkpoint['config']
        self.embedding_dim = config['embedding_dim']
        self.hidden_layers = config['hidden_layers']
        self.dropout = config['dropout']

        # Initialize and load model
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        self.model = NCFModel(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
