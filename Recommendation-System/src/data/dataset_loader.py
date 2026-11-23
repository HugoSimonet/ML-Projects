"""Dataset loader for multiple domains."""

import os
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlretrieve

import pandas as pd
from loguru import logger
from tqdm import tqdm

from ..utils.config_loader import get_data_path, load_config


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class DatasetLoader:
    """
    Unified dataset loader supporting multiple domains.

    Supports:
        - MovieLens (100K, 1M, 10M)
        - Last.fm music dataset
        - Amazon product reviews
    """

    def __init__(self, dataset_name: str, version: Optional[str] = None):
        """
        Initialize dataset loader.

        Args:
            dataset_name: Name of the dataset ('movielens', 'lastfm', 'amazon').
            version: Version of the dataset (specific to dataset type).
        """
        self.dataset_name = dataset_name.lower()
        self.version = version
        self.config = load_config()
        self.raw_data_path = get_data_path("raw")
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized loader for {dataset_name} dataset")

    def download_and_extract(self, url: str, extract_path: Path) -> None:
        """
        Download and extract dataset.

        Args:
            url: URL to download from.
            extract_path: Path to extract files to.
        """
        filename = url.split("/")[-1]
        filepath = self.raw_data_path / filename

        if not filepath.exists():
            logger.info(f"Downloading {filename}...")
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                urlretrieve(url, filepath, reporthook=t.update_to)
            logger.info(f"Downloaded {filename}")
        else:
            logger.info(f"{filename} already exists, skipping download")

        # Extract if it's a zip file
        if filename.endswith('.zip'):
            logger.info(f"Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            logger.info(f"Extracted {filename}")

    def load_movielens(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load MovieLens dataset.

        Returns:
            Tuple of (ratings, movies, users) DataFrames.
        """
        version = self.version or self.config["datasets"]["movielens"]["default_version"]
        url = self.config["datasets"]["movielens"]["urls"][version]

        extract_path = self.raw_data_path / f"ml-{version}"

        # Download and extract if needed
        if not extract_path.exists():
            self.download_and_extract(url, self.raw_data_path)

        logger.info(f"Loading MovieLens {version} data...")

        if version == "100k":
            # MovieLens 100K format
            ratings = pd.read_csv(
                extract_path / "u.data",
                sep='\t',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
            movies = pd.read_csv(
                extract_path / "u.item",
                sep='|',
                encoding='latin-1',
                names=['item_id', 'title', 'release_date', 'video_release_date',
                       'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                       'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                       'Thriller', 'War', 'Western'],
                engine='python'
            )
            users = pd.read_csv(
                extract_path / "u.user",
                sep='|',
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                engine='python'
            )

        elif version == "1m":
            # MovieLens 1M format
            ratings = pd.read_csv(
                extract_path / "ratings.dat",
                sep='::',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
            movies = pd.read_csv(
                extract_path / "movies.dat",
                sep='::',
                encoding='latin-1',
                names=['item_id', 'title', 'genres'],
                engine='python'
            )
            users = pd.read_csv(
                extract_path / "users.dat",
                sep='::',
                names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                engine='python'
            )

        else:  # 10m
            # MovieLens 10M format
            ratings = pd.read_csv(
                extract_path / "ratings.dat",
                sep='::',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
            movies = pd.read_csv(
                extract_path / "movies.dat",
                sep='::',
                encoding='latin-1',
                names=['item_id', 'title', 'genres'],
                engine='python'
            )
            users = None  # No user data in 10M

        logger.info(f"Loaded {len(ratings)} ratings, {len(movies)} movies")
        return ratings, movies, users

    def load_lastfm(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load Last.fm dataset.

        Returns:
            Tuple of (user_artists, artists) DataFrames.

        Note:
            This is a placeholder. The actual Last.fm dataset needs manual download.
            Users should download from: http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html
        """
        data_path = self.raw_data_path / "lastfm-360K"

        if not data_path.exists():
            logger.error(
                "Last.fm dataset not found. Please download manually from:\n"
                "http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html\n"
                f"and extract to {data_path}"
            )
            raise FileNotFoundError(f"Last.fm dataset not found at {data_path}")

        logger.info("Loading Last.fm data...")

        user_artists = pd.read_csv(
            data_path / "usersha1-artmbid-artname-plays.tsv",
            sep='\t',
            names=['user_id', 'artist_id', 'artist_name', 'plays']
        )

        # Rename columns to standard format
        user_artists = user_artists.rename(columns={
            'artist_id': 'item_id',
            'plays': 'rating'
        })

        # Create artists dataframe
        artists = user_artists[['item_id', 'artist_name']].drop_duplicates()

        logger.info(f"Loaded {len(user_artists)} user-artist interactions")
        return user_artists, artists

    def load_amazon(self, category: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load Amazon reviews dataset.

        Args:
            category: Product category (Electronics, Books, etc.).

        Returns:
            Tuple of (reviews, products) DataFrames.

        Note:
            This is a placeholder for Amazon dataset.
            Users should download from: https://nijianmo.github.io/amazon/index.html
        """
        category = category or self.config["datasets"]["amazon"]["default_category"]
        data_path = self.raw_data_path / "amazon" / category

        if not data_path.exists():
            logger.error(
                f"Amazon {category} dataset not found. Please download from:\n"
                "https://nijianmo.github.io/amazon/index.html\n"
                f"and place in {data_path}"
            )
            raise FileNotFoundError(f"Amazon dataset not found at {data_path}")

        logger.info(f"Loading Amazon {category} data...")

        # This assumes the reviews are in JSON format
        reviews_file = list(data_path.glob("*.json"))[0]
        reviews = pd.read_json(reviews_file, lines=True)

        # Standardize column names
        reviews = reviews.rename(columns={
            'reviewerID': 'user_id',
            'asin': 'item_id',
            'overall': 'rating',
            'unixReviewTime': 'timestamp'
        })

        # Extract product information
        products = reviews[['item_id', 'asin']].drop_duplicates()

        logger.info(f"Loaded {len(reviews)} reviews")
        return reviews, products

    def create_synthetic_data(
        self,
        n_users: int = 1000,
        n_items: int = 500,
        n_ratings: int = 10000,
        rating_scale: Tuple[int, int] = (1, 5)
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create synthetic e-commerce data for testing.

        Args:
            n_users: Number of users.
            n_items: Number of items.
            n_ratings: Number of ratings.
            rating_scale: Min and max rating values.

        Returns:
            Tuple of (ratings, items) DataFrames.
        """
        import numpy as np

        logger.info(f"Creating synthetic data: {n_users} users, {n_items} items, {n_ratings} ratings")

        np.random.seed(42)

        # Generate ratings
        user_ids = np.random.randint(1, n_users + 1, n_ratings)
        item_ids = np.random.randint(1, n_items + 1, n_ratings)
        ratings = np.random.randint(rating_scale[0], rating_scale[1] + 1, n_ratings)
        timestamps = np.random.randint(1000000000, 1600000000, n_ratings)

        ratings_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': timestamps
        }).drop_duplicates(subset=['user_id', 'item_id'])

        # Generate items with categories
        categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports', 'Toys']
        items_df = pd.DataFrame({
            'item_id': range(1, n_items + 1),
            'title': [f"Product {i}" for i in range(1, n_items + 1)],
            'category': np.random.choice(categories, n_items),
            'price': np.random.uniform(10, 500, n_items).round(2)
        })

        logger.info(f"Created {len(ratings_df)} unique ratings")
        return ratings_df, items_df

    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load dataset based on initialized configuration.

        Returns:
            Dictionary containing loaded dataframes.
        """
        if self.dataset_name == "movielens":
            ratings, movies, users = self.load_movielens()
            return {
                'ratings': ratings,
                'items': movies,
                'users': users
            }

        elif self.dataset_name == "lastfm":
            interactions, artists = self.load_lastfm()
            return {
                'ratings': interactions,
                'items': artists
            }

        elif self.dataset_name == "amazon":
            reviews, products = self.load_amazon()
            return {
                'ratings': reviews,
                'items': products
            }

        elif self.dataset_name == "synthetic":
            ratings, items = self.create_synthetic_data()
            return {
                'ratings': ratings,
                'items': items
            }

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
