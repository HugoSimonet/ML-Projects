"""
Data Utilities
Functions for loading and processing various time series datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Data loader for common time series datasets
    """

    def __init__(self, root_path: str = './data/'):
        """
        Args:
            root_path: Root directory for datasets
        """
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def load_electricity(
        self,
        filename: str = 'electricity.csv'
    ) -> pd.DataFrame:
        """
        Load Electricity dataset

        Args:
            filename: Dataset filename

        Returns:
            DataFrame with electricity data
        """
        filepath = self.root_path / filename
        if not filepath.exists():
            print(f"Dataset not found at {filepath}")
            print("Please download from: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014")
            return None

        df = pd.read_csv(filepath)
        return df

    def load_ett(
        self,
        dataset: str = 'ETTh1'
    ) -> pd.DataFrame:
        """
        Load ETT (Electricity Transformer Temperature) dataset

        Args:
            dataset: Dataset name ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')

        Returns:
            DataFrame with ETT data
        """
        filepath = self.root_path / f'{dataset}.csv'
        if not filepath.exists():
            print(f"Dataset not found at {filepath}")
            print("Please download from: https://github.com/zhouhaoyi/ETDataset")
            return None

        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def load_weather(
        self,
        filename: str = 'weather.csv'
    ) -> pd.DataFrame:
        """
        Load Weather dataset

        Args:
            filename: Dataset filename

        Returns:
            DataFrame with weather data
        """
        filepath = self.root_path / filename
        if not filepath.exists():
            print(f"Dataset not found at {filepath}")
            return None

        df = pd.read_csv(filepath)
        return df

    def load_traffic(
        self,
        filename: str = 'traffic.csv'
    ) -> pd.DataFrame:
        """
        Load Traffic dataset

        Args:
            filename: Dataset filename

        Returns:
            DataFrame with traffic data
        """
        filepath = self.root_path / filename
        if not filepath.exists():
            print(f"Dataset not found at {filepath}")
            return None

        df = pd.read_csv(filepath)
        return df

    def load_stock_data(
        self,
        ticker: str = 'AAPL',
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load stock data using yfinance

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date (default: today)

        Returns:
            DataFrame with stock data
        """
        try:
            import yfinance as yf
        except ImportError:
            print("Please install yfinance: pip install yfinance")
            return None

        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        df.columns = df.columns.str.lower()
        return df

    def load_custom_csv(
        self,
        filepath: str,
        date_column: Optional[str] = None,
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load custom CSV file

        Args:
            filepath: Path to CSV file
            date_column: Name of date column
            parse_dates: Whether to parse dates

        Returns:
            DataFrame
        """
        if parse_dates and date_column:
            df = pd.read_csv(filepath, parse_dates=[date_column])
        else:
            df = pd.read_csv(filepath)

        return df


def generate_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 1,
    trend_type: str = 'linear',
    seasonality_period: int = 24,
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic time series data

    Args:
        n_samples: Number of time steps
        n_features: Number of features
        trend_type: Type of trend ('linear', 'quadratic', 'none')
        seasonality_period: Period of seasonal component
        noise_level: Noise standard deviation
        seed: Random seed

    Returns:
        Synthetic data [n_samples, n_features]
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.zeros((n_samples, n_features))
    t = np.arange(n_samples)

    for i in range(n_features):
        # Trend
        if trend_type == 'linear':
            trend = 0.01 * t
        elif trend_type == 'quadratic':
            trend = 0.0001 * t ** 2
        else:
            trend = np.zeros(n_samples)

        # Seasonality
        seasonality = np.sin(2 * np.pi * t / seasonality_period)

        # Noise
        noise = np.random.normal(0, noise_level, n_samples)

        # Combine
        data[:, i] = trend + seasonality + noise

    return data


def add_temporal_features(
    df: pd.DataFrame,
    date_column: str = 'date',
    freq: str = 'h'
) -> pd.DataFrame:
    """
    Add temporal features to DataFrame

    Args:
        df: Input DataFrame
        date_column: Name of date column
        freq: Frequency of data

    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()

    if date_column not in df.columns:
        print(f"Column {date_column} not found")
        return df

    # Ensure datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['weekday'] = df[date_column].dt.weekday
    df['hour'] = df[date_column].dt.hour
    df['minute'] = df[date_column].dt.minute

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    if freq in ['h', 't', 's']:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    return df


def detect_outliers(
    data: np.ndarray,
    method: str = 'iqr',
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in time series

    Args:
        data: Input data [N] or [N, features]
        method: Detection method ('iqr', 'zscore')
        threshold: Threshold for detection

    Returns:
        Binary array (1 for outlier)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    outliers = np.zeros(len(data), dtype=bool)

    for i in range(data.shape[1]):
        feature_data = data[:, i]

        if method == 'iqr':
            q1 = np.percentile(feature_data, 25)
            q3 = np.percentile(feature_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            feature_outliers = (feature_data < lower_bound) | (feature_data > upper_bound)

        elif method == 'zscore':
            mean = np.mean(feature_data)
            std = np.std(feature_data)
            z_scores = np.abs((feature_data - mean) / std)

            feature_outliers = z_scores > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        outliers = outliers | feature_outliers

    return outliers.astype(int)


def handle_missing_values(
    data: np.ndarray,
    method: str = 'interpolate'
) -> np.ndarray:
    """
    Handle missing values in time series

    Args:
        data: Input data with NaN values
        method: Method to handle missing values
            - 'interpolate': Linear interpolation
            - 'forward_fill': Forward fill
            - 'backward_fill': Backward fill
            - 'mean': Fill with mean

    Returns:
        Data with missing values handled
    """
    data = data.copy()

    if method == 'interpolate':
        # Linear interpolation
        for i in range(data.shape[1] if data.ndim > 1 else 1):
            if data.ndim > 1:
                col = data[:, i]
            else:
                col = data

            mask = np.isnan(col)
            if not mask.any():
                continue

            # Get valid indices
            valid_idx = np.where(~mask)[0]
            if len(valid_idx) == 0:
                continue

            # Interpolate
            col[mask] = np.interp(
                np.where(mask)[0],
                valid_idx,
                col[valid_idx]
            )

            if data.ndim > 1:
                data[:, i] = col
            else:
                data = col

    elif method == 'forward_fill':
        df = pd.DataFrame(data)
        data = df.fillna(method='ffill').values

    elif method == 'backward_fill':
        df = pd.DataFrame(data)
        data = df.fillna(method='bfill').values

    elif method == 'mean':
        if data.ndim > 1:
            col_means = np.nanmean(data, axis=0)
            for i in range(data.shape[1]):
                data[np.isnan(data[:, i]), i] = col_means[i]
        else:
            data[np.isnan(data)] = np.nanmean(data)

    else:
        raise ValueError(f"Unknown method: {method}")

    return data


def detrend_data(
    data: np.ndarray,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove trend from time series

    Args:
        data: Input data [N]
        method: Detrending method ('linear', 'constant')

    Returns:
        Detrended data, trend
    """
    from scipy import signal

    detrended = signal.detrend(data, type=method)
    trend = data - detrended

    return detrended, trend


def difference_data(
    data: np.ndarray,
    order: int = 1,
    seasonal_period: Optional[int] = None
) -> np.ndarray:
    """
    Apply differencing to time series

    Args:
        data: Input data [N]
        order: Order of differencing
        seasonal_period: Seasonal differencing period

    Returns:
        Differenced data
    """
    diff_data = data.copy()

    # Regular differencing
    for _ in range(order):
        diff_data = np.diff(diff_data)

    # Seasonal differencing
    if seasonal_period is not None:
        diff_data = diff_data[seasonal_period:] - diff_data[:-seasonal_period]

    return diff_data


def create_sequences(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences for forecasting

    Args:
        data: Time series data [N, features]
        seq_len: Input sequence length
        pred_len: Prediction length
        stride: Stride for sliding window

    Returns:
        X: Input sequences [num_samples, seq_len, features]
        Y: Target sequences [num_samples, pred_len, features]
    """
    X, Y = [], []

    for i in range(0, len(data) - seq_len - pred_len + 1, stride):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len:i + seq_len + pred_len])

    return np.array(X), np.array(Y)


def split_by_time(
    data: pd.DataFrame,
    date_column: str,
    split_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by time

    Args:
        data: Input DataFrame
        date_column: Name of date column
        split_date: Date to split on

    Returns:
        Before split, after split
    """
    data[date_column] = pd.to_datetime(data[date_column])
    split_date = pd.to_datetime(split_date)

    before = data[data[date_column] < split_date]
    after = data[data[date_column] >= split_date]

    return before, after
