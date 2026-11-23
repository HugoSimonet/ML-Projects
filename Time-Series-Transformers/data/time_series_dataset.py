"""
Time Series Dataset
Dataset classes for loading and processing time series data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from pathlib import Path
import warnings


class TimeSeriesDataset(Dataset):
    """
    General time series dataset for forecasting
    Supports both univariate and multivariate time series
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 24,
        features: str = 'MS',
        target: str = 'OT',
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 'h',
        scaler=None
    ):
        """
        Args:
            data: Time series data [N, C] or DataFrame
            seq_len: Input sequence length
            label_len: Start token length for decoder
            pred_len: Prediction length
            features: Forecasting task
                - 'M': Multivariate predict multivariate
                - 'S': Univariate predict univariate
                - 'MS': Multivariate predict univariate
            target: Target column name (for MS/S tasks)
            scale: Whether to scale data
            timeenc: Time encoding (0: raw, 1: timeF)
            freq: Frequency for time features
            scaler: Scaler object (fitted on training data)
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = scaler

        # Convert to numpy if DataFrame
        if isinstance(data, pd.DataFrame):
            self.df_stamp = data[['date']] if 'date' in data.columns else None
            self.data_stamp = self._process_timestamps(self.df_stamp)

            # Get data columns
            if target in data.columns and features in ['S', 'MS']:
                self.target_idx = data.columns.get_loc(target)

            self.data = data.select_dtypes(include=[np.number]).values
        else:
            self.data = data
            self.data_stamp = None

        # Data scaling
        if self.scale and self.scaler is not None:
            self.data = self.scaler.transform(self.data)
        elif self.scale:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)

        # Set dimensions
        self.num_samples = len(self.data) - seq_len - pred_len + 1

    def _process_timestamps(self, df_stamp):
        """Process timestamps to time features"""
        if df_stamp is None:
            return None

        # Convert to datetime
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            # Raw time features
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            df_stamp['minute'] = df_stamp.date.dt.minute

            # Select features based on frequency
            if self.freq == 'h':
                data_stamp = df_stamp[['month', 'day', 'weekday', 'hour']].values
            elif self.freq == 't':
                data_stamp = df_stamp[['month', 'day', 'weekday', 'hour', 'minute']].values
            elif self.freq == 'd':
                data_stamp = df_stamp[['month', 'day', 'weekday']].values
            else:
                data_stamp = df_stamp[['month', 'day', 'weekday', 'hour']].values
        else:
            # Time features from TimeFeatures (continuous encoding)
            data_stamp = self._time_features(df_stamp['date'].values, self.freq)

        return data_stamp

    def _time_features(self, dates, freq='h'):
        """Generate continuous time features"""
        dates = pd.to_datetime(dates)

        features = []

        # Hour of day (normalized to [-0.5, 0.5])
        if freq in ['h', 't', 's']:
            hour_feature = (dates.hour / 24.0) - 0.5
            features.append(hour_feature)

        # Day of week
        if freq in ['h', 't', 's', 'd', 'w']:
            dow_feature = (dates.dayofweek / 7.0) - 0.5
            features.append(dow_feature)

        # Day of month
        if freq in ['h', 't', 's', 'd']:
            dom_feature = ((dates.day - 1) / 31.0) - 0.5
            features.append(dom_feature)

        # Month of year
        if freq in ['h', 't', 's', 'd', 'w', 'm']:
            month_feature = ((dates.month - 1) / 12.0) - 0.5
            features.append(month_feature)

        return np.column_stack(features)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Returns:
            x_enc: Encoder input [seq_len, features]
            x_mark_enc: Encoder time features [seq_len, time_features]
            x_dec: Decoder input [label_len + pred_len, features]
            x_mark_dec: Decoder time features [label_len + pred_len, time_features]
            y: Target [pred_len, target_features]
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Encoder input
        x_enc = self.data[s_begin:s_end]

        # Decoder input
        x_dec = self.data[r_begin:r_end]

        # Target
        if self.features == 'M' or self.features == 'MS':
            y = self.data[s_end:r_end]
        else:
            y = self.data[s_end:r_end, self.target_idx:self.target_idx+1]

        # Time features
        if self.data_stamp is not None:
            x_mark_enc = self.data_stamp[s_begin:s_end]
            x_mark_dec = self.data_stamp[r_begin:r_end]
        else:
            # Create dummy time features based on frequency
            # Number of features needed for each frequency
            freq_feature_count = {
                's': 5, 't': 5, 'h': 4, 'd': 3, 'b': 3, 'w': 2, 'm': 1
            }
            num_features = freq_feature_count.get(self.freq, 4)
            x_mark_enc = np.zeros((self.seq_len, num_features))
            x_mark_dec = np.zeros((self.label_len + self.pred_len, num_features))

        return (
            torch.FloatTensor(x_enc),
            torch.FloatTensor(x_mark_enc),
            torch.FloatTensor(x_dec),
            torch.FloatTensor(x_mark_dec),
            torch.FloatTensor(y)
        )


class UnivariateDataset(Dataset):
    """Simple univariate time series dataset"""

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 24,
        scale: bool = True
    ):
        """
        Args:
            data: 1D array of time series values
            seq_len: Input sequence length
            pred_len: Prediction length
            scale: Whether to scale data
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale

        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Scale
        if scale:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(data)
        else:
            self.data = data
            self.scaler = None

        self.num_samples = len(self.data) - seq_len - pred_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        x = self.data[s_begin:s_end]
        y = self.data[s_end:r_end]

        return torch.FloatTensor(x), torch.FloatTensor(y)


class MultivariateDataset(Dataset):
    """Multivariate time series dataset"""

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 24,
        scale: bool = True,
        scale_per_feature: bool = False
    ):
        """
        Args:
            data: 2D array [N, C] of time series values
            seq_len: Input sequence length
            pred_len: Prediction length
            scale: Whether to scale data
            scale_per_feature: Scale each feature independently
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale

        # Scale
        if scale:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(data)
        else:
            self.data = data
            self.scaler = None

        self.num_samples = len(self.data) - seq_len - pred_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        x = self.data[s_begin:s_end]
        y = self.data[s_end:r_end]

        return torch.FloatTensor(x), torch.FloatTensor(y)


def create_data_loaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 24,
    batch_size: int = 32,
    num_workers: int = 0,
    features: str = 'MS',
    target: str = 'OT',
    freq: str = 'h'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders

    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        seq_len: Input sequence length
        label_len: Label length for decoder
        pred_len: Prediction length
        batch_size: Batch size
        num_workers: Number of workers for data loading
        features: Forecasting task type
        target: Target column
        freq: Time frequency

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_data, seq_len, label_len, pred_len,
        features, target, scale=True, freq=freq
    )

    # Use same scaler for val and test
    val_dataset = TimeSeriesDataset(
        val_data, seq_len, label_len, pred_len,
        features, target, scale=True, freq=freq,
        scaler=train_dataset.scaler
    )

    test_dataset = TimeSeriesDataset(
        test_data, seq_len, label_len, pred_len,
        features, target, scale=True, freq=freq,
        scaler=train_dataset.scaler
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def load_csv_dataset(
    filepath: str,
    target_column: str = 'target',
    date_column: str = 'date',
    train_ratio: float = 0.7,
    val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load time series data from CSV and split into train/val/test

    Args:
        filepath: Path to CSV file
        target_column: Name of target column
        date_column: Name of date column
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data

    Returns:
        train_df, val_df, test_df
    """
    # Load data
    df = pd.read_csv(filepath)

    # Sort by date if available
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

    # Split
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df
