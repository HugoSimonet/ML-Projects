"""
Data loading and preprocessing utilities.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime, timedelta


def load_market_data(
    symbol: str = 'AAPL',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: str = 'synthetic'
) -> pd.DataFrame:
    """
    Load market data from various sources.

    Args:
        symbol: Asset symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        source: Data source ('yfinance', 'synthetic')

    Returns:
        DataFrame with OHLCV data
    """
    if source == 'yfinance':
        try:
            import yfinance as yf
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            data.columns = [col.lower() for col in data.columns]
            return data
        except ImportError:
            print("yfinance not installed, using synthetic data")
            return generate_synthetic_data(symbol, start_date, end_date)
    else:
        return generate_synthetic_data(symbol, start_date, end_date)


def generate_synthetic_data(
    symbol: str = 'SYN',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    n_days: int = 1000,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0005,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic market data with realistic patterns.

    Args:
        symbol: Symbol name
        start_date: Start date
        end_date: End date
        n_days: Number of trading days
        initial_price: Starting price
        volatility: Daily volatility
        trend: Daily trend (drift)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data and technical indicators
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate dates
    if start_date:
        start = pd.to_datetime(start_date)
    else:
        start = datetime.now() - timedelta(days=n_days)

    dates = pd.date_range(start=start, periods=n_days, freq='D')

    # Generate price series using geometric Brownian motion
    returns = np.random.normal(trend, volatility, n_days)

    # Add some autocorrelation and regime changes
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Momentum
        if i % 100 == 0:  # Occasional regime shifts
            returns[i] += np.random.choice([-0.05, 0.05])

    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    high_low_range = np.abs(np.random.normal(0, volatility * 0.5, n_days))

    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, volatility * 0.2, n_days)),
        'high': prices * (1 + high_low_range),
        'low': prices * (1 - high_low_range),
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, n_days)
    })

    data.set_index('date', inplace=True)

    return data


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: Input DataFrame
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        train_data, val_data, test_data
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    return train_data, val_data, test_data


def add_noise(data: pd.DataFrame, noise_level: float = 0.01) -> pd.DataFrame:
    """
    Add random noise to data for robustness testing.

    Args:
        data: Input DataFrame
        noise_level: Noise level as fraction of value

    Returns:
        DataFrame with added noise
    """
    noisy_data = data.copy()

    for col in ['open', 'high', 'low', 'close']:
        noise = np.random.normal(0, noise_level, len(data))
        noisy_data[col] = data[col] * (1 + noise)

    return noisy_data
