"""
Feature Engineering for Trading

Implements technical indicators and feature transformations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class TechnicalIndicators:
    """
    Technical indicator calculations.

    Implements common technical indicators for trading analysis.
    """

    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            prices: Price series
            period: RSI period (default 14)

        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Moving Average Convergence Divergence.

        Returns:
            Dict with 'macd', 'signal', and 'histogram'
        """
        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)

        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Bollinger Bands.

        Returns:
            Dict with 'upper', 'middle', and 'lower' bands
        """
        middle = TechnicalIndicators.sma(prices, period)
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': (upper - lower) / middle  # Normalized width
        }

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume.

        Args:
            close: Close prices
            volume: Volume

        Returns:
            OBV values
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def stochastic_oscillator(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator.

        Returns:
            Dict with '%K' and '%D'
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=3).mean()

        return {
            '%K': k_percent,
            '%D': d_percent
        }


class FeatureEngineer:
    """
    Feature engineering pipeline for trading data.

    Generates technical indicators and derived features.
    """

    def __init__(self, indicators: Optional[List[str]] = None):
        """
        Initialize feature engineer.

        Args:
            indicators: List of indicators to calculate.
                       Default: ['rsi', 'macd', 'bb', 'atr', 'obv']
        """
        self.indicators = indicators or ['rsi', 'macd', 'bb', 'returns', 'volatility']

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and features to OHLCV data.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            DataFrame with added features
        """
        df = data.copy()

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        # RSI
        if 'rsi' in self.indicators:
            df['rsi'] = TechnicalIndicators.rsi(df['close'])
            df['rsi'] = df['rsi'] / 100  # Normalize to 0-1

        # MACD
        if 'macd' in self.indicators:
            macd_dict = TechnicalIndicators.macd(df['close'])
            df['macd'] = macd_dict['macd']
            df['macd_signal'] = macd_dict['signal']
            df['macd_hist'] = macd_dict['histogram']

            # Normalize MACD
            df['macd'] = df['macd'] / df['close']
            df['macd_signal'] = df['macd_signal'] / df['close']
            df['macd_hist'] = df['macd_hist'] / df['close']

        # Bollinger Bands
        if 'bb' in self.indicators:
            bb_dict = TechnicalIndicators.bollinger_bands(df['close'])
            df['bb_upper'] = bb_dict['upper']
            df['bb_middle'] = bb_dict['middle']
            df['bb_lower'] = bb_dict['lower']
            df['bb_width'] = bb_dict['width']

            # Normalize relative to close
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR
        if 'atr' in self.indicators and all(col in df.columns for col in ['high', 'low']):
            df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
            df['atr_pct'] = df['atr'] / df['close']  # Normalized ATR

        # OBV
        if 'obv' in self.indicators and 'volume' in df.columns:
            df['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
            # Normalize OBV
            df['obv'] = df['obv'] / df['obv'].rolling(window=50).mean()

        # Returns and volatility
        if 'returns' in self.indicators:
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        if 'volatility' in self.indicators:
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['volatility_30'] = df['returns'].rolling(window=30).std()

        # Moving averages
        if 'sma' in self.indicators:
            df['sma_10'] = TechnicalIndicators.sma(df['close'], 10)
            df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
            df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)

            # Price relative to SMA
            df['price_sma20_ratio'] = df['close'] / df['sma_20']

        # Stochastic
        if 'stochastic' in self.indicators and all(col in df.columns for col in ['high', 'low']):
            stoch_dict = TechnicalIndicators.stochastic_oscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_dict['%K'] / 100  # Normalize to 0-1
            df['stoch_d'] = stoch_dict['%D'] / 100

        # Volume features
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price momentum
        df['momentum_5'] = df['close'].pct_change(periods=5)
        df['momentum_10'] = df['close'].pct_change(periods=10)

        # Fill NaN values
        df = df.bfill().fillna(0)

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        base_features = ['open', 'high', 'low', 'close', 'volume']
        indicator_features = []

        if 'rsi' in self.indicators:
            indicator_features.append('rsi')
        if 'macd' in self.indicators:
            indicator_features.extend(['macd', 'macd_signal', 'macd_hist'])
        if 'bb' in self.indicators:
            indicator_features.extend(['bb_width', 'bb_position'])
        if 'atr' in self.indicators:
            indicator_features.append('atr_pct')
        if 'obv' in self.indicators:
            indicator_features.append('obv')
        if 'returns' in self.indicators:
            indicator_features.extend(['returns', 'log_returns'])
        if 'volatility' in self.indicators:
            indicator_features.extend(['volatility', 'volatility_30'])

        indicator_features.extend(['momentum_5', 'momentum_10'])

        return base_features + indicator_features


def normalize_features(df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize features.

    Args:
        df: DataFrame with features
        method: Normalization method ('minmax', 'zscore', 'robust')

    Returns:
        Normalized DataFrame
    """
    df_norm = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if method == 'minmax':
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val > 0:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)

    elif method == 'zscore':
        for col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_norm[col] = (df[col] - mean_val) / std_val

    elif method == 'robust':
        for col in numeric_cols:
            median_val = df[col].median()
            q75, q25 = df[col].quantile([0.75, 0.25])
            iqr = q75 - q25
            if iqr > 0:
                df_norm[col] = (df[col] - median_val) / iqr

    return df_norm
