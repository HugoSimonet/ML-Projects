"""
Time Series Specific Modules
Temporal embedding, positional encoding, and decomposition components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding for time series
    Standard sinusoidal encoding from "Attention Is All You Need"
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalEmbedding(nn.Module):
    """
    Temporal embedding for time series
    Encodes minute, hour, day, weekday, month features
    """

    def __init__(
        self,
        d_model: int,
        embed_type: str = 'fixed',
        freq: str = 'h'
    ):
        """
        Args:
            d_model: Model dimension
            embed_type: Type of embedding ('fixed' or 'learned')
            freq: Frequency of time features
                - 's': secondly
                - 't': minutely
                - 'h': hourly
                - 'd': daily
                - 'b': business days
                - 'w': weekly
                - 'm': monthly
        """
        super().__init__()

        freq_map = {
            's': ['minute', 'hour', 'weekday', 'day', 'month'],
            't': ['minute', 'hour', 'weekday', 'day', 'month'],
            'h': ['hour', 'weekday', 'day', 'month'],
            'd': ['weekday', 'day', 'month'],
            'b': ['weekday', 'day', 'month'],
            'w': ['weekday', 'month'],
            'm': ['month']
        }

        self.freq = freq
        self.embed_type = embed_type
        self.d_model = d_model

        # Embedding dimensions for each time feature
        feature_dims = {
            'minute': 60,  # 0-59
            'hour': 24,    # 0-23
            'weekday': 7,  # 0-6
            'day': 32,     # 1-31
            'month': 13    # 1-12
        }

        # Create embeddings for each time feature
        self.embeddings = nn.ModuleDict()
        for feature in freq_map[freq]:
            if embed_type == 'fixed':
                # Fixed sinusoidal embedding
                self.embeddings[feature] = nn.Linear(1, d_model)
            else:
                # Learned embedding
                self.embeddings[feature] = nn.Embedding(
                    feature_dims[feature], d_model
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Time features tensor [B, L, num_features]
               Features should be in order based on frequency

        Returns:
            Temporal embeddings [B, L, D]
        """
        B, L, _ = x.shape

        # Initialize output
        output = torch.zeros(B, L, self.d_model).to(x.device)

        # Add embeddings for each feature
        for i, (feature_name, embedding) in enumerate(self.embeddings.items()):
            if self.embed_type == 'fixed':
                output += embedding(x[:, :, i:i+1])
            else:
                output += embedding(x[:, :, i].long())

        return output


class DataEmbedding(nn.Module):
    """
    Complete data embedding combining value, temporal, and positional encoding
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = 'fixed',
        freq: str = 'h',
        dropout: float = 0.1
    ):
        """
        Args:
            c_in: Number of input features
            d_model: Model dimension
            embed_type: Type of temporal embedding
            freq: Frequency of time features
            dropout: Dropout rate
        """
        super().__init__()

        # Value embedding (project input features to d_model)
        self.value_embedding = nn.Linear(c_in, d_model)

        # Positional encoding
        self.position_embedding = PositionalEncoding(d_model, dropout=dropout)

        # Temporal embedding
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model,
            embed_type=embed_type,
            freq=freq
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input values [B, L, C]
            x_mark: Time features [B, L, num_features]

        Returns:
            Embedded output [B, L, D]
        """
        # Value embedding
        x = self.value_embedding(x)

        # Add temporal embedding if available
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)

        # Add positional encoding
        x = self.position_embedding(x)

        return self.dropout(x)


class SeasonalDecomposition(nn.Module):
    """
    Seasonal decomposition layer
    Separates time series into trend and seasonal components
    """

    def __init__(self, kernel_size: int = 25):
        """
        Args:
            kernel_size: Size of moving average kernel for trend extraction
        """
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input time series [B, L, C]

        Returns:
            seasonal: Seasonal component [B, L, C]
            trend: Trend component [B, L, C]
        """
        # Average pooling for trend
        padding = (self.kernel_size - 1) // 2

        # Apply average pooling along time dimension
        # Reshape to [B, C, L] for conv1d
        x_permute = x.permute(0, 2, 1)
        trend = F.avg_pool1d(
            x_permute,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding
        )
        trend = trend.permute(0, 2, 1)  # Back to [B, L, C]

        # Handle potential length mismatch due to padding
        if trend.shape[1] != x.shape[1]:
            trend = trend[:, :x.shape[1], :]

        # Seasonal = original - trend
        seasonal = x - trend

        return seasonal, trend


class TrendDecomposition(nn.Module):
    """
    Learnable trend decomposition
    Uses convolution to extract trend component
    """

    def __init__(
        self,
        kernel_size: int = 25,
        num_features: int = 1
    ):
        """
        Args:
            kernel_size: Convolution kernel size
            num_features: Number of input features
        """
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        # Learnable trend extraction
        self.trend_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=num_features,
            bias=False
        )

        # Initialize with average pooling
        self.trend_conv.weight.data.fill_(1.0 / kernel_size)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input time series [B, L, C]

        Returns:
            seasonal: Seasonal component [B, L, C]
            trend: Trend component [B, L, C]
        """
        # Reshape to [B, C, L]
        x_permute = x.permute(0, 2, 1)

        # Extract trend
        trend = self.trend_conv(x_permute)
        trend = trend.permute(0, 2, 1)  # Back to [B, L, C]

        # Handle length mismatch
        if trend.shape[1] != x.shape[1]:
            trend = trend[:, :x.shape[1], :]

        # Seasonal component
        seasonal = x - trend

        return seasonal, trend


class SeriesDecomp(nn.Module):
    """
    Series decomposition block from Autoformer
    Used in both encoder and decoder
    """

    def __init__(self, kernel_size: int = 25):
        """
        Args:
            kernel_size: Moving average kernel size
        """
        super().__init__()
        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor [B, L, C]

        Returns:
            res: Seasonal component [B, L, C]
            avg: Trend component [B, L, C]
        """
        # Permute for pooling
        x_permute = x.permute(0, 2, 1)  # [B, C, L]

        # Moving average (trend)
        avg = self.moving_avg(x_permute)
        avg = avg.permute(0, 2, 1)  # [B, L, C]

        # Handle length
        if avg.shape[1] != x.shape[1]:
            avg = avg[:, :x.shape[1], :]

        # Residual (seasonal)
        res = x - avg

        return res, avg


class FourierLayer(nn.Module):
    """
    Fourier layer for frequency domain analysis
    """

    def __init__(
        self,
        d_model: int,
        modes: int = 32
    ):
        """
        Args:
            d_model: Model dimension
            modes: Number of Fourier modes to keep
        """
        super().__init__()
        self.d_model = d_model
        self.modes = modes

        # Learnable weights for Fourier modes
        self.weights = nn.Parameter(
            torch.randn(modes, d_model, d_model, 2) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output in frequency domain [B, L, D]
        """
        B, L, D = x.shape

        # FFT
        x_ft = torch.fft.rfft(x, dim=1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, L // 2 + 1, D, dtype=torch.cfloat).to(x.device)

        for i in range(min(self.modes, L // 2 + 1)):
            # Complex multiplication
            out_ft[:, i, :] = torch.complex(
                torch.einsum('bd,di->bi', x_ft[:, i, :].real, self.weights[i, :, :, 0]) -
                torch.einsum('bd,di->bi', x_ft[:, i, :].imag, self.weights[i, :, :, 1]),
                torch.einsum('bd,di->bi', x_ft[:, i, :].real, self.weights[i, :, :, 1]) +
                torch.einsum('bd,di->bi', x_ft[:, i, :].imag, self.weights[i, :, :, 0])
            )

        # iFFT
        x = torch.fft.irfft(out_ft, n=L, dim=1)

        return x
