"""
Transformer Models for Time Series Forecasting
Implements Vanilla Transformer, Informer, and Autoformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .attention_mechanisms import (
    FullAttention,
    ProbSparseAttention,
    AutoCorrelation,
    AttentionLayer
)
from .time_series_modules import (
    DataEmbedding,
    SeriesDecomp,
    PositionalEncoding
)


class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer Encoder Layer
    Self-attention + Feed-forward network
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        attention_type: str = 'full'
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
            attention_type: Type of attention ('full', 'prob', 'auto')
        """
        super().__init__()

        # Attention mechanism
        if attention_type == 'full':
            attention = FullAttention(attention_dropout=dropout)
        elif attention_type == 'prob':
            attention = ProbSparseAttention(attention_dropout=dropout)
        elif attention_type == 'auto':
            attention = AutoCorrelation(attention_dropout=dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.attention = AttentionLayer(
            attention, d_model, n_heads
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            attn_mask: Attention mask

        Returns:
            Output tensor [B, L, D]
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Standard Transformer Decoder Layer
    Self-attention + Cross-attention + Feed-forward network
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        attention_type: str = 'full'
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
            attention_type: Type of attention
        """
        super().__init__()

        # Self-attention
        if attention_type == 'full':
            self_attention = FullAttention(attention_dropout=dropout)
            cross_attention = FullAttention(attention_dropout=dropout)
        elif attention_type == 'prob':
            self_attention = ProbSparseAttention(attention_dropout=dropout)
            cross_attention = FullAttention(attention_dropout=dropout)
        elif attention_type == 'auto':
            self_attention = AutoCorrelation(attention_dropout=dropout)
            cross_attention = AutoCorrelation(attention_dropout=dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.self_attention = AttentionLayer(
            self_attention, d_model, n_heads
        )
        self.cross_attention = AttentionLayer(
            cross_attention, d_model, n_heads
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input [B, L, D]
            cross: Encoder output [B, S, D]
            x_mask: Self-attention mask
            cross_mask: Cross-attention mask

        Returns:
            Output tensor [B, L, D]
        """
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x, x_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Cross-attention
        attn_out, _ = self.cross_attention(x, cross, cross, cross_mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)

        # Feed-forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)

        return x


class VanillaTransformer(nn.Module):
    """
    Vanilla Transformer for Time Series Forecasting
    Standard encoder-decoder architecture with temporal embeddings
    """

    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.1,
        embed: str = 'fixed',
        freq: str = 'h',
        activation: str = 'gelu'
    ):
        """
        Args:
            enc_in: Encoder input size
            dec_in: Decoder input size
            c_out: Output size
            seq_len: Input sequence length
            label_len: Start token length for decoder
            pred_len: Prediction length
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            embed: Embedding type
            freq: Frequency of time features
            activation: Activation function
        """
        super().__init__()
        self.pred_len = pred_len

        # Embedding
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout
        )
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout
        )

        # Encoder
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, activation, 'full'
            ) for _ in range(e_layers)
        ])

        # Decoder
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, n_heads, d_ff, dropout, activation, 'full'
            ) for _ in range(d_layers)
        ])

        # Output projection
        self.projection = nn.Linear(d_model, c_out)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x_enc: Encoder input values [B, seq_len, enc_in]
            x_mark_enc: Encoder time features [B, seq_len, time_features]
            x_dec: Decoder input values [B, label_len + pred_len, dec_in]
            x_mark_dec: Decoder time features [B, label_len + pred_len, time_features]
            enc_self_mask: Encoder self-attention mask
            dec_self_mask: Decoder self-attention mask
            dec_enc_mask: Decoder-encoder cross-attention mask

        Returns:
            Predictions [B, pred_len, c_out]
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out, enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, dec_self_mask, dec_enc_mask)

        # Project to output
        output = self.projection(dec_out)

        # Return only predictions (not the label part)
        return output[:, -self.pred_len:, :]


class Informer(nn.Module):
    """
    Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
    Uses ProbSparse attention and distilling mechanism
    """

    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        factor: int = 5,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.0,
        embed: str = 'fixed',
        freq: str = 'h',
        activation: str = 'gelu',
        distil: bool = True
    ):
        """
        Args:
            enc_in: Encoder input size
            dec_in: Decoder input size
            c_out: Output size
            seq_len: Input sequence length
            label_len: Start token length
            pred_len: Prediction length
            factor: ProbSparse attention factor
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            embed: Embedding type
            freq: Frequency
            activation: Activation function
            distil: Whether to use distilling in encoder
        """
        super().__init__()
        self.pred_len = pred_len
        self.distil = distil

        # Embedding
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout
        )
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout
        )

        # Encoder with ProbSparse attention
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, activation, 'prob'
            ) for _ in range(e_layers)
        ])

        # Distilling layers (if enabled)
        if distil:
            self.distil_layers = nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
                for _ in range(e_layers - 1)
            ])
            self.norm = nn.LayerNorm(d_model)

        # Decoder with ProbSparse self-attention and full cross-attention
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, n_heads, d_ff, dropout, activation, 'prob'
            ) for _ in range(d_layers)
        ])

        # Output projection
        self.projection = nn.Linear(d_model, c_out)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Informer

        Args:
            x_enc: Encoder input [B, seq_len, enc_in]
            x_mark_enc: Encoder time features
            x_dec: Decoder input [B, label_len + pred_len, dec_in]
            x_mark_dec: Decoder time features
            enc_self_mask: Encoder mask
            dec_self_mask: Decoder mask
            dec_enc_mask: Cross-attention mask

        Returns:
            Predictions [B, pred_len, c_out]
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        for i, layer in enumerate(self.encoder):
            enc_out = layer(enc_out, enc_self_mask)

            # Apply distilling
            if self.distil and i < len(self.encoder) - 1:
                # Distilling: reduce sequence length by half
                enc_out_permute = enc_out.permute(0, 2, 1)  # [B, D, L]
                enc_out_permute = self.distil_layers[i](enc_out_permute)
                enc_out_permute = F.max_pool1d(enc_out_permute, kernel_size=3, stride=2, padding=1)
                enc_out = enc_out_permute.permute(0, 2, 1)  # [B, L/2, D]
                enc_out = self.norm(enc_out)

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, dec_self_mask, dec_enc_mask)

        # Project to output
        output = self.projection(dec_out)

        # Return predictions
        return output[:, -self.pred_len:, :]


class Autoformer(nn.Module):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation
    Uses series decomposition and auto-correlation mechanism
    """

    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        moving_avg: int = 25,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.1,
        embed: str = 'fixed',
        freq: str = 'h',
        activation: str = 'gelu'
    ):
        """
        Args:
            enc_in: Encoder input size
            dec_in: Decoder input size
            c_out: Output size
            seq_len: Input sequence length
            label_len: Start token length
            pred_len: Prediction length
            moving_avg: Moving average window size
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            embed: Embedding type
            freq: Frequency
            activation: Activation function
        """
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # Embedding
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout
        )
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout
        )

        # Encoder with decomposition
        self.encoder = nn.ModuleList([
            AutoformerEncoderLayer(
                d_model, n_heads, d_ff, moving_avg, dropout, activation
            ) for _ in range(e_layers)
        ])

        # Decoder with decomposition
        self.decoder = nn.ModuleList([
            AutoformerDecoderLayer(
                d_model, n_heads, d_ff, moving_avg, dropout, activation
            ) for _ in range(d_layers)
        ])

        # Decomposition for decoder input
        self.decomp = SeriesDecomp(moving_avg)

        # Output projections for trend and seasonal
        self.projection = nn.Linear(d_model, c_out)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Autoformer

        Args:
            x_enc: Encoder input
            x_mark_enc: Encoder time features
            x_dec: Decoder input
            x_mark_dec: Decoder time features

        Returns:
            Predictions [B, pred_len, c_out]
        """
        # Decomposition initialization
        seasonal_init, trend_init = self.decomp(x_enc)

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out, enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        seasonal_part = dec_out
        trend_part = torch.zeros_like(dec_out)

        for layer in self.decoder:
            seasonal_part, trend_part = layer(
                seasonal_part, enc_out, trend_part, dec_self_mask, dec_enc_mask
            )

        # Combine seasonal and trend
        dec_out = seasonal_part + trend_part

        # Project to output
        output = self.projection(dec_out)

        return output[:, -self.pred_len:, :]


class AutoformerEncoderLayer(nn.Module):
    """Encoder layer for Autoformer with decomposition"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str
    ):
        super().__init__()
        self.attention = AttentionLayer(
            AutoCorrelation(attention_dropout=dropout),
            d_model, n_heads
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Auto-correlation
        new_x, _ = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)

        # Feed-forward
        new_x = self.ff(x)
        x = x + self.dropout(new_x)
        x, _ = self.decomp2(x)

        return x


class AutoformerDecoderLayer(nn.Module):
    """Decoder layer for Autoformer with decomposition"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str
    ):
        super().__init__()
        self.self_attention = AttentionLayer(
            AutoCorrelation(attention_dropout=dropout),
            d_model, n_heads
        )
        self.cross_attention = AttentionLayer(
            AutoCorrelation(attention_dropout=dropout),
            d_model, n_heads
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        trend: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ):
        # Self-attention
        new_x, _ = self.self_attention(x, x, x, x_mask)
        x = x + self.dropout(new_x)
        x, trend1 = self.decomp1(x)
        trend = trend + trend1

        # Cross-attention
        new_x, _ = self.cross_attention(x, cross, cross, cross_mask)
        x = x + self.dropout(new_x)
        x, trend2 = self.decomp2(x)
        trend = trend + trend2

        # Feed-forward
        new_x = self.ff(x)
        x = x + self.dropout(new_x)
        x, trend3 = self.decomp3(x)
        trend = trend + trend3

        return x, trend
