"""
Attention Mechanisms for Time Series Transformers
Implements various attention mechanisms including full, ProbSparse, and auto-correlation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


class FullAttention(nn.Module):
    """
    Standard full self-attention mechanism
    Complexity: O(L^2)
    """

    def __init__(
        self,
        mask_flag: bool = True,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1
    ):
        """
        Args:
            mask_flag: Whether to apply causal mask
            scale: Scaling factor (default: 1/sqrt(d_k))
            attention_dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for full attention

        Args:
            queries: Query tensor [B, L, H, D]
            keys: Key tensor [B, S, H, D]
            values: Value tensor [B, S, H, D]
            attn_mask: Attention mask [B, 1, L, S]

        Returns:
            output: Attention output [B, L, H, D]
            attn: Attention weights [B, H, L, S]
        """
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        scale = self.scale or 1. / math.sqrt(D)

        # Compute attention scores
        # [B, L, H, D] x [B, S, H, D]^T -> [B, H, L, S]
        scores = torch.einsum('blhd,bshd->bhls', queries, keys)

        # Apply mask if needed
        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        # Scale and apply softmax
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Apply attention to values
        # [B, H, L, S] x [B, S, H, D] -> [B, L, H, D]
        output = torch.einsum('bhls,bshd->blhd', attn, values)

        return output, attn


class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention from Informer paper
    Complexity: O(L log L) instead of O(L^2)

    Selects only top-u queries based on sparsity measurement
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1
    ):
        """
        Args:
            mask_flag: Whether to apply causal mask
            factor: Sampling factor for ProbSparse attention
            scale: Scaling factor
            attention_dropout: Dropout rate
        """
        super().__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        sample_k: int,
        n_top: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate sparsity measurement M and select top-u queries

        Args:
            Q: Query tensor [B, H, L, D]
            K: Key tensor [B, H, S, D]
            sample_k: Number of keys to sample
            n_top: Number of top queries to select

        Returns:
            Q_reduce: Top-u queries
            M_top: Top-u query indices
        """
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Sample keys for sparsity measurement
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # Calculate sparsity measurement M
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Find max values in sampled attention
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Select top-u queries
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # [B, H, n_top, D]

        return Q_reduce, M_top

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ProbSparse attention

        Args:
            queries: Query tensor [B, L, H, D]
            keys: Key tensor [B, S, H, D]
            values: Value tensor [B, S, H, D]
            attn_mask: Attention mask

        Returns:
            output: Attention output [B, L, H, D]
            attn: Attention weights
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # Transpose for attention computation
        queries = queries.transpose(1, 2)  # [B, H, L, D]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculate sampling parameters
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # ProbSparse attention
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Compute attention for top-u queries
        scale = self.scale or 1. / math.sqrt(D)
        scores_top = scores_top * scale

        # Get context for top queries
        context = torch.matmul(
            torch.softmax(scores_top, dim=-1),
            keys.transpose(-2, -1)
        ).transpose(-2, -1)

        # Update values at selected positions
        if attn_mask is not None:
            context = context.masked_fill(attn_mask == 0, -1e9)

        # Apply dropout
        context = self.dropout(context)

        # Create output with cumsum for remaining positions
        attn = torch.ones([B, H, L_Q, L_K]).to(queries.device) / L_K
        attn[torch.arange(B)[:, None, None],
             torch.arange(H)[None, :, None],
             index, :] = torch.softmax(scores_top, dim=-1)

        # Transpose back
        output = torch.matmul(attn, values).transpose(1, 2)  # [B, L, H, D]

        return output, attn


class AutoCorrelation(nn.Module):
    """
    Auto-Correlation mechanism from Autoformer paper
    Replaces dot-product attention with autocorrelation in frequency domain
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 1,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1
    ):
        """
        Args:
            mask_flag: Whether to apply mask
            factor: Number of top autocorrelations to select
            scale: Scaling factor
            attention_dropout: Dropout rate
        """
        super().__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(
        self,
        values: torch.Tensor,
        corr: torch.Tensor
    ) -> torch.Tensor:
        """
        Time delay aggregation for training

        Args:
            values: Value tensor [B, H, L, D]
            corr: Correlation weights [B, H, L]

        Returns:
            Aggregated values
        """
        B, H, L, D = values.shape

        # Find top-k delays
        top_k = int(self.factor * math.log(L))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)

        # Normalize weights
        tmp_corr = torch.softmax(weights, dim=-1)

        # Aggregate
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), dims=-2)
            delays_agg = delays_agg + pattern * \
                        (tmp_corr[:, :, i].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, L, D))

        return delays_agg

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for auto-correlation

        Args:
            queries: Query tensor [B, L, H, D]
            keys: Key tensor [B, S, H, D]
            values: Value tensor [B, S, H, D]
            attn_mask: Attention mask

        Returns:
            output: Auto-correlation output [B, L, H, D]
            attn: Auto-correlation weights
        """
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        # Transpose to [B, H, L, D]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute auto-correlation using FFT
        # FFT for queries and keys
        q_fft = torch.fft.rfft(queries, dim=-2)
        k_fft = torch.fft.rfft(keys, dim=-2)

        # Compute correlation in frequency domain
        res = q_fft * torch.conj(k_fft)

        # iFFT to get correlation
        corr = torch.fft.irfft(res, dim=-2)

        # Time delay aggregation
        output = self.time_delay_agg_training(values, corr)

        # Transpose back to [B, L, H, D]
        output = output.transpose(1, 2)

        # For compatibility, return correlation as attention
        attn = torch.softmax(corr, dim=-1)

        return output, attn


class AttentionLayer(nn.Module):
    """
    Attention Layer wrapper with projections
    Projects inputs to Q, K, V and applies attention mechanism
    """

    def __init__(
        self,
        attention,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None
    ):
        """
        Args:
            attention: Attention mechanism (FullAttention, ProbSparseAttention, etc.)
            d_model: Model dimension
            n_heads: Number of attention heads
            d_keys: Dimension of keys (default: d_model // n_heads)
            d_values: Dimension of values (default: d_model // n_heads)
        """
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

        # Linear projections
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            queries: Query tensor [B, L, D]
            keys: Key tensor [B, S, D]
            values: Value tensor [B, S, D]
            attn_mask: Attention mask

        Returns:
            output: Attention output [B, L, D]
            attn: Attention weights
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project to Q, K, V
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply attention
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        # Reshape and project output
        out = out.reshape(B, L, -1)
        out = self.out_projection(out)

        return out, attn
