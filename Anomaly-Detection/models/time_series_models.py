from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformers import VanillaTransformer
from .probabilistic_forecasting import QuantileHead, GaussianHead

class TemporalEmbedding(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class PointHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class Decomposition(nn.Module):
    """Simple residual decomposition: x = trend + seasonal (via moving average)."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.register_buffer("kernel", torch.ones(1, 1, kernel_size) / kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> apply per feature with conv by reshaping
        B, T, D = x.shape
        x_ = x.transpose(1, 2)  # (B, D, T)
        trend = torch.conv1d(x_, self.kernel, padding=self.kernel_size // 2, groups=1)
        trend = trend.transpose(1, 2)  # (B, T, D')
        seasonal = x - trend
        return trend, seasonal

class TransformerForecaster(nn.Module):
    """Wrapper that combines embeddings, (optional) decomposition, a Transformer backbone,
    and a configurable forecasting head (point/quantile/gaussian).
    """
    def __init__(self, in_dim: int, out_dim: int, d_model: int = 128, head_type: str = "point",
                 quantiles: Optional[List[float]] = None):
        super().__init__()
        self.embed = TemporalEmbedding(in_dim, d_model)
        self.decomp = Decomposition(kernel_size=25)
        self.backbone = VanillaTransformer(d_model=d_model)

        self.head_type = head_type
        if head_type == "point":
            self.head = PointHead(d_model, out_dim)
        elif head_type == "quantile":
            self.head = QuantileHead(d_model, out_dim, quantiles or [0.1, 0.5, 0.9])
        elif head_type == "gaussian":
            self.head = GaussianHead(d_model, out_dim)
        else:
            raise ValueError("Unknown head_type")

    def forward(self, src: torch.Tensor, tgt_init: torch.Tensor):
        # src: (B, T_in, in_dim)  tgt_init: (B, T_out, in_dim) (e.g., last obs repeated or zeros)
        src_e = self.embed(src)
        tgt_e = self.embed(tgt_init)
        trend_src, seas_src = self.decomp(src_e)
        src_cat = src_e + trend_src + seas_src
        out = self.backbone(src_cat, tgt_e)
        if self.head_type == "gaussian":
            mu, log_std = self.head(out)
            return mu, log_std
        else:
            return self.head(out)
