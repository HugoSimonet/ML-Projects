from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- Quantile Forecasting ----------
class QuantileHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.fc = nn.Linear(d_model, out_dim * len(quantiles))
        self.quantiles = quantiles
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, T, out_dim * Q)
        return self.fc(x)

    def pinball_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # pred: (B, T, out_dim*Q), y: (B, T, out_dim)
        Q = len(self.quantiles)
        B, T, _ = pred.shape
        pred = pred.view(B, T, self.out_dim, Q)
        y = y.unsqueeze(-1).expand_as(pred)
        losses = []
        for i, q in enumerate(self.quantiles):
            e = y[..., i] - pred[..., i]
            losses.append(torch.max((q - 1) * e, q * e))
        loss = torch.stack(losses, dim=-1).mean()
        return loss

# ---------- Gaussian Forecasting ----------
class GaussianHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.mean = nn.Linear(d_model, out_dim)
        self.log_std = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mean(x)
        log_std = self.log_std(x).clamp(-5, 3)  # stabilize
        return mu, log_std

    @staticmethod
    def nll(mu: torch.Tensor, log_std: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2 * log_std)
        return 0.5 * (math.log(2 * math.pi) + 2 * log_std + (y - mu) ** 2 / var).mean()

    @staticmethod
    def gaussian_crps(mu: torch.Tensor, log_std: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Closed-form CRPS for normal distribution
        std = torch.exp(log_std)
        z = (y - mu) / std
        from torch.distributions.normal import Normal
        dist = Normal(0, 1)
        crps = std * (z * (2 * dist.cdf(z) - 1) + 2 * torch.exp(-0.5 * z**2) / math.sqrt(2*math.pi) - 1 / math.sqrt(math.pi))
        return crps.mean()
