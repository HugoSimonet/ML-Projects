"""
Probabilistic Forecasting Models
Quantile regression and distributional forecasting for uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


class ProbabilisticHead(nn.Module):
    """
    Probabilistic forecasting head
    Outputs mean and variance for uncertainty quantification
    """

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        c_out: int,
        output_distribution: str = 'gaussian'
    ):
        """
        Args:
            d_model: Model dimension
            pred_len: Prediction length
            c_out: Output channels
            output_distribution: Type of output distribution
                - 'gaussian': Normal distribution (mean, std)
                - 'student': Student-t distribution (mean, std, df)
                - 'negative_binomial': For count data
        """
        super().__init__()
        self.pred_len = pred_len
        self.c_out = c_out
        self.output_distribution = output_distribution

        # Mean prediction
        self.mean_projection = nn.Linear(d_model, pred_len * c_out)

        # Variance prediction (log scale for positivity)
        self.std_projection = nn.Linear(d_model, pred_len * c_out)

        # Additional parameters for other distributions
        if output_distribution == 'student':
            self.df_projection = nn.Linear(d_model, pred_len * c_out)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input features [B, L, D] or [B, D]

        Returns:
            Dictionary with distribution parameters
        """
        # If 3D input, aggregate
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # [B, D]

        # Predict mean
        mean = self.mean_projection(x)
        mean = mean.view(-1, self.pred_len, self.c_out)

        # Predict std (use softplus to ensure positivity)
        std = self.std_projection(x)
        std = F.softplus(std)
        std = std.view(-1, self.pred_len, self.c_out)

        output = {
            'mean': mean,
            'std': std
        }

        # Additional parameters for Student-t
        if self.output_distribution == 'student':
            df = self.df_projection(x)
            df = F.softplus(df) + 2.0  # Ensure df > 2
            df = df.view(-1, self.pred_len, self.c_out)
            output['df'] = df

        return output


class QuantileForecaster(nn.Module):
    """
    Quantile Forecaster
    Predicts multiple quantiles for probabilistic forecasting
    """

    def __init__(
        self,
        base_model: nn.Module,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        d_model: int = 512,
        pred_len: int = 24,
        c_out: int = 1
    ):
        """
        Args:
            base_model: Base forecasting model (encoder)
            quantiles: List of quantiles to predict
            d_model: Model dimension
            pred_len: Prediction length
            c_out: Output channels
        """
        super().__init__()
        self.base_model = base_model
        self.quantiles = sorted(quantiles)
        self.num_quantiles = len(quantiles)
        self.pred_len = pred_len
        self.c_out = c_out

        # Separate head for each quantile
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, pred_len * c_out)
            ) for _ in range(self.num_quantiles)
        ])

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_enc: Encoder input
            x_mark_enc: Encoder time features
            x_dec: Decoder input
            x_mark_dec: Decoder time features

        Returns:
            Quantile predictions [B, pred_len, c_out, num_quantiles]
        """
        # Get base features from model
        # Assuming base_model returns features or predictions
        features = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # If features are 3D, aggregate
        if len(features.shape) == 3:
            features = features.mean(dim=1)  # [B, D]

        # Predict each quantile
        quantile_preds = []
        for head in self.quantile_heads:
            q_pred = head(features)
            q_pred = q_pred.view(-1, self.pred_len, self.c_out)
            quantile_preds.append(q_pred)

        # Stack quantiles [B, pred_len, c_out, num_quantiles]
        output = torch.stack(quantile_preds, dim=-1)

        return output

    def get_prediction_intervals(
        self,
        predictions: torch.Tensor,
        confidence_levels: List[float] = [0.5, 0.8, 0.95]
    ) -> dict:
        """
        Get prediction intervals from quantile predictions

        Args:
            predictions: Quantile predictions [B, pred_len, c_out, num_quantiles]
            confidence_levels: Confidence levels for intervals

        Returns:
            Dictionary with intervals for each confidence level
        """
        intervals = {}

        for conf in confidence_levels:
            alpha = 1 - conf
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2

            # Find closest quantiles
            lower_idx = min(range(len(self.quantiles)),
                          key=lambda i: abs(self.quantiles[i] - lower_q))
            upper_idx = min(range(len(self.quantiles)),
                          key=lambda i: abs(self.quantiles[i] - upper_q))

            intervals[f'{conf:.0%}'] = {
                'lower': predictions[..., lower_idx],
                'upper': predictions[..., upper_idx]
            }

        return intervals


class DistributionalForecaster(nn.Module):
    """
    Distributional Forecaster
    Predicts full distribution parameters for each time step
    """

    def __init__(
        self,
        base_model: nn.Module,
        distribution: str = 'gaussian',
        d_model: int = 512,
        pred_len: int = 24,
        c_out: int = 1
    ):
        """
        Args:
            base_model: Base forecasting model
            distribution: Distribution type ('gaussian', 'student', 'negative_binomial')
            d_model: Model dimension
            pred_len: Prediction length
            c_out: Output channels
        """
        super().__init__()
        self.base_model = base_model
        self.distribution = distribution
        self.pred_len = pred_len
        self.c_out = c_out

        # Distribution parameter heads
        if distribution == 'gaussian':
            # Mean and log-variance
            self.mean_head = nn.Linear(d_model, pred_len * c_out)
            self.logvar_head = nn.Linear(d_model, pred_len * c_out)

        elif distribution == 'student':
            # Mean, log-scale, and degrees of freedom
            self.mean_head = nn.Linear(d_model, pred_len * c_out)
            self.logscale_head = nn.Linear(d_model, pred_len * c_out)
            self.df_head = nn.Linear(d_model, pred_len * c_out)

        elif distribution == 'negative_binomial':
            # Total count and logits
            self.total_count_head = nn.Linear(d_model, pred_len * c_out)
            self.logits_head = nn.Linear(d_model, pred_len * c_out)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor
    ) -> dict:
        """
        Args:
            x_enc: Encoder input
            x_mark_enc: Encoder time features
            x_dec: Decoder input
            x_mark_dec: Decoder time features

        Returns:
            Dictionary with distribution parameters
        """
        # Get features
        features = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if len(features.shape) == 3:
            features = features.mean(dim=1)

        output = {}

        if self.distribution == 'gaussian':
            mean = self.mean_head(features).view(-1, self.pred_len, self.c_out)
            logvar = self.logvar_head(features).view(-1, self.pred_len, self.c_out)
            std = torch.exp(0.5 * logvar)

            output['mean'] = mean
            output['std'] = std
            output['logvar'] = logvar

        elif self.distribution == 'student':
            mean = self.mean_head(features).view(-1, self.pred_len, self.c_out)
            logscale = self.logscale_head(features).view(-1, self.pred_len, self.c_out)
            scale = torch.exp(logscale)
            df = F.softplus(self.df_head(features).view(-1, self.pred_len, self.c_out)) + 2.0

            output['mean'] = mean
            output['scale'] = scale
            output['df'] = df

        elif self.distribution == 'negative_binomial':
            total_count = F.softplus(
                self.total_count_head(features).view(-1, self.pred_len, self.c_out)
            ) + 1.0
            logits = self.logits_head(features).view(-1, self.pred_len, self.c_out)

            output['total_count'] = total_count
            output['logits'] = logits

        return output

    def sample(
        self,
        params: dict,
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        Sample from the predicted distribution

        Args:
            params: Distribution parameters
            num_samples: Number of samples to draw

        Returns:
            Samples [num_samples, B, pred_len, c_out]
        """
        if self.distribution == 'gaussian':
            mean = params['mean']
            std = params['std']
            dist = torch.distributions.Normal(mean, std)
            samples = dist.sample((num_samples,))

        elif self.distribution == 'student':
            mean = params['mean']
            scale = params['scale']
            df = params['df']
            dist = torch.distributions.StudentT(df, mean, scale)
            samples = dist.sample((num_samples,))

        elif self.distribution == 'negative_binomial':
            total_count = params['total_count']
            logits = params['logits']
            dist = torch.distributions.NegativeBinomial(total_count, logits=logits)
            samples = dist.sample((num_samples,))

        return samples

    def compute_quantiles(
        self,
        params: dict,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        num_samples: int = 1000
    ) -> torch.Tensor:
        """
        Compute quantiles from distribution

        Args:
            params: Distribution parameters
            quantiles: List of quantiles
            num_samples: Number of samples for Monte Carlo

        Returns:
            Quantile predictions [B, pred_len, c_out, num_quantiles]
        """
        # Sample from distribution
        samples = self.sample(params, num_samples)

        # Compute quantiles
        quantile_values = []
        for q in quantiles:
            q_val = torch.quantile(samples, q, dim=0)
            quantile_values.append(q_val)

        # Stack [B, pred_len, c_out, num_quantiles]
        output = torch.stack(quantile_values, dim=-1)

        return output


class DeepAR(nn.Module):
    """
    DeepAR: Probabilistic Forecasting with Autoregressive RNNs
    Amazon's approach to probabilistic time series forecasting
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        pred_len: int,
        distribution: str = 'gaussian',
        dropout: float = 0.1
    ):
        """
        Args:
            input_size: Input feature size
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            pred_len: Prediction length
            distribution: Output distribution
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.distribution = distribution

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Distribution parameter heads
        if distribution == 'gaussian':
            self.mean_head = nn.Linear(hidden_size, 1)
            self.std_head = nn.Linear(hidden_size, 1)
        elif distribution == 'student':
            self.mean_head = nn.Linear(hidden_size, 1)
            self.scale_head = nn.Linear(hidden_size, 1)
            self.df_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        future_covariates: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            x: Historical data [B, seq_len, input_size]
            future_covariates: Future covariates [B, pred_len, covariate_size]

        Returns:
            Distribution parameters
        """
        B = x.size(0)

        # Encode history
        lstm_out, (h, c) = self.lstm(x)

        # Autoregressive prediction
        predictions_mean = []
        predictions_std = []

        # Initialize with last observation
        current_input = x[:, -1:, :]

        for t in range(self.pred_len):
            # LSTM step
            lstm_out, (h, c) = self.lstm(current_input, (h, c))

            # Predict distribution parameters
            if self.distribution == 'gaussian':
                mean = self.mean_head(lstm_out)
                std = F.softplus(self.std_head(lstm_out))
                predictions_mean.append(mean)
                predictions_std.append(std)

                # Sample for next input (during training)
                if self.training:
                    dist = torch.distributions.Normal(mean, std)
                    current_input = dist.sample()
                else:
                    current_input = mean

            # Add covariates if available
            if future_covariates is not None:
                current_input = torch.cat([current_input, future_covariates[:, t:t+1, :]], dim=-1)

        # Stack predictions
        mean = torch.cat(predictions_mean, dim=1)
        std = torch.cat(predictions_std, dim=1)

        return {
            'mean': mean,
            'std': std
        }
