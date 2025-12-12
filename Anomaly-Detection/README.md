# Time Series Forecasting and Anomaly Detection

A PyTorch implementation of transformer-based time series forecasting with integrated anomaly detection capabilities. Supports point forecasting, probabilistic forecasting (quantile and distributional), and time series anomaly detection.

## Overview

This project provides transformer models for time series forecasting along with anomaly detection utilities for temporal data. The implementation focuses on clean, modular code for multivariate time series prediction and analysis.

## Features

- **Transformer Forecasting**: Vanilla transformer architecture with temporal embeddings
- **Probabilistic Forecasting**: Quantile regression and Gaussian likelihood heads
- **Anomaly Detection**: STL decomposition with z-score detection, IsolationForest wrapper
- **Forecasting Metrics**: MAE, RMSE, MAPE, SMAPE, quantile score, coverage, sharpness
- **Flexible Configuration**: Configurable input/prediction lengths, batch processing

## Components

**models/**
- `time_series_models.py` - TransformerForecaster with point/quantile/gaussian heads
- `transformers.py` - VanillaTransformer backbone
- `probabilistic_forecasting.py` - QuantileHead, GaussianHead

**data/**
- `time_series_data.py` - TimeSeriesDataset, WindowConfig, synthetic data generation

**training/**
- `forecasting_trainer.py` - ForecastingTrainer with early stopping

**evaluation/**
- `forecasting_metrics.py` - MAE, RMSE, MAPE, SMAPE, quantile scores

**analysis/**
- `anomaly_detection.py` - STL-based and IsolationForest anomaly detection
- `causal_analysis.py` - Causal analysis utilities

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Anomaly-Detection

# Install dependencies
pip install torch numpy pandas statsmodels scikit-learn
```

**Requirements:**
- Python 3.8+
- PyTorch 1.9+
- pandas, numpy
- statsmodels (for STL decomposition)
- scikit-learn (for IsolationForest)

## Usage

### Basic Forecasting

```python
from data.time_series_data import make_toy_series, TimeSeriesDataset, WindowConfig
from models.time_series_models import TransformerForecaster
from training.forecasting_trainer import ForecastingTrainer
import torch

# Generate synthetic data
df = make_toy_series(n=1000, k=3, freq="H")

# Create dataset with sliding windows
config = WindowConfig(input_len=96, pred_len=24, stride=4)
dataset = TimeSeriesDataset(df, config)

# Split data
train_size = int(len(dataset) * 0.8)
train_data, val_data = dataset[:train_size], dataset[train_size:]

# Create dataloaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize model
model = TransformerForecaster(
    in_dim=df.shape[1],
    out_dim=df.shape[1],
    d_model=64,
    head_type="point"
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
trainer = ForecastingTrainer(model, optimizer, device="cpu", head_type="point")
history = trainer.fit(train_loader, val_loader, epochs=50)
```

### Probabilistic Forecasting

```python
# Quantile forecasting
model = TransformerForecaster(
    in_dim=df.shape[1],
    out_dim=df.shape[1],
    d_model=64,
    head_type="quantile",
    quantiles=[0.1, 0.5, 0.9]
)

# Gaussian likelihood
model = TransformerForecaster(
    in_dim=df.shape[1],
    out_dim=df.shape[1],
    d_model=64,
    head_type="gaussian"
)
```

### Anomaly Detection

```python
from analysis.anomaly_detection import stl_zscore_anomalies, isolation_forest_anomalies
import pandas as pd

# STL-based anomaly detection for univariate series
series = df['x1']
anomalies = stl_zscore_anomalies(series, seasonal=24, z=3.0)
print(anomalies[anomalies['anomaly']])

# IsolationForest for multivariate data
anomalies = isolation_forest_anomalies(df, contamination=0.02)
print(f"Detected {anomalies.sum()} anomalies")
```

### Evaluation

```python
from evaluation.forecasting_metrics import mae, rmse, mape, smape

# Evaluate forecasts
y_true = ...  # actual values
y_pred = ...  # predicted values

print(f"MAE: {mae(y_true, y_pred):.4f}")
print(f"RMSE: {rmse(y_true, y_pred):.4f}")
print(f"MAPE: {mape(y_true, y_pred):.4f}%")
print(f"SMAPE: {smape(y_true, y_pred):.4f}%")
```

## Architecture

### TransformerForecaster

The main model combines:
1. **TemporalEmbedding**: Projects input features to model dimension
2. **Decomposition**: Separates trend and seasonal components
3. **VanillaTransformer**: Attention-based sequence modeling
4. **Forecasting Head**: Point, quantile, or gaussian output

Forecasting heads:
- **PointHead**: Single-value predictions
- **QuantileHead**: Multiple quantile predictions
- **GaussianHead**: Mean and log-std predictions

### Data Pipeline

`TimeSeriesDataset` creates sliding windows from time series data:
- `input_len`: History window size
- `pred_len`: Forecast horizon
- `stride`: Step size between windows

## Project Structure

```
Anomaly-Detection/
├── models/              # Transformer and forecasting models
├── data/                # Time series dataset utilities
├── training/            # Training loops and optimization
├── evaluation/          # Forecasting metrics
├── analysis/            # Anomaly detection and causal analysis
├── realtime/            # Streaming utilities
└── examples/            # Usage examples
```

## Example Script

See `examples/run_example.py` for a complete training and evaluation example.

```bash
python examples/run_example.py
```

## References

- Vaswani et al. "Attention Is All You Need" (2017)
- Cleveland et al. "STL: A Seasonal-Trend Decomposition" (1990)
- Liu et al. "Isolation Forest" (2008)

## License

MIT License
