# Time Series Forecasting with Transformers

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Transformer-based models for time series forecasting including Informer, Autoformer, and vanilla Transformer architectures.

## Overview

This project implements Transformer architectures for time series forecasting tasks. It supports univariate and multivariate forecasting, long-horizon prediction, and provides uncertainty quantification.

## Features

- Transformer architectures: Vanilla, Informer, Autoformer
- Univariate and multivariate forecasting
- Long-horizon prediction (up to 720 timesteps)
- Probabilistic forecasting with uncertainty
- Seasonal-trend decomposition
- Sparse attention for efficiency

## Models

**Vanilla Transformer** - Standard encoder-decoder with temporal positional encoding
**Informer** - Efficient attention mechanism for long sequences
**Autoformer** - Auto-correlation and seasonal decomposition

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 1.9+, numpy, pandas, scikit-learn

## Quick Start

```bash
# Train Informer on electricity dataset
python train.py \
    --model informer \
    --dataset electricity \
    --seq_len 96 \
    --pred_len 24 \
    --epochs 10

# Multivariate forecasting
python train.py \
    --model autoformer \
    --dataset weather \
    --features M \
    --seq_len 96 \
    --pred_len 192
```

## Usage

```python
from models import Informer
from utils import TimeSeriesDataset

# Create model
model = Informer(
    enc_in=7,  # input features
    dec_in=7,
    c_out=7,   # output features
    seq_len=96,
    label_len=48,
    out_len=24,
    factor=5,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=2048,
    dropout=0.05
)

# Train
dataset = TimeSeriesDataset(data, seq_len=96, pred_len=24)
dataloader = DataLoader(dataset, batch_size=32)

for batch_x, batch_y in dataloader:
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
```

## Datasets

- **ETT (Electricity Transformer Temperature)**: ETTh1, ETTh2, ETTm1, ETTm2
- **Electricity**: UCI Electricity Load
- **Weather**: Weather forecasting
- **Traffic**: Traffic flow prediction
- **Custom**: Load from CSV with timestamps

## Configuration

```yaml
model:
  type: informer
  d_model: 512
  n_heads: 8
  e_layers: 2
  d_layers: 1
  d_ff: 2048
  dropout: 0.05

data:
  dataset: electricity
  features: M  # M=multivariate, S=univariate
  target: OT   # target column
  seq_len: 96
  label_len: 48
  pred_len: 24

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.0001
  patience: 3
```

## Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Symmetric MAPE (SMAPE)

## Attention Visualization

```python
from visualization import plot_attention

# Get attention weights
attn_weights = model.get_attention_weights(input_seq)

# Visualize
plot_attention(attn_weights, save_path='attention.png')
```

## Project Structure

```
Time-Series-Transformers/
├── models/              # Transformer architectures
├── training/            # Training loops
├── evaluation/          # Metrics and evaluation
├── utils/               # Data loading, preprocessing
├── visualization/       # Plotting tools
├── configs/             # Configuration files
├── train.py             # Main training script
└── evaluate.py          # Evaluation script
```

## Implementation Notes

Uses PyTorch with custom Transformer implementations. Positional encoding combines absolute and learnable embeddings. Informer uses ProbSparse attention to reduce complexity from O(L²) to O(L log L).

Autoformer decomposes series into seasonal and trend components using moving average. Seasonal part processed by Auto-Correlation module.

For long-horizon forecasting (>96 steps), use label_len parameter for decoder warm-up.

## References

- Vaswani et al. "Attention is All You Need" (Transformer)
- Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
- Wu et al. "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"

## License

MIT License - see LICENSE file for details.
