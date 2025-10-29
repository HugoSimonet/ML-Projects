# Time Series Forecasting with Transformers

## 🎯 Project Overview

This project implements state-of-the-art Transformer architectures for time series forecasting, demonstrating advanced understanding of sequence modeling, attention mechanisms, and temporal data analysis. The project covers multiple forecasting tasks including univariate/multivariate prediction, anomaly detection, and causal inference.

## 🚀 Key Features

- **Multiple Transformer Architectures**: Vanilla Transformer, Informer, Autoformer, and custom models
- **Multi-Scale Forecasting**: Short-term, medium-term, and long-term predictions
- **Multivariate Analysis**: Handle multiple time series simultaneously
- **Anomaly Detection**: Identify unusual patterns in time series
- **Causal Inference**: Understand cause-effect relationships in temporal data
- **Real-time Prediction**: Online learning and streaming predictions

## 🧠 Technical Architecture

### Core Components

1. **Transformer Encoder-Decoder**
   - Multi-head self-attention mechanisms
   - Positional encoding for temporal information
   - Feed-forward networks with residual connections
   - Layer normalization and dropout

2. **Time Series Specific Modules**
   - Temporal embedding layers
   - Seasonal decomposition components
   - Trend extraction modules
   - Frequency domain analysis

3. **Attention Mechanisms**
   - Self-attention for temporal dependencies
   - Cross-attention for multivariate relationships
   - Probabilistic attention for uncertainty
   - Sparse attention for efficiency

4. **Forecasting Heads**
   - Point forecasting
   - Probabilistic forecasting
   - Quantile regression
   - Distributional forecasting

### Advanced Techniques

- **Informer Architecture**: Efficient attention for long sequences
- **Autoformer**: Decomposition-based forecasting
- **Probabilistic Forecasting**: Uncertainty quantification
- **Multi-Scale Learning**: Hierarchical temporal patterns
- **Causal Attention**: Causal inference in time series

## 📊 Supported Forecasting Tasks

- **Univariate Forecasting**: Single time series prediction
- **Multivariate Forecasting**: Multiple correlated time series
- **Long-term Forecasting**: Predictions far into the future
- **Short-term Forecasting**: High-frequency predictions
- **Anomaly Detection**: Identify unusual patterns
- **Causal Analysis**: Understand causal relationships

## 🛠️ Implementation Details

### Transformer Architecture
```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, seq_len, pred_len):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Forecasting head
        self.forecasting_head = nn.Linear(d_model, pred_len)
        
        # Probabilistic head
        self.probabilistic_head = nn.Linear(d_model, pred_len * 2)  # mean and std
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Forecasting
        point_forecast = self.forecasting_head(x)
        prob_forecast = self.probabilistic_head(x)
        
        return point_forecast, prob_forecast
```

### Informer Architecture
```python
class Informer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ProbSparse attention
        self.attention = ProbSparseAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            factor=config.factor
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(config) for _ in range(config.e_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            InformerDecoderLayer(config) for _ in range(config.d_layers)
        ])
    
    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Encoder
        enc_out = x_enc
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, enc_self_mask)
        
        # Decoder
        dec_out = x_dec
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, dec_self_mask, dec_enc_mask)
        
        return dec_out
```

### Probabilistic Forecasting
```python
class ProbabilisticForecaster(nn.Module):
    def __init__(self, base_model, num_quantiles=9):
        super().__init__()
        self.base_model = base_model
        self.num_quantiles = num_quantiles
        
        # Quantile regression heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(base_model.d_model, base_model.pred_len)
            for _ in range(num_quantiles)
        ])
    
    def forward(self, x):
        # Get base features
        features = self.base_model.get_features(x)
        
        # Predict quantiles
        quantiles = []
        for head in self.quantile_heads:
            quantile = head(features)
            quantiles.append(quantile)
        
        return torch.stack(quantiles, dim=-1)
```

## 📈 Performance Metrics

### Forecasting Accuracy
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Square Error)**: Square root of average squared error
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error metric
- **SMAPE (Symmetric MAPE)**: Symmetric percentage error

### Probabilistic Forecasting
- **Quantile Score**: Accuracy of quantile predictions
- **CRPS (Continuous Ranked Probability Score)**: Probabilistic accuracy
- **Coverage**: Percentage of true values within prediction intervals
- **Sharpness**: Width of prediction intervals

### Anomaly Detection
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (recommended for training)
- 16GB+ RAM recommended
- 20GB+ disk space for datasets

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Time-Series-Transformers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional time series libraries
pip install tslearn statsmodels scikit-learn

# Download datasets
python scripts/download_datasets.py
```

## 🚀 Quick Start

### 1. Basic Time Series Forecasting
```python
from models import TimeSeriesTransformer
from data import TimeSeriesDataset

# Load time series data
dataset = TimeSeriesDataset('electricity', seq_len=96, pred_len=24)
train_loader, val_loader, test_loader = dataset.get_data_loaders()

# Initialize model
model = TimeSeriesTransformer(
    input_dim=dataset.input_dim,
    d_model=512,
    n_heads=8,
    n_layers=6,
    seq_len=96,
    pred_len=24
)

# Train model
model.train_model(train_loader, val_loader, epochs=100)

# Evaluate
metrics = model.evaluate(test_loader)
print(f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
```

### 2. Probabilistic Forecasting
```python
from models import ProbabilisticForecaster

# Initialize probabilistic model
prob_model = ProbabilisticForecaster(
    base_model=model,
    num_quantiles=9
)

# Train probabilistic model
prob_model.train_probabilistic(train_loader, val_loader, epochs=100)

# Generate probabilistic forecasts
forecasts = prob_model.predict(test_loader)
quantiles = forecasts['quantiles']  # Shape: (batch, pred_len, num_quantiles)
```

### 3. Anomaly Detection
```python
from models import AnomalyDetector

# Initialize anomaly detector
detector = AnomalyDetector(
    base_model=model,
    threshold_method='iqr'
)

# Train detector
detector.train(train_loader)

# Detect anomalies
anomalies = detector.detect_anomalies(test_loader)
anomaly_scores = detector.get_anomaly_scores(test_loader)
```

## 📊 Training and Evaluation

### Training Scripts
```bash
# Basic forecasting
python train.py --model transformer --dataset electricity --epochs 100

# Informer model
python train.py --model informer --dataset weather --epochs 200

# Probabilistic forecasting
python train.py --model probabilistic --dataset traffic --epochs 150

# Anomaly detection
python train.py --model anomaly --dataset ecg --epochs 100
```

### Evaluation
```bash
# Evaluate forecasting performance
python evaluate.py --model_path checkpoints/transformer.pth --dataset electricity

# Compare different models
python compare_models.py --models transformer,informer,autoformer

# Analyze probabilistic forecasts
python analyze_probabilistic.py --results_dir results/
```

## 🎨 Visualization and Analysis

### Forecasting Visualization
```python
from visualization import TimeSeriesVisualizer

visualizer = TimeSeriesVisualizer()
visualizer.plot_forecasts(actual, predicted, quantiles)
visualizer.plot_attention_weights(attention_weights)
visualizer.plot_residuals(residuals)
visualizer.plot_anomalies(anomalies)
```

### Performance Analysis
```python
from analysis import ForecastingAnalyzer

analyzer = ForecastingAnalyzer(results)
analyzer.plot_error_distribution()
analyzer.plot_quantile_coverage()
analyzer.plot_horizon_accuracy()
```

## 🔬 Research Contributions

### Novel Techniques
1. **Causal Transformer**: Causal inference in time series
2. **Multi-Scale Attention**: Hierarchical temporal attention
3. **Adaptive Forecasting**: Dynamic model selection

### Experimental Studies
- **Long-term Forecasting**: Performance on very long horizons
- **Multivariate Analysis**: Cross-series dependencies
- **Anomaly Detection**: Robustness to different anomaly types

## 📚 Advanced Features

### Multi-Scale Forecasting
- **Hierarchical Forecasting**: Multiple time scales
- **Seasonal Decomposition**: Trend, seasonal, and residual components
- **Frequency Analysis**: Fourier and wavelet transforms
- **Adaptive Windows**: Dynamic sequence lengths

### Causal Analysis
- **Granger Causality**: Statistical causality testing
- **Causal Attention**: Causal attention mechanisms
- **Intervention Analysis**: Effect of interventions
- **Counterfactual Prediction**: What-if scenarios

### Real-time Processing
- **Streaming Forecasting**: Online learning
- **Incremental Updates**: Model updates with new data
- **Latency Optimization**: Fast inference
- **Memory Management**: Efficient memory usage

## 🚀 Deployment Considerations

### Production Deployment
- **Model Serving**: REST API for forecasting
- **Batch Processing**: Large-scale batch forecasting
- **Real-time Streaming**: Stream processing
- **Monitoring**: Performance and accuracy monitoring

### Integration
- **Time Series Databases**: Integration with InfluxDB, TimescaleDB
- **Data Pipelines**: ETL pipeline integration
- **Visualization Tools**: Integration with Grafana, Tableau

## 📚 References and Citations

### Key Papers
- Vaswani, A., et al. "Attention Is All You Need"
- Zhou, H., et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
- Wu, H., et al. "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"

### Time Series Forecasting
- Box, G. E. P., & Jenkins, G. M. "Time Series Analysis: Forecasting and Control"
- Hyndman, R. J., & Athanasopoulos, G. "Forecasting: Principles and Practice"

## 🚀 Future Enhancements

### Planned Features
- **Multi-Modal Time Series**: Combine with other data types
- **Federated Time Series**: Distributed forecasting
- **Quantum Time Series**: Quantum computing for time series
- **Causal Discovery**: Automated causal structure learning

### Research Directions
- **Foundation Models**: Large-scale pre-trained time series models
- **Few-Shot Forecasting**: Learning with limited data
- **Interpretable Forecasting**: Explainable time series models

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation standards
- Time series analysis methodologies

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The Transformer research community
- Time series forecasting researchers
- Contributors to time series libraries
- The open-source ML community

---

**Note**: This project demonstrates advanced understanding of Transformer architectures, time series analysis, and probabilistic forecasting. The implementation showcases both theoretical knowledge and practical skills in cutting-edge time series forecasting research and applications.
