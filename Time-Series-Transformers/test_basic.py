"""
Basic Test Script
Quick test to verify the implementation works
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

print("=" * 80)
print("Testing Time Series Transformers Project")
print("=" * 80)

# Test 1: Import modules
print("\n[1/7] Testing imports...")
try:
    from models import VanillaTransformer, Informer, Autoformer
    from data.time_series_dataset import TimeSeriesDataset
    from training.forecasting_trainer import ForecastingTrainer
    from evaluation.forecasting_metrics import ForecastingMetrics, evaluate_forecasting
    from visualization.time_series_visualizer import TimeSeriesVisualizer
    from utils import Config, set_seed, get_device
    from analysis import AnomalyDetector, GrangerCausality
    print("[OK] All imports successful")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create synthetic data
print("\n[2/7] Creating synthetic data...")
try:
    from utils.data_utils import generate_synthetic_data
    data = generate_synthetic_data(
        n_samples=1000,
        n_features=7,
        trend_type='linear',
        seasonality_period=24,
        noise_level=0.1,
        seed=42
    )
    print(f"[OK] Generated data shape: {data.shape}")
except Exception as e:
    print(f"[FAIL] Data generation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create dataset
print("\n[3/7] Creating dataset...")
try:
    dataset = TimeSeriesDataset(
        data=data,
        seq_len=96,
        label_len=48,
        pred_len=24,
        features='M',
        scale=True
    )
    print(f"[OK] Dataset created with {len(dataset)} samples")
except Exception as e:
    print(f"[FAIL] Dataset creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Build model
print("\n[4/7] Building VanillaTransformer model...")
try:
    device = torch.device('cpu')
    model = VanillaTransformer(
        enc_in=7,
        dec_in=7,
        c_out=7,
        seq_len=96,
        label_len=48,
        pred_len=24,
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        dropout=0.1
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created with {num_params:,} parameters")
except Exception as e:
    print(f"[FAIL] Model creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass
print("\n[5/7] Testing forward pass...")
try:
    # Get a sample from dataset
    x_enc, x_mark_enc, x_dec, x_mark_dec, y = dataset[0]

    # Add batch dimension
    x_enc = x_enc.unsqueeze(0)
    x_mark_enc = x_mark_enc.unsqueeze(0)
    x_dec = x_dec.unsqueeze(0)
    x_mark_dec = x_mark_dec.unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    print(f"[OK] Forward pass successful")
    print(f"  Input shape: {x_enc.shape}")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"[FAIL] Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Evaluation metrics
print("\n[6/7] Testing evaluation metrics...")
try:
    predictions = np.random.randn(100, 24, 7)
    actuals = np.random.randn(100, 24, 7)

    metrics = evaluate_forecasting(predictions, actuals)

    print("[OK] Metrics computed successfully:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
except Exception as e:
    print(f"[FAIL] Metrics computation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Configuration
print("\n[7/7] Testing configuration...")
try:
    config = Config()
    config.model.model_type = 'VanillaTransformer'
    config.training.epochs = 10
    config.data.dataset = 'synthetic'

    print("[OK] Configuration created successfully")
    print(f"  Model: {config.model.model_type}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Dataset: {config.data.dataset}")
except Exception as e:
    print(f"[FAIL] Configuration error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("[SUCCESS] ALL TESTS PASSED!")
print("=" * 80)
print("\nThe Time-Series-Transformers project is ready to use!")
print("\nQuick start:")
print("  1. Train a model:")
print("     python train.py --model VanillaTransformer --dataset synthetic --epochs 10")
print("\n  2. Evaluate a model:")
print("     python evaluate.py --model_path checkpoints/best_model.pth")
print("=" * 80)
