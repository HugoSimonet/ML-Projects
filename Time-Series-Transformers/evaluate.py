"""
Main Evaluation Script
Evaluate trained time series forecasting models
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.transformers import VanillaTransformer, Informer, Autoformer
from data.time_series_dataset import create_data_loaders, load_csv_dataset
from evaluation.forecasting_metrics import evaluate_forecasting
from visualization.time_series_visualizer import TimeSeriesVisualizer
from utils.config import Config
from utils.helpers import set_seed, get_device
from utils.data_utils import DataLoader, generate_synthetic_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Time Series Forecasting Model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model configuration')
    parser.add_argument('--dataset', type=str, default='ETTh1',
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset file')
    parser.add_argument('--root_path', type=str, default='./data/',
                       help='Root path for data')
    parser.add_argument('--save_dir', type=str, default='./results/',
                       help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU if available')

    return parser.parse_args()


def load_model(model_path: str, config_path: str, device: torch.device):
    """
    Load trained model

    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration
        device: Device to load model on

    Returns:
        Model, Config
    """
    # Load configuration
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        # Try to load config from checkpoint directory
        checkpoint_dir = Path(model_path).parent
        config_file = checkpoint_dir / 'config.yaml'
        if config_file.exists():
            config = Config.from_yaml(str(config_file))
        else:
            raise ValueError("No configuration file found. Please provide --config_path")

    # Build model
    model_config = config.model

    if model_config.model_type == 'VanillaTransformer':
        model = VanillaTransformer(
            enc_in=model_config.enc_in,
            dec_in=model_config.dec_in,
            c_out=model_config.c_out,
            seq_len=model_config.seq_len,
            label_len=model_config.label_len,
            pred_len=model_config.pred_len,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            e_layers=model_config.e_layers,
            d_layers=model_config.d_layers,
            d_ff=model_config.d_ff,
            dropout=model_config.dropout,
            embed=model_config.embed,
            freq=model_config.freq,
            activation=model_config.activation
        )

    elif model_config.model_type == 'Informer':
        model = Informer(
            enc_in=model_config.enc_in,
            dec_in=model_config.dec_in,
            c_out=model_config.c_out,
            seq_len=model_config.seq_len,
            label_len=model_config.label_len,
            pred_len=model_config.pred_len,
            factor=model_config.factor,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            e_layers=model_config.e_layers,
            d_layers=model_config.d_layers,
            d_ff=model_config.d_ff,
            dropout=model_config.dropout,
            embed=model_config.embed,
            freq=model_config.freq,
            activation=model_config.activation,
            distil=model_config.distil
        )

    elif model_config.model_type == 'Autoformer':
        model = Autoformer(
            enc_in=model_config.enc_in,
            dec_in=model_config.dec_in,
            c_out=model_config.c_out,
            seq_len=model_config.seq_len,
            label_len=model_config.label_len,
            pred_len=model_config.pred_len,
            moving_avg=model_config.moving_avg,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            e_layers=model_config.e_layers,
            d_layers=model_config.d_layers,
            d_ff=model_config.d_ff,
            dropout=model_config.dropout,
            embed=model_config.embed,
            freq=model_config.freq,
            activation=model_config.activation
        )

    else:
        raise ValueError(f"Unknown model type: {model_config.model_type}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, config


def evaluate_model(
    model,
    test_loader,
    device: torch.device
):
    """
    Evaluate model on test set

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device

    Returns:
        predictions, actuals
    """
    model.eval()

    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch in test_loader:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch

            x_enc = x_enc.to(device)
            x_mark_enc = x_mark_enc.to(device)
            x_dec = x_dec.to(device)
            x_mark_dec = x_mark_dec.to(device)
            y = y.to(device)

            # Forward pass
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            all_predictions.append(outputs.cpu().numpy())
            all_actuals.append(y.cpu().numpy())

    # Concatenate
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    return predictions, actuals


def main():
    """Main evaluation function"""
    args = parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get device
    if args.use_gpu:
        device = get_device()
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}\n")

    # Load model
    print("Loading model...")
    model, config = load_model(args.model_path, args.config_path, device)
    print(f"Model: {config.model.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Load data
    print("Loading data...")
    data_config = config.data

    # Check if using synthetic data
    if data_config.dataset == 'synthetic' or args.dataset == 'synthetic':
        print("Generating synthetic test data...")
        data = generate_synthetic_data(
            n_samples=2000,
            n_features=config.model.enc_in,
            trend_type='linear',
            seasonality_period=24,
            noise_level=0.1
        )
        test_data = data

    else:
        # Load real dataset
        loader = DataLoader(root_path=args.root_path)

        dataset_name = args.dataset if args.dataset != 'ETTh1' else data_config.dataset

        if dataset_name.startswith('ETT'):
            df = loader.load_ett(dataset_name)
        elif dataset_name == 'electricity':
            df = loader.load_electricity()
        elif dataset_name == 'weather':
            df = loader.load_weather()
        elif dataset_name == 'traffic':
            df = loader.load_traffic()
        else:
            # Try loading as custom CSV
            filepath = Path(args.root_path) / (args.data_path or data_config.data_path)
            if filepath.exists():
                _, _, test_df = load_csv_dataset(
                    str(filepath),
                    target_column=data_config.target,
                    train_ratio=data_config.train_ratio,
                    val_ratio=data_config.val_ratio
                )
            else:
                raise ValueError(f"Dataset {dataset_name} not found")

        if 'test_df' not in locals():
            # Split the loaded df
            n = len(df)
            test_start = int(n * (data_config.train_ratio + data_config.val_ratio))
            test_df = df[test_start:]

        test_data = test_df.select_dtypes(include=['number']).values

    # Create data loader
    from data.time_series_dataset import TimeSeriesDataset
    from torch.utils.data import DataLoader

    test_dataset = TimeSeriesDataset(
        data=test_data,
        seq_len=config.model.seq_len,
        label_len=config.model.label_len,
        pred_len=config.model.pred_len,
        features=data_config.features,
        target=data_config.target,
        scale=data_config.scale,
        freq=data_config.freq
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print(f"Test batches: {len(test_loader)}\n")

    # Evaluate
    print("Evaluating model...")
    predictions, actuals = evaluate_model(model, test_loader, device)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Actuals shape: {actuals.shape}\n")

    # Calculate metrics
    print("Computing metrics...")
    metrics = evaluate_forecasting(predictions, actuals)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():<15}: {value:.6f}")
    print("=" * 60 + "\n")

    # Save metrics
    metrics_path = save_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    # Save predictions
    np.save(save_dir / 'predictions.npy', predictions)
    np.save(save_dir / 'actuals.npy', actuals)
    print(f"Predictions saved to: {save_dir}")

    # Visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        visualizer = TimeSeriesVisualizer()

        # Flatten for univariate visualization
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        actual_flat = actuals.reshape(-1, actuals.shape[-1])

        # Plot first variable
        visualizer.plot_forecasts(
            actual=actual_flat[:500, 0],
            predicted=pred_flat[:500, 0],
            title='Forecast: First 500 Steps',
            save_path=str(save_dir / 'forecast_plot.png')
        )

        # Plot residuals
        residuals = (pred_flat - actual_flat).flatten()
        visualizer.plot_residuals(
            residuals=residuals,
            title='Forecast Residuals',
            save_path=str(save_dir / 'residuals.png')
        )

        # Plot metrics
        visualizer.plot_metric_comparison(
            metrics=metrics,
            title='Model Performance Metrics',
            save_path=str(save_dir / 'metrics_comparison.png')
        )

        print(f"Visualizations saved to: {save_dir}")

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
