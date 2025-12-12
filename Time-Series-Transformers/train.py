"""
Main Training Script
Train time series forecasting models
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.transformers import VanillaTransformer, Informer, Autoformer
from data.time_series_dataset import create_data_loaders, load_csv_dataset
from training.forecasting_trainer import ForecastingTrainer
from utils.config import Config, get_default_config
from utils.helpers import set_seed, get_device, print_model_summary, create_directories
from utils.data_utils import DataLoader, generate_synthetic_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Time Series Forecasting Model')

    # Model arguments
    parser.add_argument('--model', type=str, default='VanillaTransformer',
                       choices=['VanillaTransformer', 'Informer', 'Autoformer'],
                       help='Model architecture')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='ETTh1',
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset file')
    parser.add_argument('--root_path', type=str, default='./data/',
                       help='Root path for data')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')

    # Model hyperparameters
    parser.add_argument('--seq_len', type=int, default=96,
                       help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                       help='Start token length')
    parser.add_argument('--pred_len', type=int, default=24,
                       help='Prediction length')
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                       help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                       help='Number of decoder layers')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                       help='Directory to save checkpoints')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU if available')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')

    return parser.parse_args()


def build_model(config: Config, device: torch.device):
    """
    Build model from configuration

    Args:
        config: Configuration object
        device: Device to build model on

    Returns:
        Model
    """
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

    return model.to(device)


def prepare_data(config: Config):
    """
    Prepare data loaders

    Args:
        config: Configuration object

    Returns:
        train_loader, val_loader, test_loader
    """
    data_config = config.data

    # Check if using synthetic data
    if data_config.dataset == 'synthetic':
        print("Generating synthetic data...")
        data = generate_synthetic_data(
            n_samples=10000,
            n_features=config.model.enc_in,
            trend_type='linear',
            seasonality_period=24,
            noise_level=0.1,
            seed=config.experiment.seed
        )

        # Split data
        n = len(data)
        train_end = int(n * data_config.train_ratio)
        val_end = int(n * (data_config.train_ratio + data_config.val_ratio))

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

    else:
        # Load real dataset
        loader = DataLoader(root_path=data_config.root_path)

        if data_config.dataset.startswith('ETT'):
            df = loader.load_ett(data_config.dataset)
        elif data_config.dataset == 'electricity':
            df = loader.load_electricity()
        elif data_config.dataset == 'weather':
            df = loader.load_weather()
        elif data_config.dataset == 'traffic':
            df = loader.load_traffic()
        else:
            # Try loading as custom CSV
            filepath = Path(data_config.root_path) / data_config.data_path
            if filepath.exists():
                train_df, val_df, test_df = load_csv_dataset(
                    str(filepath),
                    target_column=data_config.target,
                    train_ratio=data_config.train_ratio,
                    val_ratio=data_config.val_ratio
                )
            else:
                raise ValueError(f"Dataset {data_config.dataset} not found")
                return None

        # Create data loaders
        if 'train_df' not in locals():
            # Split the loaded df
            n = len(df)
            train_end = int(n * data_config.train_ratio)
            val_end = int(n * (data_config.train_ratio + data_config.val_ratio))

            train_df = df[:train_end]
            val_df = df[train_end:val_end]
            test_df = df[val_end:]

        # Extract numeric data
        train_data = train_df.select_dtypes(include=['number']).values
        val_data = val_df.select_dtypes(include=['number']).values
        test_data = test_df.select_dtypes(include=['number']).values

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        seq_len=config.model.seq_len,
        label_len=config.model.label_len,
        pred_len=config.model.pred_len,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        features=data_config.features,
        target=data_config.target,
        freq=data_config.freq
    )

    return train_loader, val_loader, test_loader


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Load or create configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_default_config(args.model)

        # Update config with command line arguments
        config.model.model_type = args.model
        config.model.seq_len = args.seq_len
        config.model.label_len = args.label_len
        config.model.pred_len = args.pred_len
        config.model.d_model = args.d_model
        config.model.n_heads = args.n_heads
        config.model.e_layers = args.e_layers
        config.model.d_layers = args.d_layers

        config.training.epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.lr
        config.training.early_stopping_patience = args.patience
        config.training.use_amp = args.use_amp

        config.data.dataset = args.dataset
        config.data.root_path = args.root_path
        if args.data_path:
            config.data.data_path = args.data_path

        config.experiment.seed = args.seed
        config.experiment.save_dir = args.save_dir

    # Set seed
    set_seed(config.experiment.seed)

    # Create directories
    create_directories([
        config.experiment.save_dir,
        config.experiment.log_dir,
        config.experiment.result_dir
    ])

    # Print configuration
    print(config)

    # Get device
    if args.use_gpu:
        device = get_device(config.training.gpu_id)
    else:
        device = torch.device('cpu')

    print(f"\nUsing device: {device}\n")

    # Prepare data
    print("Loading data...")
    train_loader, val_loader, test_loader = prepare_data(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}\n")

    # Build model
    print("Building model...")
    model = build_model(config, device)
    print_model_summary(model)

    # Create learning rate scheduler
    scheduler = None
    if config.training.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,  # Will be created in trainer
            T_max=config.training.epochs
        )
    elif config.training.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            trainer.optimizer,
            step_size=30,
            gamma=0.1
        )

    # Create trainer
    trainer = ForecastingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=config.training.criterion,
        optimizer=config.training.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        device=str(device),
        save_dir=config.experiment.save_dir,
        use_amp=config.training.use_amp
    )

    # Train
    print("\nStarting training...")
    history = trainer.train(
        epochs=config.training.epochs,
        scheduler=scheduler,
        early_stopping_patience=config.training.early_stopping_patience,
        gradient_clip=config.training.gradient_clip,
        log_interval=config.experiment.log_interval,
        save_best=config.experiment.save_best
    )

    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")

    # Save configuration
    config.to_yaml(Path(config.experiment.save_dir) / 'config.yaml')

    return trainer, history


if __name__ == '__main__':
    trainer, history = main()
