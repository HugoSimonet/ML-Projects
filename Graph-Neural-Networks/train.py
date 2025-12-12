"""
Training Script for Graph Neural Networks

Supports multiple tasks:
- Node classification
- Link prediction
- Community detection
- Influence maximization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import argparse
import os
from tqdm import tqdm

from models import GCN, GAT, GraphSAGE, GraphTransformer, CommunityDetector, InfluenceMaximizer
from data import SocialNetworkDataset, GraphDataLoader
from evaluation import GraphEvaluator
from utils import load_config, setup_logger, save_checkpoint, EarlyStopping, set_seed
from visualization import TrainingVisualizer


def train_node_classification(
    model: nn.Module,
    data: Data,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """Train one epoch for node classification."""
    model.train()
    data = data.to(device)

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_node_classification(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    device: str
) -> dict:
    """Evaluate node classification."""
    evaluator = GraphEvaluator(model, device)
    metrics = evaluator.evaluate_node_classification(data, mask)
    return metrics


def train_community_detection(
    model: CommunityDetector,
    data: Data,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train one epoch for community detection."""
    model.train()
    data = data.to(device)

    optimizer.zero_grad()

    # Forward pass
    community_logits, node_embeddings = model(data.x, data.edge_index)

    # Classification loss
    if hasattr(data, 'y'):
        cls_loss = nn.functional.cross_entropy(community_logits, data.y)
    else:
        cls_loss = 0

    # Modularity loss
    mod_loss = model.modularity_loss(community_logits, data.edge_index)

    # Total loss
    loss = cls_loss - 0.1 * mod_loss  # Maximize modularity

    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser(description='Train Graph Neural Networks')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--task', type=str, default='node_classification',
                       choices=['node_classification', 'link_prediction',
                               'community_detection', 'influence_maximization'],
                       help='Task to train')
    parser.add_argument('--dataset', type=str, default='cora',
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphsage', 'transformer'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (overrides config)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.device is not None:
        config['system']['device'] = args.device

    # Setup logger
    logger = setup_logger('gnn_training', log_dir='logs')
    logger.info(f"Starting training for task: {args.task}")
    logger.info(f"Dataset: {args.dataset}, Model: {args.model}")

    # Determine device
    device = config['system']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = SocialNetworkDataset(args.dataset, root='data/')
    data = dataset[0]

    logger.info(f"Dataset: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

    # Create model
    input_dim = data.x.size(1)
    hidden_dim = config['model']['hidden_dim']
    output_dim = dataset.num_classes if args.task == 'node_classification' else hidden_dim

    logger.info(f"Creating {args.model.upper()} model")

    if args.task == 'node_classification':
        if args.model == 'gcn':
            model = GCN(input_dim, hidden_dim, output_dim,
                       num_layers=config['model']['num_layers'],
                       dropout=config['model']['dropout'])
        elif args.model == 'gat':
            model = GAT(input_dim, hidden_dim, output_dim,
                       num_layers=config['model']['num_layers'],
                       num_heads=config['model']['num_heads'],
                       dropout=config['model']['dropout'])
        elif args.model == 'graphsage':
            model = GraphSAGE(input_dim, hidden_dim, output_dim,
                            num_layers=config['model']['num_layers'],
                            dropout=config['model']['dropout'])
        elif args.model == 'transformer':
            model = GraphTransformer(input_dim, hidden_dim, output_dim,
                                   num_layers=config['model']['num_layers'],
                                   num_heads=config['model']['num_heads'],
                                   dropout=config['model']['dropout'])
    elif args.task == 'community_detection':
        num_communities = config['community']['num_communities']
        model = CommunityDetector(input_dim, hidden_dim, num_communities,
                                 gnn_type=args.model,
                                 num_layers=config['model']['num_layers'],
                                 dropout=config['model']['dropout'])
    elif args.task == 'influence_maximization':
        model = InfluenceMaximizer(input_dim, hidden_dim,
                                  gnn_type=args.model,
                                  num_layers=config['model']['num_layers'],
                                  dropout=config['model']['dropout'])

    model = model.to(device)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )

    # Training loop
    logger.info("Starting training...")

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0

    for epoch in range(1, config['training']['epochs'] + 1):
        # Training
        if args.task == 'node_classification':
            train_loss = train_node_classification(model, data, optimizer, criterion, device)
        elif args.task == 'community_detection':
            train_loss = train_community_detection(model, data, optimizer, device)

        train_losses.append(train_loss)

        # Validation
        if args.task == 'node_classification' and hasattr(data, 'val_mask'):
            val_metrics = evaluate_node_classification(model, data, data.val_mask, device)
            val_loss = val_metrics.get('accuracy', 0)  # Using accuracy as proxy
            val_acc = val_metrics['accuracy']

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Logging
            if epoch % config['system']['log_interval'] == 0:
                logger.info(f"Epoch {epoch}/{config['training']['epochs']}: "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}, "
                          f"Val F1: {val_metrics['f1_score']:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_metrics,
                    f'checkpoints/{args.task}_{args.model}_best.pth',
                    scheduler
                )
                logger.info(f"Saved best model with validation accuracy: {val_acc:.4f}")

            # Early stopping
            if early_stopping(-val_acc):  # Negative because we want to maximize accuracy
                logger.info(f"Early stopping at epoch {epoch}")
                break
        else:
            if epoch % config['system']['log_interval'] == 0:
                logger.info(f"Epoch {epoch}/{config['training']['epochs']}: "
                          f"Train Loss: {train_loss:.4f}")

        # Update scheduler
        scheduler.step()

    # Test evaluation
    if args.task == 'node_classification' and hasattr(data, 'test_mask'):
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_node_classification(model, data, data.test_mask, device)
        logger.info(f"Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

    # Plot training curves
    visualizer = TrainingVisualizer()
    visualizer.plot_training_curves(
        train_losses,
        val_losses if val_losses else None,
        save_path=f'results/{args.task}_{args.model}_training.png'
    )

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
