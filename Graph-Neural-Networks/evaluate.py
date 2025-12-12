"""
Evaluation Script for Graph Neural Networks

Evaluates trained models on various tasks.
"""

import torch
import argparse
import os

from models import GCN, GAT, GraphSAGE, GraphTransformer
from data import SocialNetworkDataset
from evaluation import GraphEvaluator
from utils import load_config, load_checkpoint, setup_logger
from visualization import TrainingVisualizer


def main():
    parser = argparse.ArgumentParser(description='Evaluate Graph Neural Networks')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default='cora',
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphsage', 'transformer'],
                       help='Model architecture')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('gnn_evaluation', log_dir='logs')
    logger.info(f"Evaluating model: {args.model_path}")

    # Load config
    config = load_config(args.config)

    # Determine device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = SocialNetworkDataset(args.dataset, root='data/')
    data = dataset[0]

    # Create model
    input_dim = data.x.size(1)
    hidden_dim = config['model']['hidden_dim']
    output_dim = dataset.num_classes

    logger.info(f"Creating {args.model.upper()} model")

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

    model = model.to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.model_path}")
    checkpoint_info = load_checkpoint(args.model_path, model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")

    # Create evaluator
    evaluator = GraphEvaluator(model, device)

    # Evaluate on train, validation, and test sets
    logger.info("\nEvaluating on different splits...")

    # Train set
    if hasattr(data, 'train_mask'):
        train_metrics = evaluator.evaluate_node_classification(data, data.train_mask)
        logger.info("\nTrain Set Results:")
        for metric, value in train_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

    # Validation set
    if hasattr(data, 'val_mask'):
        val_metrics = evaluator.evaluate_node_classification(data, data.val_mask)
        logger.info("\nValidation Set Results:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

    # Test set
    if hasattr(data, 'test_mask'):
        test_metrics = evaluator.evaluate_node_classification(data, data.test_mask)
        logger.info("\nTest Set Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

    # Visualization
    visualizer = TrainingVisualizer()
    visualizer.plot_metric_comparison(
        test_metrics,
        title=f'{args.model.upper()} Test Performance',
        save_path=f'results/{args.model}_test_metrics.png'
    )

    logger.info("\nEvaluation completed!")


if __name__ == '__main__':
    main()
