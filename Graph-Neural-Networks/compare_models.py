"""
Model Comparison Script

Compares different GNN architectures on the same dataset.
"""

import torch
import argparse
from typing import Dict

from models import GCN, GAT, GraphSAGE, GraphTransformer
from data import SocialNetworkDataset
from evaluation import GraphEvaluator
from utils import load_config, setup_logger, set_seed
from visualization import TrainingVisualizer


def train_and_evaluate_model(
    model_name: str,
    model: torch.nn.Module,
    data,
    config: Dict,
    device: str,
    logger
) -> Dict[str, float]:
    """Train and evaluate a single model."""
    logger.info(f"\nTraining {model_name.upper()}...")

    model = model.to(device)
    data = data.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Training
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            logger.info(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

    # Evaluation
    evaluator = GraphEvaluator(model, device)
    test_metrics = evaluator.evaluate_node_classification(data, data.test_mask)

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Compare GNN Models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default='cora',
                       help='Dataset name')
    parser.add_argument('--models', type=str, default='gcn,gat,graphsage,transformer',
                       help='Comma-separated list of models to compare')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Setup logger
    logger = setup_logger('model_comparison', log_dir='logs')
    logger.info(f"Comparing models on dataset: {args.dataset}")

    # Load config
    config = load_config(args.config)
    config['training']['epochs'] = args.epochs

    # Determine device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = SocialNetworkDataset(args.dataset, root='data/')
    data = dataset[0]

    input_dim = data.x.size(1)
    hidden_dim = config['model']['hidden_dim']
    output_dim = dataset.num_classes

    # Parse model list
    model_names = [m.strip() for m in args.models.split(',')]

    # Train and evaluate each model
    results = {}

    for model_name in model_names:
        # Create model
        if model_name == 'gcn':
            model = GCN(input_dim, hidden_dim, output_dim,
                       num_layers=config['model']['num_layers'],
                       dropout=config['model']['dropout'])
        elif model_name == 'gat':
            model = GAT(input_dim, hidden_dim, output_dim,
                       num_layers=config['model']['num_layers'],
                       num_heads=config['model']['num_heads'],
                       dropout=config['model']['dropout'])
        elif model_name == 'graphsage':
            model = GraphSAGE(input_dim, hidden_dim, output_dim,
                            num_layers=config['model']['num_layers'],
                            dropout=config['model']['dropout'])
        elif model_name == 'transformer':
            model = GraphTransformer(input_dim, hidden_dim, output_dim,
                                   num_layers=config['model']['num_layers'],
                                   num_heads=config['model']['num_heads'],
                                   dropout=config['model']['dropout'])
        else:
            logger.warning(f"Unknown model: {model_name}, skipping...")
            continue

        # Train and evaluate
        metrics = train_and_evaluate_model(
            model_name, model, data, config, device, logger
        )

        results[model_name] = metrics

        logger.info(f"{model_name.upper()} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

    # Compare results
    logger.info("\n" + "="*50)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*50)

    for model_name, metrics in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  AUC-ROC: {metrics.get('auc_roc', 0):.4f}")

    # Visualize comparison
    visualizer = TrainingVisualizer()
    visualizer.plot_model_comparison(
        results,
        metric='accuracy',
        save_path='results/model_comparison_accuracy.png'
    )
    visualizer.plot_model_comparison(
        results,
        metric='f1_score',
        save_path='results/model_comparison_f1.png'
    )

    logger.info("\nComparison completed!")


if __name__ == '__main__':
    main()
