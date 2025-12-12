"""
Evaluation script for federated learning models
Loads trained models and evaluates performance
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import logging
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from config import DatasetType

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str, data_path: str = './data'):
    """Load test dataset"""
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_dataset = datasets.MNIST(
            data_path,
            train=False,
            download=True,
            transform=transform
        )

    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_dataset = datasets.CIFAR10(
            data_path,
            train=False,
            download=True,
            transform=transform
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return test_dataset


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu',
    criterion: nn.Module = None
) -> dict:
    """
    Evaluate model on test dataset

    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Computing device
        criterion: Loss function

    Returns:
        Dictionary of evaluation metrics
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0
    total = 0

    # Per-class accuracy
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Per-class statistics
            for i in range(len(target)):
                label = target[i].item()
                pred = predicted[i].item()

                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0

                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    # Overall metrics
    metrics = {
        'test_loss': test_loss / len(test_loader),
        'test_accuracy': 100. * correct / total,
        'total_samples': total,
        'correct_predictions': correct
    }

    # Per-class accuracy
    class_accuracies = {
        label: 100. * class_correct[label] / class_total[label]
        for label in class_total.keys()
    }
    metrics['class_accuracies'] = class_accuracies

    return metrics


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Federated Learning Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to evaluate on')
    parser.add_argument('--model_name', type=str, default='mnist_net',
                       help='Model architecture name')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to dataset')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save evaluation results JSON')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 80)

    # Check if model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load checkpoint
    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)

    # Get model parameters
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        round_num = checkpoint.get('round', 'unknown')
        logger.info(f"Loaded checkpoint from round: {round_num}")
    else:
        model_state = checkpoint
        logger.info("Loaded model state dictionary")

    # Determine number of classes
    num_classes = 10  # Default for MNIST and CIFAR-10

    # Create model
    logger.info(f"Creating model: {args.model_name}...")
    if args.dataset == 'mnist':
        model = create_model(args.model_name, num_classes=num_classes, input_channels=1)
    else:  # cifar10
        model = create_model(args.model_name, num_classes=num_classes, input_channels=3)

    # Load model weights
    model.load_state_dict(model_state)
    logger.info(f"Model loaded successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load test dataset
    test_dataset = load_dataset(args.dataset, args.data_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    logger.info(f"Loaded test dataset: {len(test_dataset)} samples")

    # Evaluate model
    logger.info("\nEvaluating model...")
    metrics = evaluate_model(model, test_loader, args.device)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Test Loss: {metrics['test_loss']:.4f}")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.2f}%")
    logger.info(f"Correct Predictions: {metrics['correct_predictions']}/{metrics['total_samples']}")

    logger.info("\nPer-Class Accuracy:")
    for label, acc in sorted(metrics['class_accuracies'].items()):
        logger.info(f"  Class {label}: {acc:.2f}%")

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert metrics to JSON-serializable format
        results = {
            'model_path': str(model_path),
            'dataset': args.dataset,
            'model_name': args.model_name,
            'test_loss': float(metrics['test_loss']),
            'test_accuracy': float(metrics['test_accuracy']),
            'total_samples': int(metrics['total_samples']),
            'correct_predictions': int(metrics['correct_predictions']),
            'class_accuracies': {
                int(k): float(v) for k, v in metrics['class_accuracies'].items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
