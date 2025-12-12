"""
Evaluation Script for Medical CV Diagnosis System
"""

import torch
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import MedicalImageProcessor
from data import MedicalImageDataset
from models import create_medical_classifier, create_segmentation_model
from models.explainability import GradCAM, get_target_layer
from models.uncertainty_quantification import UncertaintyEstimator
from evaluation import MedicalMetrics
from utils import Logger, CheckpointManager, get_device
from visualization import MedicalVisualizer


def evaluate_model(
    model,
    test_loader,
    device,
    task='classification',
    num_classes=2,
    uncertainty=False,
    explainability=False,
    architecture='resnet50'
):
    """
    Evaluate model on test set

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device
        task: Task type
        num_classes: Number of classes
        uncertainty: Enable uncertainty quantification
        explainability: Enable explainability
        architecture: Model architecture
    """
    model.eval()

    # Metrics
    metrics_calculator = MedicalMetrics(task=task, num_classes=num_classes)

    # Uncertainty estimator
    if uncertainty:
        uncertainty_estimator = UncertaintyEstimator(model, method='mc_dropout', num_samples=30)

    # Explainability
    if explainability and task == 'classification':
        target_layer = get_target_layer(model, architecture)
        grad_cam = GradCAM(model, target_layer)

    visualizer = MedicalVisualizer(output_dir='./evaluation_results')

    # Evaluate
    all_uncertainties = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 2:
                images, targets = batch
            else:
                images, targets, _ = batch

            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            if uncertainty:
                predictions, unc, _ = uncertainty_estimator.predict(images, return_uncertainty=True)
                all_uncertainties.extend(unc.cpu().numpy())
            else:
                outputs = model(images)
                predictions = torch.softmax(outputs, dim=1)

            # Update metrics
            pred_classes = predictions.argmax(dim=1)
            metrics_calculator.update(pred_classes, targets, predictions)

            # Visualize first few samples with Grad-CAM
            if explainability and batch_idx < 5:
                for i in range(min(2, images.shape[0])):
                    img = images[i:i+1]
                    cam = grad_cam.generate_cam(img)

                    # Save visualization
                    img_np = img[0, 0].cpu().numpy()
                    visualizer.plot_grad_cam(
                        img_np,
                        cam,
                        save_path=f'./evaluation_results/gradcam_batch{batch_idx}_sample{i}.png'
                    )

    # Compute metrics
    metrics = metrics_calculator.compute()

    return metrics, all_uncertainties


def main():
    parser = argparse.ArgumentParser(description='Medical CV Diagnosis Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/test', help='Test data directory')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'segmentation'])
    parser.add_argument('--architecture', type=str, default='resnet50')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--uncertainty', action='store_true', help='Enable uncertainty quantification')
    parser.add_argument('--explainability', action='store_true', help='Enable explainability')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Setup logger
    logger = Logger(name='medical_cv_eval')
    logger.info("Starting Medical CV Diagnosis Evaluation")
    logger.info(f"Arguments: {vars(args)}")

    # Get device
    device = get_device(prefer_gpu=(args.device == 'cuda'))

    # Load model
    logger.info(f"Loading model: {args.architecture}")

    if args.task == 'classification':
        model = create_medical_classifier(
            architecture=args.architecture,
            num_classes=args.num_classes,
            in_channels=1,
            pretrained=False
        )
    else:
        model = create_segmentation_model(
            architecture='unet',
            in_channels=1,
            num_classes=args.num_classes
        )

    # Load checkpoint
    checkpoint_manager = CheckpointManager()
    checkpoint_manager.load(model, optimizer=None, filename=args.checkpoint, device=str(device))

    model = model.to(device)

    # Load test data
    logger.info(f"Loading test data from {args.data_dir}")

    processor = MedicalImageProcessor(target_size=(224, 224))

    test_dataset = MedicalImageDataset(
        data_dir=args.data_dir,
        image_processor=processor
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Evaluate
    logger.info("Evaluating model...")

    metrics, uncertainties = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        task=args.task,
        num_classes=args.num_classes,
        uncertainty=args.uncertainty,
        explainability=args.explainability,
        architecture=args.architecture
    )

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    for metric_name, value in sorted(metrics.items()):
        logger.info(f"{metric_name:.<40} {value:.4f}")

    if uncertainties:
        mean_unc = np.mean(uncertainties)
        std_unc = np.std(uncertainties)
        logger.info(f"{'Mean Uncertainty':.<40} {mean_unc:.4f} ± {std_unc:.4f}")

    logger.info("=" * 60)
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
