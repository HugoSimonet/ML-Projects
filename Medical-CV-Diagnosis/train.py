"""
Main Training Script for Medical CV Diagnosis System
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import MedicalImageProcessor, MedicalAugmentation
from data import MedicalImageDataset, split_medical_dataset, create_medical_dataloader
from models import create_medical_classifier, create_segmentation_model
from training import MedicalTrainer, SegmentationTrainer, FocalLoss, DiceLoss
from evaluation import MedicalMetrics
from utils import Logger, CheckpointManager, get_device, MedicalConfig
from visualization import MedicalVisualizer

import numpy as np
import random


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Medical CV Diagnosis Training')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'segmentation'])
    parser.add_argument('--architecture', type=str, default='resnet50')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup logger
    logger = Logger(name='medical_cv_train')
    logger.info("Starting Medical CV Diagnosis Training")
    logger.info(f"Arguments: {vars(args)}")

    # Get device
    device = get_device(prefer_gpu=(args.device == 'cuda'))

    # Load data
    logger.info(f"Loading data from {args.data_dir}")

    # Create image processor
    processor = MedicalImageProcessor(target_size=(224, 224), normalize_method='min_max')

    # Create augmentation
    train_transform = MedicalAugmentation.get_training_augmentation()
    val_transform = MedicalAugmentation.get_validation_augmentation()

    # Load dataset
    full_dataset = MedicalImageDataset(
        data_dir=args.data_dir,
        transform=None,
        image_processor=processor
    )

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_medical_dataset(
        full_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=args.seed
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create model
    logger.info(f"Creating model: {args.architecture}")

    if args.task == 'classification':
        model = create_medical_classifier(
            architecture=args.architecture,
            num_classes=args.num_classes,
            in_channels=1,
            pretrained=True
        )
        criterion = FocalLoss(gamma=2.0)
    else:
        model = create_segmentation_model(
            architecture='unet',
            in_channels=1,
            num_classes=args.num_classes
        )
        criterion = DiceLoss()

    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Metrics
    metrics_calculator = MedicalMetrics(task=args.task, num_classes=args.num_classes)

    # Trainer
    if args.task == 'classification':
        trainer = MedicalTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            metrics_calculator=metrics_calculator,
            scheduler=scheduler,
            checkpoint_dir='./checkpoints',
            log_dir='./logs',
            mixed_precision=True
        )
    else:
        trainer = SegmentationTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            metrics_calculator=metrics_calculator,
            scheduler=scheduler,
            checkpoint_dir='./checkpoints',
            log_dir='./logs',
            mixed_precision=True
        )

    # Resume if specified
    if args.resume:
        checkpoint_manager = CheckpointManager()
        checkpoint_manager.load(model, optimizer, args.resume, device=str(device))
        logger.info(f"Resumed from {args.resume}")

    # Train
    logger.info("Starting training...")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_best=True,
        early_stopping_patience=15
    )

    logger.info("Training completed!")

    # Final evaluation
    logger.info("Final validation...")
    final_metrics = trainer.validate(val_loader)
    logger.info(f"Final metrics: {final_metrics}")


if __name__ == '__main__':
    main()
