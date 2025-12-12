"""
Data Augmentation Example
Demonstrates how to use GAN-based augmentation for downstream tasks
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import DCGANGenerator
from augmentation import GANAugmenter, ImageAugmenter, ConditionalAugmenter
from utils import get_dataloader


class SimpleClassifier(nn.Module):
    """Simple CNN classifier for demonstration"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_classifier(model, dataloader, criterion, optimizer, device):
    """Train classifier for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate_classifier(model, dataloader, criterion, device):
    """Evaluate classifier"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='GAN Augmentation Example')
    parser.add_argument('--gan_model', type=str, required=True,
                       help='Path to trained GAN checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset to augment')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to dataset')
    parser.add_argument('--augmentation_ratio', type=float, default=0.5,
                       help='Ratio of synthetic to real data')
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Use GAN augmentation')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load dataset
    print('Loading dataset...')
    train_loader = get_dataloader(
        args.dataset,
        data_path=args.data_path,
        image_size=64,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Apply GAN augmentation if specified
    if args.use_augmentation:
        print('Loading GAN for augmentation...')

        # Load generator
        generator = DCGANGenerator(latent_dim=100, output_channels=3, image_size=64)
        checkpoint = torch.load(args.gan_model)
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)

        # Create augmenter
        augmenter = ImageAugmenter(generator, latent_dim=100, device=device)

        # Augment dataset
        print(f'Augmenting dataset with ratio {args.augmentation_ratio}...')
        augmented_dataset = augmenter.augment_dataset(
            train_loader.dataset,
            augmentation_ratio=args.augmentation_ratio
        )

        # Create new dataloader
        train_loader = DataLoader(
            augmented_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

        print(f'Augmented dataset size: {len(augmented_dataset)}')

    # Create classifier
    print('Creating classifier...')
    num_classes = 10  # Adjust based on dataset
    model = SimpleClassifier(num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print('Starting training...')
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')

        # Train
        train_loss, train_acc = train_classifier(
            model, train_loader, criterion, optimizer, device
        )

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

        if train_acc > best_acc:
            best_acc = train_acc

    print(f'\nTraining complete! Best accuracy: {best_acc:.2f}%')

    if args.use_augmentation:
        print(f'With GAN augmentation (ratio={args.augmentation_ratio})')
    else:
        print('Without augmentation (baseline)')


if __name__ == '__main__':
    main()
