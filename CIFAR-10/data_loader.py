import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class CIFAR10DataLoader:
    """
    Data loader class for CIFAR-10 dataset with preprocessing and augmentation
    """
    
    def __init__(self, batch_size=128, data_dir='./data'):
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Define transforms
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        
    def load_data(self):
        """Load CIFAR-10 training and test datasets"""
        print("Loading CIFAR-10 dataset...")
        
        # Load training dataset
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.transform_train
        )
        
        # Load test dataset
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.transform_test
        )
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        
        return self.train_loader, self.test_loader
    
    def visualize_samples(self, num_samples=8):
        """Visualize sample images from the dataset"""
        # Get a batch of data
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        
        # Create subplot
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # Denormalize image for visualization
            img = images[i]
            img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + \
                  torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            
            # Convert to numpy and transpose for matplotlib
            img_np = img.numpy().transpose(1, 2, 0)
            
            axes[i].imshow(img_np)
            axes[i].set_title(f'{self.class_names[labels[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_distribution(self):
        """Get class distribution in training set"""
        if self.train_dataset is None:
            self.load_data()
        
        class_counts = [0] * 10
        for _, label in self.train_dataset:
            class_counts[label] += 1
        
        return class_counts
