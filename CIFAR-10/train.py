import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from data_loader import CIFAR10DataLoader
from models import get_model, count_parameters

class CIFAR10Trainer:
    """
    Training class for CIFAR-10 models with comprehensive logging and visualization
    """
    
    def __init__(self, model_name='cnn', batch_size=128, learning_rate=0.001, 
                 num_epochs=50, device=None, save_dir='./checkpoints'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize data loader
        self.data_loader = CIFAR10DataLoader(batch_size=batch_size)
        self.train_loader, self.test_loader = self.data_loader.load_data()
        
        # Initialize model
        self.model = get_model(model_name).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(f'runs/cifar10_{model_name}_{int(time.time())}')
        
        print(f"Model: {model_name}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {num_epochs}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def test_epoch(self):
        """Test for one epoch"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{test_loss/(len(pbar)):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print("=" * 60)
        
        best_test_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Test
            test_loss, test_acc = self.test_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Test', test_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Test', test_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.save_model(f'best_model_{self.model_name}.pth')
                print(f"New best model saved! Test accuracy: {test_acc:.2f}%")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}_{self.model_name}.pth')
        
        print(f"\nTraining completed!")
        print(f"Best test accuracy: {best_test_acc:.2f}%")
        
        # Close TensorBoard writer
        self.writer.close()
    
    def save_model(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies,
            'model_name': self.model_name,
            'best_test_acc': max(self.test_accuracies) if self.test_accuracies else 0
        }, filepath)
    
    def load_model(self, filename):
        """Load model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_losses = checkpoint['test_losses']
        self.test_accuracies = checkpoint['test_accuracies']
        
        print(f"Model loaded from {filepath}")
        print(f"Best test accuracy: {checkpoint['best_test_acc']:.2f}%")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.test_losses, label='Test Loss', color='red')
        ax1.set_title('Training and Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.test_accuracies, label='Test Accuracy', color='red')
        ax2.set_title('Training and Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        print("\nEvaluating model...")
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Evaluation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Classification report
        print("\nClassification Report:")
        print("=" * 50)
        print(classification_report(all_targets, all_predictions, 
                                  target_names=self.data_loader.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.data_loader.class_names,
                   yticklabels=self.data_loader.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return all_predictions, all_targets

def main():
    """Main training function"""
    # You can experiment with different models: 'cnn', 'simple', 'resnet'
    trainer = CIFAR10Trainer(
        model_name='cnn',
        batch_size=128,
        learning_rate=0.001,
        num_epochs=50
    )
    
    # Visualize some training samples
    trainer.data_loader.visualize_samples()
    
    # Start training
    trainer.train()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate the model
    trainer.evaluate_model()

if __name__ == "__main__":
    main()
