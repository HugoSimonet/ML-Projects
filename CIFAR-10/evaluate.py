import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import seaborn as sns
from data_loader import CIFAR10DataLoader
from models import get_model

class CIFAR10Evaluator:
    """
    Comprehensive evaluation class for CIFAR-10 models
    """
    
    def __init__(self, model_path, model_name='cnn', device=None):
        self.model_name = model_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data
        self.data_loader = CIFAR10DataLoader(batch_size=100)
        self.train_loader, self.test_loader = self.data_loader.load_data()
        
        # Load model
        self.model = get_model(model_name).to(self.device)
        self.load_model(model_path)
        
        print(f"Model loaded from: {model_path}")
        print(f"Using device: {self.device}")
    
    def load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def predict_batch(self, data_loader):
        """Get predictions for entire dataset"""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def evaluate_accuracy(self):
        """Calculate overall accuracy"""
        predictions, targets, _ = self.predict_batch(self.test_loader)
        accuracy = accuracy_score(targets, predictions)
        return accuracy, predictions, targets
    
    def evaluate_per_class_metrics(self):
        """Calculate per-class metrics"""
        predictions, targets, _ = self.predict_batch(self.test_loader)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None
        )
        
        class_names = self.data_loader.class_names
        
        # Create metrics DataFrame
        metrics_data = []
        for i, class_name in enumerate(class_names):
            metrics_data.append({
                'Class': class_name,
                'Precision': precision[i],
                'Recall': recall[i],
                'F1-Score': f1[i],
                'Support': support[i]
            })
        
        return metrics_data, predictions, targets
    
    def plot_confusion_matrix(self, predictions, targets):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.data_loader.class_names,
                   yticklabels=self.data_loader.class_names)
        plt.title('Confusion Matrix - CIFAR-10')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def plot_class_distribution(self, predictions, targets):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True distribution
        true_counts = np.bincount(targets)
        ax1.bar(range(10), true_counts, color='skyblue', alpha=0.7)
        ax1.set_title('True Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(10))
        ax1.set_xticklabels(self.data_loader.class_names, rotation=45)
        
        # Predicted distribution
        pred_counts = np.bincount(predictions)
        ax2.bar(range(10), pred_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(self.data_loader.class_names, rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_misclassifications(self, predictions, targets, probabilities, num_samples=10):
        """Analyze misclassified samples"""
        misclassified_indices = np.where(predictions != targets)[0]
        
        print(f"\nTotal misclassified samples: {len(misclassified_indices)}")
        print(f"Accuracy: {(1 - len(misclassified_indices)/len(targets))*100:.2f}%")
        
        # Get some misclassified samples
        sample_indices = misclassified_indices[:num_samples]
        
        # Get the corresponding images
        all_images = []
        all_true_labels = []
        all_pred_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for idx in sample_indices:
                # Find which batch this index belongs to
                batch_idx = idx // 100
                sample_idx = idx % 100
                
                # Get the batch
                for i, (data, target) in enumerate(self.test_loader):
                    if i == batch_idx:
                        image = data[sample_idx]
                        true_label = target[sample_idx]
                        pred_label = predictions[idx]
                        confidence = probabilities[idx][pred_label]
                        
                        all_images.append(image)
                        all_true_labels.append(true_label.item())
                        all_pred_labels.append(pred_label)
                        all_confidences.append(confidence)
                        break
        
        # Visualize misclassified samples
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i in range(min(len(all_images), 10)):
            # Denormalize image
            img = all_images[i]
            img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + \
                  torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            img_np = img.numpy().transpose(1, 2, 0)
            
            axes[i].imshow(img_np)
            true_class = self.data_loader.class_names[all_true_labels[i]]
            pred_class = self.data_loader.class_names[all_pred_labels[i]]
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {all_confidences[i]:.2f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("=" * 60)
        print("CIFAR-10 MODEL EVALUATION REPORT")
        print("=" * 60)
        
        # Overall accuracy
        accuracy, predictions, targets = self.evaluate_accuracy()
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class metrics
        metrics_data, _, _ = self.evaluate_per_class_metrics()
        
        print(f"\nPer-Class Metrics:")
        print("-" * 50)
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
        print("-" * 50)
        
        for metric in metrics_data:
            print(f"{metric['Class']:<12} {metric['Precision']:<10.4f} {metric['Recall']:<10.4f} "
                  f"{metric['F1-Score']:<10.4f} {metric['Support']:<8}")
        
        # Confusion matrix
        print(f"\nGenerating confusion matrix...")
        cm = self.plot_confusion_matrix(predictions, targets)
        
        # Class distribution
        print(f"\nGenerating class distribution plots...")
        self.plot_class_distribution(predictions, targets)
        
        # Misclassification analysis
        print(f"\nAnalyzing misclassifications...")
        probabilities = self.predict_batch(self.test_loader)[2]
        self.analyze_misclassifications(predictions, targets, probabilities)
        
        return {
            'accuracy': accuracy,
            'metrics': metrics_data,
            'confusion_matrix': cm,
            'predictions': predictions,
            'targets': targets
        }

def main():
    """Main evaluation function"""
    # Example usage - you'll need to provide the path to your trained model
    model_path = "./checkpoints/best_model_cnn.pth"  # Update this path
    
    try:
        evaluator = CIFAR10Evaluator(model_path, model_name='cnn')
        results = evaluator.generate_report()
        
        print(f"\nEvaluation completed successfully!")
        print(f"Final accuracy: {results['accuracy']*100:.2f}%")
        
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please train a model first using train.py")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
