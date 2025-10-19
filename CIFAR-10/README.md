# CIFAR-10 Image Classification

A comprehensive machine learning project for CIFAR-10 image classification using PyTorch. This project implements multiple CNN architectures and provides extensive training, evaluation, and visualization capabilities.

## Overview

CIFAR-10 is a classic computer vision dataset containing 60,000 32x32 color images in 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

This project provides:
- Multiple CNN architectures (Standard CNN, Simple CNN, ResNet-like)
- Comprehensive data loading and preprocessing
- Advanced training with metrics tracking and visualization
- Detailed model evaluation and analysis
- TensorBoard integration for monitoring training

## Project Structure

```
CIFAR-10/
├── data_loader.py      # Data loading and preprocessing utilities
├── models.py           # CNN model architectures
├── train.py           # Training script with comprehensive logging
├── evaluate.py        # Model evaluation and analysis
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

## Quick Start

### 1. Train a Model

**Basic training with default settings:**
```bash
python train.py
```

**Custom training parameters:**
```python
from train import CIFAR10Trainer

trainer = CIFAR10Trainer(
    model_name='cnn',        # 'cnn', 'simple', or 'resnet'
    batch_size=128,          # Batch size
    learning_rate=0.001,     # Learning rate
    num_epochs=50           # Number of epochs
)

trainer.train()
```

### 2. Evaluate a Trained Model

```bash
python evaluate.py
```

Or programmatically:
```python
from evaluate import CIFAR10Evaluator

evaluator = CIFAR10Evaluator(
    model_path='./checkpoints/best_model_cnn.pth',
    model_name='cnn'
)

results = evaluator.generate_report()
```

### 3. Visualize Training Data

```python
from data_loader import CIFAR10DataLoader

data_loader = CIFAR10DataLoader()
data_loader.load_data()
data_loader.visualize_samples(num_samples=8)
```

## Model Architectures

### 1. Standard CNN (`cnn`)
- **Architecture**: Multi-layer CNN with BatchNorm and Dropout
- **Parameters**: ~1.2M parameters
- **Best for**: Balanced performance and training time
- **Expected Accuracy**: 85-90%

### 2. Simple CNN (`simple`)
- **Architecture**: Lightweight CNN for quick experimentation
- **Parameters**: ~200K parameters
- **Best for**: Fast training and prototyping
- **Expected Accuracy**: 75-80%

### 3. ResNet-like (`resnet`)
- **Architecture**: Residual connections for deeper networks
- **Parameters**: ~1.5M parameters
- **Best for**: Maximum accuracy
- **Expected Accuracy**: 88-92%

## Training Features

### Advanced Training Capabilities
- **Data Augmentation**: Random crops, horizontal flips, normalization
- **Learning Rate Scheduling**: Step decay for better convergence
- **Model Checkpointing**: Automatic saving of best models
- **TensorBoard Logging**: Real-time training monitoring
- **Progress Tracking**: Detailed progress bars and metrics

### Training Monitoring
```bash
# Start TensorBoard to monitor training
tensorboard --logdir=runs

# View training progress at http://localhost:6006
```

### Training Output
The training script provides:
- Real-time loss and accuracy updates
- Epoch-by-epoch progress tracking
- Automatic best model saving
- Training history visualization
- Comprehensive evaluation reports

## Evaluation Features

### Comprehensive Analysis
- **Overall Accuracy**: Test set performance
- **Per-Class Metrics**: Precision, recall, F1-score for each class
- **Confusion Matrix**: Visual representation of classification errors
- **Misclassification Analysis**: Detailed analysis of wrong predictions
- **Class Distribution**: Comparison of true vs predicted distributions

### Visualization Tools
- Training/validation loss and accuracy curves
- Confusion matrix heatmap
- Class distribution comparisons
- Misclassified sample visualization
- Sample image displays

## Usage Examples

### Example 1: Quick Training
```python
from train import CIFAR10Trainer

# Train with default settings
trainer = CIFAR10Trainer()
trainer.train()
```

### Example 2: Custom Model Training
```python
from train import CIFAR10Trainer

# Train ResNet-like model
trainer = CIFAR10Trainer(
    model_name='resnet',
    batch_size=64,
    learning_rate=0.0005,
    num_epochs=100
)

trainer.train()
trainer.plot_training_history()
```

### Example 3: Model Evaluation
```python
from evaluate import CIFAR10Evaluator

# Evaluate trained model
evaluator = CIFAR10Evaluator(
    model_path='./checkpoints/best_model_resnet.pth',
    model_name='resnet'
)

results = evaluator.generate_report()
print(f"Final accuracy: {results['accuracy']*100:.2f}%")
```

### Example 4: Data Exploration
```python
from data_loader import CIFAR10DataLoader

# Load and explore data
data_loader = CIFAR10DataLoader()
train_loader, test_loader = data_loader.load_data()

# Visualize samples
data_loader.visualize_samples(num_samples=16)

# Check class distribution
class_counts = data_loader.get_class_distribution()
print("Class distribution:", class_counts)
```

## Performance Tips

### For Better Accuracy
1. **Use ResNet architecture** for maximum performance
2. **Increase training epochs** (100+ epochs)
3. **Use data augmentation** (already included)
4. **Experiment with learning rates** (0.0001 to 0.01)
5. **Try different optimizers** (Adam, SGD with momentum)

### For Faster Training
1. **Use Simple CNN** for quick experiments
2. **Increase batch size** (if memory allows)
3. **Use GPU acceleration** (automatic if available)
4. **Reduce image resolution** (modify transforms)

### For Better Generalization
1. **Use dropout and batch normalization** (included)
2. **Implement early stopping** (monitor validation loss)
3. **Use learning rate scheduling** (included)
4. **Regularize with weight decay**

## Troubleshooting

### Common Issues

**CUDA out of memory:**
```python
# Reduce batch size
trainer = CIFAR10Trainer(batch_size=32)
```

**Slow training:**
```python
# Use Simple CNN for faster training
trainer = CIFAR10Trainer(model_name='simple')
```

**Poor accuracy:**
```python
# Increase training epochs and use ResNet
trainer = CIFAR10Trainer(
    model_name='resnet',
    num_epochs=100,
    learning_rate=0.0005
)
```

### Performance Expectations

| Model | Parameters | Training Time* | Expected Accuracy |
|-------|------------|----------------|-------------------|
| Simple CNN | ~200K | 10-15 min | 75-80% |
| Standard CNN | ~1.2M | 30-45 min | 85-90% |
| ResNet-like | ~1.5M | 45-60 min | 88-92% |

*Training time on GPU (NVIDIA GTX 1060 or better)

## File Descriptions

- **`data_loader.py`**: Handles CIFAR-10 data loading, preprocessing, and visualization
- **`models.py`**: Contains CNN architectures (Standard, Simple, ResNet-like)
- **`train.py`**: Main training script with comprehensive logging and monitoring
- **`evaluate.py`**: Model evaluation with detailed analysis and visualization
- **`requirements.txt`**: Python package dependencies

## Contributing

Feel free to extend this project by:
- Adding new model architectures
- Implementing additional data augmentation techniques
- Adding support for other datasets
- Improving visualization capabilities
- Adding hyperparameter optimization

## License

This project is open source and available under the MIT License.

## Acknowledgments

- CIFAR-10 dataset creators
- PyTorch team for the excellent framework
- Computer vision research community
