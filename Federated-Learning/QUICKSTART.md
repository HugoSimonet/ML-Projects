# Quick Start Guide

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Default Settings

Train on MNIST with default configuration:

```bash
python train_federated.py --dataset mnist --num_clients 10 --num_rounds 50
```

### Using Configuration Files

```bash
# Basic federated averaging
python train_federated.py --config configs/basic_fl.yaml

# With differential privacy
python train_federated.py --config configs/dp_fl.yaml

# Non-IID data distribution
python train_federated.py --config configs/heterogeneous_fl.yaml
```

## Training Examples

### FedAvg on MNIST

```bash
python train_federated.py \
    --dataset mnist \
    --num_clients 10 \
    --num_rounds 50 \
    --local_epochs 5 \
    --algorithm fedavg
```

### Privacy-Preserving Training

```bash
python train_federated.py \
    --dataset mnist \
    --num_clients 10 \
    --num_rounds 50 \
    --algorithm fedavg \
    --privacy
```

### Non-IID Data with FedProx

```bash
python train_federated.py \
    --dataset mnist \
    --num_clients 10 \
    --num_rounds 50 \
    --algorithm fedprox \
    --non_iid
```

### CIFAR-10 Training

```bash
python train_federated.py \
    --dataset cifar10 \
    --num_clients 20 \
    --num_rounds 100 \
    --local_epochs 5 \
    --algorithm fedavg
```

## Evaluation

### Model Evaluation

```bash
python evaluate.py \
    --model_path checkpoints/final_model.pt \
    --dataset mnist \
    --model_name mnist_net \
    --output results/evaluation.json
```

### Visualization

```python
from visualization import FLVisualizer

# Load training history
visualizer = FLVisualizer('checkpoints/training_log.json')

# Generate plots
visualizer.plot_all(output_dir='./plots')

# Individual plots
visualizer.plot_accuracy_curves(save_path='accuracy.png')
visualizer.plot_loss_curves(save_path='loss.png')
visualizer.plot_client_participation(save_path='participation.png')
```

### Privacy Analysis

```python
from privacy_analysis import PrivacyAnalyzer

analyzer = PrivacyAnalyzer('checkpoints/training_log.json')
analyzer.plot_all(output_dir='./privacy_analysis')
analyzer.generate_report(output_path='privacy_report.txt')
```

## Configuration Options

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | None |
| `--dataset` | Dataset (mnist, cifar10) | mnist |
| `--num_clients` | Number of clients | 10 |
| `--num_rounds` | Training rounds | 50 |
| `--local_epochs` | Local epochs per round | 5 |
| `--algorithm` | Algorithm (fedavg, fedprox, fednova) | fedavg |
| `--privacy` | Enable differential privacy | False |
| `--non_iid` | Use non-IID data | False |

### Custom Configuration

Create a YAML file with desired settings:

```yaml
server:
  num_clients: 20
  min_clients_per_round: 5
  max_clients_per_round: 10
  aggregation_algorithm: fedavg
  checkpoint_frequency: 10

client:
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.01
  optimizer: sgd
  momentum: 0.9

data:
  dataset: mnist
  num_clients: 20
  iid: false
  alpha: 0.5

training:
  num_rounds: 100
  evaluation_frequency: 1

model:
  model_name: mnist_net
  num_classes: 10
  input_channels: 1

privacy:
  enable_privacy: false
  epsilon: 1.0
  delta: 1.0e-5
```

Run with:

```bash
python train_federated.py --config my_config.yaml
```

## Output Structure

Training generates the following directory structure:

```
Federated-Learning/
├── checkpoints/
│   ├── final_model.pt          # Final global model
│   ├── latest_model.pt         # Latest checkpoint
│   ├── training_log.json       # Training history
│   └── global_model_round_*.pt # Periodic checkpoints
├── logs/
│   └── federated_training.log  # Training logs
├── plots/                      # Visualization outputs
└── results/                    # Evaluation results
```

## Common Scenarios

### Quick MNIST Test

```bash
# Train for 30 rounds with 10 clients
python train_federated.py --dataset mnist --num_rounds 30 --num_clients 10

# Evaluate
python evaluate.py --model_path checkpoints/final_model.pt --dataset mnist
```

### Privacy-Preserving CIFAR-10

```bash
# Train with differential privacy
python train_federated.py \
    --dataset cifar10 \
    --num_clients 20 \
    --num_rounds 100 \
    --privacy \
    --config configs/dp_fl.yaml

# Analyze privacy budget
python -m privacy_analysis.privacy_analyzer checkpoints/training_log.json
```

### Heterogeneous Data Distribution

```bash
# Train with non-IID data using FedProx
python train_federated.py \
    --dataset mnist \
    --num_clients 15 \
    --num_rounds 50 \
    --algorithm fedprox \
    --non_iid

# Visualize results
python -m visualization.fl_visualizer checkpoints/training_log.json
```

## Troubleshooting

**CUDA out of memory**: Reduce batch size or use CPU with `--device cpu`

**Slow training**: Reduce number of clients or local epochs

**Poor convergence with non-IID data**: Try FedProx algorithm or adjust learning rate

## Next Steps

- Experiment with different algorithms (FedAvg, FedProx, FedNova)
- Tune hyperparameters (learning rate, batch size, local epochs)
- Enable differential privacy and adjust epsilon/delta
- Analyze results using visualization tools
- Implement custom models in `models/simple_models.py`
- Add custom datasets by extending data loading in `train_federated.py`

See README.md for detailed architecture and implementation information.
