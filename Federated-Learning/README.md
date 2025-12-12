# Federated Learning System

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A privacy-preserving distributed machine learning system implementing multiple federated learning algorithms with differential privacy support.

## Overview

This system enables training machine learning models across distributed clients without centralizing raw data. It supports heterogeneous data distributions, secure aggregation, and privacy-preserving techniques suitable for research and production environments.

## Features

- Multiple aggregation algorithms (FedAvg, FedProx, FedNova)
- Differential privacy with Rényi DP accounting
- Non-IID data handling (Dirichlet, pathological splitting)
- Client selection strategies (random, round-robin, uniform)
- Comprehensive evaluation metrics (convergence, fairness, privacy)
- YAML-based configuration system
- Checkpoint management and recovery

## Architecture

### Core Components

**Central Server** - Coordinates training rounds, performs model aggregation, manages client selection, and tracks global model state.

**Federated Clients** - Execute local training on private data, compute and transmit model updates, handle dropout/reconnection scenarios.

**Aggregation Layer** - Implements FedAvg (weighted averaging), FedProx (proximal term for heterogeneity), and FedNova (normalized averaging for variable local epochs).

**Privacy Engine** - Applies differential privacy via gradient clipping and noise calibration, tracks cumulative privacy budget using RDP composition.

**Communication Layer** - Manages secure channels, message serialization, compression, and retry logic.

### Data Distribution

- **IID**: Uniform random data distribution across clients
- **Dirichlet Non-IID**: Label distribution controlled by concentration parameter alpha
- **Pathological Non-IID**: Each client receives limited label classes

## Supported Datasets

- MNIST (handwritten digits)
- CIFAR-10 (image classification)

Models include CNNs (MNISTNet, CIFAR10Net), MLPs, and ResNet18 variants.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 1.9+, CUDA 11.0+ (optional)

## Quick Start

### Basic Training

```bash
python train_federated.py --dataset mnist --num_clients 10 --num_rounds 50
```

### With Configuration File

```bash
python train_federated.py --config configs/basic_fl.yaml
```

### Privacy-Preserving Training

```bash
python train_federated.py --config configs/dp_fl.yaml
```

### Non-IID Data

```bash
python train_federated.py --config configs/heterogeneous_fl.yaml
```

## Configuration

See `config.py` for all available options. Key parameters:

```yaml
server:
  num_clients: 10
  aggregation_algorithm: fedavg

client:
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.01

data:
  dataset: mnist
  iid: false
  alpha: 0.5  # Dirichlet concentration

privacy:
  enable_privacy: true
  epsilon: 1.0
  delta: 1.0e-5
```

## Evaluation

```bash
# Evaluate trained model
python evaluate.py --model_path checkpoints/final_model.pt --dataset mnist

# Generate visualizations
python -m visualization.fl_visualizer checkpoints/training_log.json

# Privacy analysis
python -m privacy_analysis.privacy_analyzer checkpoints/training_log.json
```

## Metrics

**Convergence**: Global accuracy, loss, rounds to convergence, training time

**Fairness**: Jain index, Gini coefficient, client accuracy variance

**Privacy**: Epsilon consumed, delta, privacy-utility tradeoff

**Communication**: Total bytes transmitted, compression ratio

## Project Structure

```
Federated-Learning/
├── algorithms/          # FedAvg, FedProx, FedNova implementations
├── client/              # Federated client logic
├── server/              # Central server and aggregation
├── privacy/             # Differential privacy mechanisms
├── communication/       # Network layer (gRPC, HTTP, WebSocket)
├── evaluation/          # Metrics and analysis
├── models/              # Neural network architectures
├── utils/               # Data splitting, helpers
├── configs/             # YAML configuration templates
├── train_federated.py   # Main training script
└── evaluate.py          # Evaluation script
```

## Implementation Notes

The system uses PyTorch for model training and supports GPU acceleration. Privacy is implemented via gradient clipping and Gaussian noise addition with moment accountant for RDP tracking. Secure aggregation uses placeholder implementations (homomorphic encryption and secret sharing require additional dependencies).

Communication currently operates in simulation mode. For distributed deployment, the communication layer supports gRPC, HTTP, and WebSocket protocols but requires additional network configuration.

## Testing

```bash
# Run test suite
pytest tests/

# Quick verification
python train_federated.py --dataset mnist --num_clients 3 --num_rounds 2 --local_epochs 1
```

See `TEST_RESULTS.md` for validation results.

## Results

Experimental results on MNIST dataset with 5 clients (12,000 samples each):

| Round | Train Accuracy | Test Accuracy | Test Loss |
|-------|----------------|---------------|-----------|
| 0 | 82.29% | 97.38% | 0.1692 |
| 1 | 95.54% | 98.70% | 0.0398 |
| 2 | 97.25% | 99.10% | 0.0289 |
| 3 | 97.79% | 99.14% | 0.0268 |
| 4 | 98.13% | 99.33% | 0.0214 |

![Federated Training Curves](results/federated_training_curves.png)

The system achieves 99.33% test accuracy after 5 communication rounds using FedAvg aggregation. All clients show consistent performance (98%+ accuracy) with 100% participation rate.

![Client Performance](results/client_performance.png)

See [RESULTS.md](RESULTS.md) for detailed analysis including:
- Accuracy and loss convergence curves
- Per-client performance metrics
- Client accuracy evolution across rounds
- Privacy-preserving training analysis

## References

- McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- Li et al. "Federated Optimization in Heterogeneous Networks" (FedProx)
- Wang et al. "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization" (FedNova)
- Abadi et al. "Deep Learning with Differential Privacy"
- Mironov "Rényi Differential Privacy"

## License

MIT License - see LICENSE file for details.
