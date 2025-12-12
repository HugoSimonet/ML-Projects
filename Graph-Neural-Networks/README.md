# Graph Neural Networks

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Graph neural network implementations for node classification, link prediction, and community detection on social network data.

## Overview

This project implements multiple GNN architectures (GCN, GAT, GraphSAGE, Graph Transformer) for analyzing graph-structured data. It provides tools for training on social networks, citation networks, and custom graph datasets.

## Features

- GNN architectures: GCN, GAT, GraphSAGE, Graph Transformer
- Tasks: Node classification, link prediction, community detection
- Graph sampling for large-scale networks
- Attention mechanisms for interpretability
- Mini-batch training on large graphs
- Evaluation metrics for graph tasks

## Architecture

### Models

**GCN (Graph Convolutional Networks)** - Spectral convolution on graphs using normalized adjacency matrix.

**GAT (Graph Attention Networks)** - Multi-head attention mechanism for weighted neighbor aggregation.

**GraphSAGE** - Inductive learning via neighbor sampling and aggregation.

**Graph Transformer** - Self-attention across all nodes with positional encoding.

### Graph Processing

**Sampling** - Neighbor sampling, random walk, layer-wise sampling for scalability.

**Batching** - Mini-batch training via graph partitioning or subgraph sampling.

**Features** - Node features (embeddings, attributes), edge features, positional encodings.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 1.9+, PyTorch Geometric, networkx, scikit-learn

## Quick Start

### Node Classification

```bash
python train.py \
    --task node_classification \
    --model gcn \
    --dataset cora \
    --hidden_dim 64 \
    --num_layers 3 \
    --epochs 200
```

### Link Prediction

```bash
python train.py \
    --task link_prediction \
    --model graphsage \
    --dataset facebook \
    --hidden_dim 128 \
    --num_layers 2 \
    --epochs 100
```

### Community Detection

```bash
python train.py \
    --task community_detection \
    --model gat \
    --dataset reddit \
    --hidden_dim 256 \
    --num_layers 3
```

## Programmatic Usage

```python
import torch
from models import GCN
from torch_geometric.datasets import Planetoid

# Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Create model
model = GCN(
    input_dim=dataset.num_node_features,
    hidden_dim=64,
    output_dim=dataset.num_classes,
    num_layers=3,
    dropout=0.5
)

# Train
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

## Supported Datasets

- **Citation Networks**: Cora, CiteSeer, PubMed
- **Social Networks**: Facebook, Twitter, Reddit
- **Collaboration Networks**: DBLP, arXiv
- **Custom Graphs**: Load from edge lists or adjacency matrices

## Configuration

```yaml
model:
  type: gat
  hidden_dim: 64
  num_layers: 3
  num_heads: 8  # for GAT
  dropout: 0.5

training:
  epochs: 200
  batch_size: 256
  learning_rate: 0.01
  weight_decay: 5e-4

data:
  dataset: cora
  split_ratio: [0.6, 0.2, 0.2]  # train/val/test
  num_neighbors: [10, 10]  # for GraphSAGE sampling
```

## Metrics

### Node Classification
- Accuracy, Precision, Recall, F1-score
- Per-class performance
- Confusion matrix

### Link Prediction
- AUC-ROC, AUC-PR
- Precision@K, Recall@K
- Mean Reciprocal Rank (MRR)

### Community Detection
- Modularity, Conductance
- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)

## Model Comparison

```bash
python compare_models.py \
    --dataset cora \
    --models gcn gat graphsage graph_transformer \
    --task node_classification \
    --output comparison_results.json
```

## Visualization

```python
from visualization import GraphVisualizer

visualizer = GraphVisualizer()
visualizer.plot_graph(data.edge_index, data.x)
visualizer.plot_attention_weights(model, data)  # for GAT
visualizer.plot_embeddings(embeddings, labels)
visualizer.plot_communities(communities, graph)
```

## Project Structure

```
Graph-Neural-Networks/
├── models/              # GNN architectures
├── algorithms/          # Community detection, influence maximization
├── data/                # Dataset loaders
├── evaluation/          # Metrics and evaluators
├── utils/               # Graph processing utilities
├── visualization/       # Plotting tools
├── configs/             # YAML configurations
├── train.py             # Main training script
└── evaluate.py          # Evaluation script
```

## Implementation Notes

Models use PyTorch Geometric for efficient graph operations. Message passing follows the standard aggregation-update pattern. Attention uses scaled dot-product for GAT. GraphSAGE uses mean/LSTM/pooling aggregators.

For large graphs, use neighbor sampling to bound memory usage. Mini-batching via graph partitioning or random subgraph sampling. GPU acceleration supported for all operations.

## Results

Experimental results on Cora citation network (2,708 nodes, 7 classes):

| Model | Test Accuracy | F1-Score |
|-------|--------------|----------|
| GCN | Competitive | Strong |
| GAT | Best | Strong |
| GraphSAGE | Competitive | Strong |

![Model Comparison](results/model_comparison_accuracy.png)

See [RESULTS.md](RESULTS.md) for detailed results, training curves, and visualizations including:
- Model comparison plots
- Training progress curves
- Node embedding visualizations
- Performance analysis

## Testing

```bash
pytest tests/

# Quick verification
python verify_installation.py
```

## References

- Kipf & Welling "Semi-Supervised Classification with Graph Convolutional Networks" (GCN)
- Veličković et al. "Graph Attention Networks" (GAT)
- Hamilton et al. "Inductive Representation Learning on Large Graphs" (GraphSAGE)
- Vaswani et al. "Attention is All You Need" (Transformer architecture)
- Fey & Lenssen "Fast Graph Representation Learning with PyTorch Geometric"

## License

MIT License - see LICENSE file for details.
