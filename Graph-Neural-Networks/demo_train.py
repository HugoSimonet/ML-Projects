"""
Simple Demo Training Script for Graph Neural Networks

This script demonstrates basic training without requiring all dependencies.
Perfect for quick testing and learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import time


# Simple GCN implementation for demo
class SimpleGCN(nn.Module):
    """Simplified GCN for demonstration."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass with adjacency matrix."""
        # First layer
        x = torch.mm(adj, x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second layer
        x = torch.mm(adj, x)
        x = self.conv2(x)

        return x


def create_simple_graph() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a simple graph dataset for demonstration.

    Returns:
        features, adjacency matrix, labels, masks
    """
    # Create a small graph (Karate Club-like)
    num_nodes = 34
    num_features = 16
    num_classes = 4

    # Random node features
    features = torch.randn(num_nodes, num_features)

    # Create adjacency matrix (simple community structure)
    adj = torch.zeros(num_nodes, num_nodes)

    # Community 1 (nodes 0-9)
    for i in range(10):
        for j in range(10):
            if i != j and torch.rand(1) > 0.6:
                adj[i, j] = 1

    # Community 2 (nodes 10-19)
    for i in range(10, 20):
        for j in range(10, 20):
            if i != j and torch.rand(1) > 0.6:
                adj[i, j] = 1

    # Community 3 (nodes 20-26)
    for i in range(20, 27):
        for j in range(20, 27):
            if i != j and torch.rand(1) > 0.6:
                adj[i, j] = 1

    # Community 4 (nodes 27-33)
    for i in range(27, 34):
        for j in range(27, 34):
            if i != j and torch.rand(1) > 0.6:
                adj[i, j] = 1

    # Add some inter-community edges
    adj[9, 10] = adj[10, 9] = 1
    adj[19, 20] = adj[20, 19] = 1
    adj[26, 27] = adj[27, 26] = 1

    # Normalize adjacency matrix (add self-loops and normalize)
    adj = adj + torch.eye(num_nodes)
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    adj = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)

    # Create labels based on communities
    labels = torch.zeros(num_nodes, dtype=torch.long)
    labels[0:10] = 0
    labels[10:20] = 1
    labels[20:27] = 2
    labels[27:34] = 3

    # Create train/val/test masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Split data
    train_mask[0:20] = True
    val_mask[20:27] = True
    test_mask[27:34] = True

    return features, adj, labels, (train_mask, val_mask, test_mask)


def train_epoch(model: nn.Module, features: torch.Tensor, adj: torch.Tensor,
                labels: torch.Tensor, train_mask: torch.Tensor,
                optimizer: torch.optim.Optimizer) -> float:
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(features, adj)

    # Compute loss only on training nodes
    loss = F.cross_entropy(output[train_mask], labels[train_mask])

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model: nn.Module, features: torch.Tensor, adj: torch.Tensor,
             labels: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    """Evaluate the model."""
    model.eval()

    with torch.no_grad():
        output = model(features, adj)
        predictions = output[mask].argmax(dim=1)

        # Compute accuracy
        correct = (predictions == labels[mask]).sum().item()
        accuracy = correct / mask.sum().item()

        # Compute loss
        loss = F.cross_entropy(output[mask], labels[mask]).item()

    return accuracy, loss


def main():
    """Main training function."""
    print("=" * 60)
    print("Graph Neural Network - Demo Training")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create dataset
    print("\n[1/5] Creating synthetic graph dataset...")
    features, adj, labels, (train_mask, val_mask, test_mask) = create_simple_graph()
    print(f"  [OK] Nodes: {features.size(0)}")
    print(f"  [OK] Features: {features.size(1)}")
    print(f"  [OK] Classes: {labels.max().item() + 1}")
    print(f"  [OK] Edges: {(adj > 0).sum().item() // 2}")

    # Create model
    print("\n[2/5] Creating GCN model...")
    model = SimpleGCN(
        input_dim=features.size(1),
        hidden_dim=32,
        output_dim=labels.max().item() + 1
    )
    print(f"  [OK] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training
    print("\n[3/5] Training model...")
    epochs = 100
    best_val_acc = 0.0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, features, adj, labels, train_mask, optimizer)

        # Evaluate
        train_acc, _ = evaluate(model, features, adj, labels, train_mask)
        val_acc, val_loss = evaluate(model, features, adj, labels, val_mask)

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Print progress
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, "
                  f"Val Acc={val_acc:.4f}")

    training_time = time.time() - start_time

    print(f"\n  [OK] Training completed in {training_time:.2f} seconds")
    print(f"  [OK] Best validation accuracy: {best_val_acc:.4f}")

    # Final evaluation on test set
    print("\n[4/5] Evaluating on test set...")
    test_acc, test_loss = evaluate(model, features, adj, labels, test_mask)
    print(f"  [OK] Test Accuracy: {test_acc:.4f}")
    print(f"  [OK] Test Loss: {test_loss:.4f}")

    # Summary
    print("\n[5/5] Training Summary")
    print("=" * 60)
    print(f"  Best Validation Accuracy:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Test Accuracy:             {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Training Time:             {training_time:.2f} seconds")
    print(f"  Epochs:                    {epochs}")
    print("=" * 60)
    print("\n[SUCCESS] Demo training completed successfully!")
    print("\nNext steps:")
    print("  1. Try different hyperparameters (learning rate, hidden dim)")
    print("  2. Experiment with more layers")
    print("  3. Train on real datasets (Cora, CiteSeer)")
    print("  4. Compare different GNN architectures (GAT, GraphSAGE)")
    print("\n")


if __name__ == '__main__':
    main()
