# Graph Neural Networks - Results

## Dataset
- **Dataset**: Cora citation network
- **Nodes**: 2,708 papers
- **Edges**: 5,429 citations
- **Features**: 1,433-dimensional bag-of-words
- **Classes**: 7 categories

## Model Comparison

### Accuracy Comparison
![Model Comparison - Accuracy](results/model_comparison_accuracy.png)

The comparison shows different GNN architectures (GCN, GAT, GraphSAGE) on node classification.

### F1-Score Comparison
![Model Comparison - F1](results/model_comparison_f1.png)

Performance metrics across different models demonstrating the effectiveness of attention mechanisms.

## Training Results

### GCN Training Progress
![GCN Training](results/node_classification_gcn_training.png)

Training and validation accuracy over epochs showing convergence and generalization.

### Node Embeddings Visualization
![Node Embeddings](results/Figure_1.png)

t-SNE visualization of learned node embeddings showing clear cluster separation by class.

### Additional Visualizations
![Visualization 2](results/Figure_2.png)

![Visualization 3](results/Figure_3.png)

## Key Findings

- **Best Model**: GAT achieved highest accuracy through multi-head attention
- **Convergence**: Models converge within 50-100 epochs
- **Generalization**: Low gap between train and validation accuracy
- **Embeddings**: Learned representations show meaningful clustering by topic
