"""
Graph Visualizer

Provides tools for visualizing graph structures and analysis results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Optional, Tuple
import seaborn as sns


class GraphVisualizer:
    """Visualizer for graph structures and community detection."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100
    ):
        """
        Initialize graph visualizer.

        Args:
            figsize: Figure size
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style('whitegrid')

    def plot_graph(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_labels: Optional[torch.Tensor] = None,
        node_colors: Optional[List[int]] = None,
        layout: str = 'spring',
        save_path: Optional[str] = None
    ):
        """
        Plot graph structure.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            node_labels: Node labels for coloring
            node_colors: Custom node colors
            layout: Graph layout algorithm
            save_path: Path to save figure
        """
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            G.add_edge(src, dst)

        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Prepare colors
        if node_colors is None and node_labels is not None:
            node_colors = node_labels.cpu().numpy()

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            cmap='Set3',
            node_size=50,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            alpha=0.2,
            ax=ax
        )

        ax.set_title('Graph Structure', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_communities(
        self,
        edge_index: torch.Tensor,
        communities: Dict[int, int],
        num_nodes: int,
        layout: str = 'spring',
        save_path: Optional[str] = None
    ):
        """
        Plot graph with community structure.

        Args:
            edge_index: Edge indices [2, num_edges]
            communities: Community assignments
            num_nodes: Number of nodes
            layout: Graph layout algorithm
            save_path: Path to save figure
        """
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            G.add_edge(src, dst)

        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        else:
            pos = getattr(nx, f'{layout}_layout')(G)

        # Prepare colors
        node_colors = [communities.get(node, 0) for node in range(num_nodes)]
        num_communities = len(set(communities.values()))

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            cmap='Set3',
            node_size=100,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            alpha=0.2,
            width=0.5,
            ax=ax
        )

        ax.set_title(
            f'Community Structure ({num_communities} communities)',
            fontsize=14,
            fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_influence_map(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        seed_nodes: List[int],
        influenced_nodes: Optional[List[int]] = None,
        layout: str = 'spring',
        save_path: Optional[str] = None
    ):
        """
        Plot influence propagation map.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            seed_nodes: Seed node indices
            influenced_nodes: Influenced node indices
            layout: Graph layout algorithm
            save_path: Path to save figure
        """
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            G.add_edge(src, dst)

        # Compute layout
        pos = getattr(nx, f'{layout}_layout')(G)

        # Prepare node colors
        node_colors = ['lightgray'] * num_nodes
        for node in seed_nodes:
            node_colors[node] = 'red'

        if influenced_nodes:
            for node in influenced_nodes:
                if node not in seed_nodes:
                    node_colors[node] = 'lightblue'

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=100,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            alpha=0.1,
            ax=ax
        )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Seed Nodes'),
            Patch(facecolor='lightblue', label='Influenced Nodes'),
            Patch(facecolor='lightgray', label='Uninfluenced Nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title('Influence Propagation Map', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_degree_distribution(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        save_path: Optional[str] = None
    ):
        """
        Plot degree distribution of graph.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            save_path: Path to save figure
        """
        # Compute degree
        degree = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            degree[src] += 1

        degree_np = degree.cpu().numpy()

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Histogram
        ax1.hist(degree_np, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Degree', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Degree Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Log-log plot
        degree_counts = np.bincount(degree_np)
        degrees = np.arange(len(degree_counts))
        mask = degree_counts > 0

        ax2.loglog(degrees[mask], degree_counts[mask], 'o', color='coral', alpha=0.6)
        ax2.set_xlabel('Degree (log)', fontsize=12)
        ax2.set_ylabel('Count (log)', fontsize=12)
        ax2.set_title('Degree Distribution (log-log)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()


def plot_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    **kwargs
):
    """
    Convenience function for plotting graph.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        **kwargs: Additional arguments for GraphVisualizer.plot_graph
    """
    visualizer = GraphVisualizer()
    visualizer.plot_graph(edge_index, num_nodes, **kwargs)


def plot_communities(
    edge_index: torch.Tensor,
    communities: Dict[int, int],
    num_nodes: int,
    **kwargs
):
    """
    Convenience function for plotting communities.

    Args:
        edge_index: Edge indices [2, num_edges]
        communities: Community assignments
        num_nodes: Number of nodes
        **kwargs: Additional arguments for GraphVisualizer.plot_communities
    """
    visualizer = GraphVisualizer()
    visualizer.plot_communities(edge_index, communities, num_nodes, **kwargs)
