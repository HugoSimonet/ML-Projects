"""
Influence Maximization Algorithms

Implements influence maximization algorithms for social networks including:
- Greedy algorithm
- Independent Cascade Model
- Linear Threshold Model
"""

import torch
import numpy as np
from typing import List, Set, Dict
from collections import deque


class IndependentCascadeModel:
    """Independent Cascade Model for influence diffusion."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        propagation_prob: float = 0.1
    ):
        """
        Initialize Independent Cascade Model.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            propagation_prob: Probability of influence propagation
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.propagation_prob = propagation_prob

        # Build adjacency list
        self.adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            self.adj_list[src].append(dst)

    def simulate(
        self,
        seed_set: List[int],
        num_simulations: int = 1000
    ) -> float:
        """
        Simulate influence spread from seed set.

        Args:
            seed_set: List of seed node indices
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Average influence spread (number of influenced nodes)
        """
        total_influenced = 0

        for _ in range(num_simulations):
            influenced = self._single_simulation(seed_set)
            total_influenced += len(influenced)

        return total_influenced / num_simulations

    def _single_simulation(self, seed_set: List[int]) -> Set[int]:
        """Run single influence propagation simulation."""
        influenced = set(seed_set)
        active_nodes = deque(seed_set)

        while active_nodes:
            current = active_nodes.popleft()

            # Try to influence neighbors
            for neighbor in self.adj_list[current]:
                if neighbor not in influenced:
                    # Propagation succeeds with probability
                    if np.random.rand() < self.propagation_prob:
                        influenced.add(neighbor)
                        active_nodes.append(neighbor)

        return influenced


class LinearThresholdModel:
    """Linear Threshold Model for influence diffusion."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        threshold: float = 0.5
    ):
        """
        Initialize Linear Threshold Model.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            threshold: Activation threshold
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.threshold = threshold

        # Build adjacency list and compute weights
        self.adj_list = [[] for _ in range(num_nodes)]
        self.in_degree = np.zeros(num_nodes)

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            self.adj_list[src].append(dst)
            self.in_degree[dst] += 1

        # Edge weights (normalized by in-degree)
        self.edge_weights = {}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            weight = 1.0 / max(self.in_degree[dst], 1)
            self.edge_weights[(src, dst)] = weight

    def simulate(
        self,
        seed_set: List[int],
        num_simulations: int = 1000
    ) -> float:
        """
        Simulate influence spread from seed set.

        Args:
            seed_set: List of seed node indices
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Average influence spread
        """
        total_influenced = 0

        for _ in range(num_simulations):
            influenced = self._single_simulation(seed_set)
            total_influenced += len(influenced)

        return total_influenced / num_simulations

    def _single_simulation(self, seed_set: List[int]) -> Set[int]:
        """Run single influence propagation simulation."""
        # Random thresholds for each node
        thresholds = np.random.rand(self.num_nodes) * self.threshold

        influenced = set(seed_set)
        active_nodes = deque(seed_set)
        influence_received = np.zeros(self.num_nodes)

        while active_nodes:
            current = active_nodes.popleft()

            # Propagate influence to neighbors
            for neighbor in self.adj_list[current]:
                if neighbor not in influenced:
                    weight = self.edge_weights.get((current, neighbor), 0)
                    influence_received[neighbor] += weight

                    # Check if threshold exceeded
                    if influence_received[neighbor] >= thresholds[neighbor]:
                        influenced.add(neighbor)
                        active_nodes.append(neighbor)

        return influenced


class GreedyInfluenceMaximization:
    """Greedy algorithm for influence maximization."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        diffusion_model: str = 'independent_cascade',
        model_params: Dict = None
    ):
        """
        Initialize greedy influence maximization.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            diffusion_model: Diffusion model ('independent_cascade', 'linear_threshold')
            model_params: Parameters for diffusion model
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        # Initialize diffusion model
        if model_params is None:
            model_params = {}

        if diffusion_model == 'independent_cascade':
            self.model = IndependentCascadeModel(edge_index, num_nodes, **model_params)
        elif diffusion_model == 'linear_threshold':
            self.model = LinearThresholdModel(edge_index, num_nodes, **model_params)
        else:
            raise ValueError(f"Unknown diffusion model: {diffusion_model}")

    def select_seeds(
        self,
        k: int,
        num_simulations: int = 100
    ) -> List[int]:
        """
        Select k seed nodes using greedy algorithm.

        Args:
            k: Number of seeds to select
            num_simulations: Number of simulations per evaluation

        Returns:
            List of selected seed node indices
        """
        seeds = []

        for _ in range(k):
            best_node = None
            best_marginal_gain = -1

            # Try adding each node
            for node in range(self.num_nodes):
                if node in seeds:
                    continue

                # Compute marginal gain
                current_spread = self.model.simulate(seeds, num_simulations)
                new_spread = self.model.simulate(seeds + [node], num_simulations)
                marginal_gain = new_spread - current_spread

                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_node = node

            if best_node is not None:
                seeds.append(best_node)

        return seeds


class InfluenceSpreadSimulator:
    """Simulator for analyzing influence spread patterns."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ):
        """
        Initialize influence spread simulator.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        # Build adjacency list
        self.adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            self.adj_list[src].append(dst)

    def simulate_cascade(
        self,
        seed_nodes: List[int],
        propagation_probs: torch.Tensor,
        max_steps: int = 100
    ) -> List[Set[int]]:
        """
        Simulate influence cascade with node-specific propagation probabilities.

        Args:
            seed_nodes: Initial seed nodes
            propagation_probs: Propagation probability for each node [num_nodes]
            max_steps: Maximum propagation steps

        Returns:
            List of influenced node sets at each step
        """
        influenced_history = []
        influenced = set(seed_nodes)
        active_nodes = set(seed_nodes)

        for step in range(max_steps):
            if not active_nodes:
                break

            influenced_history.append(influenced.copy())
            new_active = set()

            for node in active_nodes:
                prob = propagation_probs[node].item()

                for neighbor in self.adj_list[node]:
                    if neighbor not in influenced:
                        if np.random.rand() < prob:
                            influenced.add(neighbor)
                            new_active.add(neighbor)

            active_nodes = new_active

        return influenced_history

    def evaluate_seed_quality(
        self,
        seed_set: List[int],
        num_simulations: int = 1000,
        propagation_prob: float = 0.1
    ) -> Dict[str, float]:
        """
        Evaluate quality of seed set.

        Args:
            seed_set: Seed node indices
            num_simulations: Number of simulations
            propagation_prob: Propagation probability

        Returns:
            Dictionary of quality metrics
        """
        model = IndependentCascadeModel(
            self.edge_index,
            self.num_nodes,
            propagation_prob
        )

        # Average influence spread
        avg_spread = model.simulate(seed_set, num_simulations)

        # Coverage (fraction of network influenced)
        coverage = avg_spread / self.num_nodes

        # Influence per seed
        influence_per_seed = avg_spread / len(seed_set) if len(seed_set) > 0 else 0

        return {
            'average_spread': avg_spread,
            'coverage': coverage,
            'influence_per_seed': influence_per_seed,
            'num_seeds': len(seed_set)
        }
