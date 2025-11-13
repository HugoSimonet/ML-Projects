"""
Reinforcement Learning Trading Agents

Implements DQN, PPO, and A3C agents for algorithmic trading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict
import random

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network Architecture for trading.

    Separates value and advantage streams for better Q-value estimation.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, state):
        features = self.feature_net(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling aggregation: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.

    Samples experiences based on TD-error priority for more efficient learning.
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(self, experience: Experience, priority: float = None):
        if priority is None:
            priority = max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        experiences = [self.buffer[idx] for idx in indices]
        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero priority

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent with Double DQN and Dueling architecture.

    Uses experience replay and target networks for stable training.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Networks
        self.policy_net = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Experience replay
        self.memory = PrioritizedReplayBuffer(capacity=buffer_capacity)

        # Training stats
        self.steps = 0
        self.losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.add(experience)

    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from memory
        experiences, indices, weights = self.memory.sample(self.batch_size)

        # Prepare batch tensors (convert to numpy first for speed)
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Double DQN: use policy network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculate TD errors for priority update
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Compute loss with importance sampling weights
        loss = (weights_tensor * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.losses.append(loss.item())
        return loss.item()

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for PPO.

    Outputs both policy (actor) and value function (critic).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()

        # Shared feature extraction
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Actor (policy) head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (value) head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        features = self.shared_net(state)
        policy = self.actor_head(features)
        value = self.critic_head(features)
        return policy, value


class PPOAgent:
    """
    Proximal Policy Optimization Agent.

    Uses clipped surrogate objective for stable policy updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs_per_update: int = 10,
        batch_size: int = 64,
        device: str = 'cpu'
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Network
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Trajectory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action from policy and return action, log_prob, and value."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, value = self.network(state_tensor)

        if training:
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        else:
            action = policy.argmax()
            log_prob = torch.log(policy[0, action])

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, done, log_prob, value):
        """Store transition for PPO update."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return np.array(advantages), np.array(returns)

    def train_step(self, next_value: float) -> Dict[str, float]:
        """Perform PPO update."""
        if len(self.states) == 0:
            return {}

        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # PPO update for multiple epochs
        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(self.epochs_per_update):
            # Get current policy and values
            policies, values = self.network(states)

            # Calculate new log probs
            action_dist = torch.distributions.Categorical(policies)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()

            # Calculate ratio and clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = F.mse_loss(values.squeeze(), returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        # Clear trajectory buffer
        self.clear_trajectory()

        return {
            'actor_loss': total_actor_loss / self.epochs_per_update,
            'critic_loss': total_critic_loss / self.epochs_per_update
        }

    def clear_trajectory(self):
        """Clear trajectory buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class A3CAgent:
    """
    Asynchronous Advantage Actor-Critic Agent (simplified single-thread version).

    Uses n-step returns and advantage estimation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        n_steps: int = 5,
        device: str = 'cpu'
    ):
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = torch.device(device)

        # Network (shared for simplicity)
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # N-step buffer
        self.trajectory = []

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, value = self.network(state_tensor)

        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), value.item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in n-step buffer."""
        self.trajectory.append((state, action, reward, next_state, done))

        # Train when trajectory reaches n_steps or episode ends
        if len(self.trajectory) >= self.n_steps or done:
            return True
        return False

    def train_step(self) -> Dict[str, float]:
        """Perform A3C update using n-step returns."""
        if len(self.trajectory) == 0:
            return {}

        states, actions, rewards, next_states, dones = zip(*self.trajectory)

        # Compute n-step return
        R = 0
        if not dones[-1]:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_states[-1]).unsqueeze(0).to(self.device)
                _, R = self.network(next_state_tensor)
                R = R.item()

        # Calculate returns
        returns = []
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)

        # Prepare tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Forward pass
        policies, values = self.network(states_tensor)
        values = values.squeeze()

        # Calculate advantages
        advantages = returns_tensor - values.detach()

        # Actor loss
        action_dist = torch.distributions.Categorical(policies)
        log_probs = action_dist.log_prob(actions_tensor)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss
        critic_loss = F.mse_loss(values, returns_tensor)

        # Entropy bonus for exploration
        entropy = action_dist.entropy().mean()

        # Total loss
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        # Clear trajectory
        self.trajectory = []

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Alias for main trading agent (using DQN as default)
TradingAgent = DQNAgent
