"""
Quick test for A3C agent only to complete the agent comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging
from datetime import datetime

from agents import A3CAgent
from environments import TradingEnvironment
from features import FeatureEngineer
from evaluation import PerformanceAnalyzer
from utils import generate_synthetic_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_agent(agent, env, n_episodes=100, verbose=True):
    """Train A3C agent."""
    logger.info(f"Starting training for {n_episodes} episodes")

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action - now with training parameter
            action, value = agent.select_action(state, training=True)

            # Execute action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # Store and potentially train
            should_train = agent.store_transition(state, action, reward, next_state, done)
            if should_train:
                agent.train_step()

            state = next_state

        if verbose and (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode+1}/{n_episodes}")

    logger.info(f"Training completed: {n_episodes} episodes finished")

def evaluate_agent(agent, env, n_episodes=5):
    """Evaluate A3C agent."""
    portfolio_values_list = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            action, _ = agent.select_action(state, training=False)
            state, _, done, _ = env.step(action)

        portfolio_values_list.append(env.portfolio_values)

    return portfolio_values_list[0]  # Return first episode for analysis

def main():
    logger.info("="*80)
    logger.info("A3C Agent Test")
    logger.info("="*80)

    # Generate same data as original test (same seed)
    logger.info("Generating synthetic market data...")
    data = generate_synthetic_data(
        n_days=1000,
        initial_price=100,
        volatility=0.02,
        trend=0.0005,
        seed=42
    )

    # Prepare data
    feature_engineer = FeatureEngineer()
    data_with_features = feature_engineer.add_features(data)

    # Create environment
    env = TradingEnvironment(
        data=data_with_features,
        initial_balance=100000,
        commission_rate=0.001,
        lookback_window=50
    )

    # Create A3C agent
    agent = A3CAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=0.0003,
        gamma=0.99,
        n_steps=5
    )

    logger.info(f"Environment: State dim={env.state_dim}, Action dim={env.action_dim}")
    logger.info("Testing A3C agent...")

    # Train
    train_agent(agent, env, n_episodes=100, verbose=True)

    # Evaluate
    logger.info("Evaluating A3C agent...")
    portfolio_values = evaluate_agent(agent, env, n_episodes=5)

    # Analyze performance
    analyzer = PerformanceAnalyzer()
    result = analyzer.analyze(np.array(portfolio_values))

    logger.info("\nA3C Results:")
    analyzer.print_summary(result)

    logger.info("\n" + "="*80)
    logger.info("A3C TEST COMPLETE!")
    logger.info("="*80)
    logger.info(f"A3C Sharpe Ratio: {result.sharpe_ratio:.3f}")
    logger.info(f"A3C Total Return: {result.total_return:.2%}")
    logger.info(f"A3C Max Drawdown: {result.max_drawdown:.2%}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
