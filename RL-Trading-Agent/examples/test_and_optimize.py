"""
Comprehensive Testing and Hyperparameter Optimization for RL Trading Agents

This script tests different RL agents and hyperparameter configurations,
evaluates their trading performance, and identifies optimal settings.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

from agents import DQNAgent, PPOAgent, A3CAgent
from environments import TradingEnvironment
from risk import RiskManager, RiskLimits
from features import FeatureEngineer
from evaluation import PerformanceAnalyzer, PerformanceResult
from utils import generate_synthetic_data, split_data

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("RL Trading Agent - Comprehensive Testing & Optimization")
print("="*80)
logger.info("="*80)
logger.info("RL Trading Agent - Comprehensive Testing & Optimization")
logger.info(f"Log file: {log_file}")
logger.info("="*80)

# Configuration spaces for hyperparameter optimization
AGENT_CONFIGS = {
    "DQN": {
        "learning_rate": [0.00001, 0.0001, 0.0003, 0.001],
        "gamma": [0.95, 0.99, 0.995],
        "epsilon_decay": [0.99, 0.995, 0.999],
        "buffer_capacity": [50000, 100000],
        "batch_size": [32, 64, 128],
        "target_update_freq": [500, 1000, 2000]
    },
    "PPO": {
        "learning_rate": [0.0001, 0.0003, 0.001],
        "gamma": [0.95, 0.99],
        "clip_epsilon": [0.1, 0.2, 0.3],
        "epochs_per_update": [5, 10, 20]
    },
    "A3C": {
        "learning_rate": [0.0001, 0.0003, 0.001],
        "gamma": [0.95, 0.99],
        "n_steps": [3, 5, 10]
    }
}

ENV_CONFIGS = {
    "commission_rate": [0.0001, 0.001, 0.002],
    "reward_scaling": [1e-4, 1e-3],
    "lookback_window": [20, 50, 100]
}


def train_agent(
    agent,
    env: TradingEnvironment,
    n_episodes: int = 100,
    verbose: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Train an RL agent.

    Args:
        agent: RL agent instance
        env: Trading environment
        n_episodes: Number of training episodes
        verbose: Print training progress

    Returns:
        episode_rewards, episode_portfolio_values
    """
    episode_rewards = []
    episode_portfolio_values = []

    logger.info(f"Starting training for {n_episodes} episodes")

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            # Select action
            if hasattr(agent, 'select_action'):
                if isinstance(agent, PPOAgent):
                    action, log_prob, value = agent.select_action(state, training=True)
                    # Store transition for PPO
                    next_state_temp = state.copy()  # Placeholder
                else:
                    action = agent.select_action(state, training=True)
            else:
                action = np.random.randint(env.action_dim)

            # Execute action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # Store experience and train
            if isinstance(agent, DQNAgent):
                agent.store_experience(state, action, reward, next_state, done)
                loss = agent.train_step()
            elif isinstance(agent, PPOAgent):
                agent.store_transition(state, action, reward, done, log_prob, value)
                if done:
                    agent.train_step(0)  # next_value = 0 at episode end
            elif isinstance(agent, A3CAgent):
                should_train = agent.store_transition(state, action, reward, next_state, done)
                if should_train:
                    agent.train_step()

            state = next_state
            step_count += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(env.portfolio_value)

        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_value = np.mean(episode_portfolio_values[-10:])
            msg = f"Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | Avg Portfolio Value: ${avg_value:,.0f}"
            print(msg)
            logger.info(msg)
        elif (episode + 1) % 10 == 0:
            logger.info(f"Completed episode {episode+1}/{n_episodes}")

    logger.info(f"Training completed: {n_episodes} episodes finished")
    return episode_rewards, episode_portfolio_values


def evaluate_agent(
    agent,
    env: TradingEnvironment,
    n_episodes: int = 10
) -> Dict:
    """
    Evaluate trained agent.

    Args:
        agent: Trained agent
        env: Trading environment
        n_episodes: Number of evaluation episodes

    Returns:
        Evaluation metrics
    """
    portfolio_values_list = []
    metrics_list = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            # Select action (no exploration)
            if hasattr(agent, 'select_action'):
                if isinstance(agent, PPOAgent):
                    action, _, _ = agent.select_action(state, training=False)
                else:
                    action = agent.select_action(state, training=False)
            else:
                action = 0  # Default to hold

            state, _, done, _ = env.step(action)

        portfolio_values_list.append(env.portfolio_values)
        metrics_list.append(env.get_metrics())

    # Average metrics across episodes
    avg_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m]
        avg_metrics[key] = np.mean(values) if values else 0

    avg_metrics['portfolio_values'] = portfolio_values_list[0]  # Use first episode for analysis

    return avg_metrics


def run_baseline_test(data: pd.DataFrame) -> Dict:
    """Run baseline test with default hyperparameters."""
    print("\n" + "="*80)
    print("BASELINE TEST - Default Hyperparameters (DQN)")
    print("="*80)
    logger.info("="*80)
    logger.info("PHASE 1: BASELINE TEST - Default Hyperparameters (DQN)")
    logger.info("="*80)

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

    # Create agent
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_decay=0.995
    )

    print(f"\nEnvironment: State dim={state_dim}, Action dim={action_dim}")
    print(f"Agent: DQN with default parameters")
    print(f"Training for 100 episodes...")
    logger.info(f"Environment: State dim={state_dim}, Action dim={action_dim}")
    logger.info("Starting baseline DQN training (100 episodes)...")

    # Train
    episode_rewards, episode_values = train_agent(agent, env, n_episodes=100, verbose=True)

    # Evaluate
    print("\nEvaluating...")
    logger.info("Baseline training complete. Starting evaluation...")
    eval_metrics = evaluate_agent(agent, env, n_episodes=5)

    # Analyze performance
    analyzer = PerformanceAnalyzer()
    portfolio_values = np.array(eval_metrics['portfolio_values'])
    result = analyzer.analyze(portfolio_values)

    print("\nBaseline Results:")
    analyzer.print_summary(result)

    return {
        'agent_type': 'DQN',
        'config': 'baseline',
        'training_rewards': episode_rewards,
        'training_values': episode_values,
        'eval_metrics': eval_metrics,
        'performance': result
    }


def optimize_hyperparameters(
    data: pd.DataFrame,
    agent_type: str = 'DQN',
    n_trials: int = 10
) -> Tuple[Dict, List[Dict]]:
    """
    Optimize hyperparameters for a specific agent type.

    Args:
        data: Market data
        agent_type: Agent type to optimize
        n_trials: Number of hyperparameter trials

    Returns:
        best_config, all_results
    """
    print("\n" + "="*80)
    print(f"HYPERPARAMETER OPTIMIZATION - {agent_type}")
    print("="*80)

    # Prepare data
    feature_engineer = FeatureEngineer()
    data_with_features = feature_engineer.add_features(data)

    configs = AGENT_CONFIGS[agent_type]
    all_results = []
    best_sharpe = -np.inf
    best_config = None

    print(f"\nTesting {n_trials} hyperparameter configurations...")

    for trial in range(n_trials):
        # Sample random configuration
        config = {}
        for param, values in configs.items():
            config[param] = np.random.choice(values)

        print(f"\n[{trial+1}/{n_trials}] Testing config: {config}")

        # Create environment
        env = TradingEnvironment(
            data=data_with_features,
            initial_balance=100000,
            commission_rate=0.001,
            lookback_window=50
        )

        # Create agent with sampled config
        if agent_type == 'DQN':
            agent = DQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                **config
            )
        elif agent_type == 'PPO':
            agent = PPOAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                **config
            )
        elif agent_type == 'A3C':
            agent = A3CAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                **config
            )

        # Train
        train_agent(agent, env, n_episodes=50, verbose=False)

        # Evaluate
        eval_metrics = evaluate_agent(agent, env, n_episodes=3)

        # Analyze
        analyzer = PerformanceAnalyzer()
        portfolio_values = np.array(eval_metrics['portfolio_values'])
        result = analyzer.analyze(portfolio_values)

        print(f"    Sharpe: {result.sharpe_ratio:.3f} | "
              f"Return: {result.total_return:.2%} | "
              f"Max DD: {result.max_drawdown:.2%}")

        # Store results
        trial_result = {
            'config': config,
            'performance': result,
            'eval_metrics': eval_metrics
        }
        all_results.append(trial_result)

        # Update best
        if result.sharpe_ratio > best_sharpe:
            best_sharpe = result.sharpe_ratio
            best_config = config
            print(f"    *** New best configuration! ***")

    print("\n" + "="*80)
    print(f"OPTIMIZATION COMPLETE - {agent_type}")
    print("="*80)
    print(f"\nBest Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")

    best_result = next(r for r in all_results if r['config'] == best_config)
    print(f"\nBest Performance:")
    analyzer = PerformanceAnalyzer()
    analyzer.print_summary(best_result['performance'])

    return best_config, all_results


def compare_agents(data: pd.DataFrame) -> Dict:
    """Compare different agent types."""
    print("\n" + "="*80)
    print("AGENT COMPARISON - DQN vs PPO vs A3C")
    print("="*80)

    # Prepare data
    feature_engineer = FeatureEngineer()
    data_with_features = feature_engineer.add_features(data)

    results = {}

    for agent_type in ['DQN', 'PPO', 'A3C']:
        print(f"\n\nTesting {agent_type}...")

        env = TradingEnvironment(
            data=data_with_features,
            initial_balance=100000,
            commission_rate=0.001,
            lookback_window=50
        )

        if agent_type == 'DQN':
            agent = DQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                learning_rate=0.0001,
                gamma=0.99
            )
        elif agent_type == 'PPO':
            agent = PPOAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                learning_rate=0.0003,
                gamma=0.99
            )
        elif agent_type == 'A3C':
            agent = A3CAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                learning_rate=0.0003,
                gamma=0.99
            )

        # Train
        train_agent(agent, env, n_episodes=100, verbose=True)

        # Evaluate
        eval_metrics = evaluate_agent(agent, env, n_episodes=5)

        # Analyze
        analyzer = PerformanceAnalyzer()
        portfolio_values = np.array(eval_metrics['portfolio_values'])
        result = analyzer.analyze(portfolio_values)

        results[agent_type] = result

        print(f"\n{agent_type} Results:")
        analyzer.print_summary(result)

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
    print(f"\n{'Metric':<20} {'DQN':>12} {'PPO':>12} {'A3C':>12}")
    print("-"*60)

    for metric in metrics:
        dqn_val = getattr(results['DQN'], metric)
        ppo_val = getattr(results['PPO'], metric)
        a3c_val = getattr(results['A3C'], metric)

        if 'return' in metric or 'drawdown' in metric or 'rate' in metric:
            print(f"{metric.replace('_', ' ').title():<20} {dqn_val:>11.2%} {ppo_val:>11.2%} {a3c_val:>11.2%}")
        else:
            print(f"{metric.replace('_', ' ').title():<20} {dqn_val:>12.3f} {ppo_val:>12.3f} {a3c_val:>12.3f}")

    return results


def save_results(results: Dict, filename: str = "optimization_results.json"):
    """Save results to JSON file."""
    # Convert numpy/torch types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, PerformanceResult):
            return {
                'total_return': float(obj.total_return),
                'annual_return': float(obj.annual_return),
                'sharpe_ratio': float(obj.sharpe_ratio),
                'sortino_ratio': float(obj.sortino_ratio),
                'max_drawdown': float(obj.max_drawdown),
                'calmar_ratio': float(obj.calmar_ratio),
                'win_rate': float(obj.win_rate),
                'profit_factor': float(obj.profit_factor),
                'total_trades': int(obj.total_trades)
            }
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj

    results_serializable = convert_types(results)

    output_path = Path(__file__).parent.parent / "results" / filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main execution function."""
    print("\nGenerating synthetic market data...")
    logger.info("Generating synthetic market data...")
    data = generate_synthetic_data(
        n_days=1000,
        initial_price=100,
        volatility=0.02,
        trend=0.0005,
        seed=42
    )

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    logger.info(f"Data generated: {data.shape[0]} days")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Phase 1: Baseline test
    baseline_results = run_baseline_test(data)
    logger.info("PHASE 1 COMPLETE: Baseline test finished")

    # Phase 2: Hyperparameter optimization for DQN
    print("\n" + "="*80)
    print("Starting hyperparameter optimization...")
    print("="*80)
    logger.info("="*80)
    logger.info("PHASE 2: Starting hyperparameter optimization (10 trials)...")
    logger.info("="*80)

    best_dqn_config, dqn_trials = optimize_hyperparameters(data, agent_type='DQN', n_trials=10)
    logger.info("PHASE 2 COMPLETE: Hyperparameter optimization finished")

    # Phase 3: Compare agents
    logger.info("="*80)
    logger.info("PHASE 3: Starting agent comparison (DQN vs PPO vs A3C)...")
    logger.info("="*80)
    comparison_results = compare_agents(data)
    logger.info("PHASE 3 COMPLETE: Agent comparison finished")

    # Save all results
    all_results = {
        'baseline': baseline_results,
        'dqn_optimization': {
            'best_config': best_dqn_config,
            'trials': dqn_trials
        },
        'agent_comparison': comparison_results
    }

    save_results(all_results)

    print("\n" + "="*80)
    print("TESTING AND OPTIMIZATION COMPLETE!")
    print("="*80)
    logger.info("="*80)
    logger.info("ALL PHASES COMPLETE! TESTING AND OPTIMIZATION FINISHED!")
    logger.info("="*80)
    print("\nKey Findings:")
    print(f"  - Baseline DQN Sharpe Ratio: {baseline_results['performance'].sharpe_ratio:.3f}")
    print(f"  - Optimized DQN Sharpe Ratio: {dqn_trials[-1]['performance'].sharpe_ratio:.3f}")
    print(f"  - Best Agent: {max(comparison_results.items(), key=lambda x: x[1].sharpe_ratio)[0]}")
    print("\nResults saved to results/optimization_results.json")
    logger.info(f"Baseline DQN Sharpe Ratio: {baseline_results['performance'].sharpe_ratio:.3f}")
    logger.info(f"Optimized DQN Sharpe Ratio: {dqn_trials[-1]['performance'].sharpe_ratio:.3f}")
    logger.info(f"Best Agent: {max(comparison_results.items(), key=lambda x: x[1].sharpe_ratio)[0]}")
    logger.info("Results saved to results/optimization_results.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
