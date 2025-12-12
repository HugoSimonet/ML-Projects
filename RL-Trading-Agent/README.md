# RL Trading Agent

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Reinforcement learning agent for algorithmic trading using DQN, PPO, and A3C with risk management and backtesting.

## Overview

This project implements RL agents that learn trading strategies through interaction with financial markets. It includes DQN, PPO, A3C algorithms, realistic market simulation, risk management, and backtesting framework for stocks, forex, and cryptocurrency.

## Features

- RL algorithms: DQN, PPO, A3C, A2C
- Market simulation with order book dynamics
- Risk management (position sizing, stop-loss, drawdown control)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Backtesting with transaction costs and slippage
- Multi-asset portfolio management
- Live trading integration (paper and real)

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 1.9+, gym, pandas, numpy, yfinance

## Quick Start

### Train Agent

```bash
# Train DQN agent
python train.py \
    --algorithm dqn \
    --symbol AAPL \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --episodes 1000

# Train PPO agent
python train.py \
    --algorithm ppo \
    --symbol BTC-USD \
    --episodes 500
```

### Backtest Strategy

```bash
python backtest.py \
    --model checkpoints/dqn_agent.pth \
    --symbol AAPL \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

### Live Trading

```bash
python live_trade.py \
    --model checkpoints/ppo_agent.pth \
    --symbol AAPL \
    --mode paper  # or 'real' for live trading
```

## Usage

```python
from agents import DQNAgent
from environments import TradingEnvironment
import gym

# Create environment
env = TradingEnvironment(
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_balance=100000
)

# Create agent
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=256,
    learning_rate=1e-4
)

# Train
for episode in range(1000):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.update()

        state = next_state
        episode_reward += reward

    print(f"Episode {episode}: Reward = {episode_reward:.2f}")
```

## Algorithms

### DQN (Deep Q-Network)

Value-based RL using deep neural network to approximate Q-function. Includes experience replay and target networks.

### PPO (Proximal Policy Optimization)

Policy gradient method with clipped surrogate objective for stable training. Supports continuous action spaces.

### A3C (Asynchronous Advantage Actor-Critic)

Parallel training with multiple workers. Actor-critic architecture with advantage estimation.

## Trading Environment

### State Space

- Price features: Open, high, low, close, volume
- Technical indicators: RSI, MACD, Bollinger Bands, moving averages
- Position info: Current holdings, available cash, unrealized P&L
- Market features: Volatility, momentum, trend indicators

### Action Space

**Discrete**: Hold, Buy, Sell (with different position sizes)
**Continuous**: Position size as continuous value [-1, 1]

### Reward Function

```python
reward = portfolio_return - transaction_costs - risk_penalty
```

Risk penalty includes drawdown, volatility, and position concentration penalties.

## Risk Management

```python
from risk import RiskManager

risk_manager = RiskManager(
    max_position_size=0.2,      # 20% of portfolio per trade
    stop_loss_pct=0.05,         # 5% stop loss
    take_profit_pct=0.10,       # 10% take profit
    max_drawdown=0.20,          # 20% max drawdown
    max_leverage=1.0            # No leverage
)

# Apply risk controls
action = risk_manager.apply_constraints(action, portfolio_state)
```

## Backtesting

```python
from backtesting import Backtest

backtest = Backtest(
    agent=agent,
    data=historical_data,
    initial_capital=100000,
    commission=0.001,           # 0.1% commission
    slippage=0.0005            # 0.05% slippage
)

results = backtest.run()
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Metrics

**Performance**: Total return, annualized return, Sharpe ratio, Sortino ratio
**Risk**: Max drawdown, volatility, value at risk (VaR)
**Trading**: Win rate, profit factor, average trade, number of trades

## Features Engineering

```python
from features import FeatureEngine

features = FeatureEngine(
    indicators=['rsi', 'macd', 'bbands', 'sma', 'ema'],
    lookback_window=20,
    normalize=True
)

state = features.compute(price_data)
```

## Market Simulation

Realistic simulation including:
- Bid-ask spread
- Transaction costs (commission + slippage)
- Market impact (for large orders)
- Order execution delays
- Partial fills

## Configuration

```yaml
agent:
  algorithm: dqn
  hidden_dim: 256
  learning_rate: 0.0001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995

environment:
  symbol: AAPL
  start_date: 2020-01-01
  end_date: 2023-12-31
  initial_balance: 100000
  transaction_cost: 0.001

risk:
  max_position_size: 0.2
  stop_loss: 0.05
  max_drawdown: 0.20

training:
  episodes: 1000
  batch_size: 64
  update_frequency: 4
  target_update_frequency: 1000
```

## Project Structure

```
RL-Trading-Agent/
├── agents/              # RL algorithms (DQN, PPO, A3C)
├── environments/        # Trading environments
├── features/            # Feature engineering
├── risk/                # Risk management
├── backtesting/         # Backtesting framework
├── data/                # Data loading and processing
├── utils/               # Utilities
├── configs/             # Configuration files
├── train.py             # Training script
└── backtest.py          # Backtesting script
```

## Implementation Notes

Uses PyTorch for neural networks. Gym for environment interface. yfinance for historical data. Handles multiple asset classes with unified interface.

DQN uses prioritized experience replay for better sample efficiency. PPO uses GAE for advantage estimation. A3C runs multiple workers in parallel.

Risk management applied before executing trades. Position sizing based on Kelly criterion or fixed fraction. Stop-loss and take-profit executed at market.

## Testing

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Backtest validation
python validate_backtest.py
```

## References

- Mnih et al. "Playing Atari with Deep Reinforcement Learning" (DQN)
- Schulman et al. "Proximal Policy Optimization Algorithms"
- Mnih et al. "Asynchronous Methods for Deep Reinforcement Learning" (A3C)
- Moody & Saffell "Learning to Trade via Direct Reinforcement"

## License

MIT License - see LICENSE file for details.
