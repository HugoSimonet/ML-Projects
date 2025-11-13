"""
Trading Environment for Reinforcement Learning

Implements a realistic trading environment with market dynamics,
transaction costs, and risk management features.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from enum import Enum
from dataclasses import dataclass


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = 'market'
    LIMIT = 'limit'


class TradeAction(Enum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Trade:
    """Trade record."""
    timestamp: int
    action: str
    price: float
    quantity: float
    value: float
    commission: float
    balance: float
    portfolio_value: float


class TradingEnvironment:
    """
    Gym-compatible trading environment for RL agents.

    Supports realistic market simulation with transaction costs,
    slippage, and position management.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        max_position_size: float = 1.0,
        lookback_window: int = 50,
        reward_scaling: float = 1e-4
    ):
        """
        Initialize trading environment.

        Args:
            data: Market data with OHLCV columns
            initial_balance: Starting cash balance
            commission_rate: Commission as fraction of trade value
            slippage_rate: Slippage as fraction of price
            max_position_size: Maximum position as fraction of balance
            lookback_window: Number of historical steps in state
            reward_scaling: Scaling factor for rewards
        """
        self.data = data
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Number of shares held
        self.entry_price = 0.0
        self.portfolio_value = initial_balance
        self.previous_portfolio_value = initial_balance

        # Trading history
        self.trades: List[Trade] = []
        self.portfolio_values = [initial_balance]
        self.returns = []

        # State and action dimensions
        self.n_features = 10  # OHLCV + indicators
        self.state_dim = self.lookback_window * self.n_features + 3  # + position, balance, value
        self.action_dim = 3  # Hold, Buy, Sell

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance

        self.trades = []
        self.portfolio_values = [self.initial_balance]
        self.returns = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Get current environment state.

        Returns:
            State vector combining price history and portfolio status
        """
        # Price and volume features (normalized)
        window_data = self.data.iloc[self.current_step - self.lookback_window:self.current_step]

        # Extract OHLCV
        features = []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in window_data.columns:
                values = window_data[col].values
                # Normalize
                if col != 'volume':
                    values = values / values[0] - 1.0  # Returns from first price
                else:
                    values = values / (values.mean() + 1e-8)  # Volume relative to mean
                features.append(values)

        # Technical indicators if available
        for col in ['rsi', 'macd', 'bb_width', 'returns', 'volatility']:
            if col in window_data.columns:
                features.append(window_data[col].values)

        # Flatten price features
        price_features = np.concatenate(features) if features else np.zeros(self.lookback_window * 5)

        # Portfolio features (normalized)
        current_price = self.data.iloc[self.current_step]['close']
        position_value = self.position * current_price
        total_value = self.balance + position_value

        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            position_value / self.initial_balance,  # Normalized position value
            self.position / (self.initial_balance / current_price)  # Normalized position size
        ])

        # Combine all features
        state = np.concatenate([price_features, portfolio_features])

        # Pad or truncate to fixed size
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]

        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step.

        Args:
            action: Trading action (0=Hold, 1=Buy, 2=Sell)

        Returns:
            next_state, reward, done, info
        """
        # Store previous portfolio value
        self.previous_portfolio_value = self.portfolio_value

        # Execute action
        current_price = self.data.iloc[self.current_step]['close']
        trade_info = self._execute_action(action, current_price)

        # Move to next step
        self.current_step += 1

        # Calculate portfolio value
        if self.current_step < len(self.data):
            new_price = self.data.iloc[self.current_step]['close']
            position_value = self.position * new_price
            self.portfolio_value = self.balance + position_value
        else:
            self.portfolio_value = self.balance + self.position * current_price

        # Calculate reward
        reward = self._calculate_reward()

        # Check if done
        done = self.current_step >= len(self.data) - 1

        # Record portfolio value
        self.portfolio_values.append(self.portfolio_value)

        # Calculate return
        period_return = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        self.returns.append(period_return)

        # Get next state
        next_state = self._get_state() if not done else np.zeros(self.state_dim)

        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'position_value': self.position * (new_price if not done else current_price),
            'return': period_return,
            'trade': trade_info
        }

        return next_state, reward, done, info

    def _execute_action(self, action: int, price: float) -> Dict:
        """Execute trading action."""
        trade_info = {'action': TradeAction(action).name, 'quantity': 0, 'value': 0, 'commission': 0}

        # Apply slippage to price
        if action == TradeAction.BUY.value:
            execution_price = price * (1 + self.slippage_rate)
        elif action == TradeAction.SELL.value:
            execution_price = price * (1 - self.slippage_rate)
        else:
            return trade_info

        if action == TradeAction.BUY.value and self.position == 0:
            # Buy: use available balance
            max_shares = self.balance / execution_price
            shares_to_buy = max_shares * 0.95  # Use 95% of available balance

            if shares_to_buy > 0:
                cost = shares_to_buy * execution_price
                commission = cost * self.commission_rate
                total_cost = cost + commission

                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.position += shares_to_buy
                    self.entry_price = execution_price

                    trade_info = {
                        'action': 'BUY',
                        'quantity': shares_to_buy,
                        'price': execution_price,
                        'value': cost,
                        'commission': commission
                    }

                    # Record trade
                    self.trades.append(Trade(
                        timestamp=self.current_step,
                        action='BUY',
                        price=execution_price,
                        quantity=shares_to_buy,
                        value=cost,
                        commission=commission,
                        balance=self.balance,
                        portfolio_value=self.portfolio_value
                    ))

        elif action == TradeAction.SELL.value and self.position > 0:
            # Sell: sell all position
            revenue = self.position * execution_price
            commission = revenue * self.commission_rate
            net_revenue = revenue - commission

            self.balance += net_revenue

            trade_info = {
                'action': 'SELL',
                'quantity': self.position,
                'price': execution_price,
                'value': revenue,
                'commission': commission,
                'profit': net_revenue - (self.position * self.entry_price)
            }

            # Record trade
            self.trades.append(Trade(
                timestamp=self.current_step,
                action='SELL',
                price=execution_price,
                quantity=self.position,
                value=revenue,
                commission=commission,
                balance=self.balance,
                portfolio_value=self.portfolio_value
            ))

            self.position = 0
            self.entry_price = 0

        return trade_info

    def _calculate_reward(self) -> float:
        """
        Calculate reward for the current step.

        Uses a combination of:
        - Portfolio value change (returns)
        - Risk-adjusted returns (Sharpe-like)
        - Drawdown penalty
        """
        # Basic return-based reward
        portfolio_return = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value

        # Scale reward
        reward = portfolio_return / self.reward_scaling

        # Penalize drawdown
        if len(self.portfolio_values) > 1:
            peak = max(self.portfolio_values)
            drawdown = (peak - self.portfolio_value) / peak
            reward -= drawdown * 10  # Drawdown penalty

        # Small penalty for holding to encourage action
        if self.position == 0 and len(self.trades) > 0:
            reward -= 0.01

        return reward

    def get_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if len(self.portfolio_values) < 2:
            return {}

        portfolio_values = np.array(self.portfolio_values)
        returns = np.array(self.returns) if self.returns else np.diff(portfolio_values) / portfolio_values[:-1]

        # Total return
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Sharpe ratio (annualized, assuming daily data)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = drawdown.max()

        # Win rate
        profitable_trades = sum(1 for t in self.trades if hasattr(t, 'profit') and t.profit > 0)
        total_trades = len([t for t in self.trades if t.action == 'SELL'])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t.profit for t in self.trades if hasattr(t, 'profit') and t.profit > 0)
        gross_loss = abs(sum(t.profit for t in self.trades if hasattr(t, 'profit') and t.profit < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'final_portfolio_value': portfolio_values[-1]
        }


class MultiAssetTradingEnvironment(TradingEnvironment):
    """
    Trading environment supporting multiple assets.

    Manages portfolio of multiple assets with rebalancing.
    """

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        initial_balance: float = 100000,
        **kwargs
    ):
        """
        Initialize multi-asset environment.

        Args:
            data_dict: Dictionary mapping asset symbols to market data
            initial_balance: Starting cash balance
            **kwargs: Additional parameters for TradingEnvironment
        """
        self.assets = list(data_dict.keys())
        self.data_dict = data_dict
        self.positions = {asset: 0.0 for asset in self.assets}

        # Use first asset's data for base initialization
        super().__init__(data_dict[self.assets[0]], initial_balance, **kwargs)

        # Override action space for multi-asset
        self.action_dim = len(self.assets) * 3  # Buy/Sell/Hold for each asset

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.positions = {asset: 0.0 for asset in self.assets}
        return super().reset()

    # Additional methods for multi-asset management would go here
    # (simplified for brevity)
