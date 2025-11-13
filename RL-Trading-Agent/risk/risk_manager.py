"""
Risk Management System

Implements position sizing, stop-loss, and portfolio risk controls.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_position_size: float = 0.2  # Max 20% of portfolio in one position
    max_portfolio_risk: float = 0.02  # Max 2% portfolio risk per trade
    max_drawdown: float = 0.15  # Max 15% drawdown before stopping
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.06  # 6% take profit
    max_daily_loss: float = 0.05  # Max 5% daily loss
    max_leverage: float = 1.0  # No leverage by default


class PositionSizer:
    """
    Position sizing calculator using various methods.

    Implements Kelly Criterion, fixed fractional, and volatility-based sizing.
    """

    @staticmethod
    def kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_fraction: float = 0.25
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size
            max_fraction: Maximum position size cap

        Returns:
            Position size as fraction of portfolio
        """
        if avg_loss == 0 or win_rate == 0:
            return 0.0

        # Kelly formula: f* = (p*b - q) / b
        # where p = win_rate, q = 1-p, b = avg_win/avg_loss
        b = avg_win / abs(avg_loss)
        q = 1 - win_rate

        kelly_fraction = (win_rate * b - q) / b

        # Apply safety factor (typically use half Kelly)
        safe_kelly = kelly_fraction * 0.5

        # Cap at max fraction
        return max(0.0, min(safe_kelly, max_fraction))

    @staticmethod
    def fixed_fractional(
        portfolio_value: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Fixed fractional position sizing.

        Args:
            portfolio_value: Current portfolio value
            risk_per_trade: Risk per trade as fraction (e.g., 0.02 = 2%)

        Returns:
            Maximum risk amount in currency
        """
        return portfolio_value * risk_per_trade

    @staticmethod
    def volatility_based(
        portfolio_value: float,
        volatility: float,
        target_volatility: float = 0.15,
        max_fraction: float = 0.3
    ) -> float:
        """
        Volatility-based position sizing.

        Args:
            portfolio_value: Current portfolio value
            volatility: Asset volatility (std of returns)
            target_volatility: Target portfolio volatility
            max_fraction: Maximum position size

        Returns:
            Position size as fraction of portfolio
        """
        if volatility == 0:
            return 0.0

        # Scale position inversely with volatility
        position_fraction = target_volatility / volatility

        return min(position_fraction, max_fraction)


class RiskManager:
    """
    Comprehensive risk management system.

    Manages position sizing, stop-loss, drawdown limits, and risk monitoring.
    """

    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager.

        Args:
            risk_limits: Risk limit configuration
        """
        self.limits = risk_limits or RiskLimits()

        # State tracking
        self.peak_portfolio_value = 0
        self.daily_pnl = 0
        self.day_start_value = 0
        self.trade_history = []

    def check_can_trade(
        self,
        portfolio_value: float,
        current_drawdown: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits.

        Args:
            portfolio_value: Current portfolio value
            current_drawdown: Current drawdown (optional, will calculate if not provided)

        Returns:
            (can_trade, reason)
        """
        # Update peak
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        # Calculate drawdown if not provided
        if current_drawdown is None:
            current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value

        # Check drawdown limit
        if current_drawdown > self.limits.max_drawdown:
            return False, f"Drawdown {current_drawdown:.2%} exceeds limit {self.limits.max_drawdown:.2%}"

        # Check daily loss limit
        daily_loss = (self.day_start_value - portfolio_value) / self.day_start_value if self.day_start_value > 0 else 0
        if daily_loss > self.limits.max_daily_loss:
            return False, f"Daily loss {daily_loss:.2%} exceeds limit {self.limits.max_daily_loss:.2%}"

        return True, "OK"

    def calculate_position_size(
        self,
        portfolio_value: float,
        current_price: float,
        volatility: float,
        signal_strength: float = 1.0,
        method: str = 'fixed_fractional'
    ) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            portfolio_value: Current portfolio value
            current_price: Current asset price
            volatility: Asset volatility
            signal_strength: Trading signal strength (0-1)
            method: Position sizing method ('kelly', 'fixed_fractional', 'volatility')

        Returns:
            Number of shares/units to trade
        """
        if method == 'kelly' and len(self.trade_history) > 10:
            # Calculate Kelly based on trade history
            wins = [t for t in self.trade_history if t['pnl'] > 0]
            losses = [t for t in self.trade_history if t['pnl'] < 0]

            if len(wins) > 0 and len(losses) > 0:
                win_rate = len(wins) / len(self.trade_history)
                avg_win = np.mean([t['pnl'] for t in wins])
                avg_loss = abs(np.mean([t['pnl'] for t in losses]))

                fraction = PositionSizer.kelly_criterion(
                    win_rate, avg_win, avg_loss, self.limits.max_position_size
                )
            else:
                fraction = self.limits.max_portfolio_risk
        elif method == 'volatility':
            fraction = PositionSizer.volatility_based(
                portfolio_value, volatility, max_fraction=self.limits.max_position_size
            )
        else:  # fixed_fractional
            fraction = self.limits.max_portfolio_risk

        # Apply signal strength
        fraction *= signal_strength

        # Calculate shares
        position_value = portfolio_value * fraction
        shares = position_value / current_price

        return shares

    def calculate_stop_loss(self, entry_price: float, is_long: bool = True) -> float:
        """
        Calculate stop-loss price.

        Args:
            entry_price: Entry price
            is_long: True for long positions, False for short

        Returns:
            Stop-loss price
        """
        if is_long:
            return entry_price * (1 - self.limits.stop_loss_pct)
        else:
            return entry_price * (1 + self.limits.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, is_long: bool = True) -> float:
        """
        Calculate take-profit price.

        Args:
            entry_price: Entry price
            is_long: True for long positions, False for short

        Returns:
            Take-profit price
        """
        if is_long:
            return entry_price * (1 + self.limits.take_profit_pct)
        else:
            return entry_price * (1 - self.limits.take_profit_pct)

    def check_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        is_long: bool = True
    ) -> bool:
        """
        Check if stop-loss has been hit.

        Args:
            entry_price: Entry price
            current_price: Current price
            is_long: True for long positions

        Returns:
            True if stop-loss hit
        """
        stop_loss_price = self.calculate_stop_loss(entry_price, is_long)

        if is_long:
            return current_price <= stop_loss_price
        else:
            return current_price >= stop_loss_price

    def check_take_profit(
        self,
        entry_price: float,
        current_price: float,
        is_long: bool = True
    ) -> bool:
        """
        Check if take-profit has been hit.

        Args:
            entry_price: Entry price
            current_price: Current price
            is_long: True for long positions

        Returns:
            True if take-profit hit
        """
        take_profit_price = self.calculate_take_profit(entry_price, is_long)

        if is_long:
            return current_price >= take_profit_price
        else:
            return current_price <= take_profit_price

    def record_trade(self, entry_price: float, exit_price: float, quantity: float, is_long: bool = True):
        """Record trade for statistics."""
        pnl = (exit_price - entry_price) * quantity if is_long else (entry_price - exit_price) * quantity
        self.trade_history.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'is_long': is_long
        })

    def update_daily(self, portfolio_value: float):
        """Update daily tracking (call at start of each day)."""
        self.day_start_value = portfolio_value
        self.daily_pnl = 0

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        current_drawdown = (self.peak_portfolio_value - self.day_start_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0

        metrics = {
            'current_drawdown': current_drawdown,
            'peak_portfolio_value': self.peak_portfolio_value,
            'daily_pnl': self.daily_pnl,
            'total_trades': len(self.trade_history)
        }

        if len(self.trade_history) > 0:
            wins = [t for t in self.trade_history if t['pnl'] > 0]
            metrics['win_rate'] = len(wins) / len(self.trade_history)
            metrics['avg_win'] = np.mean([t['pnl'] for t in wins]) if wins else 0
            metrics['avg_loss'] = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] < 0]) if len(wins) < len(self.trade_history) else 0

        return metrics
