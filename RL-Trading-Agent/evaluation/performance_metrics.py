"""
Trading Performance Metrics

Comprehensive performance evaluation for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PerformanceResult:
    """Performance metrics result."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    var_95: float
    cvar_95: float


class TradingMetrics:
    """
    Trading performance metrics calculator.

    Implements standard financial performance metrics.
    """

    @staticmethod
    def calculate_returns(portfolio_values: np.ndarray) -> np.ndarray:
        """Calculate returns from portfolio values."""
        return np.diff(portfolio_values) / portfolio_values[:-1]

    @staticmethod
    def total_return(portfolio_values: np.ndarray) -> float:
        """Calculate total return."""
        return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    @staticmethod
    def annualized_return(portfolio_values: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return.

        Args:
            portfolio_values: Array of portfolio values
            periods_per_year: Trading periods per year (252 for daily, 12 for monthly)

        Returns:
            Annualized return
        """
        total_return = TradingMetrics.total_return(portfolio_values)
        n_periods = len(portfolio_values)
        years = n_periods / periods_per_year
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe Ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Annualized Sharpe Ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation).

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Annualized Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        downside_std = downside_returns.std()
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std

    @staticmethod
    def maximum_drawdown(portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Returns:
            Maximum drawdown as a fraction (positive value)
        """
        if len(portfolio_values) == 0:
            return 0.0

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return drawdown.max()

    @staticmethod
    def calmar_ratio(
        portfolio_values: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio (annual return / max drawdown).

        Args:
            portfolio_values: Array of portfolio values
            periods_per_year: Trading periods per year

        Returns:
            Calmar Ratio
        """
        annual_return = TradingMetrics.annualized_return(portfolio_values, periods_per_year)
        max_dd = TradingMetrics.maximum_drawdown(portfolio_values)

        return annual_return / max_dd if max_dd > 0 else 0.0

    @staticmethod
    def win_rate(trades: List[Dict]) -> float:
        """
        Calculate win rate from trades.

        Args:
            trades: List of trade dictionaries with 'profit' or 'pnl' key

        Returns:
            Win rate (0-1)
        """
        if len(trades) == 0:
            return 0.0

        profitable = sum(1 for t in trades if t.get('profit', t.get('pnl', 0)) > 0)
        return profitable / len(trades)

    @staticmethod
    def profit_factor(trades: List[Dict]) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            trades: List of trade dictionaries with 'profit' or 'pnl' key

        Returns:
            Profit factor
        """
        if len(trades) == 0:
            return 0.0

        profits = [t.get('profit', t.get('pnl', 0)) for t in trades]
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))

        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Array of returns
            confidence: Confidence level (0.95 = 95%)

        Returns:
            VaR at confidence level (positive value = loss)
        """
        if len(returns) == 0:
            return 0.0

        return -np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: Array of returns
            confidence: Confidence level

        Returns:
            CVaR at confidence level
        """
        if len(returns) == 0:
            return 0.0

        var = TradingMetrics.value_at_risk(returns, confidence)
        return -returns[returns <= -var].mean() if len(returns[returns <= -var]) > 0 else 0.0


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer.

    Analyzes trading performance and generates detailed metrics.
    """

    def __init__(self, periods_per_year: int = 252, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.

        Args:
            periods_per_year: Trading periods per year
            risk_free_rate: Annual risk-free rate
        """
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    def analyze(
        self,
        portfolio_values: np.ndarray,
        trades: Optional[List[Dict]] = None
    ) -> PerformanceResult:
        """
        Perform comprehensive performance analysis.

        Args:
            portfolio_values: Array of portfolio values over time
            trades: Optional list of individual trades

        Returns:
            PerformanceResult with all metrics
        """
        if len(portfolio_values) < 2:
            return self._empty_result()

        # Calculate returns
        returns = TradingMetrics.calculate_returns(portfolio_values)

        # Calculate metrics
        total_return = TradingMetrics.total_return(portfolio_values)
        annual_return = TradingMetrics.annualized_return(portfolio_values, self.periods_per_year)
        sharpe = TradingMetrics.sharpe_ratio(returns, self.risk_free_rate, self.periods_per_year)
        sortino = TradingMetrics.sortino_ratio(returns, self.risk_free_rate, self.periods_per_year)
        max_dd = TradingMetrics.maximum_drawdown(portfolio_values)
        calmar = TradingMetrics.calmar_ratio(portfolio_values, self.periods_per_year)

        # Trade-based metrics
        if trades and len(trades) > 0:
            win_rate = TradingMetrics.win_rate(trades)
            profit_factor = TradingMetrics.profit_factor(trades)
            total_trades = len(trades)
            avg_trade_return = np.mean([t.get('profit', t.get('pnl', 0)) for t in trades])
        else:
            win_rate = 0.0
            profit_factor = 0.0
            total_trades = 0
            avg_trade_return = 0.0

        # Risk metrics
        volatility = returns.std() * np.sqrt(self.periods_per_year)
        var_95 = TradingMetrics.value_at_risk(returns, 0.95)
        cvar_95 = TradingMetrics.conditional_var(returns, 0.95)

        return PerformanceResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95
        )

    def _empty_result(self) -> PerformanceResult:
        """Return empty result for invalid data."""
        return PerformanceResult(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_trade_return=0.0,
            volatility=0.0,
            var_95=0.0,
            cvar_95=0.0
        )

    def print_summary(self, result: PerformanceResult):
        """Print formatted performance summary."""
        print("="*70)
        print("TRADING PERFORMANCE SUMMARY")
        print("="*70)
        print(f"\nReturns:")
        print(f"  Total Return:        {result.total_return:>10.2%}")
        print(f"  Annual Return:       {result.annual_return:>10.2%}")
        print(f"\nRisk-Adjusted Returns:")
        print(f"  Sharpe Ratio:        {result.sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:       {result.sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:        {result.calmar_ratio:>10.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Volatility (Annual): {result.volatility:>10.2%}")
        print(f"  Maximum Drawdown:    {result.max_drawdown:>10.2%}")
        print(f"  VaR (95%):          {result.var_95:>10.2%}")
        print(f"  CVaR (95%):         {result.cvar_95:>10.2%}")
        print(f"\nTrading Statistics:")
        print(f"  Total Trades:        {result.total_trades:>10}")
        print(f"  Win Rate:            {result.win_rate:>10.2%}")
        print(f"  Profit Factor:       {result.profit_factor:>10.2f}")
        print(f"  Avg Trade Return:    {result.avg_trade_return:>10.2f}")
        print("="*70)
