import pandas as pd
import numpy as np
from typing import List, Dict, Any
from bot_core.position_manager import Position
from bot_core.logger import get_logger

logger = get_logger(__name__)

class PerformanceAnalyzer:
    @staticmethod
    def generate_report(closed_positions: List[Position], equity_curve: List[Dict[str, Any]], initial_capital: float) -> Dict[str, Any]:
        """
        Generates a comprehensive performance report.
        
        Args:
            closed_positions: List of closed Position objects.
            equity_curve: List of dicts {'timestamp': datetime, 'equity': float}.
            initial_capital: Starting capital.
            
        Returns:
            Dictionary containing performance metrics.
        """
        if not equity_curve:
            return {"error": "No equity curve data available."}

        # Convert equity curve to DataFrame
        df_equity = pd.DataFrame(equity_curve)
        df_equity.set_index('timestamp', inplace=True)
        df_equity.sort_index(inplace=True)
        
        # 1. Basic Return Metrics
        final_equity = df_equity['equity'].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        total_return_pct = total_return * 100
        
        # 2. Drawdown Analysis
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak']
        max_drawdown = df_equity['drawdown'].min()
        max_drawdown_pct = max_drawdown * 100
        
        # 3. Risk-Adjusted Metrics (Sharpe & Sortino)
        # Calculate returns per period (assuming the curve is sampled regularly, e.g., per candle)
        df_equity['returns'] = df_equity['equity'].pct_change().fillna(0)
        
        mean_return = df_equity['returns'].mean()
        std_return = df_equity['returns'].std()
        
        # Annualization factor (approximate based on sample frequency)
        # We'll assume the backtest interval is consistent. 
        # If we have N samples over T days, we can estimate.
        if len(df_equity) > 1:
            duration = df_equity.index[-1] - df_equity.index[0]
            days = max(1, duration.total_seconds() / 86400)
            samples_per_year = len(df_equity) / (days / 365.25)
        else:
            samples_per_year = 252 * 24 * 12 # Fallback

        risk_free_rate = 0.0 # Simplified
        
        if std_return > 0:
            sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(samples_per_year)
        else:
            sharpe_ratio = 0.0
            
        # Sortino: Downside deviation only
        downside_returns = df_equity['returns'][df_equity['returns'] < 0]
        downside_std = downside_returns.std()
        
        if downside_std > 0:
            sortino_ratio = (mean_return - risk_free_rate) / downside_std * np.sqrt(samples_per_year)
        else:
            sortino_ratio = 0.0 if mean_return <= 0 else 100.0 # High if no downside

        # 4. Trade Analysis
        total_trades = len(closed_positions)
        winning_trades = [p for p in closed_positions if p.pnl > 0]
        losing_trades = [p for p in closed_positions if p.pnl <= 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0
        
        gross_profit = sum(p.pnl for p in winning_trades)
        gross_loss = abs(sum(p.pnl for p in losing_trades))
        
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        average_trade = (gross_profit - gross_loss) / total_trades if total_trades > 0 else 0.0

        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "average_trade_usd": round(average_trade, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2)
        }

    @staticmethod
    def print_report(metrics: Dict[str, Any]):
        """Prints the performance report to the console."""
        print("\n" + "="*40)
        print("      BACKTEST PERFORMANCE REPORT      ")
        print("="*40)
        
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return

        print(f"Initial Capital:   ${metrics['initial_capital']:,.2f}")
        print(f"Final Equity:      ${metrics['final_equity']:,.2f}")
        print(f"Total Return:      {metrics['total_return_pct']}%")
        print("-"*40)
        print(f"Max Drawdown:      {metrics['max_drawdown_pct']}%")
        print(f"Sharpe Ratio:      {metrics['sharpe_ratio']}")
        print(f"Sortino Ratio:     {metrics['sortino_ratio']}")
        print("-"*40)
        print(f"Total Trades:      {metrics['total_trades']}")
        print(f"Win Rate:          {metrics['win_rate_pct']}%")
        print(f"Profit Factor:     {metrics['profit_factor']}")
        print(f"Avg Trade:         ${metrics['average_trade_usd']:,.2f}")
        print("="*40 + "\n")
