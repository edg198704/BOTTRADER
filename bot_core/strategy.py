# bot_core/strategy.py
import abc
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TradingStrategy(abc.ABC):
    """Abstract Base Class for a trading strategy."""

    def __init__(self, config: Dict[str, Any]):
        self.symbol = config.get("symbol", "BTCUSDT")
        self.interval = config.get("interval", 60) # seconds
        self.trade_quantity = config.get("trade_quantity", 0.001)
        logger.info(f"Strategy initialized for {self.symbol} with interval {self.interval}s, quantity {self.trade_quantity}")

    @abc.abstractmethod
    async def analyze_market(self, market_data: Dict[str, Any], open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Analyzes market data and current positions to determine if a trade should be made.
        Returns a dictionary with trade details (e.g., {'action': 'BUY', 'quantity': 0.001, 'price': 30000})
        or None if no trade is recommended.
        """
        pass

    @abc.abstractmethod
    async def manage_positions(self, market_data: Dict[str, Any], open_positions: List[Any]) -> List[Dict[str, Any]]:
        """
        Manages existing open positions (e.g., setting stop-loss, take-profit, or closing).
        Returns a list of dictionaries with actions to take (e.g., [{'action': 'CLOSE', 'position_id': 123, 'price': 30500}]).
        """
        pass

class SimpleMACrossoverStrategy(TradingStrategy):
    """
    A simple Moving Average Crossover strategy.
    Buys when fast MA crosses above slow MA, sells when fast MA crosses below slow MA.
    This is a simplified example and does not actually calculate MAs from historical data.
    It uses current price and a 'simulated' MA for demonstration.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fast_ma_period = config.get("fast_ma_period", 10) # Not actually used for real MA calculation here
        self.slow_ma_period = config.get("slow_ma_period", 20) # Not actually used for real MA calculation here
        self.last_price = None
        self.simulated_fast_ma = None
        self.simulated_slow_ma = None
        self.ma_alpha_fast = 2 / (self.fast_ma_period + 1)
        self.ma_alpha_slow = 2 / (self.slow_ma_period + 1)
        logger.info(f"SimpleMACrossoverStrategy initialized for {self.symbol}. Fast MA: {self.fast_ma_period}, Slow MA: {self.slow_ma_period}")

    async def _update_simulated_mas(self, current_price: float):
        if self.simulated_fast_ma is None:
            self.simulated_fast_ma = current_price
            self.simulated_slow_ma = current_price
        else:
            self.simulated_fast_ma = (current_price * self.ma_alpha_fast) + (self.simulated_fast_ma * (1 - self.ma_alpha_fast))
            self.simulated_slow_ma = (current_price * self.ma_alpha_slow) + (self.simulated_slow_ma * (1 - self.ma_alpha_slow))
        self.last_price = current_price

    async def analyze_market(self, market_data: Dict[str, Any], open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        current_price = float(market_data.get("lastPrice", 0.0))
        if current_price == 0.0:
            logger.warning(f"No valid lastPrice in market data for {self.symbol}.")
            return None

        await self._update_simulated_mas(current_price)

        if self.simulated_fast_ma is None or self.simulated_slow_ma is None:
            logger.debug("Waiting for MA initialization.")
            return None

        logger.debug(f"Market analysis for {self.symbol}: Price={current_price:.2f}, FastMA={self.simulated_fast_ma:.2f}, SlowMA={self.simulated_slow_ma:.2f}")

        has_open_buy = any(p.side == 'BUY' for p in open_positions)
        has_open_sell = any(p.side == 'SELL' for p in open_positions)

        # Buy signal: Fast MA crosses above Slow MA
        if self.simulated_fast_ma > self.simulated_slow_ma and not has_open_buy:
            if has_open_sell:
                logger.info(f"Strategy: Sell signal detected, but an open SELL position exists. Will manage existing position first.")
                return None # Prioritize managing existing position
            logger.info(f"Strategy: BUY signal for {self.symbol} at {current_price:.2f}")
            return {'action': 'BUY', 'quantity': self.trade_quantity, 'price': current_price, 'order_type': 'MARKET'}

        # Sell signal: Fast MA crosses below Slow MA
        elif self.simulated_fast_ma < self.simulated_slow_ma and not has_open_sell:
            if has_open_buy:
                logger.info(f"Strategy: Buy signal detected, but an open BUY position exists. Will manage existing position first.")
                return None # Prioritize managing existing position
            logger.info(f"Strategy: SELL signal for {self.symbol} at {current_price:.2f}")
            return {'action': 'SELL', 'quantity': self.trade_quantity, 'price': current_price, 'order_type': 'MARKET'}

        return None

    async def manage_positions(self, market_data: Dict[str, Any], open_positions: List[Any]) -> List[Dict[str, Any]]:
        actions = []
        current_price = float(market_data.get("lastPrice", 0.0))
        if current_price == 0.0:
            return actions

        await self._update_simulated_mas(current_price)

        if self.simulated_fast_ma is None or self.simulated_slow_ma is None:
            return actions

        for position in open_positions:
            # Simple exit logic: close position if MA crossover reverses
            if position.side == 'BUY' and self.simulated_fast_ma < self.simulated_slow_ma:
                logger.info(f"Strategy: Closing BUY position {position.id} for {position.symbol} due to MA crossover reversal.")
                actions.append({'action': 'CLOSE', 'position_id': position.id, 'price': current_price, 'order_type': 'MARKET'})
            elif position.side == 'SELL' and self.simulated_fast_ma < self.simulated_slow_ma:
                logger.info(f"Strategy: Closing SELL position {position.id} for {position.symbol} due to MA crossover reversal.")
                actions.append({'action': 'CLOSE', 'position_id': position.id, 'price': current_price, 'order_type': 'MARKET'})
            # Add more sophisticated position management here (e.g., stop-loss, take-profit)
            # For example:
            # if position.side == 'BUY' and current_price <= position.entry_price * (1 - 0.01): # 1% stop loss
            #     logger.info(f"Strategy: Closing BUY position {position.id} due to stop loss.")
            #     actions.append({'action': 'CLOSE', 'position_id': position.id, 'price': current_price, 'order_type': 'MARKET'})
            # elif position.side == 'BUY' and current_price >= position.entry_price * (1 + 0.02): # 2% take profit
            #     logger.info(f"Strategy: Closing BUY position {position.id} due to take profit.")
            #     actions.append({'action': 'CLOSE', 'position_id': position.id, 'price': current_price, 'order_type': 'MARKET'})

        return actions
