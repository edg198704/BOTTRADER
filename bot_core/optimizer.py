import asyncio
import json
import os
import pandas as pd
from typing import List, Dict, Any
from bot_core.logger import get_logger
from bot_core.config import BotConfig, OptimizerConfig
from bot_core.position_manager import PositionManager, Position

logger = get_logger(__name__)

class StrategyOptimizer:
    """
    Periodically analyzes trade history to optimize strategy parameters (specifically confidence thresholds)
    based on realized performance in different market regimes.
    Persists learned parameters to disk to ensure adaptation survives restarts.
    """
    def __init__(self, config: BotConfig, position_manager: PositionManager):
        self.config = config
        self.opt_config = config.optimizer
        self.position_manager = position_manager
        self.running = False
        
        # Load saved state on init to restore learned thresholds
        self._load_state()
        
        logger.info("StrategyOptimizer initialized.")

    def _get_state_path(self) -> str:
        return self.opt_config.state_file_path

    def _load_state(self):
        """Loads optimized thresholds from the state file and applies them to the config."""
        path = self._get_state_path()
        if not os.path.exists(path):
            return

        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            # We modify the config object in-place. Since it's passed by reference,
            # the Strategy component (which holds a reference to the same config object)
            # will see these updates immediately.
            mr_config = self.config.strategy.params.market_regime
            updates = 0
            
            for regime, threshold in state.get('regime_thresholds', {}).items():
                attr_name = f"{regime}_confidence_threshold"
                if hasattr(mr_config, attr_name):
                    setattr(mr_config, attr_name, threshold)
                    updates += 1
            
            if updates > 0:
                logger.info(f"Restored {updates} optimized thresholds from state file.", path=path)
                
        except Exception as e:
            logger.error("Failed to load optimizer state", error=str(e))

    def _save_state(self):
        """Saves the current optimized thresholds to the state file."""
        path = self._get_state_path()
        mr_config = self.config.strategy.params.market_regime
        
        state = {
            'last_updated': str(pd.Timestamp.now()),
            'regime_thresholds': {}
        }
        
        for regime in ['bull', 'bear', 'volatile', 'sideways']:
            attr_name = f"{regime}_confidence_threshold"
            if hasattr(mr_config, attr_name):
                val = getattr(mr_config, attr_name)
                if val is not None:
                    state['regime_thresholds'][regime] = val
        
        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug("Optimizer state saved.", path=path)
        except Exception as e:
            logger.error("Failed to save optimizer state", error=str(e))

    async def run(self):
        if not self.opt_config.enabled:
            logger.info("Optimizer disabled in config.")
            return

        self.running = True
        logger.info("Starting StrategyOptimizer loop.")
        
        # Initial delay to allow some trades to happen or DB to load
        await asyncio.sleep(self.opt_config.interval_hours * 3600)

        while self.running:
            try:
                await self.optimize_regime_thresholds()
            except asyncio.CancelledError:
                logger.info("StrategyOptimizer loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in StrategyOptimizer loop", error=str(e))
            
            # Sleep for the configured interval
            await asyncio.sleep(self.opt_config.interval_hours * 3600)

    async def optimize_regime_thresholds(self):
        logger.info("Running regime threshold optimization...")
        
        # Fetch recent closed positions
        all_closed = await self.position_manager.get_all_closed_positions()
        
        if not all_closed:
            logger.info("No closed positions to analyze.")
            return

        # Filter for recent trades based on lookback window
        lookback_count = self.opt_config.lookback_trades
        recent_trades = all_closed[-lookback_count:]
        
        if len(recent_trades) < self.opt_config.min_trades_for_adjustment:
            logger.info("Insufficient trades for optimization.", count=len(recent_trades), required=self.opt_config.min_trades_for_adjustment)
            return

        # Group by Regime
        regime_stats = {'bull': [], 'bear': [], 'sideways': [], 'volatile': []}
        
        for pos in recent_trades:
            if not pos.strategy_metadata:
                continue
            try:
                meta = json.loads(pos.strategy_metadata)
                regime = meta.get('regime')
                if regime in regime_stats:
                    regime_stats[regime].append(pos)
            except json.JSONDecodeError:
                continue

        # Analyze and Adjust
        if not hasattr(self.config.strategy.params, 'market_regime'):
            logger.warning("Strategy does not have market_regime configuration. Skipping optimization.")
            return
            
        mr_config = self.config.strategy.params.market_regime
        any_changes = False
        
        for regime, trades in regime_stats.items():
            if len(trades) < 5: # Minimum sample per regime to be statistically relevant
                continue

            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(wins) / len(trades)
            gross_profit = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            logger.info(f"Regime Stats: {regime.upper()}", win_rate=f"{win_rate:.2f}", profit_factor=f"{profit_factor:.2f}", trades=len(trades))

            # Get current threshold
            attr_name = f"{regime}_confidence_threshold"
            if not hasattr(mr_config, attr_name):
                continue
                
            current_thresh = getattr(mr_config, attr_name)
            # If None, it uses base threshold. We should set a specific one now.
            if current_thresh is None:
                current_thresh = self.config.strategy.params.confidence_threshold

            new_thresh = current_thresh
            action = "MAINTAINED"
            
            # Adjustment Logic
            if profit_factor < self.opt_config.min_profit_factor:
                # Performance bad -> Increase threshold (stricter)
                new_thresh = min(current_thresh + self.opt_config.adjustment_step, self.opt_config.max_threshold_cap)
                action = "INCREASED"
            elif profit_factor > self.opt_config.high_performance_pf:
                # Performance good -> Decrease threshold (looser)
                new_thresh = max(current_thresh - self.opt_config.adjustment_step, self.opt_config.min_threshold_floor)
                action = "DECREASED"

            if new_thresh != current_thresh:
                setattr(mr_config, attr_name, new_thresh)
                any_changes = True
                logger.info(f"Optimizer {action} {regime} threshold.", old=current_thresh, new=new_thresh, reason=f"PF={profit_factor:.2f}")

        if any_changes:
            self._save_state()

    async def stop(self):
        self.running = False
        logger.info("StrategyOptimizer stopped.")
