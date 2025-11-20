import asyncio
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from bot_core.logger import get_logger
from bot_core.config import BotConfig, OptimizerConfig
from bot_core.position_manager import PositionManager, Position

logger = get_logger(__name__)

class StrategyOptimizer:
    """
    Periodically analyzes trade history to optimize strategy parameters:
    1. Confidence Thresholds (Entry Strictness)
    2. Reward-to-Risk Ratios (Exit Targets)
    3. ATR Stop Multipliers (Risk Width)
    
    Adjustments are based on realized performance (Win Rate & Profit Factor) per market regime.
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
        """Loads optimized parameters from the state file and applies them to the config."""
        path = self._get_state_path()
        if not os.path.exists(path):
            return

        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            updates = 0
            
            # 1. Restore Confidence Thresholds
            mr_config = self.config.strategy.params.market_regime
            for regime, threshold in state.get('regime_thresholds', {}).items():
                attr_name = f"{regime}_confidence_threshold"
                if hasattr(mr_config, attr_name):
                    setattr(mr_config, attr_name, threshold)
                    updates += 1
            
            # 2. Restore Risk Parameters
            rm_config = self.config.risk_management.regime_based_risk
            for regime, params in state.get('regime_risk_params', {}).items():
                if hasattr(rm_config, regime):
                    regime_override = getattr(rm_config, regime)
                    if 'reward_to_risk_ratio' in params:
                        regime_override.reward_to_risk_ratio = params['reward_to_risk_ratio']
                        updates += 1
                    if 'atr_stop_multiplier' in params:
                        regime_override.atr_stop_multiplier = params['atr_stop_multiplier']
                        updates += 1

            if updates > 0:
                logger.info(f"Restored {updates} optimized parameters from state file.", path=path)
                
        except Exception as e:
            logger.error("Failed to load optimizer state", error=str(e))

    def _save_state(self):
        """Saves the current optimized parameters to the state file."""
        path = self._get_state_path()
        mr_config = self.config.strategy.params.market_regime
        rm_config = self.config.risk_management.regime_based_risk
        
        state = {
            'last_updated': str(pd.Timestamp.now()),
            'regime_thresholds': {},
            'regime_risk_params': {}
        }
        
        regimes = ['bull', 'bear', 'volatile', 'sideways']
        
        for regime in regimes:
            # Save Confidence Thresholds
            attr_name = f"{regime}_confidence_threshold"
            if hasattr(mr_config, attr_name):
                val = getattr(mr_config, attr_name)
                if val is not None:
                    state['regime_thresholds'][regime] = val
            
            # Save Risk Params
            if hasattr(rm_config, regime):
                regime_override = getattr(rm_config, regime)
                params = {}
                if regime_override.reward_to_risk_ratio is not None:
                    params['reward_to_risk_ratio'] = regime_override.reward_to_risk_ratio
                if regime_override.atr_stop_multiplier is not None:
                    params['atr_stop_multiplier'] = regime_override.atr_stop_multiplier
                
                if params:
                    state['regime_risk_params'][regime] = params
        
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
                await self.optimize_strategy_parameters()
            except asyncio.CancelledError:
                logger.info("StrategyOptimizer loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in StrategyOptimizer loop", error=str(e))
            
            # Sleep for the configured interval
            await asyncio.sleep(self.opt_config.interval_hours * 3600)

    async def optimize_strategy_parameters(self):
        logger.info("Running strategy parameter optimization...")
        
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
        rm_config = self.config.risk_management.regime_based_risk
        any_changes = False
        
        for regime, trades in regime_stats.items():
            if len(trades) < 5: # Minimum sample per regime
                continue

            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(wins) / len(trades)
            gross_profit = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            logger.info(f"Regime Stats: {regime.upper()}", win_rate=f"{win_rate:.2f}", profit_factor=f"{profit_factor:.2f}", trades=len(trades))

            # --- 1. Optimize Confidence Threshold (Entry) ---
            attr_name = f"{regime}_confidence_threshold"
            if hasattr(mr_config, attr_name):
                current_thresh = getattr(mr_config, attr_name)
                if current_thresh is None:
                    current_thresh = self.config.strategy.params.confidence_threshold

                new_thresh = current_thresh
                
                # Logic: 
                # If PF is bad (< 1.0) AND Win Rate is low (< 40%), the entry is the problem -> Tighten Entry.
                # If PF is good (> 1.5) AND Win Rate is high (> 60%), we are too picky -> Loosen Entry.
                
                if profit_factor < self.opt_config.min_profit_factor and win_rate < 0.40:
                    new_thresh = min(current_thresh + self.opt_config.adjustment_step, self.opt_config.max_threshold_cap)
                    if new_thresh != current_thresh:
                        setattr(mr_config, attr_name, new_thresh)
                        any_changes = True
                        logger.info(f"Optimizer INCREASED {regime} confidence.", old=current_thresh, new=new_thresh, reason="Low WR & PF")
                
                elif profit_factor > self.opt_config.high_performance_pf and win_rate > 0.60:
                    new_thresh = max(current_thresh - self.opt_config.adjustment_step, self.opt_config.min_threshold_floor)
                    if new_thresh != current_thresh:
                        setattr(mr_config, attr_name, new_thresh)
                        any_changes = True
                        logger.info(f"Optimizer DECREASED {regime} confidence.", old=current_thresh, new=new_thresh, reason="High WR & PF")

            # --- 2. Optimize Risk Parameters (Exit/Sizing) ---
            if self.opt_config.optimize_risk_params and hasattr(rm_config, regime):
                regime_risk = getattr(rm_config, regime)
                
                # Initialize if None
                if regime_risk.reward_to_risk_ratio is None:
                    regime_risk.reward_to_risk_ratio = self.config.risk_management.reward_to_risk_ratio
                if regime_risk.atr_stop_multiplier is None:
                    regime_risk.atr_stop_multiplier = self.config.risk_management.atr_stop_multiplier
                
                # Logic: 
                # If PF is bad (< 1.0) BUT Win Rate is decent (> 40%), the payoff is the problem -> Increase R:R.
                # If PF is good (> 1.5) BUT Win Rate is low (< 40%), we are getting stopped out too often -> Widen Stop (Increase ATR Mult).
                
                # Adjust Reward-to-Risk
                if profit_factor < self.opt_config.min_profit_factor and win_rate >= 0.40:
                    old_rr = regime_risk.reward_to_risk_ratio
                    new_rr = min(old_rr + self.opt_config.risk_adjustment_step, self.opt_config.max_reward_to_risk)
                    if new_rr != old_rr:
                        regime_risk.reward_to_risk_ratio = new_rr
                        any_changes = True
                        logger.info(f"Optimizer INCREASED {regime} R:R.", old=old_rr, new=new_rr, reason="Low PF, Decent WR")
                
                # Adjust ATR Multiplier (Stop Width)
                # Note: Widening stop improves Win Rate but hurts R:R. Only do this if PF is healthy enough to absorb it.
                if profit_factor > 1.2 and win_rate < 0.40:
                    old_atr = regime_risk.atr_stop_multiplier
                    new_atr = min(old_atr + self.opt_config.risk_adjustment_step, self.opt_config.max_atr_multiplier)
                    if new_atr != old_atr:
                        regime_risk.atr_stop_multiplier = new_atr
                        any_changes = True
                        logger.info(f"Optimizer WIDENED {regime} Stop.", old=old_atr, new=new_atr, reason="Low WR, Good PF")

        if any_changes:
            self._save_state()

    async def stop(self):
        self.running = False
        logger.info("StrategyOptimizer stopped.")
