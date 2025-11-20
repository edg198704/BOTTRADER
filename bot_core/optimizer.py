import asyncio
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from bot_core.logger import get_logger
from bot_core.config import BotConfig, OptimizerConfig
from bot_core.position_manager import PositionManager, Position, PositionStatus

logger = get_logger(__name__)

class StrategyOptimizer:
    """
    Periodically analyzes trade history to optimize strategy parameters:
    1. Confidence Thresholds (Entry Strictness)
    2. Reward-to-Risk Ratios (Exit Targets)
    3. ATR Stop Multipliers (Risk Width)
    4. Risk Per Trade % (Position Sizing via Realized Kelly)
    5. Execution Parameters (Limit Offsets, Chase Aggressiveness)
    
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
                    if 'risk_per_trade_pct' in params:
                        regime_override.risk_per_trade_pct = params['risk_per_trade_pct']
                        updates += 1
            
            # 3. Restore Execution Parameters
            exec_config = self.config.execution
            exec_state = state.get('execution_params', {})
            if 'limit_price_offset_pct' in exec_state:
                exec_config.limit_price_offset_pct = exec_state['limit_price_offset_pct']
                updates += 1
            if 'chase_aggressiveness_pct' in exec_state:
                exec_config.chase_aggressiveness_pct = exec_state['chase_aggressiveness_pct']
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
        exec_config = self.config.execution
        
        state = {
            'last_updated': str(pd.Timestamp.now()),
            'regime_thresholds': {},
            'regime_risk_params': {},
            'execution_params': {}
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
                if regime_override.risk_per_trade_pct is not None:
                    params['risk_per_trade_pct'] = regime_override.risk_per_trade_pct
                
                if params:
                    state['regime_risk_params'][regime] = params
        
        # Save Execution Params
        state['execution_params'] = {
            'limit_price_offset_pct': exec_config.limit_price_offset_pct,
            'chase_aggressiveness_pct': exec_config.chase_aggressiveness_pct
        }
        
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
                if self.opt_config.execution.enabled:
                    await self.optimize_execution_parameters()
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
                if regime_override_atr := regime_risk.atr_stop_multiplier:
                     pass # Just checking existence
                else:
                    regime_risk.atr_stop_multiplier = self.config.risk_management.atr_stop_multiplier
                
                # Adjust Reward-to-Risk
                if profit_factor < self.opt_config.min_profit_factor and win_rate >= 0.40:
                    old_rr = regime_risk.reward_to_risk_ratio
                    new_rr = min(old_rr + self.opt_config.risk_adjustment_step, self.opt_config.max_reward_to_risk)
                    if new_rr != old_rr:
                        regime_risk.reward_to_risk_ratio = new_rr
                        any_changes = True
                        logger.info(f"Optimizer INCREASED {regime} R:R.", old=old_rr, new=new_rr, reason="Low PF, Decent WR")
                
                # Adjust ATR Multiplier (Stop Width)
                if profit_factor > 1.2 and win_rate < 0.40:
                    old_atr = regime_risk.atr_stop_multiplier
                    new_atr = min(old_atr + self.opt_config.risk_adjustment_step, self.opt_config.max_atr_multiplier)
                    if new_atr != old_atr:
                        regime_risk.atr_stop_multiplier = new_atr
                        any_changes = True
                        logger.info(f"Optimizer WIDENED {regime} Stop.", old=old_atr, new=new_atr, reason="Low WR, Good PF")

            # --- 3. Optimize Risk Sizing (Realized Kelly) ---
            if self.opt_config.optimize_risk_sizing and hasattr(rm_config, regime):
                regime_risk = getattr(rm_config, regime)
                
                # Initialize if None
                if regime_risk.risk_per_trade_pct is None:
                    regime_risk.risk_per_trade_pct = self.config.risk_management.risk_per_trade_pct
                
                current_risk = regime_risk.risk_per_trade_pct
                
                # Calculate Realized Kelly
                # K = W - (1-W)/R
                # R = AvgWin / AvgLoss
                avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
                avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0.0
                
                if avg_loss > 0 and win_rate > 0:
                    realized_r = avg_win / avg_loss
                    kelly = win_rate - (1.0 - win_rate) / realized_r
                    
                    # Target Risk = Kelly * Fraction
                    target_risk = kelly * self.opt_config.target_kelly_fraction
                    
                    # Clamp Target
                    target_risk = max(self.opt_config.min_risk_pct, min(target_risk, self.opt_config.max_risk_pct))
                    
                    # Move current risk towards target (Dampening)
                    step = self.opt_config.risk_size_step
                    new_risk = current_risk
                    
                    if target_risk > current_risk + (step * 0.5):
                        new_risk = min(current_risk + step, target_risk)
                    elif target_risk < current_risk - (step * 0.5):
                        new_risk = max(current_risk - step, target_risk)
                    
                    if new_risk != current_risk:
                        regime_risk.risk_per_trade_pct = round(new_risk, 5)
                        any_changes = True
                        logger.info(f"Optimizer ADJUSTED {regime} Risk Size.", 
                                    old=current_risk, new=new_risk, 
                                    kelly=f"{kelly:.2f}", realized_r=f"{realized_r:.2f}")
                elif avg_loss == 0 and win_rate > 0:
                    # Infinite R (No losses yet), slowly scale up
                    new_risk = min(current_risk + self.opt_config.risk_size_step, self.opt_config.max_risk_pct)
                    if new_risk != current_risk:
                        regime_risk.risk_per_trade_pct = round(new_risk, 5)
                        any_changes = True
                        logger.info(f"Optimizer INCREASED {regime} Risk Size (No Losses).", old=current_risk, new=new_risk)
                else:
                    # No wins or terrible performance, scale down to min
                    new_risk = max(current_risk - self.opt_config.risk_size_step, self.opt_config.min_risk_pct)
                    if new_risk != current_risk:
                        regime_risk.risk_per_trade_pct = round(new_risk, 5)
                        any_changes = True
                        logger.info(f"Optimizer DECREASED {regime} Risk Size (Poor Perf).", old=current_risk, new=new_risk)

        if any_changes:
            self._save_state()

    async def optimize_execution_parameters(self):
        """Analyzes fill rates and slippage to optimize execution settings."""
        logger.info("Running execution parameter optimization...")
        
        # Fetch recent terminal positions (CLOSED + FAILED)
        history = await self.position_manager.get_recent_execution_history(limit=100)
        if not history:
            return

        # 1. Calculate Fill Rate
        # Fill Rate = (Closed + Open) / (Closed + Open + Failed)
        # Note: Open positions are not in history list, but we can infer success from CLOSED vs FAILED
        # Actually, we only care about entry success. 
        # CLOSED positions were successfully entered. FAILED positions were not.
        
        successful_entries = [p for p in history if p.status == PositionStatus.CLOSED]
        failed_entries = [p for p in history if p.status == PositionStatus.FAILED]
        
        total_attempts = len(successful_entries) + len(failed_entries)
        if total_attempts < 10:
            return
            
        fill_rate = len(successful_entries) / total_attempts
        
        # 2. Calculate Average Slippage (on successful entries)
        slippage_values = [p.slippage_pct for p in successful_entries if p.slippage_pct is not None]
        avg_slippage = sum(slippage_values) / len(slippage_values) if slippage_values else 0.0
        
        logger.info("Execution Stats", fill_rate=f"{fill_rate:.2%}", avg_slippage=f"{avg_slippage:.4%}")
        
        exec_config = self.config.execution
        opt_exec = self.opt_config.execution
        any_changes = False
        
        # --- Optimization Logic ---
        
        # A. Low Fill Rate -> Increase Aggressiveness
        if fill_rate < opt_exec.target_fill_rate:
            # 1. Increase Chase Aggressiveness
            old_chase = exec_config.chase_aggressiveness_pct
            new_chase = min(old_chase + opt_exec.chase_adjustment_step, opt_exec.max_chase_aggro)
            if new_chase != old_chase:
                exec_config.chase_aggressiveness_pct = new_chase
                any_changes = True
                logger.info("Optimizer INCREASED Chase Aggressiveness.", old=old_chase, new=new_chase, reason="Low Fill Rate")
            
            # 2. Make Limit Offset More Aggressive (Lower/Negative)
            # Note: Lower offset = Higher Buy Price = More Aggressive
            old_offset = exec_config.limit_price_offset_pct
            new_offset = max(old_offset - opt_exec.offset_adjustment_step, opt_exec.min_limit_offset)
            if new_offset != old_offset:
                exec_config.limit_price_offset_pct = new_offset
                any_changes = True
                logger.info("Optimizer INCREASED Limit Aggressiveness (Lower Offset).", old=old_offset, new=new_offset, reason="Low Fill Rate")

        # B. High Slippage -> Decrease Aggressiveness
        # If we are paying too much spread, back off.
        # Note: Slippage here is (Fill - Decision). High positive slippage on BUY means we paid way more than decision price.
        if abs(avg_slippage) > opt_exec.max_slippage_tolerance:
            # 1. Decrease Chase Aggressiveness
            old_chase = exec_config.chase_aggressiveness_pct
            new_chase = max(old_chase - opt_exec.chase_adjustment_step, opt_exec.min_chase_aggro)
            if new_chase != old_chase:
                exec_config.chase_aggressiveness_pct = new_chase
                any_changes = True
                logger.info("Optimizer DECREASED Chase Aggressiveness.", old=old_chase, new=new_chase, reason="High Slippage")

            # 2. Make Limit Offset More Passive (Higher)
            old_offset = exec_config.limit_price_offset_pct
            new_offset = min(old_offset + opt_exec.offset_adjustment_step, opt_exec.max_limit_offset)
            if new_offset != old_offset:
                exec_config.limit_price_offset_pct = new_offset
                any_changes = True
                logger.info("Optimizer DECREASED Limit Aggressiveness (Higher Offset).", old=old_offset, new=new_offset, reason="High Slippage")

        if any_changes:
            self._save_state()

    async def stop(self):
        self.running = False
        logger.info("StrategyOptimizer stopped.")
