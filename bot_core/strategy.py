import abc
import logging
import os
import joblib
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Define dummy classes if ML libraries are not available
    class VotingClassifier: pass
    class RandomForestClassifier: pass
    class GradientBoostingClassifier: pass
    class LogisticRegression: pass
    class XGBClassifier: pass
    class nn:
        class Module: pass

logger = logging.getLogger(__name__)

# --- Strategy Interface ---

class TradingStrategy(abc.ABC):
    """Abstract Base Class for a trading strategy."""
    def __init__(self, config: Dict[str, Any]):
        self.symbol = config.get("symbol", "BTC/USDT")
        self.interval_seconds = config.get("interval_seconds", 60)
        logger.info(f"{self.__class__.__name__} initialized for {self.symbol}.")

    @abc.abstractmethod
    async def analyze_market(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        """Analyzes market data to generate a new trade signal (BUY/SELL)."""
        pass

    @abc.abstractmethod
    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        """Manages existing open positions (e.g., signals to close)."""
        pass

# --- Simple MA Crossover Strategy ---

class SimpleMACrossoverStrategy(TradingStrategy):
    """A simple Moving Average Crossover strategy."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        ma_config = config.get('simple_ma', {})
        self.fast_ma_period = ma_config.get("fast_ma_period", 10)
        self.slow_ma_period = ma_config.get("slow_ma_period", 20)

    async def analyze_market(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        if ohlcv_df.empty or len(ohlcv_df) < self.slow_ma_period:
            return None

        df = ohlcv_df.copy()
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        has_open_position = any(p.symbol == self.symbol for p in open_positions)

        # Golden Cross
        if not has_open_position and last_row['fast_ma'] > last_row['slow_ma'] and prev_row['fast_ma'] <= prev_row['slow_ma']:
            logger.info(f"Strategy: BUY signal for {self.symbol} at {last_row['close']:.2f}")
            return {'action': 'BUY', 'symbol': self.symbol, 'confidence': 0.75}

        return None

    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        actions = []
        position_to_manage = next((p for p in open_positions if p.symbol == self.symbol), None)
        if ohlcv_df.empty or not position_to_manage:
            return actions

        df = ohlcv_df.copy()
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        # Death Cross
        if position_to_manage.side == 'BUY' and last_row['fast_ma'] < last_row['slow_ma'] and prev_row['fast_ma'] >= prev_row['slow_ma']:
            logger.info(f"Strategy: Closing BUY position {position_to_manage.id} for {position_to_manage.symbol}.")
            actions.append({'action': 'CLOSE', 'position_id': position_to_manage.id})
        
        return actions

# --- AI Components (Consolidated from monolithic files for integration) ---
# NOTE: In a full refactor, these large classes would be in their own files.

class _LegacyAIConfig:
    """A temporary bridge class to make legacy AI components work with the new config."""
    def __init__(self, config_dict: Dict[str, Any]):
        self.symbols = [config_dict.get('symbol', 'BTC/USDT')]
        # Add other fields with defaults as needed by the legacy classes
        self.ensemble_models = config_dict.get('ensemble_models', ['random_forest', 'gradient_boost', 'logistic_regression', 'xgboost'])

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class CompleteMarketRegimeDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def detect_regime(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 50:
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
        
        try:
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50

            trend_strength = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0

            if trend_strength > 0.01: regime = MarketRegime.BULL
            elif trend_strength < -0.01: regime = MarketRegime.BEAR
            elif volatility > df['close'].pct_change().rolling(50).std().mean() * 1.5: regime = MarketRegime.VOLATILE
            else: regime = MarketRegime.SIDEWAYS

            confidence = min(1.0, abs(trend_strength) * 20 + (abs(rsi - 50) / 50))
            return {'regime': regime.value, 'confidence': confidence}
        except Exception as e:
            logger.error(f"Error in regime detection for {symbol}: {e}")
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}

class AdvancedEnsembleLearner:
    def __init__(self, config: Dict[str, Any]):
        if not ML_AVAILABLE: raise ImportError("ML libraries not available")
        self.config = config
        self.models: Dict[str, Any] = {}
        self.is_trained = False
        self._initialize_models()

    def _initialize_models(self):
        self.models['gb'] = XGBClassifier(n_estimators=10, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.models['technical'] = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)),
            ('lr', LogisticRegression(max_iter=100, random_state=42))
        ], voting='soft')
        logger.info("AI ensemble models initialized.")
        asyncio.create_task(self._load_models())

    async def _load_models(self):
        model_path = self.config.get('model_path', 'models/ensemble')
        symbol_safe = self.config.get('symbol', '').replace('/', '_')
        try:
            gb_path = os.path.join(model_path, f"gb_model.pkl")
            tech_path = os.path.join(model_path, f"technical_model.pkl")
            if os.path.exists(gb_path) and os.path.exists(tech_path):
                self.models['gb'] = joblib.load(gb_path)
                self.models['technical'] = joblib.load(tech_path)
                self.is_trained = True
                logger.info(f"AI models for {self.config.get('symbol')} loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load pre-trained AI models: {e}. Models will need training.")
            self.is_trained = False

    async def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained: return {'action': 'hold', 'confidence': 0.0}
        try:
            feature_cols = self.config.get('feature_columns', ['close', 'rsi', 'macd', 'volume'])
            # Ensure all feature columns exist, fill with 0 if not
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0
            latest_features = df[feature_cols].iloc[-1:].values
            latest_features = np.nan_to_num(latest_features, nan=0.0)

            gb_pred = self.models['gb'].predict_proba(latest_features)[0]
            tech_pred = self.models['technical'].predict_proba(latest_features)[0]

            ensemble_pred = np.average([gb_pred, tech_pred], weights=[0.6, 0.4], axis=0)
            action_idx = np.argmax(ensemble_pred)
            confidence = float(ensemble_pred[action_idx])
            action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            return {'action': action_map.get(action_idx, 'hold'), 'confidence': confidence}
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {'action': 'hold', 'confidence': 0.0}

class CompletePPOAgent:
    def __init__(self, config: Dict[str, Any]): pass
    async def act(self, df: pd.DataFrame) -> Tuple[int, float]: return (1, 0.5)

# --- AI Ensemble Strategy ---

class AIEnsembleStrategy(TradingStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not installed for AIEnsembleStrategy.")
        
        self.ai_config = config.get('ai_ensemble', {})
        self.confidence_threshold = self.ai_config.get('confidence_threshold', 0.6)
        self.use_regime_filter = self.ai_config.get('use_regime_filter', True)
        self.use_ppo_agent = self.ai_config.get('use_ppo_agent', False)
        
        # Pass the nested ai_ensemble config to the components
        ai_component_config = {**self.ai_config, 'symbol': self.symbol}

        self.regime_detector = CompleteMarketRegimeDetector(ai_component_config)
        self.ensemble_learner = AdvancedEnsembleLearner(ai_component_config)
        self.ppo_agent = CompletePPOAgent(ai_component_config) if self.use_ppo_agent else None

    async def analyze_market(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        if ohlcv_df is None or ohlcv_df.empty or not self.ensemble_learner.is_trained:
            return None

        if any(p.symbol == self.symbol for p in open_positions):
            return None

        # 1. Detect Market Regime
        regime = await self.regime_detector.detect_regime(self.symbol, ohlcv_df)
        logger.debug(f"Market regime for {self.symbol}: {regime}")

        # 2. Get Ensemble Prediction
        ensemble_prediction = await self.ensemble_learner.predict(ohlcv_df)
        logger.debug(f"Ensemble prediction for {self.symbol}: {ensemble_prediction}")

        # 3. Combine signals into a final decision
        ensemble_action = ensemble_prediction['action']
        ensemble_confidence = ensemble_prediction['confidence']

        if ensemble_confidence < self.confidence_threshold:
            return None

        # 4. Apply Regime Filter
        final_action = ensemble_action
        final_confidence = ensemble_confidence
        if self.use_regime_filter:
            current_regime = regime.get('regime')
            if (current_regime == MarketRegime.BULL.value and ensemble_action == 'sell') or \
               (current_regime == MarketRegime.BEAR.value and ensemble_action == 'buy'):
                logger.info(f"Regime filter overrides action. Regime: {current_regime}, Action: {ensemble_action}")
                return None # Contradictory signal, do not trade
            final_confidence *= regime.get('confidence', 0.5)

        if final_action != 'hold':
            logger.info(f"Final AI signal for {self.symbol}: {final_action.upper()} with confidence {final_confidence:.2f}")
            return {'action': final_action.upper(), 'symbol': self.symbol, 'confidence': final_confidence}

        return None

    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        # This strategy relies on stop-loss and take-profit set at trade execution.
        # Future enhancements could use AI signals for early exit.
        return []
