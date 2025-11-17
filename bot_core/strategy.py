import abc
import logging
import os
import joblib
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Define dummy classes if ML libraries are not available
    class VotingClassifier: pass
    class RandomForestClassifier: pass
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
        self.fast_ma_period = config.get("fast_ma_period", 10)
        self.slow_ma_period = config.get("slow_ma_period", 20)

    async def analyze_market(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        if ohlcv_df.empty or len(ohlcv_df) < self.slow_ma_period:
            return None

        df = ohlcv_df.copy()
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        has_open_position = any(p.symbol == self.symbol for p in open_positions)

        if not has_open_position and last_row['fast_ma'] > last_row['slow_ma'] and prev_row['fast_ma'] <= prev_row['slow_ma']:
            logger.info(f"Strategy: BUY signal for {self.symbol} at {last_row['close']:.2f}")
            return {'action': 'BUY', 'confidence': 0.75}

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

        if position_to_manage.side == 'BUY' and last_row['fast_ma'] < last_row['slow_ma'] and prev_row['fast_ma'] >= prev_row['slow_ma']:
            logger.info(f"Strategy: Closing BUY position {position_to_manage.id} for {position_to_manage.symbol}.")
            actions.append({'action': 'CLOSE', 'position_id': position_to_manage.id})
        
        return actions

# --- AI Components (Migrated from monolithic files) ---

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class CompleteMarketRegimeDetector:
    async def detect_regime(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 50:
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
        
        try:
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
            rsi = df['rsi'].iloc[-1]

            trend_strength = (sma_20 - sma_50) / sma_50

            if trend_strength > 0.01: regime = MarketRegime.BULL
            elif trend_strength < -0.01: regime = MarketRegime.BEAR
            elif volatility > df['close'].pct_change().rolling(50).std().mean() * 1.5: regime = MarketRegime.VOLATILE
            else: regime = MarketRegime.SIDEWAYS

            confidence = min(1.0, abs(trend_strength) * 20 + (rsi if rsi > 70 or rsi < 30 else 50) / 100)
            return {'regime': regime.value, 'confidence': confidence}
        except Exception as e:
            logger.error(f"Error in regime detection for {symbol}: {e}")
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}

class AdvancedEnsembleLearner:
    def __init__(self, config: Dict[str, Any]):
        if not ML_AVAILABLE:
            raise ImportError("ML libraries not available for AdvancedEnsembleLearner")
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: Dict[str, Any] = {}
        self.is_trained = False
        self._initialize_models()
        self._load_models()

    def _initialize_models(self):
        self.models['gb'] = XGBClassifier(n_estimators=10, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.models['technical'] = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)),
            ('lr', LogisticRegression(max_iter=100, random_state=42))
        ], voting='soft')
        logger.info("AI ensemble models initialized.")

    def _load_models(self):
        model_path = self.config.get('model_path', 'models/ensemble')
        symbol_safe = self.config.get('symbol', '').replace('/', '_')
        try:
            gb_path = os.path.join(model_path, f"{symbol_safe}_gb.joblib")
            tech_path = os.path.join(model_path, f"{symbol_safe}_technical.joblib")
            if os.path.exists(gb_path) and os.path.exists(tech_path):
                self.models['gb'] = joblib.load(gb_path)
                self.models['technical'] = joblib.load(tech_path)
                self.is_trained = True
                logger.info(f"AI models for {self.config.get('symbol')} loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading AI models: {e}")
            self.is_trained = False

    async def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained: return {'action': 'hold', 'confidence': 0.0}
        try:
            feature_cols = self.config.get('feature_columns', ['close', 'rsi', 'macd', 'volume'])
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
    def __init__(self, config: Dict[str, Any]):
        pass # Placeholder for brevity, full implementation is complex
    async def act(self, df: pd.DataFrame) -> Tuple[int, float]:
        # Dummy implementation: Returns 'hold' with neutral confidence
        return (1, 0.5) # 0: sell, 1: hold, 2: buy

# --- AI Ensemble Strategy ---

class AIEnsembleStrategy(TradingStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not installed for AIEnsembleStrategy.")
        
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        # Instantiate AI components
        self.regime_detector = CompleteMarketRegimeDetector()
        self.ensemble_learner = AdvancedEnsembleLearner(config)
        self.ppo_agent = CompletePPOAgent(config)

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

        # 3. Get PPO Agent Action
        ppo_action_idx, ppo_confidence = await self.ppo_agent.act(ohlcv_df)
        ppo_action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
        ppo_action = ppo_action_map.get(ppo_action_idx, 'hold')
        logger.debug(f"PPO action for {self.symbol}: {ppo_action}")

        # 4. Combine signals into a final decision
        final_action = 'hold'
        final_confidence = 0.0

        # Simple combination logic: Trust ensemble, but filter based on regime.
        # PPO agent is ignored for now as it's a dummy.
        ensemble_action = ensemble_prediction['action']
        ensemble_confidence = ensemble_prediction['confidence']

        if ensemble_confidence < self.confidence_threshold:
            return None

        # Regime filter
        current_regime = regime.get('regime')
        if (current_regime == MarketRegime.BULL.value and ensemble_action == 'buy') or \
           (current_regime == MarketRegime.BEAR.value and ensemble_action == 'sell'):
            final_action = ensemble_action
            final_confidence = ensemble_confidence * regime.get('confidence', 0.5)
        elif current_regime == MarketRegime.SIDEWAYS.value or current_regime == MarketRegime.VOLATILE.value:
            # Be more cautious in non-trending markets
            if ensemble_confidence > (self.confidence_threshold + 0.1):
                 final_action = ensemble_action
                 final_confidence = ensemble_confidence * regime.get('confidence', 0.5)

        if final_action != 'hold':
            logger.info(f"Final AI signal for {self.symbol}: {final_action.upper()} with confidence {final_confidence:.2f}")
            return {'action': final_action.upper(), 'confidence': final_confidence}

        return None

    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        # For now, position management is handled by SL/TP in the main bot loop.
        # A future enhancement would be to use AI signals to close positions early.
        return []
