import abc
import os
import joblib
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import pickle
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from bot_core.logger import get_logger
from bot_core.config import AIStrategyConfig, SimpleMAStrategyConfig
from bot_core.data_handler import MarketEvent, SignalEvent, FillEvent

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

logger = get_logger(__name__)

# --- Strategy Interface ---

class TradingStrategy(abc.ABC):
    """Abstract Base Class for a trading strategy."""
    def __init__(self, event_queue: asyncio.Queue, config: Dict[str, Any]):
        self.event_queue = event_queue
        self.symbol = config.get("symbol", "BTC/USDT")
        self.interval_seconds = config.get("interval_seconds", 60)
        logger.info("Strategy initialized", strategy_name=self.__class__.__name__, symbol=self.symbol)

    @abc.abstractmethod
    async def on_market_event(self, event: MarketEvent, open_positions: List[Any]):
        """Reacts to new market data to generate trade signals."""
        pass

    @abc.abstractmethod
    async def on_fill_event(self, event: FillEvent):
        """Reacts to fill events to manage position state within the strategy."""
        pass

# --- Simple MA Crossover Strategy ---

class SimpleMACrossoverStrategy(TradingStrategy):
    """A simple Moving Average Crossover strategy."""
    def __init__(self, event_queue: asyncio.Queue, config: Dict[str, Any]):
        super().__init__(event_queue, config)
        ma_config = SimpleMAStrategyConfig(**config.get('simple_ma', {}))
        self.fast_ma_period = ma_config.fast_ma_period
        self.slow_ma_period = ma_config.slow_ma_period

    async def on_market_event(self, event: MarketEvent, open_positions: List[Any]):
        if event.symbol != self.symbol:
            return

        ohlcv_df = event.ohlcv_df
        if ohlcv_df.empty or len(ohlcv_df) < self.slow_ma_period:
            return

        df = ohlcv_df.copy()
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        has_open_position = any(p.symbol == self.symbol for p in open_positions)

        # Golden Cross (Buy Signal)
        if not has_open_position and last_row['fast_ma'] > last_row['slow_ma'] and prev_row['fast_ma'] <= prev_row['slow_ma']:
            logger.info("BUY signal generated", strategy="SimpleMACrossover", symbol=self.symbol, price=last_row['close'])
            signal = SignalEvent(symbol=self.symbol, action='BUY', confidence=0.75)
            await self.event_queue.put(signal)

        # Death Cross (Sell Signal to close position)
        position_to_manage = next((p for p in open_positions if p.symbol == self.symbol), None)
        if position_to_manage and position_to_manage.side == 'BUY' and last_row['fast_ma'] < last_row['slow_ma'] and prev_row['fast_ma'] >= prev_row['slow_ma']:
            logger.info("CLOSE signal generated for BUY position", strategy="SimpleMACrossover", position_id=position_to_manage.id)
            signal = SignalEvent(symbol=self.symbol, action='SELL', confidence=0.75) # Signal to sell/close
            await self.event_queue.put(signal)

    async def on_fill_event(self, event: FillEvent):
        # Simple strategy does not need to react to fills
        pass

# --- AI Components (Consolidated for integration) ---

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class _MarketRegimeDetector:
    def __init__(self, config: AIStrategyConfig):
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
            logger.error("Error in regime detection", symbol=symbol, error=str(e))
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}

class _EnsembleLearner:
    class LSTMPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 3)

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return F.softmax(self.fc(hn[0]), dim=1)

    class AttentionNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True), 
                num_layers=2
            )
            self.fc = nn.Linear(hidden_dim, 3)

        def forward(self, x):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            x = self.embedding(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return F.softmax(self.fc(x), dim=1)

    def __init__(self, config: AIStrategyConfig):
        if not ML_AVAILABLE: raise ImportError("ML libraries not available")
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.symbol_models: Dict[str, Dict[str, Any]] = {}
        self.is_trained = False
        logger.info("AI ensemble model templates initialized.")
        asyncio.create_task(self._load_models())

    def _get_or_create_symbol_models(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self.symbol_models:
            logger.info("Creating new model set for symbol", symbol=symbol)
            self.symbol_models[symbol] = {
                'lstm': self.LSTMPredictor(len(self.config.feature_columns), 64).to(self.device),
                'gb': XGBClassifier(n_estimators=10, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss'),
                'attention': self.AttentionNetwork(len(self.config.feature_columns), 64).to(self.device),
                'technical': VotingClassifier(estimators=[
                    ('rf', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)),
                    ('lr', LogisticRegression(max_iter=100, random_state=42))
                ], voting='soft')
            }
        return self.symbol_models[symbol]

    async def _load_models(self):
        base_path = self.config.model_path
        if not os.path.isdir(base_path):
            logger.warning("Model path does not exist, cannot load models.", path=base_path)
            return

        for symbol_dir in os.listdir(base_path):
            symbol_path = os.path.join(base_path, symbol_dir)
            if os.path.isdir(symbol_path):
                symbol = symbol_dir.replace('_', '/')
                try:
                    models = self._get_or_create_symbol_models(symbol)
                    models['gb'] = joblib.load(os.path.join(symbol_path, "gb_model.pkl"))
                    models['technical'] = joblib.load(os.path.join(symbol_path, "technical_model.pkl"))
                    models['lstm'].load_state_dict(torch.load(os.path.join(symbol_path, "lstm_model.pth")))
                    models['attention'].load_state_dict(torch.load(os.path.join(symbol_path, "attention_model.pth")))
                    self.is_trained = True
                    logger.info("Loaded models for symbol", symbol=symbol)
                except Exception as e:
                    logger.warning("Could not load models for symbol", symbol=symbol, error=str(e))

    async def predict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if not self.is_trained:
            logger.debug("Predict called but models are not trained or loaded.")
            return {'action': 'hold', 'confidence': 0.0}
        try:
            models = self._get_or_create_symbol_models(symbol)
            feature_cols = self.config.feature_columns
            
            current_features = {col: df[col].iloc[-1] if col in df.columns else 0 for col in feature_cols}
            features_df = pd.DataFrame([current_features], columns=feature_cols)
            features = np.nan_to_num(features_df.values, nan=0.0)

            predictions = []
            weights = []

            gb_pred = models['gb'].predict_proba(features)[0]
            predictions.append(gb_pred)
            weights.append(0.25)

            tech_pred = models['technical'].predict_proba(features)[0]
            predictions.append(tech_pred)
            weights.append(0.15)

            features_tensor = torch.FloatTensor(features).unsqueeze(1).to(self.device)
            with torch.no_grad():
                lstm_pred = models['lstm'](features_tensor).cpu().numpy()[0]
                predictions.append(lstm_pred)
                weights.append(0.3)

                attn_pred = models['attention'](features_tensor).cpu().numpy()[0]
                predictions.append(attn_pred)
                weights.append(0.3)

            ensemble_pred = np.average(predictions, weights=weights, axis=0)
            action_idx = np.argmax(ensemble_pred)
            confidence = float(ensemble_pred[action_idx])
            action_map = {0: 'sell', 1: 'hold', 2: 'buy'} # Assuming 0:sell, 1:hold, 2:buy
            return {'action': action_map.get(action_idx, 'hold'), 'confidence': confidence}
        except Exception as e:
            logger.error("Error during prediction", symbol=symbol, error=str(e), exc_info=True)
            return {'action': 'hold', 'confidence': 0.0}

# --- AI Ensemble Strategy ---

class AIEnsembleStrategy(TradingStrategy):
    def __init__(self, event_queue: asyncio.Queue, config: Dict[str, Any]):
        super().__init__(event_queue, config)
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not installed for AIEnsembleStrategy.")
        
        self.ai_config = AIStrategyConfig(**config.get('ai_ensemble', {}))
        
        self.confidence_threshold = self.ai_config.confidence_threshold
        self.use_regime_filter = self.ai_config.use_regime_filter
        
        self.regime_detector = _MarketRegimeDetector(self.ai_config)
        self.ensemble_learner = _EnsembleLearner(self.ai_config)

    async def on_market_event(self, event: MarketEvent, open_positions: List[Any]):
        if event.symbol != self.symbol or not self.ensemble_learner.is_trained:
            return

        if any(p.symbol == self.symbol for p in open_positions):
            logger.debug("Skipping signal generation, position already open.", symbol=self.symbol)
            return

        ohlcv_df = event.ohlcv_df
        regime = await self.regime_detector.detect_regime(self.symbol, ohlcv_df)
        logger.debug("Market regime detected", symbol=self.symbol, regime=regime)

        ensemble_prediction = await self.ensemble_learner.predict(ohlcv_df, self.symbol)
        logger.debug("Ensemble prediction received", symbol=self.symbol, prediction=ensemble_prediction)

        ensemble_action = ensemble_prediction['action']
        ensemble_confidence = ensemble_prediction['confidence']

        if ensemble_confidence < self.confidence_threshold:
            logger.debug("Signal ignored due to low confidence.", confidence=ensemble_confidence, threshold=self.confidence_threshold)
            return

        final_action = ensemble_action
        final_confidence = ensemble_confidence
        if self.use_regime_filter:
            current_regime = regime.get('regime')
            if (current_regime == MarketRegime.BULL.value and ensemble_action == 'sell') or \
               (current_regime == MarketRegime.BEAR.value and ensemble_action == 'buy'):
                logger.info("Regime filter overrides action", regime=current_regime, action=ensemble_action)
                return
            final_confidence *= regime.get('confidence', 0.5)

        if final_action != 'hold':
            logger.info("Final AI signal generated", symbol=self.symbol, action=final_action.upper(), confidence=final_confidence)
            signal = SignalEvent(symbol=self.symbol, action=final_action.upper(), confidence=final_confidence)
            await self.event_queue.put(signal)

    async def on_fill_event(self, event: FillEvent):
        # AI strategy could use fill events to update internal state or trigger retraining
        logger.debug("AI Strategy received fill event", symbol=event.symbol)
