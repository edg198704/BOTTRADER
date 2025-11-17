import abc
import logging
import os
import pickle
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Define dummy classes if ML libraries are not available
    class nn:
        class Module: pass
    class RandomForestClassifier: pass
    class GradientBoostingClassifier: pass
    class VotingClassifier: pass
    class LogisticRegression: pass
    class XGBClassifier: pass

logger = logging.getLogger(__name__)

# --- Strategy Interfaces ---

class TradingStrategy(abc.ABC):
    """Abstract Base Class for a trading strategy."""
    def __init__(self, config: Dict[str, Any]):
        self.symbol = config.get("symbol", "BTC/USDT")
        self.interval_seconds = config.get("interval_seconds", 60)
        logger.info(f"{self.__class__.__name__} initialized for {self.symbol}.")

    @abc.abstractmethod
    async def analyze_market(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        pass

# --- AI Ensemble Strategy ---

class AIEnsembleStrategy(TradingStrategy):
    """Trading strategy based on an ensemble of AI models."""

    class LSTMPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 3) # sell, hold, buy

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return self.fc(hn[0])

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not ML_AVAILABLE:
            raise ImportError("Required machine learning libraries are not installed.")
        
        self.feature_columns = config.get('feature_columns', ['close', 'rsi', 'macd', 'volume'])
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.model_path = config.get('model_path', 'models/ensemble')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models = {}
        self.is_trained = False
        self.is_training = False
        self._initialize_models()
        self._load_models()

    def _initialize_models(self):
        self.models['lstm'] = self.LSTMPredictor(len(self.feature_columns), 64).to(self.device)
        self.models['gb'] = XGBClassifier(n_estimators=50, max_depth=5, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='mlogloss')
        self.models['technical'] = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
            ('lr', LogisticRegression(max_iter=200, random_state=42))
        ], voting='soft')
        logger.info(f"AI models initialized on device: {self.device}")

    async def _ensure_model_trained(self, df: pd.DataFrame):
        if not self.is_trained and not self.is_training:
            self.is_training = True
            try:
                logger.info("Initial model training required. Starting training...")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.train, df)
                self._save_models()
            finally:
                self.is_training = False

    async def analyze_market(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        df = self._calculate_technical_indicators(ohlcv_df)
        if df is None or df.empty:
            return None

        await self._ensure_model_trained(df)

        if not self.is_trained or any(p.symbol == self.symbol for p in open_positions):
            return None

        prediction = self._predict(df)
        signal = prediction['signal']
        confidence = prediction['confidence']

        logger.debug(f"AI prediction for {self.symbol}: signal={signal}, confidence={confidence:.2f}")

        if signal == 1 and confidence > self.confidence_threshold: # BUY
            return {'action': 'BUY', 'confidence': confidence}
        elif signal == -1 and confidence > self.confidence_threshold: # SELL
            return {'action': 'SELL', 'confidence': confidence}
        
        return None

    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        # Position management logic (e.g., closing on opposite signal) can be added here.
        # For now, we rely on the RiskManager and main bot loop for stop-loss and take-profit.
        return []

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df.dropna(inplace=True)
        return df

    def train(self, df: pd.DataFrame):
        logger.info("Starting AI model training...")
        df = df.copy()
        future_returns = df['close'].pct_change(periods=5).shift(-5)
        df['target'] = 0 # HOLD
        df.loc[future_returns > 0.005, 'target'] = 1 # BUY threshold
        df.loc[future_returns < -0.005, 'target'] = -1 # SELL threshold
        df.dropna(inplace=True)

        if len(df) < 100 or len(df['target'].unique()) < 3:
            logger.warning("Not enough data or class diversity to train AI model.")
            return

        X = df[self.feature_columns]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train non-neural models
        for name in ['gb', 'technical']:
            self.models[name].fit(X_train, y_train)
            score = self.models[name].score(X_test, y_test)
            logger.info(f"Model '{name}' trained with accuracy: {score:.2f}")

        # Train LSTM
        X_train_torch = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_train_torch = torch.tensor(y_train.values + 1, dtype=torch.long).to(self.device) # map -1,0,1 to 0,1,2
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.models['lstm'].parameters(), lr=0.001)
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = self.models['lstm'](X_train_torch)
            loss = criterion(outputs, y_train_torch)
            loss.backward()
            optimizer.step()
        logger.info("Model 'lstm' trained.")

        self.is_trained = True
        logger.info("AI model training complete.")

    def _predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        latest_features = df[self.feature_columns].iloc[-1:]
        
        # Non-neural predictions
        gb_pred_proba = self.models['gb'].predict_proba(latest_features)[0]
        tech_pred_proba = self.models['technical'].predict_proba(latest_features)[0]

        # LSTM prediction
        lstm_features = torch.tensor(latest_features.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            lstm_output = self.models['lstm'](lstm_features)
            lstm_pred_proba = F.softmax(lstm_output, dim=1).cpu().numpy()[0]

        # Ensemble probabilities (sell, hold, buy)
        ensemble_proba = (gb_pred_proba + tech_pred_proba + lstm_pred_proba) / 3
        final_signal_idx = np.argmax(ensemble_proba)
        confidence = np.max(ensemble_proba)

        reverse_map = {0: -1, 1: 0, 2: 1} # Array index to our target (-1 sell, 0 hold, 1 buy)
        final_signal = reverse_map[final_signal_idx]

        return {'signal': final_signal, 'confidence': confidence}

    def _save_models(self):
        os.makedirs(self.model_path, exist_ok=True)
        for name, model in self.models.items():
            try:
                if isinstance(model, nn.Module):
                    torch.save(model.state_dict(), f"{self.model_path}/{name}.pth")
                else:
                    with open(f"{self.model_path}/{name}.pkl", 'wb') as f:
                        pickle.dump(model, f)
            except Exception as e:
                logger.error(f"Error saving model {name}: {e}")
        logger.info("AI models saved.")

    def _load_models(self):
        if not os.path.exists(self.model_path):
            return
        for name, model in self.models.items():
            try:
                path = f"{self.model_path}/{name}.{'pth' if isinstance(model, nn.Module) else 'pkl'}"
                if not os.path.exists(path):
                    continue
                if isinstance(model, nn.Module):
                    model.load_state_dict(torch.load(path, map_location=self.device))
                else:
                    with open(path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                self.is_trained = True
            except Exception as e:
                logger.error(f"Error loading model {name}: {e}")
        if self.is_trained:
            logger.info("AI models loaded from disk.")


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

        # Buy signal: Fast MA crosses above Slow MA
        if not has_open_position and last_row['fast_ma'] > last_row['slow_ma'] and prev_row['fast_ma'] <= prev_row['slow_ma']:
            logger.info(f"Strategy: BUY signal for {self.symbol} at {last_row['close']:.2f}")
            return {'action': 'BUY'}

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

        # Close signal: Fast MA crosses below Slow MA
        if position_to_manage.side == 'BUY' and last_row['fast_ma'] < last_row['slow_ma'] and prev_row['fast_ma'] >= prev_row['slow_ma']:
            logger.info(f"Strategy: Closing BUY position {position_to_manage.id} for {position_to_manage.symbol}.")
            actions.append({'action': 'CLOSE', 'position_id': position_to_manage.id})
        
        return actions
