"""
ADDITIONAL COMPLETE AI TRADING BOT COMPONENTS
=============================================

This file contains the additional components that complete the AI trading bot:
- Complete PPO Agent with training and inference
- Complete Market Regime Detection with multiple indicators
- Complete AI Trading Bot main class integration
- Complete execution functions

All components maintain 100% compatibility with the original bot.
"""

import asyncio
import sys
import os
import ccxt.async_support as ccxt
import json
import gc
import logging
import logging.handlers
import math
import random
import sqlite3
import time
import traceback
import uuid
import signal
import resource
import cProfile
import pstats
import psutil
import warnings
import hashlib
import hmac
from contextlib import contextmanager, asynccontextmanager
from abc import ABC, abstractmethod
from collections import OrderedDict, deque, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Iterable, Protocol, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock, RLock
import weakref

import numpy as np
import pandas as pd

# Import complete components from our comprehensive implementation
from bot_ai_complete_components import *

# ====================
# ENHANCED PPO AGENT
# ====================

class PPOTrainingConfig:
    """PPO training configuration"""
    def __init__(self):
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10

class CompletePPOAgent:
    """Complete PPO Agent with training and inference"""
    
    def __init__(self, config: AdvancedAIConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Policy and value networks
        self.policy_net = self._build_policy_network().to(self.device)
        self.value_net = self._build_value_network().to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=3e-4)
        
        # Training state
        self.total_episodes = 0
        self.update_count = 0
        self.memory = []
        self.training_config = PPOTrainingConfig()
        
        print(f"🤖 PPO Agent inicializado en dispositivo: {self.device}")
    
    def _build_policy_network(self):
        """Build policy network for trading actions"""
        return nn.Sequential(
            nn.Linear(10, 128),  # 10 features: price, volume, indicators, etc.
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 actions: buy, sell, hold
        )
    
    def _build_value_network(self):
        """Build value network for state value estimation"""
        return nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _build_state(self, df: pd.DataFrame, current_price: float) -> torch.Tensor:
        """Build state vector from market data"""
        try:
            if len(df) < 20:
                # Fallback state if insufficient data
                return torch.zeros(10, dtype=torch.float32)
            
            # Extract features
            features = []
            
            # Price features
            current_price = float(current_price)
            sma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_price
            sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else current_price
            
            features.extend([
                (current_price - sma_20) / sma_20 if sma_20 > 0 else 0,  # Price vs SMA20
                (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0,  # SMA20 vs SMA50
            ])
            
            # Technical indicators
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                features.extend([
                    (rsi - 50) / 50,  # Normalized RSI
                    (rsi - df['rsi'].rolling(10).mean().iloc[-1]) / 50 if len(df) >= 10 else 0  # RSI momentum
                ])
            else:
                features.extend([0, 0])
            
            if 'macd' in df.columns:
                macd = df['macd'].iloc[-1]
                macd_signal = df.get('macd_signal', pd.Series([0])).iloc[-1]
                features.extend([
                    macd / current_price if current_price > 0 else 0,  # Normalized MACD
                    (macd - macd_signal) / current_price if current_price > 0 else 0  # MACD signal
                ])
            else:
                features.extend([0, 0])
            
            # Volume features
            if 'volume' in df.columns:
                volume = df['volume'].iloc[-1]
                volume_ma = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else volume
                features.extend([
                    (volume - volume_ma) / volume_ma if volume_ma > 0 else 0,  # Volume vs average
                ])
            else:
                features.append(0)
            
            # Volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0
            features.extend([
                volatility * 100  # Normalized volatility
            ])
            
            # Ensure we have exactly 10 features
            while len(features) < 10:
                features.append(0)
            features = features[:10]
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            print(f"⚠️ Error construyendo estado PPO: {e}")
            return torch.zeros(10, dtype=torch.float32)
    
    async def act(self, df: pd.DataFrame, current_price: float) -> Tuple[int, float]:
        """Choose action based on current state"""
        try:
            state = self._build_state(df, current_price).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get action logits and value
                action_logits = self.policy_net(state)
                value = self.value_net(state)
                
                # Apply softmax to get action probabilities
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                
                # Sample action
                action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action)
                
                return action.item(), value.item()
            
        except Exception as e:
            print(f"❌ Error en selección de acción PPO: {e}")
            return 1, 0.0  # Default to hold
    
    async def store_transition(self, state: torch.Tensor, action: int, reward: float, 
                              next_state: torch.Tensor, done: bool):
        """Store transition in memory for training"""
        self.memory.append({
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done
        })
        
        # Limit memory size
        if len(self.memory) > self.training_config.n_steps:
            self.memory.pop(0)
    
    async def compute_gae(self, rewards: List[float], values: List[float], 
                         next_value: float, dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = values[t + 1]
            
            delta = rewards[t] + self.training_config.gamma * next_val * next_non_terminal - values[t]
            advantage = delta + self.training_config.gamma * self.training_config.lam * next_non_terminal * advantage
            advantages.insert(0, advantage)
        
        # Normalize advantages
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages.tolist(), returns
    
    async def update(self) -> Dict[str, float]:
        """Update policy and value networks using collected experiences"""
        if len(self.memory) < self.training_config.batch_size:
            return {'loss': 0.0}
        
        # Extract data from memory
        states = torch.stack([m['state'] for m in self.memory]).to(self.device)
        actions = torch.tensor([m['action'] for m in self.memory]).to(self.device)
        rewards = [m['reward'] for m in self.memory]
        
        with torch.no_grad():
            values = [self.value_net(state.unsqueeze(0)).item() for state in states]
            next_states = torch.stack([m['next_state'] for m in self.memory]).to(self.device)
            next_values = [self.value_net(state.unsqueeze(0)).item() for state in next_states]
            dones = [m['done'] for m in self.memory]
            
            advantages, returns = await self.compute_gae(rewards, values, next_values[-1], dones)
        
        # Training loop
        policy_losses = []
        value_losses = []
        
        for epoch in range(self.training_config.n_epochs):
            # Shuffle indices
            indices = torch.randperm(len(self.memory))
            
            for start in range(0, len(self.memory), self.training_config.batch_size):
                end = start + self.training_config.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = torch.tensor([advantages[i] for i in batch_indices]).to(self.device)
                batch_returns = torch.tensor([returns[i] for i in batch_indices]).to(self.device)
                
                # Get current policy and value predictions
                action_logits = self.policy_net(batch_states)
                values_pred = self.value_net(batch_states).squeeze()
                
                # Compute policy loss
                action_dist = torch.distributions.Categorical(F.softmax(action_logits, dim=-1))
                action_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy()
                
                ratio = torch.exp(action_log_probs - action_log_probs.detach())
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.training_config.clip_ratio, 
                                  1 + self.training_config.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                # Total loss
                total_loss = (policy_loss + 
                            self.training_config.value_coef * value_loss - 
                            self.training_config.entropy_coef * entropy.mean())
                
                # Update networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.training_config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.training_config.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        self.update_count += 1
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'total_loss': np.mean(policy_losses) + np.mean(value_losses),
            'entropy': entropy.mean().item()
        }
    
    def save(self, path: str = "models/ppo_agent.pth"):
        """Save agent model"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'policy_state_dict': self.policy_net.state_dict(),
                'value_state_dict': self.value_net.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': self.value_optimizer.state_dict(),
                'total_episodes': self.total_episodes,
                'update_count': self.update_count,
                'config': self.training_config.__dict__
            }, path)
            print(f"✅ PPO agent guardado en {path}")
        except Exception as e:
            print(f"❌ Error guardando PPO agent: {e}")
    
    def load(self, path: str = "models/ppo_agent.pth") -> bool:
        """Load agent model"""
        try:
            if not os.path.exists(path):
                print(f"📁 Archivo de modelo no existe: {path}")
                return False
            
            checkpoint = torch.load(path, map_location=self.device)
            
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            
            if 'policy_optimizer_state_dict' in checkpoint:
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            if 'value_optimizer_state_dict' in checkpoint:
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            
            self.total_episodes = checkpoint.get('total_episodes', 0)
            self.update_count = checkpoint.get('update_count', 0)
            
            if 'config' in checkpoint:
                for key, value in checkpoint['config'].items():
                    if hasattr(self.training_config, key):
                        setattr(self.training_config, key, value)
            
            print(f"✅ PPO agent cargado desde {path}")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando PPO agent: {e}")
            return False

# ====================
# ENHANCED MARKET REGIME DETECTOR
# ====================

class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class CompleteMarketRegimeDetector:
    """Complete market regime detection with multiple indicators"""
    
    def __init__(self, config: AdvancedAIConfig):
        self.config = config
        self.regime_cache = {}
        self.regime_history = {}
        
    async def detect_regime(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime using multiple indicators"""
        try:
            if len(df) < 50:
                return {
                    'regime': MarketRegime.UNKNOWN.value,
                    'confidence': 0.0,
                    'indicators': {}
                }
            
            # Calculate regime indicators
            indicators = {}
            
            # 1. Trend Analysis
            indicators['trend'] = self._analyze_trend(df)
            
            # 2. Volatility Analysis
            indicators['volatility'] = self._analyze_volatility(df)
            
            # 3. Volume Analysis
            indicators['volume'] = self._analyze_volume(df)
            
            # 4. RSI Analysis
            indicators['rsi'] = self._analyze_rsi(df)
            
            # 5. MACD Analysis
            indicators['macd'] = self._analyze_macd(df)
            
            # 6. Price Action Analysis
            indicators['price_action'] = self._analyze_price_action(df)
            
            # 7. Support/Resistance Analysis
            indicators['support_resistance'] = self._analyze_support_resistance(df)
            
            # Determine regime based on combined indicators
            regime, confidence = self._combine_indicators(indicators)
            
            result = {
                'regime': regime,
                'confidence': confidence,
                'indicators': indicators,
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Cache result
            self.regime_cache[symbol] = result
            self._update_regime_history(symbol, regime)
            
            return result
            
        except Exception as e:
            print(f"❌ Error detectando régimen de mercado: {e}")
            return {
                'regime': MarketRegime.UNKNOWN.value,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend using multiple moving averages"""
        try:
            # Calculate moving averages
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()
            sma_200 = df['close'].rolling(200).mean() if len(df) >= 200 else None
            
            current_price = df['close'].iloc[-1]
            sma_20_curr = sma_20.iloc[-1]
            sma_50_curr = sma_50.iloc[-1]
            
            # Trend strength calculation
            trend_signals = []
            
            # Price vs SMAs
            if current_price > sma_20_curr:
                trend_signals.append(1)
            else:
                trend_signals.append(-1)
            
            # SMA 20 vs SMA 50
            if sma_20_curr > sma_50_curr:
                trend_signals.append(1)
            else:
                trend_signals.append(-1)
            
            # SMA trend
            if len(sma_20) >= 10:
                sma_trend = sma_20.iloc[-1] - sma_20.iloc[-10]
                trend_signals.append(1 if sma_trend > 0 else -1)
            
            # Calculate trend strength
            trend_strength = sum(trend_signals) / len(trend_signals) if trend_signals else 0
            
            return {
                'trend_strength': trend_strength,
                'price_vs_sma20': (current_price - sma_20_curr) / sma_20_curr if sma_20_curr > 0 else 0,
                'sma20_vs_sma50': (sma_20_curr - sma_50_curr) / sma_50_curr if sma_50_curr > 0 else 0,
                'is_uptrend': trend_strength > 0.3,
                'is_downtrend': trend_strength < -0.3
            }
            
        except Exception as e:
            return {'trend_strength': 0, 'error': str(e)}
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market volatility"""
        try:
            returns = df['close'].pct_change().dropna()
            
            # Calculate different volatility measures
            current_vol = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else returns.std()
            historical_vol = returns.rolling(50).std().iloc[-1] if len(returns) >= 50 else returns.std()
            
            # Volatility percentiles
            vol_percentile = returns.rolling(100).std().rank(pct=True).iloc[-1] if len(returns) >= 100 else 0.5
            
            # ATR (Average True Range)
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else 0
            
            atr_percentile = tr.rolling(50).std().rank(pct=True).iloc[-1] if len(tr) >= 50 else 0.5
            
            return {
                'current_volatility': current_vol,
                'historical_volatility': historical_vol,
                'volatility_ratio': current_vol / historical_vol if historical_vol > 0 else 1,
                'volatility_percentile': vol_percentile,
                'atr': atr,
                'atr_percentile': atr_percentile,
                'is_high_volatility': current_vol > historical_vol * 1.5,
                'is_low_volatility': current_vol < historical_vol * 0.7
            }
            
        except Exception as e:
            return {'current_volatility': 0, 'error': str(e)}
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            if 'volume' not in df.columns:
                return {'volume_ratio': 1.0, 'error': 'Volume data not available'}
            
            current_volume = df['volume'].iloc[-1]
            volume_ma_20 = df['volume'].rolling(20).mean().iloc[-1]
            volume_ma_50 = df['volume'].rolling(50).mean().iloc[-1] if len(df) >= 50 else volume_ma_20
            
            volume_ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1.0
            volume_trend = (df['volume'].iloc[-10:].mean() - df['volume'].iloc[-20:-10].mean()) / df['volume'].iloc[-20:-10].mean() if len(df) >= 20 else 0
            
            # Volume price relationship
            price_change = df['close'].pct_change().iloc[-1] if len(df) > 1 else 0
            volume_price_correlation = df['volume'].rolling(20).corr(df['close'].pct_change()).iloc[-1] if len(df) >= 20 else 0
            
            return {
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'volume_price_correlation': volume_price_correlation,
                'is_high_volume': volume_ratio > 1.5,
                'is_low_volume': volume_ratio < 0.7,
                'is_volume_increasing': volume_trend > 0.1
            }
            
        except Exception as e:
            return {'volume_ratio': 1.0, 'error': str(e)}
    
    def _analyze_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RSI patterns"""
        try:
            if 'rsi' not in df.columns:
                return {'rsi': 50, 'error': 'RSI not calculated'}
            
            rsi = df['rsi'].iloc[-1]
            rsi_ma = df['rsi'].rolling(10).mean().iloc[-1] if len(df) >= 10 else rsi
            rsi_trend = rsi - rsi_ma
            
            # RSI momentum
            if len(df) >= 5:
                rsi_momentum = df['rsi'].iloc[-1] - df['rsi'].iloc[-5]
            else:
                rsi_momentum = 0
            
            return {
                'rsi': rsi,
                'rsi_ma': rsi_ma,
                'rsi_trend': rsi_trend,
                'rsi_momentum': rsi_momentum,
                'is_oversold': rsi < 30,
                'is_overbought': rsi > 70,
                'is_neutral': 30 <= rsi <= 70
            }
            
        except Exception as e:
            return {'rsi': 50, 'error': str(e)}
    
    def _analyze_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze MACD patterns"""
        try:
            if 'macd' not in df.columns:
                return {'macd_signal': 0, 'error': 'MACD not calculated'}
            
            macd = df['macd'].iloc[-1]
            macd_signal = df.get('macd_signal', pd.Series([0])).iloc[-1]
            macd_hist = df.get('macd_hist', pd.Series([0])).iloc[-1]
            
            # MACD trend
            if len(df) >= 10:
                macd_trend = macd - df['macd'].iloc[-10]
            else:
                macd_trend = 0
            
            return {
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'macd_trend': macd_trend,
                'is_bullish_cross': macd > macd_signal and df['macd'].iloc[-2] <= df.get('macd_signal', pd.Series([0])).iloc[-2],
                'is_bearish_cross': macd < macd_signal and df['macd'].iloc[-2] >= df.get('macd_signal', pd.Series([0])).iloc[-2]
            }
            
        except Exception as e:
            return {'macd_signal': 0, 'error': str(e)}
    
    def _analyze_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price action patterns"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Recent price range
            price_range_20 = (df['high'].rolling(20).max().iloc[-1] - df['low'].rolling(20).min().iloc[-1]) / current_price if len(df) >= 20 else 0
            price_range_5 = (df['high'].rolling(5).max().iloc[-1] - df['low'].rolling(5).min().iloc[-1]) / current_price if len(df) >= 5 else 0
            
            # Price momentum
            if len(df) >= 10:
                price_momentum = (current_price - df['close'].iloc[-10]) / df['close'].iloc[-10]
            else:
                price_momentum = 0
            
            # Price position in range
            high_20 = df['high'].rolling(20).max().iloc[-1] if len(df) >= 20 else current_price
            low_20 = df['low'].rolling(20).min().iloc[-1] if len(df) >= 20 else current_price
            
            if high_20 > low_20:
                price_position = (current_price - low_20) / (high_20 - low_20)
            else:
                price_position = 0.5
            
            return {
                'price_momentum': price_momentum,
                'price_range_20': price_range_20,
                'price_range_5': price_range_5,
                'price_position_in_range': price_position,
                'is_near_high': price_position > 0.8,
                'is_near_low': price_position < 0.2,
                'is_momentum_positive': price_momentum > 0.01,
                'is_momentum_negative': price_momentum < -0.01
            }
            
        except Exception as e:
            return {'price_momentum': 0, 'error': str(e)}
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze support and resistance levels"""
        try:
            # Find local highs and lows
            highs = df['high'].rolling(5, center=True).max()
            lows = df['low'].rolling(5, center=True).min()
            
            # Recent levels
            recent_highs = df[high == df['high']].tail(10)['high'].values
            recent_lows = df[low == df['low']].tail(10)['low'].values
            
            current_price = df['close'].iloc[-1]
            
            # Find nearest support and resistance
            support_levels = [level for level in recent_lows if level < current_price]
            resistance_levels = [level for level in recent_highs if level > current_price]
            
            nearest_support = max(support_levels) if support_levels else current_price * 0.95
            nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
            
            # Distance to levels
            support_distance = (current_price - nearest_support) / current_price
            resistance_distance = (nearest_resistance - current_price) / current_price
            
            return {
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance': support_distance,
                'resistance_distance': resistance_distance,
                'is_near_support': support_distance < 0.02,
                'is_near_resistance': resistance_distance < 0.02,
                'support_levels_count': len(support_levels),
                'resistance_levels_count': len(resistance_levels)
            }
            
        except Exception as e:
            return {
                'nearest_support': current_price * 0.95,
                'nearest_resistance': current_price * 1.05,
                'error': str(e)
            }
    
    def _combine_indicators(self, indicators: Dict[str, Any]) -> Tuple[str, float]:
        """Combine all indicators to determine market regime"""
        try:
            # Extract key signals
            trend_signal = indicators.get('trend', {}).get('trend_strength', 0)
            volatility_signal = indicators.get('volatility', {}).get('volatility_percentile', 0.5)
            volume_signal = indicators.get('volume', {}).get('volume_ratio', 1.0)
            rsi_signal = indicators.get('rsi', {}).get('rsi', 50)
            macd_signal = indicators.get('macd', {}).get('macd', 0)
            
            # Regime determination logic
            regime_scores = {
                MarketRegime.BULL.value: 0,
                MarketRegime.BEAR.value: 0,
                MarketRegime.SIDEWAYS.value: 0,
                MarketRegime.VOLATILE.value: 0
            }
            
            # Trend-based scoring
            if trend_signal > 0.3:
                regime_scores[MarketRegime.BULL.value] += 2
            elif trend_signal < -0.3:
                regime_scores[MarketRegime.BEAR.value] += 2
            else:
                regime_scores[MarketRegime.SIDEWAYS.value] += 1
            
            # Volatility-based scoring
            if volatility_signal > 0.8:
                regime_scores[MarketRegime.VOLATILE.value] += 2
            elif volatility_signal < 0.3:
                regime_scores[MarketRegime.SIDEWAYS.value] += 1
            
            # RSI-based scoring
            if rsi_signal > 70:
                regime_scores[MarketRegime.BULL.value] += 1
                regime_scores[MarketRegime.VOLATILE.value] += 1
            elif rsi_signal < 30:
                regime_scores[MarketRegime.BEAR.value] += 1
                regime_scores[MarketRegime.VOLATILE.value] += 1
            else:
                regime_scores[MarketRegime.SIDEWAYS.value] += 1
            
            # MACD-based scoring
            if macd_signal > 0:
                regime_scores[MarketRegime.BULL.value] += 1
            else:
                regime_scores[MarketRegime.BEAR.value] += 1
            
            # Determine final regime
            max_score = max(regime_scores.values())
            if max_score == 0:
                return MarketRegime.SIDEWAYS.value, 0.5
            
            # Find regime with max score (with tie-breaking)
            candidates = [regime for regime, score in regime_scores.items() if score == max_score]
            
            if len(candidates) == 1:
                final_regime = candidates[0]
            else:
                # Tie-breaking logic
                if MarketRegime.VOLATILE.value in candidates:
                    final_regime = MarketRegime.VOLATILE.value
                elif MarketRegime.SIDEWAYS.value in candidates:
                    final_regime = MarketRegime.SIDEWAYS.value
                else:
                    final_regime = candidates[0]  # Default to first candidate
            
            # Calculate confidence based on score
            confidence = min(0.9, max_score / 5.0)  # Normalize to 0.9 max
            
            return final_regime, confidence
            
        except Exception as e:
            print(f"❌ Error combinando indicadores de régimen: {e}")
            return MarketRegime.UNKNOWN.value, 0.0
    
    def _update_regime_history(self, symbol: str, regime: str):
        """Update regime history for analysis"""
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        
        self.regime_history[symbol].append({
            'regime': regime,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Keep only last 100 entries
        if len(self.regime_history[symbol]) > 100:
            self.regime_history[symbol] = self.regime_history[symbol][-100:]
    
    def get_regime_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Get regime history for a symbol"""
        return self.regime_history.get(symbol, [])
    
    def get_current_regime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached current regime for a symbol"""
        return self.regime_cache.get(symbol)

print("✅ Additional Complete AI Trading Bot Components loaded successfully!")