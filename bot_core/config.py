from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ExchangeConfig(BaseModel):
    name: str = "MockExchange"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True

class AIStrategyConfig(BaseModel):
    feature_columns: List[str] = Field(default_factory=lambda: ['close', 'rsi', 'macd', 'volume'])
    confidence_threshold: float = 0.60
    model_path: str = "models/ensemble"
    use_regime_filter: bool = True
    use_ppo_agent: bool = False # Disabled by default as it's more experimental
    retrain_interval_hours: int = 24
    training_epochs: int = 10

class SimpleMAStrategyConfig(BaseModel):
    fast_ma_period: int = 10
    slow_ma_period: int = 20

class StrategyConfig(BaseModel):
    name: str = "AIEnsembleStrategy"
    symbol: str = "BTC/USDT"
    interval_seconds: int = 60
    ai_ensemble: AIStrategyConfig = Field(default_factory=AIStrategyConfig)
    simple_ma: SimpleMAStrategyConfig = Field(default_factory=SimpleMAStrategyConfig)

class RiskManagementConfig(BaseModel):
    max_position_size_usd: float = 1000.0
    max_daily_loss_usd: float = 500.0
    max_open_positions: int = 5
    circuit_breaker_threshold: float = -0.10 # -10% portfolio drawdown
    use_trailing_stop: bool = True
    atr_stop_multiplier: float = 2.0
    stop_loss_fallback_pct: float = 0.05
    risk_per_trade_pct: float = 0.01 # Risk 1% of portfolio equity per trade

class DatabaseConfig(BaseModel):
    path: str = "position_ledger.db"

class TelegramConfig(BaseModel):
    bot_token: Optional[str] = None
    admin_chat_ids: List[int] = Field(default_factory=list)

class BotConfig(BaseModel):
    initial_capital: float = 10000.0
    exchange: ExchangeConfig
    strategy: StrategyConfig
    risk_management: RiskManagementConfig
    database: DatabaseConfig
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)

    @validator('strategy')
    def validate_strategy_symbol(cls, v):
        if '/' not in v.symbol:
            raise ValueError('Strategy symbol should be a valid trading pair, e.g., BTC/USDT')
        return v
