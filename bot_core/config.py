from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ExchangeConfig(BaseModel):
    name: str = "MockExchange"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True

class StrategyConfig(BaseModel):
    name: str = "AIEnsembleStrategy"
    symbol: str = "BTC/USDT"
    interval_seconds: int = 60
    trade_quantity: float = 0.001
    # AIEnsembleStrategy specific parameters
    feature_columns: List[str] = Field(default_factory=lambda: ['close', 'rsi', 'macd', 'volume'])
    confidence_threshold: float = 0.60
    model_path: str = "models/ensemble"
    # SimpleMACrossoverStrategy specific parameters
    fast_ma_period: int = 10
    slow_ma_period: int = 20

class RiskManagementConfig(BaseModel):
    max_position_size_usd: float = 1000.0
    max_daily_loss_usd: float = 500.0
    max_open_positions: int = 5
    circuit_breaker_threshold: float = -0.10 # -10% portfolio drawdown
    use_trailing_stop: bool = True
    atr_stop_multiplier: float = 2.0
    stop_loss_fallback_pct: float = 0.05

class DatabaseConfig(BaseModel):
    path: str = "position_ledger.db"

class BotConfig(BaseModel):
    exchange: ExchangeConfig
    strategy: StrategyConfig
    risk_management: RiskManagementConfig
    database: DatabaseConfig

    @validator('strategy')
    def validate_strategy_symbol(cls, v):
        if '/' not in v.symbol:
            raise ValueError('Strategy symbol should be a valid trading pair, e.g., BTC/USDT')
        return v
