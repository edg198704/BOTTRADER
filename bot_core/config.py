from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ExchangeConfig(BaseModel):
    name: str = "MockExchange"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True

class StrategyConfig(BaseModel):
    name: str = "SimpleMACrossoverStrategy"
    symbol: str = "BTC/USDT"
    interval_seconds: int = 60
    trade_quantity: float = 0.001
    # AIEnsembleStrategy specific parameters
    feature_columns: List[str] = Field(default_factory=lambda: ['close', 'rsi', 'macd', 'volume'])
    buy_threshold: float = 0.002
    sell_threshold: float = -0.0015
    # SimpleMACrossoverStrategy specific parameters
    fast_ma_period: int = 10
    slow_ma_period: int = 20

class RiskManagementConfig(BaseModel):
    max_position_size_usd: float = 1000.0
    max_daily_loss_usd: float = 500.0
    max_open_positions: int = 5
    circuit_breaker_threshold: float = -0.10 # -10% portfolio drawdown

class DatabaseConfig(BaseModel):
    path: str = "position_ledger.db"

class BotConfig(BaseModel):
    exchange: ExchangeConfig
    strategy: StrategyConfig
    risk_management: RiskManagementConfig
    database: DatabaseConfig

    @validator('strategy')
    def validate_strategy_symbol(cls, v):
        if '/' not in v.symbol and 'USDT' not in v.symbol:
            # A simple check, can be made more robust
            raise ValueError('Strategy symbol should be a valid trading pair, e.g., BTC/USDT')
        return v
