from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ExchangeConfig(BaseModel):
    name: str = "MockExchange"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True

class XGBoostConfig(BaseModel):
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8

class RandomForestConfig(BaseModel):
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_leaf: int = 5

class LogisticRegressionConfig(BaseModel):
    max_iter: int = 200
    C: float = 1.0

class EnsembleWeights(BaseModel):
    xgboost: float = 0.3
    technical_ensemble: float = 0.2
    lstm: float = 0.25
    attention: float = 0.25

class MarketRegimeConfig(BaseModel):
    trend_strength_threshold: float = 0.01
    volatility_multiplier: float = 1.5

class AIStrategyConfig(BaseModel):
    feature_columns: List[str] = Field(default_factory=lambda: ['close', 'rsi', 'macd', 'volume'])
    confidence_threshold: float = 0.60
    model_path: str = "models/ensemble"
    use_regime_filter: bool = True
    use_ppo_agent: bool = False # Disabled by default as it's more experimental
    retrain_interval_hours: int = 24
    training_epochs: int = 10

    # Configurable model parameters
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    logistic_regression: LogisticRegressionConfig = Field(default_factory=LogisticRegressionConfig)
    ensemble_weights: EnsembleWeights = Field(default_factory=EnsembleWeights)
    market_regime: MarketRegimeConfig = Field(default_factory=MarketRegimeConfig)

class SimpleMAStrategyConfig(BaseModel):
    fast_ma_period: int = 10
    slow_ma_period: int = 20

class StrategyConfig(BaseModel):
    name: str = "AIEnsembleStrategy"
    symbols: List[str] = Field(default_factory=lambda: ["BTC/USDT"])
    interval_seconds: int = 60
    ai_ensemble: AIStrategyConfig = Field(default_factory=AIStrategyConfig)
    simple_ma: SimpleMAStrategyConfig = Field(default_factory=SimpleMAStrategyConfig)

class RiskManagementConfig(BaseModel):
    max_position_size_usd: float = 1000.0
    max_daily_loss_usd: float = 500.0
    max_open_positions: int = 5
    circuit_breaker_threshold: float = -0.10 # -10% portfolio drawdown
    use_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.02 # e.g., 2% profit before trailing
    trailing_stop_pct: float = 0.015 # e.g., trail by 1.5%
    atr_stop_multiplier: float = 2.0
    stop_loss_fallback_pct: float = 0.05
    risk_per_trade_pct: float = 0.01 # Risk 1% of portfolio equity per trade

class DatabaseConfig(BaseModel):
    path: str = "position_ledger.db"

class TelegramConfig(BaseModel):
    bot_token: Optional[str] = None
    admin_chat_ids: List[int] = Field(default_factory=list)

class LoggingConfig(BaseModel):
    level: str = "INFO"
    file_path: Optional[str] = "logs/trading_bot.log"
    use_json: bool = True

class BotConfig(BaseModel):
    initial_capital: float = 10000.0
    exchange: ExchangeConfig
    strategy: StrategyConfig
    risk_management: RiskManagementConfig
    database: DatabaseConfig
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @validator('strategy')
    def validate_strategy_symbols(cls, v):
        if not v.symbols:
            raise ValueError('Strategy symbols list cannot be empty')
        for symbol in v.symbols:
            if '/' not in symbol:
                raise ValueError(f'Strategy symbol "{symbol}" should be a valid trading pair, e.g., BTC/USDT')
        return v
