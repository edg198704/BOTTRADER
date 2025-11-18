from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

# --- AI Strategy Sub-configs ---

class XGBoostConfig(BaseModel):
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8

class RandomForestConfig(BaseModel):
    n_estimators: int = 100
    max_depth: Optional[int] = 10
    min_samples_leaf: int = 5

class LogisticRegressionConfig(BaseModel):
    max_iter: int = 1000
    C: float = 1.0

class EnsembleWeightsConfig(BaseModel):
    xgboost: float = 0.3
    technical_ensemble: float = 0.3
    lstm: float = 0.2
    attention: float = 0.2

class MarketRegimeConfig(BaseModel):
    trend_strength_threshold: float = 0.015
    volatility_multiplier: float = 1.5

class AITrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 5
    validation_split: float = Field(0.15, ge=0.0, lt=1.0)

class AILSTMConfig(BaseModel):
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = Field(0.2, ge=0.0, lt=1.0)

class AIAttentionConfig(BaseModel):
    hidden_dim: int = 64
    num_layers: int = 2
    nhead: int = 4
    dropout: float = Field(0.2, ge=0.0, lt=1.0)

class AIStrategyConfig(BaseModel):
    feature_columns: List[str]
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    model_path: str
    use_regime_filter: bool
    retrain_interval_hours: int
    sequence_length: int
    training_data_limit: int = 5000
    labeling_horizon: int = 5
    labeling_threshold: float = 0.005

    # Nested model configs
    xgboost: XGBoostConfig = XGBoostConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    ensemble_weights: EnsembleWeightsConfig = EnsembleWeightsConfig()
    market_regime: MarketRegimeConfig = MarketRegimeConfig()
    training: AITrainingConfig = AITrainingConfig()
    lstm: AILSTMConfig = AILSTMConfig()
    attention: AIAttentionConfig = AIAttentionConfig()

# --- Simple Strategy Sub-configs ---

class SimpleMAConfig(BaseModel):
    fast_ma_period: int
    slow_ma_period: int

# --- Core Component Configs ---

class RetryConfig(BaseModel):
    max_attempts: int
    delay_seconds: int
    backoff_factor: int

class ExchangeConfig(BaseModel):
    name: str
    testnet: bool
    retry: RetryConfig
    api_key: Optional[str] = Field(None, env="BOT_EXCHANGE_API_KEY")
    api_secret: Optional[str] = Field(None, env="BOT_EXCHANGE_API_SECRET")

class ExecutionConfig(BaseModel):
    default_order_type: str
    limit_price_offset_pct: float
    order_fill_timeout_seconds: int
    use_order_chasing: bool
    chase_interval_seconds: int
    max_chase_attempts: int
    chase_aggressiveness_pct: float
    execute_on_timeout: bool

class DataHandlerConfig(BaseModel):
    history_limit: int
    update_interval_multiplier: float

class StrategyConfig(BaseModel):
    name: str
    symbols: List[str]
    interval_seconds: int
    timeframe: str
    ai_ensemble: AIStrategyConfig
    simple_ma: SimpleMAConfig

class RegimeRiskOverride(BaseModel):
    risk_per_trade_pct: Optional[float] = None
    atr_stop_multiplier: Optional[float] = None
    reward_to_risk_ratio: Optional[float] = None

class RegimeBasedRiskConfig(BaseModel):
    bull: RegimeRiskOverride = RegimeRiskOverride()
    bear: RegimeRiskOverride = RegimeRiskOverride()
    volatile: RegimeRiskOverride = RegimeRiskOverride()
    sideways: RegimeRiskOverride = RegimeRiskOverride()

class RiskManagementConfig(BaseModel):
    max_position_size_usd: float
    max_daily_loss_usd: float
    max_open_positions: int
    circuit_breaker_threshold: float
    use_trailing_stop: bool
    atr_stop_multiplier: float
    stop_loss_fallback_pct: float
    risk_per_trade_pct: float
    reward_to_risk_ratio: float
    trailing_stop_activation_pct: float
    trailing_stop_pct: float
    regime_based_risk: RegimeBasedRiskConfig

class DatabaseConfig(BaseModel):
    path: str

class TelegramConfig(BaseModel):
    admin_chat_ids: List[int]
    bot_token: Optional[str] = Field(None, env="BOT_TELEGRAM_BOT_TOKEN")

class LoggingConfig(BaseModel):
    level: str
    file_path: str
    use_json: bool

# --- Top-Level Bot Config ---

class BotConfig(BaseModel):
    initial_capital: float
    exchange: ExchangeConfig
    execution: ExecutionConfig
    data_handler: DataHandlerConfig
    strategy: StrategyConfig
    risk_management: RiskManagementConfig
    database: DatabaseConfig
    telegram: TelegramConfig
    logging: LoggingConfig
