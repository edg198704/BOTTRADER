from pydantic import BaseModel, Field, SecretStr, validator
from typing import List, Optional, Dict, Any, Union, Literal

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
    auto_tune: bool = True  # If True, weights are learned from validation performance

class MarketRegimeConfig(BaseModel):
    trend_strength_threshold: float = 0.015
    volatility_multiplier: float = 1.5
    # Column aliases to use for detection (must match aliases in strategy.indicators)
    trend_fast_ma_col: str = "sma_fast"
    trend_slow_ma_col: str = "sma_slow"
    volatility_col: str = "atr"
    rsi_col: str = "rsi"

class AITrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 5
    validation_split: float = Field(0.15, ge=0.0, lt=1.0)
    min_precision_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum precision required on validation set to accept a new model.")

class AILSTMConfig(BaseModel):
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = Field(0.2, ge=0.0, lt=1.0)

class AIAttentionConfig(BaseModel):
    hidden_dim: int = 64
    num_layers: int = 2
    nhead: int = 4
    dropout: float = Field(0.2, ge=0.0, lt=1.0)

class AIFeatureEngineeringConfig(BaseModel):
    sequence_length: int
    labeling_horizon: int = 5
    labeling_threshold: float = 0.005
    normalization_window: int = 100
    # Dynamic Labeling Parameters (Legacy/Simple)
    use_dynamic_labeling: bool = False
    labeling_atr_multiplier: float = 1.0
    # Triple Barrier Labeling Parameters (Advanced)
    use_triple_barrier: bool = False
    triple_barrier_tp_multiplier: float = 2.0
    triple_barrier_sl_multiplier: float = 2.0

class AIPerformanceConfig(BaseModel):
    enabled: bool = True
    window_size: int = 50 # Number of past predictions to evaluate
    min_accuracy: float = 0.45 # Trigger retrain if accuracy drops below this

class AIHyperparameters(BaseModel):
    xgboost: XGBoostConfig = XGBoostConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    lstm: AILSTMConfig = AILSTMConfig()
    attention: AIAttentionConfig = AIAttentionConfig()

# --- Strategy-Specific Parameter Models ---

class StrategyParamsBase(BaseModel):
    """Base model for strategy-specific parameters."""
    name: str

class SimpleMACrossoverStrategyParams(StrategyParamsBase):
    name: Literal["SimpleMACrossoverStrategy"]
    fast_ma_period: int
    slow_ma_period: int

class AIEnsembleStrategyParams(StrategyParamsBase):
    name: Literal["AIEnsembleStrategy"]
    feature_columns: List[str]
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    model_path: str
    use_regime_filter: bool
    retrain_interval_hours: int
    training_data_limit: int = 5000

    # Nested, structured configs
    features: AIFeatureEngineeringConfig
    training: AITrainingConfig = AITrainingConfig()
    hyperparameters: AIHyperparameters = AIHyperparameters()
    ensemble_weights: EnsembleWeightsConfig = EnsembleWeightsConfig()
    market_regime: MarketRegimeConfig = MarketRegimeConfig()
    performance: AIPerformanceConfig = AIPerformanceConfig()

# --- Core Component Configs ---

class RetryConfig(BaseModel):
    max_attempts: int
    delay_seconds: int
    backoff_factor: int

class ExchangeConfig(BaseModel):
    name: str
    testnet: bool
    retry: RetryConfig
    api_key: Optional[SecretStr] = Field(None, env="BOT_EXCHANGE_API_KEY")
    api_secret: Optional[SecretStr] = Field(None, env="BOT_EXCHANGE_API_SECRET")

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
    symbols: List[str]
    interval_seconds: int
    timeframe: str
    # This list of dictionaries will directly drive the pandas-ta indicator calculation.
    # Each dict should be a valid input for pandas-ta, e.g., {"kind": "rsi", "length": 14}
    indicators: List[Dict[str, Any]]
    params: Union[AIEnsembleStrategyParams, SimpleMACrossoverStrategyParams] = Field(..., discriminator='name')

class RegimeRiskOverride(BaseModel):
    risk_per_trade_pct: Optional[float] = None
    atr_stop_multiplier: Optional[float] = None
    reward_to_risk_ratio: Optional[float] = None

class RegimeBasedRiskConfig(BaseModel):
    bull: RegimeRiskOverride = RegimeRiskOverride()
    bear: RegimeRiskOverride = RegimeRiskOverride()
    volatile: RegimeRiskOverride = RegimeRiskOverride()
    sideways: RegimeRiskOverride = RegimeRiskOverride()

class TimeBasedExitConfig(BaseModel):
    enabled: bool = False
    max_hold_time_hours: int = 24
    threshold_pct: float = 0.005 # If PnL % is below this after max time, close.

class CorrelationConfig(BaseModel):
    enabled: bool = True
    max_correlation: float = 0.8
    penalty_factor: float = 0.5 # Multiply position size by this if correlated
    lookback_periods: int = 50

class RiskManagementConfig(BaseModel):
    max_position_size_usd: float
    max_daily_loss_usd: float
    max_open_positions: int
    circuit_breaker_threshold: float
    close_positions_on_halt: bool = True
    use_trailing_stop: bool
    atr_stop_multiplier: float
    stop_loss_fallback_pct: float
    risk_per_trade_pct: float
    reward_to_risk_ratio: float
    trailing_stop_activation_pct: float
    trailing_stop_pct: float
    
    # Confidence-Based Sizing
    confidence_scaling_factor: float = 0.0 # Multiplier for the surplus confidence. 0.0 = disabled.
    max_confidence_risk_multiplier: float = 1.0 # Hard cap on the risk multiplier (e.g., 1.5x)

    regime_based_risk: RegimeBasedRiskConfig
    time_based_exit: TimeBasedExitConfig = TimeBasedExitConfig()
    correlation: CorrelationConfig = CorrelationConfig()

class DatabaseConfig(BaseModel):
    path: str

class TelegramConfig(BaseModel):
    admin_chat_ids: List[int]
    bot_token: Optional[SecretStr] = Field(None, env="BOT_TELEGRAM_BOT_TOKEN")

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
