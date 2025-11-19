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
    auto_tune: bool = True
    use_stacking: bool = True
    # Penalize confidence if models disagree (StdDev of probs * penalty)
    disagreement_penalty: float = 0.5

class MarketRegimeConfig(BaseModel):
    # Static Fallbacks
    trend_strength_threshold: float = 0.015
    volatility_multiplier: float = 1.5
    
    # Column Mappings
    trend_fast_ma_col: str = "sma_fast"
    trend_slow_ma_col: str = "sma_slow"
    volatility_col: str = "atr"
    rsi_col: str = "rsi"
    
    # --- ADX Filter Settings (New) ---
    use_adx_filter: bool = False
    adx_col: str = "adx"
    adx_threshold: float = 25.0
    
    # --- Adaptive Regime Settings ---
    use_dynamic_thresholds: bool = False
    dynamic_window: int = 500
    trend_percentile: float = 0.75 # Top 25% of absolute trend values define a trend
    volatility_percentile: float = 0.80 # Top 20% of volatility values define volatile regime
    
    # --- Dynamic Confidence Thresholds (Entry) ---
    bull_confidence_threshold: Optional[float] = None
    bear_confidence_threshold: Optional[float] = None
    volatile_confidence_threshold: Optional[float] = None
    sideways_confidence_threshold: Optional[float] = None

    # --- Dynamic Confidence Thresholds (Exit) ---
    bull_exit_threshold: Optional[float] = None
    bear_exit_threshold: Optional[float] = None
    volatile_exit_threshold: Optional[float] = None
    sideways_exit_threshold: Optional[float] = None

class AITrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 5
    validation_split: float = Field(0.15, ge=0.0, lt=1.0)
    min_precision_threshold: float = Field(0.5, ge=0.0, le=1.0)
    # --- Profitability Gates (New) ---
    min_profit_factor: float = 1.05 # Require positive expectancy (Gross Profit / Gross Loss)
    min_sharpe_ratio: float = 0.05 # Require slightly positive risk-adjusted return
    min_improvement_pct: float = 0.02 # Require 2% improvement over previous model to replace it
    
    # --- Hyperparameter Optimization ---
    auto_tune_models: bool = False
    n_iter_search: int = 10
    # --- Data Handling ---
    use_class_weighting: bool = True

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
    use_dynamic_labeling: bool = False
    labeling_atr_multiplier: float = 1.0
    use_triple_barrier: bool = False
    triple_barrier_tp_multiplier: float = 2.0
    triple_barrier_sl_multiplier: float = 2.0
    
    # --- Temporal Features ---
    lag_features: List[str] = [] # List of column names to generate lags for
    lag_depth: int = 0 # Number of past periods to include as features

class AIPerformanceConfig(BaseModel):
    enabled: bool = True
    window_size: int = 50
    min_accuracy: float = 0.45

class AIHyperparameters(BaseModel):
    xgboost: XGBoostConfig = XGBoostConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    lstm: AILSTMConfig = AILSTMConfig()
    attention: AIAttentionConfig = AIAttentionConfig()

class StrategyParamsBase(BaseModel):
    name: str

class SimpleMACrossoverStrategyParams(StrategyParamsBase):
    name: Literal["SimpleMACrossoverStrategy"]
    fast_ma_period: int
    slow_ma_period: int

class AIEnsembleStrategyParams(StrategyParamsBase):
    name: Literal["AIEnsembleStrategy"]
    feature_columns: List[str]
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    exit_confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    model_path: str
    use_regime_filter: bool
    retrain_interval_hours: int
    training_data_limit: int = 5000
    signal_cooldown_candles: int = 1
    inference_workers: int = 2
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
    # Fees for estimation if not returned by exchange
    maker_fee_pct: float = 0.001 # 0.1%
    taker_fee_pct: float = 0.001 # 0.1%

class ExecutionConfig(BaseModel):
    default_order_type: str
    limit_price_offset_pct: float
    order_fill_timeout_seconds: int
    use_order_chasing: bool
    chase_interval_seconds: int
    max_chase_attempts: int
    chase_aggressiveness_pct: float
    max_chase_slippage_pct: float = 0.02 # Max 2% deviation from initial price during chase
    execute_on_timeout: bool

class DataHandlerConfig(BaseModel):
    history_limit: int
    update_interval_multiplier: float

class StrategyConfig(BaseModel):
    symbols: List[str]
    interval_seconds: int
    timeframe: str
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
    threshold_pct: float = 0.005

class CorrelationConfig(BaseModel):
    enabled: bool = True
    max_correlation: float = 0.8
    penalty_factor: float = 0.5
    lookback_periods: int = 50

class BreakevenConfig(BaseModel):
    enabled: bool = False
    activation_pct: float = 0.015 # 1.5% profit triggers BE
    buffer_pct: float = 0.001 # Lock in 0.1% profit to cover fees

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
    
    # --- Confidence-Based Sizing ---
    confidence_scaling_factor: float = 0.0
    max_confidence_risk_multiplier: float = 1.0
    
    # --- Confidence-Based Reward Targeting ---
    confidence_rr_scaling_factor: float = 0.0
    max_confidence_rr_multiplier: float = 1.5

    # --- Kelly Criterion Sizing (New) ---
    use_kelly_criterion: bool = False
    kelly_fraction: float = 0.5 # Half-Kelly by default for safety

    regime_based_risk: RegimeBasedRiskConfig
    time_based_exit: TimeBasedExitConfig = TimeBasedExitConfig()
    correlation: CorrelationConfig = CorrelationConfig()
    breakeven: BreakevenConfig = BreakevenConfig()

class DatabaseConfig(BaseModel):
    path: str

class TelegramConfig(BaseModel):
    admin_chat_ids: List[int]
    bot_token: Optional[SecretStr] = Field(None, env="BOT_TELEGRAM_BOT_TOKEN")

class LoggingConfig(BaseModel):
    level: str
    file_path: str
    use_json: bool

class BacktestConfig(BaseModel):
    enabled: bool = False
    initial_balance: float = 10000.0
    maker_fee_pct: float = 0.001 # 0.1%
    taker_fee_pct: float = 0.001 # 0.1%
    slippage_pct: float = 0.0005 # 0.05% 
    model_path: str = "backtest_models"

class OptimizerConfig(BaseModel):
    enabled: bool = False
    interval_hours: int = 24
    lookback_trades: int = 100
    min_trades_for_adjustment: int = 10
    min_profit_factor: float = 1.0
    high_performance_pf: float = 1.5
    adjustment_step: float = 0.02
    max_threshold_cap: float = 0.90
    min_threshold_floor: float = 0.55

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
    backtest: BacktestConfig = BacktestConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
