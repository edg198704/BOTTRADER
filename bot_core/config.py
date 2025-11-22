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
    optimization_method: str = 'slsqp'
    disagreement_penalty: float = 0.5
    
    # --- Online Learning Settings ---
    use_regime_specific_weights: bool = True
    use_dynamic_weighting: bool = True
    dynamic_window: int = 25
    dynamic_smoothing_factor: float = 0.1
    adaptive_weight_learning_rate: float = 0.05
    learning_algorithm: Literal['momentum', 'exponentiated_gradient'] = 'exponentiated_gradient'

class MetaLabelingConfig(BaseModel):
    enabled: bool = True
    probability_threshold: float = 0.55
    use_primary_confidence_feature: bool = True
    n_estimators: int = 100
    max_depth: int = 5

class DriftDetectionConfig(BaseModel):
    enabled: bool = True
    contamination: float = 0.05
    confidence_penalty: float = 0.2
    block_trade: bool = False
    max_consecutive_anomalies: int = 12

class TechnicalGuardrailsConfig(BaseModel):
    enabled: bool = True
    rsi_overbought: float = 85.0
    rsi_oversold: float = 15.0
    min_volume_percentile: float = 0.10
    max_spread_pct: float = 0.002
    adx_min_strength: float = 15.0
    require_trend_alignment: bool = False

class MarketRegimeConfig(BaseModel):
    trend_strength_threshold: float = 0.015
    volatility_multiplier: float = 1.5
    trend_fast_ma_col: str = "sma_fast"
    trend_slow_ma_col: str = "sma_slow"
    volatility_col: str = "atr"
    rsi_col: str = "rsi"
    use_adx_filter: bool = False
    adx_col: str = "adx"
    adx_threshold: float = 25.0
    efficiency_period: int = 10
    efficiency_threshold: float = 0.3
    use_hurst_filter: bool = True
    hurst_window: int = 100
    hurst_mean_reversion_threshold: float = 0.45
    hurst_trending_threshold: float = 0.55
    use_dynamic_thresholds: bool = False
    dynamic_window: int = 500
    trend_percentile: float = 0.75
    volatility_percentile: float = 0.80
    regime_confirmation_window: int = 3
    bull_confidence_threshold: Optional[float] = None
    bear_confidence_threshold: Optional[float] = None
    volatile_confidence_threshold: Optional[float] = None
    sideways_confidence_threshold: Optional[float] = None
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
    cv_splits: int = 5
    min_precision_threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_profit_factor: float = 1.05
    min_sharpe_ratio: float = 0.05
    min_improvement_pct: float = 0.02
    use_probabilistic_sharpe: bool = True
    purge_overlap: bool = True
    auto_tune_models: bool = False
    n_iter_search: int = 10
    use_class_weighting: bool = True
    sample_weighting_mode: Literal['balanced', 'return_based', 'hybrid', 'none'] = 'balanced'
    calibration_method: str = 'isotonic'
    optimize_entry_threshold: bool = True

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
    lag_features: List[str] = []
    lag_depth: int = 0
    use_time_features: bool = False
    use_price_action_features: bool = False
    use_volatility_estimators: bool = False
    use_microstructure_features: bool = False
    use_frac_diff: bool = False
    frac_diff_d: float = 0.4
    frac_diff_thres: float = 1e-4
    use_leader_features: bool = False
    use_order_book_features: bool = False
    use_feature_selection: bool = True
    max_active_features: int = 20
    scaling_method: Literal['zscore', 'robust'] = 'zscore'

class AIPerformanceConfig(BaseModel):
    enabled: bool = True
    window_size: int = 50
    min_accuracy: float = 0.45
    auto_rollback: bool = True
    critical_accuracy_threshold: float = 0.30

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
    model_monitor_interval_seconds: int = 60
    training_data_limit: int = 5000
    signal_cooldown_candles: int = 1
    inference_workers: int = 2
    market_leader_symbol: Optional[str] = "BTC/USDT"
    features: AIFeatureEngineeringConfig
    training: AITrainingConfig = AITrainingConfig()
    hyperparameters: AIHyperparameters = AIHyperparameters()
    ensemble_weights: EnsembleWeightsConfig = EnsembleWeightsConfig()
    market_regime: MarketRegimeConfig = MarketRegimeConfig()
    performance: AIPerformanceConfig = AIPerformanceConfig()
    drift: DriftDetectionConfig = DriftDetectionConfig()
    meta_labeling: MetaLabelingConfig = MetaLabelingConfig()
    guardrails: TechnicalGuardrailsConfig = TechnicalGuardrailsConfig()

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
    maker_fee_pct: float = 0.001
    taker_fee_pct: float = 0.001

class ExecutionProfile(BaseModel):
    """Defines behavior for a specific execution urgency."""
    order_type: Literal['LIMIT', 'MARKET']
    limit_offset_pct: float  # Positive = Passive, Negative = Aggressive
    use_chasing: bool
    chase_interval_seconds: float
    max_chase_attempts: int
    chase_aggressiveness_pct: float
    max_slippage_pct: float
    execute_on_timeout: bool
    post_only: bool

class ExecutionConfig(BaseModel):
    """Dynamic Execution Configuration."""
    default_profile: str = "neutral"
    max_entry_spread_pct: float = 0.001
    max_impact_cost_pct: float = 0.005
    order_fill_timeout_seconds: int = 45
    
    # Profiles for different urgencies
    profiles: Dict[str, ExecutionProfile] = {
        "passive": ExecutionProfile(
            order_type="LIMIT", limit_offset_pct=0.0005, use_chasing=True, 
            chase_interval_seconds=10.0, max_chase_attempts=3, chase_aggressiveness_pct=0.0001, 
            max_slippage_pct=0.005, execute_on_timeout=False, post_only=True
        ),
        "neutral": ExecutionProfile(
            order_type="LIMIT", limit_offset_pct=0.0, use_chasing=True, 
            chase_interval_seconds=5.0, max_chase_attempts=5, chase_aggressiveness_pct=0.0002, 
            max_slippage_pct=0.01, execute_on_timeout=True, post_only=False
        ),
        "aggressive": ExecutionProfile(
            order_type="LIMIT", limit_offset_pct=-0.0005, use_chasing=True, 
            chase_interval_seconds=2.0, max_chase_attempts=5, chase_aggressiveness_pct=0.0005, 
            max_slippage_pct=0.02, execute_on_timeout=True, post_only=False
        ),
        "sniper": ExecutionProfile(
            order_type="MARKET", limit_offset_pct=0.0, use_chasing=False, 
            chase_interval_seconds=0.0, max_chase_attempts=0, chase_aggressiveness_pct=0.0, 
            max_slippage_pct=0.03, execute_on_timeout=False, post_only=False
        )
    }

    # Legacy fields for backward compatibility (mapped to 'neutral' profile internally if needed)
    default_order_type: str = "LIMIT"
    limit_price_offset_pct: float = 0.0
    use_order_chasing: bool = True
    chase_interval_seconds: int = 5
    max_chase_attempts: int = 3
    chase_aggressiveness_pct: float = 0.0002
    max_chase_slippage_pct: float = 0.01
    execute_on_timeout: bool = False
    post_only: bool = False

class DataHandlerConfig(BaseModel):
    history_limit: int
    update_interval_multiplier: float
    ticker_update_interval_seconds: float = 2.0
    order_book_depth: int = 20

class StrategyConfig(BaseModel):
    symbols: List[str]
    interval_seconds: int
    timeframe: str
    secondary_timeframes: List[str] = []
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
    activation_pct: float = 0.015
    buffer_pct: float = 0.001

class VaRConfig(BaseModel):
    enabled: bool = True
    confidence_level: float = 0.95
    lookback_periods: int = 100
    max_portfolio_var_pct: float = 0.02
    method: Literal['historical', 'parametric'] = 'historical'

class RiskManagementConfig(BaseModel):
    max_position_size_usd: float
    max_position_size_portfolio_pct: float = 1.0
    max_portfolio_risk_pct: float = 0.05
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
    stop_loss_type: Literal["ATR", "SWING"] = "ATR"
    swing_lookback: int = 20
    swing_buffer_atr_multiplier: float = 1.0
    use_atr_for_trailing: bool = False
    atr_trailing_multiplier: float = 2.0
    atr_column_name: str = "atr"
    confidence_scaling_factor: float = 0.0
    max_confidence_risk_multiplier: float = 1.0
    confidence_rr_scaling_factor: float = 0.0
    max_confidence_rr_multiplier: float = 1.5
    use_kelly_criterion: bool = False
    kelly_fraction: float = 0.5
    max_volume_participation_pct: float = 0.01
    volume_lookback_periods: int = 20
    max_consecutive_losses: int = 3
    consecutive_loss_cooldown_minutes: int = 60
    monitor_interval_seconds: float = 2.0
    regime_based_risk: RegimeBasedRiskConfig
    time_based_exit: TimeBasedExitConfig = TimeBasedExitConfig()
    correlation: CorrelationConfig = CorrelationConfig()
    breakeven: BreakevenConfig = BreakevenConfig()
    var: VaRConfig = VaRConfig()

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
    maker_fee_pct: float = 0.001
    taker_fee_pct: float = 0.001
    slippage_pct: float = 0.0005
    model_path: str = "backtest_models"

class ExecutionOptimizerConfig(BaseModel):
    enabled: bool = True
    target_fill_rate: float = 0.90
    max_slippage_tolerance: float = 0.001
    offset_adjustment_step: float = 0.0001
    chase_adjustment_step: float = 0.0001
    min_limit_offset: float = -0.001
    max_limit_offset: float = 0.005
    min_chase_aggro: float = 0.0
    max_chase_aggro: float = 0.005

class OptimizerConfig(BaseModel):
    enabled: bool = False
    interval_hours: int = 24
    lookback_trades: int = 100
    min_trades_for_adjustment: int = 10
    state_file_path: str = "optimizer_state.json"
    min_profit_factor: float = 1.0
    high_performance_pf: float = 1.5
    adjustment_step: float = 0.02
    max_threshold_cap: float = 0.90
    min_threshold_floor: float = 0.55
    optimize_risk_params: bool = True
    risk_adjustment_step: float = 0.1
    optimize_risk_sizing: bool = True
    target_kelly_fraction: float = 0.25
    min_risk_pct: float = 0.005
    max_risk_pct: float = 0.05
    risk_size_step: float = 0.0025
    min_reward_to_risk: float = 1.0
    max_reward_to_risk: float = 3.0
    min_atr_multiplier: float = 1.5
    max_atr_multiplier: float = 4.0
    execution: ExecutionOptimizerConfig = ExecutionOptimizerConfig()

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
