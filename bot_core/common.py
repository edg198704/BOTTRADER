from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Literal, Optional, Dict, Any, List, Union

class TradeSignal(BaseModel):
    """
    Standardized signal object passed from Strategy to Bot/Executor.
    """
    symbol: str
    action: Literal['BUY', 'SELL']
    regime: Optional[str] = None
    confidence: float = 0.0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_name: str
    metadata: Dict[str, Any] = {}

class AIInferenceResult(BaseModel):
    """
    Strictly typed result from the AI Ensemble Learner.
    """
    action: Literal['buy', 'sell', 'hold']
    confidence: float
    model_version: str
    active_weights: Dict[str, float]
    top_features: Dict[str, float]
    metrics: Dict[str, Any]
    optimized_weights: Optional[Dict[str, float]] = None
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    optimized_threshold: Optional[float] = None
    individual_predictions: Dict[str, Any] = {}
    meta_probability: Optional[float] = None
