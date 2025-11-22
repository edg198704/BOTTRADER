from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Literal, Optional, Dict, Any, List, Union
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_UP

# --- Global Precision Settings ---
getcontext().prec = 28

ZERO = Decimal("0")
ONE = Decimal("1")

def to_decimal(value: Union[float, str, int, Decimal]) -> Decimal:
    """Safely converts a value to Decimal, handling float artifacts via string conversion."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)

class TradeSignal(BaseModel):
    """
    Standardized signal object passed from Strategy to Bot/Executor.
    Includes execution urgency to guide the Execution Engine.
    """
    symbol: str
    action: Literal['BUY', 'SELL']
    regime: Optional[str] = None
    confidence: float = 0.0
    urgency: Literal['passive', 'neutral', 'aggressive', 'sniper'] = 'neutral'
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
