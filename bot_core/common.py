from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
from typing import Literal, Optional, Dict, Any, List, Union
from decimal import Decimal, ROUND_HALF_UP, Context
from enum import Enum

# --- Financial Types ---

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"

class Arith:
    """Centralized arithmetic operations for financial precision."""
    CTX = Context(prec=28, rounding=ROUND_HALF_UP)

    @staticmethod
    def quantize(value: Union[Decimal, float, str], precision: Union[Decimal, float, str]) -> Decimal:
        """Quantizes a value to a specific precision (step size)."""
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        if not isinstance(precision, Decimal):
            precision = Decimal(str(precision))
        return value.quantize(precision, context=Arith.CTX)

    @staticmethod
    def decimal(value: Union[float, str, int, Decimal]) -> Decimal:
        """Safe conversion to Decimal."""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

# --- Data Models ---

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

    class Config:
        arbitrary_types_allowed = True

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
