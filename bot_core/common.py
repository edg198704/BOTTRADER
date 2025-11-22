from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Literal, Optional, Dict, Any, Union
from decimal import Decimal, getcontext, ROUND_DOWN, InvalidOperation

# --- Global Precision Settings ---
# Set global precision to 50 places (Quantum Standard)
getcontext().prec = 50
getcontext().rounding = ROUND_DOWN

# Type Alias for Financial Calculations
Dec = Decimal

ZERO = Dec("0")
ONE = Dec("1")

def setup_math_context():
    """Ensures the Decimal context is configured correctly for the process."""
    getcontext().prec = 50
    getcontext().rounding = ROUND_DOWN

setup_math_context()

def to_decimal(value: Union[float, str, int, Decimal, None]) -> Decimal:
    """
    Safely converts a value to Decimal, handling float artifacts via string conversion.
    Returns ZERO if input is None or invalid.
    """
    if value is None:
        return ZERO
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        # Convert to string first to avoid float precision artifacts
        # We limit to 20 decimal places to strip garbage precision from floats
        return Decimal(f"{value:.20f}".rstrip('0').rstrip('.'))
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return ZERO

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

class TradeExecutionResult(BaseModel):
    """
    Result of a trade initiation request.
    Since execution is async, 'status' indicates if the process started successfully.
    """
    symbol: str
    action: str
    quantity: Decimal
    price: Optional[Decimal]
    trade_id: str
    status: Literal['INITIATED', 'REJECTED', 'ERROR']
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True
