"""API schemas for Pythia Signals API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class IngestRow(BaseModel):
    date: str
    ticker: str
    features: Dict[str, float]


class IngestRequest(BaseModel):
    rows: List[IngestRow]


class PredictResponse(BaseModel):
    ticker: str
    horizon: str = "1d"
    prob_up: Optional[float] = None
    signal: Optional[str] = None
    last_close: float
    predicted_return: Optional[float] = None
    model: str = "gradient_boosting"
    task: str = "classifier"


class AllocateRequest(BaseModel):
    budget: float = Field(gt=0)
    tickers: Optional[List[str]] = None
    k_top: int = 5
    threshold: float = 0.55


class TradePlan(BaseModel):
    ticker: str
    shares: int
    reference_price: float


class AllocateResponse(BaseModel):
    budget: float
    used: float
    plan: List[TradePlan]


class SimRequest(BaseModel):
    start: str
    end: str
    budget: float = Field(gt=0)
    threshold: float = 0.55
    k_top: int = 5


class SimResponse(BaseModel):
    cagr: float
    max_drawdown: float
    total_return: float


# Multi-model schemas

class ModelInfo(BaseModel):
    """Information about an available model."""
    model: str
    task: str
    trained: bool
    has_metrics: bool
    metrics: Optional[Dict[str, Any]] = None


class ModelPrediction(BaseModel):
    """Single model's prediction."""
    model: str
    task: str
    prob_up: Optional[float] = None
    signal: Optional[str] = None
    predicted_return: Optional[float] = None
    confidence: Optional[float] = None


class MultiModelPredictRequest(BaseModel):
    """Request prediction from specific model(s)."""
    ticker: str
    models: Optional[List[str]] = None
    tasks: Optional[List[str]] = None
    horizon: str = "1d"


class MultiModelPredictResponse(BaseModel):
    """Response with predictions from multiple models."""
    ticker: str
    last_close: float
    predictions: List[ModelPrediction]
    consensus_signal: Optional[str] = None


class ModelComparisonResponse(BaseModel):
    """Model comparison results."""
    task: str
    metric: str
    models: Dict[str, Dict[str, Any]]
    ranking: Dict[str, int]


class FeatureImportanceResponse(BaseModel):
    """Feature importance results."""
    model: str
    task: str
    feature_importance: Dict[str, float]


# Auth schemas

class LoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str


class UserResponse(BaseModel):
    """User info response."""
    id: int
    username: str
    tier: str


class LoginResponse(BaseModel):
    """Login response with JWT token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


# Analyze schemas

class AnalyzeRequest(BaseModel):
    """Request for stock analysis."""
    tickers: List[str] = Field(..., min_length=1)
    model: str = "gradient_boosting"
    task: str = "classifier"
    period: str = "1M"  # 1D, 1W, 1M, 3M, 6M, 1Y, 5Y
    horizon: str = "1d"


class AnalyzeResultItem(BaseModel):
    """Single stock analysis result."""
    ticker: str
    last_close: float
    prob_up: Optional[float] = None
    signal: Optional[str] = None
    predicted_return: Optional[float] = None


class AnalyzeResponse(BaseModel):
    """Analysis response with results."""
    results: List[AnalyzeResultItem]
    metadata: Dict[str, Any]


class UserFeaturesResponse(BaseModel):
    """User tier features and limits."""
    tier: str
    features: Dict[str, bool]
    limits: Dict[str, Any]


class UniverseResponse(BaseModel):
    """Stock universe response."""
    stocks: List[str]
    categories: Dict[str, List[str]]
    total: int


class ModelsAvailableResponse(BaseModel):
    """Available models response."""
    models: List[str]
    tasks: List[str]
    can_use_custom: bool
