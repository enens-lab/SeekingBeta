"""
Pydantic models for authentication and users
"""
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, EmailStr, Field


class SubscriptionTier(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"


# Tier configuration
TIER_CONFIG = {
    SubscriptionTier.FREE: {
        "name": "Free",
        "price": 0,
        "stocks_limit": 5,
        "timeframes": ["1d"],
        "features": [
            "5 stocks from universe",
            "Daily predictions only",
            "Basic signal alerts",
        ],
        # Analysis settings
        "daily_requests": 10,
        "max_stocks_per_request": 5,
        "max_historical_days": 30,
        "models": ["gradient_boosting", "linear_regression"],
        "tasks": ["classifier"],
        "export_csv": False,
    },
    SubscriptionTier.BASIC: {
        "name": "Basic",
        "price": 19,
        "stocks_limit": 15,
        "timeframes": ["1d", "2d", "3d", "4h", "1h", "30m"],
        "features": [
            "15 stocks from universe",
            "Multi-day, daily, and intraday predictions",
            "Email alerts",
            "Historical prediction accuracy",
        ],
        # Analysis settings
        "daily_requests": 50,
        "max_stocks_per_request": 10,
        "max_historical_days": 90,
        "models": ["gradient_boosting", "linear_regression", "random_forest"],
        "tasks": ["classifier", "regressor"],
        "export_csv": True,
    },
    SubscriptionTier.PRO: {
        "name": "Pro",
        "price": 49,
        "stocks_limit": -1,  # unlimited
        "timeframes": ["1mo", "1w", "3d", "2d", "1d", "4h", "1h", "30m", "15m", "5m", "1m"],
        "features": [
            "Full universe access (43+ stocks)",
            "All timeframes (1m to monthly)",
            "Priority email alerts",
            "API access",
            "Historical prediction accuracy",
            "Custom watchlists",
        ],
        # Analysis settings
        "daily_requests": None,  # unlimited
        "max_stocks_per_request": 50,
        "max_historical_days": 365,
        "models": ["gradient_boosting", "linear_regression", "random_forest", "lstm"],
        "tasks": ["classifier", "regressor"],
        "export_csv": True,
    },
}


class UserBase(BaseModel):
    email: EmailStr
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    tier: SubscriptionTier = SubscriptionTier.FREE


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(UserBase):
    id: str
    tier: SubscriptionTier
    email_verified: bool
    created_at: datetime


class UserInDB(UserBase):
    id: str
    hashed_password: str
    tier: SubscriptionTier
    email_verified: bool
    verification_token: str | None
    created_at: datetime
    updated_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TierInfo(BaseModel):
    tier: SubscriptionTier
    name: str
    price: int
    stocks_limit: int
    timeframes: list[str]
    features: list[str]


class MessageResponse(BaseModel):
    message: str


class VerifyEmailRequest(BaseModel):
    token: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class PredictResponse(BaseModel):
    ticker: str
    horizon: str
    prob_up: float
    signal: str
    last_close: float


# ============================================================
# User Oracle (Watchlist) Models
# ============================================================

class UserOracle(BaseModel):
    """User's personalized oracle settings (watchlist and timeframes)."""
    user_id: str
    watchlist: list[str] = []
    timeframes: list[str] = ["1d"]
    updated_at: datetime


class OracleResponse(BaseModel):
    """Response for oracle endpoints."""
    watchlist: list[str]
    timeframes: list[str]
    available_stocks: list[str]
    available_categories: dict[str, list[str]]
    available_timeframes: list[str]


class UpdateWatchlistRequest(BaseModel):
    """Request to update entire watchlist."""
    watchlist: list[str]


class AddToWatchlistRequest(BaseModel):
    """Request to add a single ticker to watchlist."""
    ticker: str


class UpdateTimeframesRequest(BaseModel):
    """Request to update preferred timeframes."""
    timeframes: list[str]


# ============================================================
# Analysis Models
# ============================================================

class AnalyzeRequest(BaseModel):
    """Request to run stock analysis."""
    tickers: list[str] = Field(..., min_length=1)
    model: str = "gradient_boosting"
    task: str = "classifier"
    period: str = "1M"
    horizon: str = "1d"


class AnalyzeResultItem(BaseModel):
    """Single result item from analysis."""
    ticker: str
    last_close: float | None
    prob_up: float | None
    signal: str | None
    predicted_return: float | None = None


class AnalyzeResponse(BaseModel):
    """Response from analysis endpoint."""
    results: list[AnalyzeResultItem]
    metadata: dict


class ModelsAvailableResponse(BaseModel):
    """Available models for user's tier."""
    models: list[str]
    tasks: list[str]
    can_use_custom: bool = False


class UserFeaturesResponse(BaseModel):
    """User tier features and limits."""
    tier: str
    features: dict
    limits: dict


# ============================================================
# Company Models
# ============================================================

class CompanyResponse(BaseModel):
    """Basic company info."""
    ticker: str
    name: str | None = None
    asset_type: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class CompanyInfoResponse(BaseModel):
    """Company fundamentals."""
    ticker: str
    sector: str | None = None
    industry: str | None = None
    description: str | None = None
    market_cap: int | None = None
    enterprise_value: int | None = None
    employees: int | None = None
    website: str | None = None
    country: str | None = None
    state: str | None = None
    city: str | None = None
    exchange: str | None = None
    currency: str | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    avg_volume: int | None = None
    fetched_at: str | None = None
    updated_at: str | None = None


class CompanyNewsItem(BaseModel):
    """Single news article."""
    article_id: str | None = None
    title: str | None = None
    publisher: str | None = None
    link: str | None = None
    published_at: str | None = None
    article_type: str | None = None
    thumbnail_url: str | None = None
    related_tickers: list[str] | None = None


class CompanyDetailResponse(BaseModel):
    """Full company detail including info and recent news."""
    company: CompanyResponse
    info: CompanyInfoResponse | None = None
    news: list[CompanyNewsItem] = []
