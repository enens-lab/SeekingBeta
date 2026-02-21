"""
Pydantic models for authentication and users
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict
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
        "models": ["lstm_5d", "lstm_jackpot"],
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
        "models": ["lstm_5d", "lstm_jackpot"],
        "tasks": ["classifier"],
        "export_csv": True,
    },
    SubscriptionTier.PRO: {
        "name": "Pro",
        "price": 49,
        "stocks_limit": -1,  # unlimited
        "timeframes": ["1mo", "1w", "3d", "2d", "1d", "4h", "1h", "30m", "15m", "5m", "1m"],
        "features": [
            "Full universe access (6,000+ stocks)",
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
        "models": ["lstm_5d", "lstm_jackpot"],
        "tasks": ["classifier"],
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
    verification_token: Optional[str]
    password_reset_token: Optional[str] = None
    password_reset_expiry: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


class ErrorResponse(BaseModel):
    error: str
    code: Optional[str] = None
    message: Optional[str] = None


class TierInfo(BaseModel):
    tier: SubscriptionTier
    name: str
    price: int
    stocks_limit: int
    timeframes: List[str]
    features: List[str]


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
    watchlist: List[str] = []
    timeframes: List[str] = ["1d"]
    updated_at: datetime


class OracleResponse(BaseModel):
    """Response for oracle endpoints."""
    watchlist: List[str]
    timeframes: List[str]
    available_stocks: List[str]
    available_categories: Dict[str, List[str]]
    available_timeframes: List[str]


class UpdateWatchlistRequest(BaseModel):
    """Request to update entire watchlist."""
    watchlist: List[str]


class AddToWatchlistRequest(BaseModel):
    """Request to add a single ticker to watchlist."""
    ticker: str


class UpdateTimeframesRequest(BaseModel):
    """Request to update preferred timeframes."""
    timeframes: List[str]


# ============================================================
# Analysis Models
# ============================================================

class AnalyzeRequest(BaseModel):
    """Request to run stock analysis."""
    tickers: List[str] = Field(..., min_length=1)
    model: str = "lstm_5d"
    task: str = "classifier"
    period: str = "1M"
    horizon: str = "1d"


class AnalyzeResultItem(BaseModel):
    """Single result item from analysis."""
    ticker: str
    last_close: Optional[float]
    prob_up: Optional[float]
    signal: Optional[str]
    predicted_return: Optional[float] = None


class AnalyzeResponse(BaseModel):
    """Response from analysis endpoint."""
    results: List[AnalyzeResultItem]
    metadata: dict


class ModelsAvailableResponse(BaseModel):
    """Available models for user's tier."""
    models: List[str]
    tasks: List[str]
    can_use_custom: bool = False


class UserFeaturesResponse(BaseModel):
    """User tier features and limits."""
    tier: str
    features: dict
    limits: dict


# ============================================================
# Performance / Track Record Models
# ============================================================

class RegimeBreakdown(BaseModel):
    """Win/loss stats for a market regime bucket."""
    trades: int
    wins: int
    losses: int
    win_rate: Optional[float] = None
    avg_return_net: Optional[float] = None


class TrackRecordSummary(BaseModel):
    """Computed performance summary from backtest/live records."""
    source_file: str
    as_of: str
    sample_size: int
    transaction_cost_bps: float
    hit_rate: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return_gross: Optional[float] = None
    total_return_net: Optional[float] = None
    avg_trade_return_net: Optional[float] = None
    avg_holding_days: Optional[float] = None
    regime_breakdown: Dict[str, RegimeBreakdown] = {}
    notes: List[str] = []


class TrackRecordResponse(BaseModel):
    """Response wrapper for public performance track record."""
    available: bool
    summary: Optional[TrackRecordSummary] = None
    message: Optional[str] = None


# ============================================================
# Company Models
# ============================================================

class CompanyResponse(BaseModel):
    """Basic company info."""
    ticker: str
    name: Optional[str] = None
    asset_type: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CompanyInfoResponse(BaseModel):
    """Company fundamentals."""
    ticker: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    market_cap: Optional[int] = None
    enterprise_value: Optional[int] = None
    employees: Optional[int] = None
    website: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    avg_volume: Optional[int] = None
    fetched_at: Optional[str] = None
    updated_at: Optional[str] = None


class CompanyNewsItem(BaseModel):
    """Single news article."""
    article_id: Optional[str] = None
    title: Optional[str] = None
    publisher: Optional[str] = None
    link: Optional[str] = None
    published_at: Optional[str] = None
    article_type: Optional[str] = None
    thumbnail_url: Optional[str] = None
    related_tickers: Optional[List[str]] = None


class CompanyDetailResponse(BaseModel):
    """Full company detail including info and recent news."""
    company: CompanyResponse
    info: Optional[CompanyInfoResponse] = None
    news: List[CompanyNewsItem] = []


# ============================================================
# User Preferences Models
# ============================================================

class UserPreferences(BaseModel):
    """User preference settings."""
    theme: str = "light"  # light or dark
    email_alerts_enabled: bool = True
    daily_digest_enabled: bool = False
    newsletter_enabled: bool = False
    two_factor_enabled: bool = False
    language: str = "en"
    timezone: str = "UTC"
    notifications_enabled: bool = True


class UserPreferencesResponse(UserPreferences):
    """Response with user preferences."""
    user_id: str
    updated_at: datetime


class UpdatePreferencesRequest(BaseModel):
    """Request to update user preferences."""
    theme: Optional[str] = None
    email_alerts_enabled: Optional[bool] = None
    daily_digest_enabled: Optional[bool] = None
    newsletter_enabled: Optional[bool] = None
    language: Optional[str] = None
    timezone: Optional[str] = None
    notifications_enabled: Optional[bool] = None
