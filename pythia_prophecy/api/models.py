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


MODEL_HORIZONS = ["5d", "20d"]


# Tier configuration
TIER_CONFIG = {
    SubscriptionTier.FREE: {
        "name": "Free",
        "price": 0.0,
        "stocks_limit": 5,
        "timeframes": MODEL_HORIZONS,
        "features": [
            "Watch up to 5 stocks",
            "See both live stock boards",
            "Use the dashboard, watchlist, and analysis pages",
            "Run analysis 10 times per day",
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
        "price": 9.99,
        "stocks_limit": 15,
        "timeframes": MODEL_HORIZONS,
        "features": [
            "Watch up to 15 stocks",
            "Download results as CSV",
            "Check up to 10 stocks at a time",
            "Look back up to 90 days",
            "Run analysis 50 times per day",
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
        "price": 19.99,
        "stocks_limit": -1,  # unlimited
        "timeframes": MODEL_HORIZONS,
        "features": [
            "See the full stock list",
            "Unlimited watchlist",
            "Check up to 50 stocks at a time",
            "Look back up to 365 days",
            "Higher export and usage limits",
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
    accept_terms: bool = False
    accept_privacy: bool = False
    policy_version: str = Field(..., min_length=1, max_length=64)
    marketing_opt_in: bool = False


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


class OAuthName(BaseModel):
    """Human name a provider supplies. Apple sends this only on the FIRST
    authorization (never again), so the client forwards it for that one call."""
    first: Optional[str] = None
    last: Optional[str] = None


class OAuthVerifyRequest(BaseModel):
    """Body for POST /api/auth/oauth/{provider}. `credential` is the provider
    token (Google ID token / Apple identityToken / Facebook access token)."""
    credential: str = Field(..., min_length=1)
    nonce: Optional[str] = None
    name: Optional[OAuthName] = None


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class DeleteAccountRequest(BaseModel):
    password: str
    confirm_text: str = Field(..., min_length=6, max_length=12)


class ErrorResponse(BaseModel):
    error: str
    code: Optional[str] = None
    message: Optional[str] = None


class TierInfo(BaseModel):
    tier: SubscriptionTier
    name: str
    price: float
    stocks_limit: int
    timeframes: List[str]
    features: List[str]


class MessageResponse(BaseModel):
    message: str


class VerifyEmailRequest(BaseModel):
    token: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class BetaTesterSignupRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    role: Optional[str] = Field(default=None, max_length=120)
    organization: Optional[str] = Field(default=None, max_length=160)
    investing_experience: Optional[str] = Field(default=None, max_length=80)
    testing_focus: str = Field(..., min_length=12, max_length=1200)
    accept_contact: bool
    source: Optional[str] = Field(default=None, max_length=120)


class BetaTesterSignupResponse(BaseModel):
    message: str
    discord_url: str


class PredictResponse(BaseModel):
    ticker: str
    horizon: str
    prob_up: float
    signal: str
    last_close: float


class MarketHistorySupplementalContext(BaseModel):
    sentiment_score: Optional[float] = None
    sentiment_articles: Optional[int] = None
    sentiment_source: Optional[str] = None


class MarketHistoryBar(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketHistoryResponse(BaseModel):
    ticker: str
    bars: List[MarketHistoryBar]
    supplemental_context: Optional[MarketHistorySupplementalContext] = None


# ============================================================
# User Oracle (Watchlist) Models
# ============================================================

class UserOracle(BaseModel):
    """User's personalized oracle settings (watchlist and timeframes)."""
    user_id: str
    watchlist: List[str] = []
    timeframes: List[str] = ["5d"]
    updated_at: datetime


class OracleResponse(BaseModel):
    """Response for oracle endpoints."""
    watchlist: List[str]
    timeframes: List[str]
    available_stocks: List[str]
    available_categories: Dict[str, List[str]]
    available_timeframes: List[str]


class WatchlistInsight(BaseModel):
    """Single watchlist insight card payload."""
    ticker: str
    chart_url: str
    source: str
    updated_at: str
    price: Optional[float] = None
    change_pct: Optional[float] = None
    rsi: Optional[float] = None
    sma20: Optional[float] = None
    sma50: Optional[float] = None
    sma200: Optional[float] = None
    volume: Optional[float] = None
    rel_volume: Optional[float] = None
    atr: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    trend: Optional[str] = None
    summary: Optional[str] = None


class WatchlistInsightsResponse(BaseModel):
    """Watchlist insights response for Oracle page."""
    watchlist: List[str]
    finviz_enabled: bool
    insights: List[WatchlistInsight]


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
    horizon: str = "5d"


class AnalyzeResultItem(BaseModel):
    """Single result item from analysis."""
    ticker: str
    last_close: Optional[float]
    prob_up: Optional[float]
    signal: Optional[str]
    predicted_return: Optional[float] = None
    error: Optional[str] = None


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


class BillingCheckoutSessionRequest(BaseModel):
    """Request payload for creating Stripe Checkout session."""
    tier: SubscriptionTier


class BillingCheckoutSessionResponse(BaseModel):
    """Checkout session response."""
    checkout_url: str
    session_id: str


class BillingPortalSessionResponse(BaseModel):
    """Billing portal session response."""
    portal_url: str


class BillingChangeSubscriptionRequest(BaseModel):
    """Request payload for switching subscription tiers."""
    tier: SubscriptionTier


class BillingChangeSubscriptionResponse(BaseModel):
    """Result for subscription change flow."""
    mode: str
    message: str
    checkout_url: Optional[str] = None


class AppleVerifyRequest(BaseModel):
    """Payload from the iOS app after a StoreKit-verified purchase."""
    transaction_id: str
    original_transaction_id: Optional[str] = None
    product_id: str
    app_account_token: Optional[str] = None
    signed_transaction_info: str


class GoogleVerifyRequest(BaseModel):
    """Payload from the Android app after a Play-Billing purchase. The server
    re-verifies the purchase_token against the Google Play Developer API; the
    client-supplied fields are only cross-checks."""
    purchase_token: str
    product_id: str  # the subscription product id (e.g. seekingbeta_subscription)
    base_plan_id: Optional[str] = None  # e.g. basic-monthly / pro-monthly
    package_name: Optional[str] = None
    order_id: Optional[str] = None
    app_account_token: Optional[str] = None  # obfuscatedExternalAccountId


class RegisterAccountTokenRequest(BaseModel):
    """Lets the client register its per-install account token up front, so an
    RTDN-only purchase (client verify never landed) can still be attributed."""
    app_account_token: str


class BillingStatusResponse(BaseModel):
    """Current billing/subscription state for authenticated user."""
    billing_enabled: bool
    user_tier: SubscriptionTier
    effective_tier: SubscriptionTier
    plan_tier: SubscriptionTier
    subscription_status: str
    tier_stripe: Optional[SubscriptionTier] = None
    tier_apple: Optional[SubscriptionTier] = None
    tier_google: Optional[SubscriptionTier] = None
    billing_provider: str = "none"
    entitlement_source: Optional[str] = None
    stripe_status: Optional[str] = None
    stripe_expires_at: Optional[datetime] = None
    apple_product_id: Optional[str] = None
    apple_subscription_status: Optional[str] = None
    apple_expires_at: Optional[datetime] = None
    google_product_id: Optional[str] = None
    google_subscription_status: Optional[str] = None
    google_expires_at: Optional[datetime] = None
    is_active: bool = False
    cancel_at_period_end: bool
    current_period_end: Optional[datetime] = None
    legacy_grace_expires_at: Optional[datetime] = None
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    price_id: Optional[str] = None
    grace_active: bool = False


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
    benchmark_return: Optional[float] = None
    avg_trade_return_net: Optional[float] = None
    avg_win_return_net: Optional[float] = None
    avg_loss_return_net: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_holding_days: Optional[float] = None
    regime_breakdown: Dict[str, RegimeBreakdown] = {}
    notes: List[str] = []


class TrackRecordResponse(BaseModel):
    """Response wrapper for public performance track record."""
    available: bool
    summary: Optional[TrackRecordSummary] = None
    message: Optional[str] = None


class TrackRecordCurvePoint(BaseModel):
    """Single point in model-vs-benchmark equity curve."""
    date: str
    model_value: float
    benchmark_value: Optional[float] = None


class TrackRecordCurveResponse(BaseModel):
    """Curve data for homepage performance visualization."""
    available: bool
    series: List[TrackRecordCurvePoint] = []
    model_label: str
    benchmark_label: str
    start_value: Optional[float] = None
    end_value: Optional[float] = None
    benchmark_end_value: Optional[float] = None
    message: Optional[str] = None


# ============================================================
# Sports Board Models
# ============================================================

class SportsAvailabilitySummary(BaseModel):
    ilAdds14: Optional[int] = None
    ilActivations14: Optional[int] = None
    rosterMoves14: Optional[int] = None


class SportsProjectedLineupContext(BaseModel):
    awayCoverage: Optional[float] = None
    homeCoverage: Optional[float] = None
    awayContinuity: Optional[float] = None
    homeContinuity: Optional[float] = None


class SportsBoardDateOption(BaseModel):
    dateKey: str
    label: str
    gameCount: int


class SportsBoardSeasonSummary(BaseModel):
    year: int
    sampleSize: int
    topPickHits: int
    topPickAccuracy: Optional[float] = None
    top3Hits: Optional[int] = None
    top3Accuracy: Optional[float] = None
    top5Hits: Optional[int] = None
    top5Accuracy: Optional[float] = None


class SportsTeamDetails(BaseModel):
    teamId: Optional[int] = None
    abbreviation: Optional[str] = None
    logoUrl: Optional[str] = None
    wordmarkUrl: Optional[str] = None
    primaryColor: Optional[str] = None
    secondaryColor: Optional[str] = None
    recordPrior: Optional[str] = None
    recentForm: Optional[str] = None
    bullpenSummary: Optional[str] = None
    availabilitySummary: Optional[str] = None
    lineupContinuity: Optional[str] = None
    venue: Optional[str] = None
    weather: Optional[str] = None


class SportsRadarMetric(BaseModel):
    label: str
    value: float


class SportsPlayerStat(BaseModel):
    label: str
    value: str


class SportsPlayerProfile(BaseModel):
    imageUrl: Optional[str] = None
    subtitle: Optional[str] = None
    country: Optional[str] = None
    stats: List[SportsPlayerStat] = []


class SportsLineupPlayer(BaseModel):
    playerId: Optional[int] = None
    playerName: str
    lineupSlot: Optional[int] = None
    position: Optional[str] = None
    batSide: Optional[str] = None
    performanceSummary: Optional[str] = None
    profile: Optional[SportsPlayerProfile] = None
    radarMetrics: List[SportsRadarMetric] = []


class SportsBoardPrediction(BaseModel):
    rank: int
    playerName: str
    winProbability: float
    actualWinner: Optional[bool] = None
    side: Optional[str] = None
    profile: Optional[SportsPlayerProfile] = None
    radarMetrics: List[SportsRadarMetric] = []


# ---- World Cup / Olympics detail blocks (all optional, attached per board) ----

class SportsHeadToHeadMatch(BaseModel):
    date: Optional[str] = None          # ISO yyyy-mm-dd
    homeTeam: Optional[str] = None
    awayTeam: Optional[str] = None
    homeScore: Optional[int] = None
    awayScore: Optional[int] = None
    competition: Optional[str] = None   # e.g. "FIFA World Cup", "Friendly"


class SportsHeadToHead(BaseModel):
    summary: Optional[str] = None       # e.g. "Mexico 2-1-1 South Africa (4 mtgs)"
    homeWins: Optional[int] = None
    awayWins: Optional[int] = None
    draws: Optional[int] = None
    matches: List[SportsHeadToHeadMatch] = []


class SportsTeamFormResult(BaseModel):
    date: Optional[str] = None
    opponent: Optional[str] = None
    result: Optional[str] = None        # "W" | "D" | "L"
    score: Optional[str] = None         # "2-1"
    competition: Optional[str] = None


class SportsTeamHistory(BaseModel):
    team: str
    recentForm: Optional[str] = None    # compact "WWDLW"
    recentResults: List[SportsTeamFormResult] = []
    pastTournament: List[str] = []      # e.g. ["2022: Round of 16", "2018: Group"]


class SportsRosterPlayer(BaseModel):
    name: str
    position: Optional[str] = None
    age: Optional[int] = None
    number: Optional[str] = None


class SportsTeamRoster(BaseModel):
    team: str
    players: List[SportsRosterPlayer] = []


class SportsMedalist(BaseModel):
    medal: str                          # "Gold" | "Silver" | "Bronze"
    name: str
    country: Optional[str] = None


class SportsOlympicDiscipline(BaseModel):
    sport: str                          # e.g. "Athletics"
    event: str                          # e.g. "Men's 100 metres"
    year: Optional[int] = None
    medalists: List[SportsMedalist] = []


class SportsUpcomingBoard(BaseModel):
    id: str
    name: str
    tour: str
    course: str
    original_name: Optional[str] = None
    scheduledDate: Optional[int] = None
    latestDate: Optional[int] = None
    venue: Optional[str] = None
    # Lifecycle state derived from the date window vs today: "upcoming" (not
    # started), "live" (in progress today), or "completed". Lets the UI badge an
    # in-progress event (e.g. a Grand Slam mid-fortnight) instead of showing it
    # as a plain upcoming card. Optional/None => UI falls back to old behavior.
    eventState: Optional[str] = None
    predictedWinner: Optional[str] = None
    # 1X2 outcome distribution for sports that can draw (soccer). Optional so
    # binary home-win sports (NFL/NBA/MLB/NHL) and ranked sports (golf/tennis)
    # leave them null and are unaffected.
    homeWinProbability: Optional[float] = None
    drawProbability: Optional[float] = None
    awayWinProbability: Optional[float] = None
    awayTeam: Optional[str] = None
    homeTeam: Optional[str] = None
    awayStarter: Optional[str] = None
    homeStarter: Optional[str] = None
    awayStarterProfile: Optional[SportsPlayerProfile] = None
    homeStarterProfile: Optional[SportsPlayerProfile] = None
    awayTeamDetails: Optional[SportsTeamDetails] = None
    homeTeamDetails: Optional[SportsTeamDetails] = None
    awayStarterRadar: Optional[List[SportsRadarMetric]] = None
    homeStarterRadar: Optional[List[SportsRadarMetric]] = None
    awayAvailability: Optional[SportsAvailabilitySummary] = None
    homeAvailability: Optional[SportsAvailabilitySummary] = None
    projectedLineupContext: Optional[SportsProjectedLineupContext] = None
    predictionSource: Optional[str] = None
    awayLineup: List[SportsLineupPlayer] = []
    homeLineup: List[SportsLineupPlayer] = []
    awayFeaturedPlayer: Optional[SportsLineupPlayer] = None
    homeFeaturedPlayer: Optional[SportsLineupPlayer] = None
    predictions: List[SportsBoardPrediction] = []
    # Set only by the ?preview=true response shaping: the size of the full
    # predictions list before it was trimmed, so UIs can keep showing the real
    # field size ("131 names") next to the trimmed top-N list.
    predictionsTotal: Optional[int] = None
    # World Cup / Olympics detail (optional; only populated for those tours).
    headToHead: Optional[SportsHeadToHead] = None
    teamHistory: List[SportsTeamHistory] = []          # [home, away] for WC matches
    rosters: List[SportsTeamRoster] = []               # [home, away] squads
    disciplines: List[SportsOlympicDiscipline] = []    # Olympics: events under a sport


class SportsHistoricalBoard(BaseModel):
    year: int
    tournament: str
    tour: str
    hitStatus: str
    predictedWinner: Optional[str] = None
    homeWinProbability: Optional[float] = None
    drawProbability: Optional[float] = None
    awayWinProbability: Optional[float] = None
    predictedTop3: List[str] = []
    predictedTop5: List[str] = []
    actualWinner: Optional[str] = None
    prob: Optional[float] = None
    venue: Optional[str] = None
    course: Optional[str] = None
    fullField: List[SportsBoardPrediction] = []
    disciplines: List[SportsOlympicDiscipline] = []   # Olympics historical browser
    latestDate: Optional[int] = None
    tournamentId: Optional[str] = None
    scheduledDate: Optional[int] = None
    awayTeam: Optional[str] = None
    homeTeam: Optional[str] = None
    awayStarter: Optional[str] = None
    homeStarter: Optional[str] = None
    awayStarterProfile: Optional[SportsPlayerProfile] = None
    homeStarterProfile: Optional[SportsPlayerProfile] = None
    awayStarterRadar: Optional[List[SportsRadarMetric]] = None
    homeStarterRadar: Optional[List[SportsRadarMetric]] = None
    awayTeamDetails: Optional[SportsTeamDetails] = None
    homeTeamDetails: Optional[SportsTeamDetails] = None
    awayLineup: List[SportsLineupPlayer] = []
    homeLineup: List[SportsLineupPlayer] = []
    awayFeaturedPlayer: Optional[SportsLineupPlayer] = None
    homeFeaturedPlayer: Optional[SportsLineupPlayer] = None


class SportsBoardCollection(BaseModel):
    upcoming: List[SportsUpcomingBoard] = []
    backtests: List[SportsHistoricalBoard] = []
    updated_at: datetime
    source: str
    selectedDate: Optional[str] = None
    availableDates: List[SportsBoardDateOption] = []
    seasonSummary: Optional[SportsBoardSeasonSummary] = None


class SportsBoardsResponse(BaseModel):
    golf: SportsBoardCollection
    tennis: SportsBoardCollection
    basketball: SportsBoardCollection
    mlb: SportsBoardCollection
    football: SportsBoardCollection
    soccer: SportsBoardCollection
    olympics: SportsBoardCollection


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
