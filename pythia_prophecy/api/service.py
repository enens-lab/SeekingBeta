"""
Pythia Claude - FastAPI Service
Serves the React frontend, prediction endpoints, and authentication
"""
from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from .logging_config import setup_logging, get_logger

# Initialize logging (may already be initialized by main.py, but safe to call again)
setup_logging()
logger = get_logger("service")
access_logger = get_logger("access")

# Ensure pythia is importable
PYTHIA_PATH = Path(__file__).parent.parent.parent / "pythia"
if PYTHIA_PATH.exists() and str(PYTHIA_PATH) not in sys.path:
    sys.path.insert(0, str(PYTHIA_PATH))

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx


from .models import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    TierInfo,
    MessageResponse,
    VerifyEmailRequest,
    ResendVerificationRequest,
    SubscriptionTier,
    TIER_CONFIG,
    PredictResponse,
    OracleResponse,
    UpdateWatchlistRequest,
    AddToWatchlistRequest,
    UpdateTimeframesRequest,
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeResultItem,
    ModelsAvailableResponse,
    UserFeaturesResponse,
    CompanyResponse,
    CompanyInfoResponse,
    CompanyNewsItem,
    CompanyDetailResponse,
)
from .database import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    get_user_by_verification_token,
    verify_user_email,
    update_user_verification_token,
    get_user_oracle,
    create_or_update_user_oracle,
    add_to_watchlist,
    remove_from_watchlist,
    update_oracle_timeframes,
    get_company,
    get_company_info,
    get_company_news,
)
from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    generate_verification_token,
)
from .email_service import send_verification_email, send_welcome_email
from .models import UserInDB

# Stock categorization function (mirrors divination logic)
def categorize_stocks(stocks: list[str]) -> dict[str, list[str]]:
    """Categorize stocks by sector."""
    tech = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC",
            "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "CRM", "ORCL", "ADBE",
            "NOW", "INTU", "IBM", "CSCO", "HPQ", "DELL"]
    financials = ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "BRK-B",
                  "AIG", "MET", "PRU", "BLK", "SCHW", "AXP", "V", "MA", "PYPL"]
    healthcare = ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD",
                  "REGN", "MRNA", "UNH", "CVS", "CI", "MDT", "ABT"]
    consumer = ["HD", "LOW", "NKE", "SBUX", "MCD", "TGT", "TJX", "BKNG", "MAR",
                "F", "GM", "LULU", "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL"]
    energy = ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY", "KMI"]
    industrials = ["CAT", "BA", "UPS", "FDX", "HON", "RTX", "LMT", "GE", "DE", "UNP"]
    etfs = ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "MDY", "XLF", "XLK",
            "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
            "BND", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "USO", "UNG",
            "EFA", "EEM", "VWO", "FXI", "EWJ"]

    categories = {}
    stock_set = set(stocks)

    if tech_in := [s for s in tech if s in stock_set]:
        categories["Technology"] = tech_in
    if fin_in := [s for s in financials if s in stock_set]:
        categories["Financials"] = fin_in
    if health_in := [s for s in healthcare if s in stock_set]:
        categories["Healthcare"] = health_in
    if cons_in := [s for s in consumer if s in stock_set]:
        categories["Consumer"] = cons_in
    if energy_in := [s for s in energy if s in stock_set]:
        categories["Energy"] = energy_in
    if ind_in := [s for s in industrials if s in stock_set]:
        categories["Industrials"] = ind_in
    if etf_in := [s for s in etfs if s in stock_set]:
        categories["ETFs"] = etf_in

    # Other - anything not categorized
    categorized = set()
    for cat_stocks in categories.values():
        categorized.update(cat_stocks)
    other = [s for s in stocks if s not in categorized]
    if other:
        categories["Other"] = other

    return categories


# Load universe from pythia_divination config.yaml
def load_divination_config() -> dict:
    """Load config from pythia_divination's config.yaml."""
    import yaml
    config_path = Path(__file__).parent.parent.parent / "pythia_divination" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


# Import from pythia_divination for predictions
try:
    import sys
    import yaml
    DIVINATION_PATH = Path(__file__).parent.parent.parent / "pythia_divination"
    if DIVINATION_PATH.exists() and str(DIVINATION_PATH) not in sys.path:
        sys.path.insert(0, str(DIVINATION_PATH))

    # Load universe directly from config.yaml (doesn't require DATABASE_URL)
    divination_config = load_divination_config()
    UNIVERSE = divination_config.get("universe", [])
    THRESHOLD = float(divination_config.get("threshold", 0.55))
    CFG_LOOKBACK = divination_config.get("lookback_download", "120d")
    DATA_SOURCE = divination_config.get("data_source", "alpaca")

    if not UNIVERSE:
        raise ValueError("No universe found in divination config")

    # Try to load ML model artifacts (may fail if dependencies missing)
    try:
        from data.fetch import fetch_ohlcv
        from features.technical import make_features
        from models.utils import load_artifacts
        MODEL, SCALER, FEATS, _ = load_artifacts()
        PYTHIA_AVAILABLE = True
    except Exception as model_err:
        logger.warning(f"Could not load ML models: {model_err}")
        PYTHIA_AVAILABLE = False
        MODEL = SCALER = FEATS = None

    # Build stock categories dynamically from universe
    STOCK_CATEGORIES = categorize_stocks(UNIVERSE)

    logger.info(f"Loaded universe with {len(UNIVERSE)} stocks from pythia_divination")
    logger.info(f"Categories: {', '.join(f'{k} ({len(v)})' for k, v in STOCK_CATEGORIES.items())}")
except Exception as e:
    logger.warning(f"Could not load from pythia_divination: {e}")
    logger.info("Running in demo mode with mock data")
    PYTHIA_AVAILABLE = False
    UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "UNH"]
    THRESHOLD = 0.55
    STOCK_CATEGORIES = categorize_stocks(UNIVERSE)

# Simple in-memory rate limiting (resets on server restart)
# In production, this should be stored in a database
from collections import defaultdict
RATE_LIMIT_STORAGE: dict[str, dict] = defaultdict(lambda: {"date": None, "count": 0})


# FastAPI app
app = FastAPI(
    title="Pythia Claude",
    description="Stock Signal Intelligence Platform",
    version="0.1.0",
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all HTTP requests with timing."""
    start_time = time.time()

    # Get client info
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000

    # Log the request (skip static assets for cleaner logs)
    if not path.startswith("/assets/") and path != "/favicon.ico":
        access_logger.info(
            f"{client_ip} | {method} {path} | {response.status_code} | {duration_ms:.1f}ms"
        )

    return response


# Path to React build output
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"


# ============================================================
# Auth Dependency
# ============================================================

async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[UserInDB]:
    """Extract and validate user from Authorization header."""
    if not authorization:
        return None

    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            return None

        payload = decode_access_token(token)
        if not payload:
            return None

        user = get_user_by_id(payload.get("sub"))
        return user
    except Exception:
        return None


async def require_auth(authorization: Optional[str] = Header(None)) -> UserInDB:
    """Require authenticated user."""
    user = await get_current_user(authorization)
    if not user:
        raise HTTPException(401, detail="Not authenticated")
    return user


async def require_verified_user(authorization: Optional[str] = Header(None)) -> UserInDB:
    """Require authenticated and email-verified user."""
    user = await require_auth(authorization)
    if not user.email_verified:
        raise HTTPException(403, detail="Email not verified")
    return user


# ============================================================
# Auth Endpoints
# ============================================================

@app.post("/api/auth/signup", response_model=MessageResponse)
async def signup(data: UserCreate):
    """Register a new user account."""
    # Check if email already exists
    existing = get_user_by_email(data.email.lower())
    if existing:
        logger.info(f"Signup attempt with existing email: {data.email.lower()}")
        raise HTTPException(400, detail="Email already registered")

    # Create user
    verification_token = generate_verification_token()
    now = datetime.utcnow()

    user = UserInDB(
        id=str(uuid.uuid4()),
        email=data.email.lower(),
        first_name=data.first_name,
        last_name=data.last_name,
        hashed_password=hash_password(data.password),
        tier=data.tier,
        email_verified=False,
        verification_token=verification_token,
        created_at=now,
        updated_at=now,
    )

    create_user(user)
    logger.info(f"New user created: {user.email} (tier={user.tier.value})")

    # Send verification email
    send_verification_email(user.email, user.first_name, verification_token)

    return MessageResponse(
        message="Account created. Please check your email to verify your account."
    )


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(data: UserLogin):
    """Login with email and password."""
    user = get_user_by_email(data.email.lower())

    if not user or not verify_password(data.password, user.hashed_password):
        logger.warning(f"Failed login attempt for: {data.email.lower()}")
        raise HTTPException(401, detail="Invalid email or password")

    token = create_access_token(user.id, user.email)
    logger.info(f"User logged in: {user.email}")

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            tier=user.tier,
            email_verified=user.email_verified,
            created_at=user.created_at,
        ),
    )


@app.post("/api/auth/verify-email", response_model=TokenResponse)
async def verify_email(data: VerifyEmailRequest):
    """Verify email with token from email link."""
    user = get_user_by_verification_token(data.token)

    if not user:
        logger.warning("Invalid verification token used")
        raise HTTPException(400, detail="Invalid or expired verification token")

    verify_user_email(user.id)
    logger.info(f"Email verified for user: {user.email}")

    # Send welcome email
    send_welcome_email(user.email, user.first_name, user.tier.value)

    # Return token so user is logged in
    token = create_access_token(user.id, user.email)

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            tier=user.tier,
            email_verified=True,
            created_at=user.created_at,
        ),
    )


@app.post("/api/auth/resend-verification", response_model=MessageResponse)
async def resend_verification(data: ResendVerificationRequest):
    """Resend verification email."""
    user = get_user_by_email(data.email.lower())

    if not user:
        # Don't reveal if email exists
        return MessageResponse(message="If the email exists, a verification link has been sent.")

    if user.email_verified:
        raise HTTPException(400, detail="Email already verified")

    # Generate new token
    new_token = generate_verification_token()
    update_user_verification_token(user.id, new_token)

    # Send email
    send_verification_email(user.email, user.first_name, new_token)

    return MessageResponse(message="If the email exists, a verification link has been sent.")


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(user: UserInDB = Depends(require_auth)):
    """Get current user info."""
    return UserResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        tier=user.tier,
        email_verified=user.email_verified,
        created_at=user.created_at,
    )


# ============================================================
# Tier Endpoints
# ============================================================

@app.get("/api/tiers", response_model=list[TierInfo])
async def get_tiers():
    """Get all subscription tiers."""
    return [
        TierInfo(
            tier=tier,
            name=config["name"],
            price=config["price"],
            stocks_limit=config["stocks_limit"],
            timeframes=config["timeframes"],
            features=config["features"],
        )
        for tier, config in TIER_CONFIG.items()
    ]


@app.get("/api/tiers/{tier}", response_model=TierInfo)
async def get_tier(tier: SubscriptionTier):
    """Get specific tier info."""
    config = TIER_CONFIG[tier]
    return TierInfo(
        tier=tier,
        name=config["name"],
        price=config["price"],
        stocks_limit=config["stocks_limit"],
        timeframes=config["timeframes"],
        features=config["features"],
    )


# ============================================================
# Prediction Endpoints
# ============================================================

def _normalize_horizon(h: str) -> str:
    """Normalize horizon tokens to canonical timeframe string."""
    if not h:
        return "1Day"
    h = h.strip().lower()
    table = {
        # Minutes
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "30m": "30Min",
        # Hours
        "1h": "1Hour",
        "60m": "1Hour",
        "4h": "4Hour",
        "240m": "4Hour",
        # Days
        "1d": "1Day",
        "1day": "1Day",
        "2d": "2Day",
        "2day": "2Day",
        "3d": "3Day",
        "3day": "3Day",
        # Weeks
        "1w": "1Week",
        "1wk": "1Week",
        "1week": "1Week",
        # Months
        "1mo": "1Month",
        "1month": "1Month",
    }
    return table.get(h, "1Day")


def _horizon_to_display(h: str) -> str:
    """Convert horizon to display format."""
    table = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "60m": "1h", "4h": "4h", "240m": "4h",
        "1d": "1d", "1day": "1d", "2d": "2d", "2day": "2d", "3d": "3d", "3day": "3d",
        "1w": "1w", "1wk": "1w", "1week": "1w",
        "1mo": "1mo", "1month": "1mo",
    }
    return table.get(h.lower(), "1d")


def predict_for_ticker(ticker: str, horizon: str = "1d"):
    """Generate prediction for a single ticker."""
    if not PYTHIA_AVAILABLE:
        import random
        prob_up = random.uniform(0.35, 0.75)
        signal = "buy" if prob_up >= THRESHOLD else ("sell" if prob_up <= 1 - THRESHOLD else "hold")
        last_close = random.uniform(100, 500)
        return prob_up, signal, last_close

    tf = _normalize_horizon(horizon)

    raw = fetch_ohlcv(
        ticker,
        period=CFG_LOOKBACK,
        data_source=DATA_SOURCE,
        timeframe=tf,
    )
    feat = make_features(raw)

    X = feat[FEATS].values
    Xs = SCALER.transform(X)
    prob_up = float(MODEL.predict_proba(Xs)[:, 1][-1])
    last_close = float(raw["Close"].iloc[-1])
    signal = "buy" if prob_up >= THRESHOLD else ("sell" if prob_up <= 1 - THRESHOLD else "hold")

    return prob_up, signal, last_close


def check_tier_access(user: Optional[UserInDB], ticker: str, horizon: str) -> bool:
    """Check if user's tier allows access to this ticker/horizon."""
    if not user:
        # Anonymous users get free tier
        tier_config = TIER_CONFIG[SubscriptionTier.FREE]
    else:
        tier_config = TIER_CONFIG[user.tier]

    # Check timeframe
    horizon_display = _horizon_to_display(horizon)
    if horizon_display not in tier_config["timeframes"]:
        return False

    # Check stock limit (for now, just check if in universe)
    stocks_limit = tier_config["stocks_limit"]
    if stocks_limit == -1:
        return True  # Unlimited

    # For limited tiers, only allow first N stocks from universe
    allowed_stocks = UNIVERSE[:stocks_limit]
    return ticker.upper() in [s.upper() for s in allowed_stocks]


@app.get("/predict/{ticker}", response_model=PredictResponse)
async def predict(
    ticker: str,
    horizon: str = "1d",
    user: Optional[UserInDB] = Depends(get_current_user),
):
    """Get prediction for a single ticker."""
    user_email = user.email if user else "anonymous"

    # Check tier access (but allow anonymous users basic access)
    if user and not check_tier_access(user, ticker, horizon):
        tier_name = user.tier.value
        logger.info(f"Prediction access denied: {ticker} ({horizon}) for {user_email} (tier={tier_name})")
        raise HTTPException(
            403,
            detail=f"Your {tier_name} tier doesn't include access to this ticker/timeframe. Please upgrade.",
        )

    try:
        prob_up, signal, last_close = predict_for_ticker(ticker.upper(), horizon=horizon)
        logger.debug(f"Prediction: {ticker.upper()} ({horizon}) = {signal} ({prob_up:.2%}) for {user_email}")
        return PredictResponse(
            ticker=ticker.upper(),
            horizon=_horizon_to_display(horizon),
            prob_up=prob_up,
            signal=signal,
            last_close=last_close,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {ticker}: {e}")
        raise HTTPException(400, detail=str(e))


@app.get("/predict/lstm_5d/{ticker}", response_model=PredictResponse)
async def predict_lstm_5d(ticker: str):
    """Proxy LSTM 5-Day predictions from divination backend."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"http://pythia-divination:8000/predict/lstm_5d/{ticker}")
            if response.status_code != 200:
                raise HTTPException(response.status_code, detail="Failed to fetch LSTM prediction")
            data = response.json()
            # Transform LSTM response to match PredictResponse schema
            return PredictResponse(
                ticker=data.get("ticker", ticker),
                horizon=data.get("horizon", "5 days"),
                prob_up=data.get("probability", 0.0) / 100.0,  # Convert percentage to decimal
                signal=data.get("signal", "hold"),
                last_close=data.get("last_close", 0.0),
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LSTM 5D prediction error for {ticker}: {e}")
        raise HTTPException(500, detail=str(e))



@app.get("/predict/lstm_jackpot/{ticker}", response_model=PredictResponse)
async def predict_lstm_jackpot(ticker: str):
    """Proxy LSTM Jackpot predictions from divination backend."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"http://pythia-divination:8000/predict/lstm_jackpot/{ticker}")
            if response.status_code != 200:
                raise HTTPException(response.status_code, detail="Failed to fetch LSTM prediction")
            data = response.json()
            # Transform LSTM response to match PredictResponse schema
            return PredictResponse(
                ticker=data.get("ticker", ticker),
                horizon=data.get("horizon", "20 days"),
                prob_up=data.get("probability", 0.0) / 100.0,  # Convert percentage to decimal
                signal=data.get("signal", "hold"),
                last_close=data.get("last_close", 0.0),
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LSTM Jackpot prediction error for {ticker}: {e}")
        raise HTTPException(500, detail=str(e))



@app.get("/api/universe")
async def get_universe(user: Optional[UserInDB] = Depends(get_current_user)):
    """Get the list of tickers available to the user based on their tier."""
    if not user:
        tier_config = TIER_CONFIG[SubscriptionTier.FREE]
    else:
        tier_config = TIER_CONFIG[user.tier]

    stocks_limit = tier_config["stocks_limit"]
    if stocks_limit == -1:
        available = UNIVERSE
    else:
        available = UNIVERSE[:stocks_limit]

    return {
        "universe": available,
        "total_available": len(UNIVERSE),
        "your_limit": stocks_limit,
        "timeframes": tier_config["timeframes"],
    }


@app.get("/healthz")
def healthz():
    """Health check endpoint with universe info."""
    return JSONResponse({
        "ok": True,
        "message": "ok",
        "universe": UNIVERSE,
        "data_source": DATA_SOURCE if PYTHIA_AVAILABLE else "demo",
    })


# ============================================================
# Oracle (Watchlist) Endpoints
# ============================================================

def _get_available_for_user(user: UserInDB) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Get available stocks, categories, and timeframes for user's tier."""
    tier_config = TIER_CONFIG[user.tier]
    stocks_limit = tier_config["stocks_limit"]
    if stocks_limit == -1:
        available_stocks = list(UNIVERSE)
    else:
        available_stocks = list(UNIVERSE[:stocks_limit])

    # Build categories filtered to available stocks
    available_set = set(available_stocks)
    available_categories = {}
    for category, stocks in STOCK_CATEGORIES.items():
        filtered = [s for s in stocks if s in available_set]
        if filtered:
            available_categories[category] = filtered

    available_timeframes = tier_config["timeframes"]
    return available_stocks, available_categories, available_timeframes


@app.get("/api/oracle", response_model=OracleResponse)
async def get_oracle(user: UserInDB = Depends(require_verified_user)):
    """Get user's oracle (watchlist and preferred timeframes)."""
    oracle = get_user_oracle(user.id)
    available_stocks, available_categories, available_timeframes = _get_available_for_user(user)

    if oracle:
        # Filter watchlist to only include stocks user has access to
        valid_watchlist = [
            t for t in oracle.watchlist
            if t.upper() in [s.upper() for s in available_stocks]
        ]
        # Filter timeframes to only include ones user has access to
        valid_timeframes = [
            tf for tf in oracle.timeframes
            if tf in available_timeframes
        ]
        if not valid_timeframes:
            valid_timeframes = [available_timeframes[0]] if available_timeframes else ["1d"]
    else:
        valid_watchlist = []
        valid_timeframes = [available_timeframes[0]] if available_timeframes else ["1d"]

    return OracleResponse(
        watchlist=valid_watchlist,
        timeframes=valid_timeframes,
        available_stocks=available_stocks,
        available_categories=available_categories,
        available_timeframes=available_timeframes,
    )


@app.put("/api/oracle/watchlist", response_model=OracleResponse)
async def update_watchlist(
    data: UpdateWatchlistRequest,
    user: UserInDB = Depends(require_verified_user),
):
    """Update entire watchlist."""
    available_stocks, available_categories, available_timeframes = _get_available_for_user(user)

    # Validate all tickers are accessible
    invalid = [
        t for t in data.watchlist
        if t.upper() not in [s.upper() for s in available_stocks]
    ]
    if invalid:
        raise HTTPException(
            400,
            detail=f"Tickers not available in your tier: {', '.join(invalid)}",
        )

    # Get current timeframes
    oracle = get_user_oracle(user.id)
    current_timeframes = oracle.timeframes if oracle else ["1d"]

    # Normalize tickers to uppercase
    normalized = [t.upper() for t in data.watchlist]

    create_or_update_user_oracle(user.id, normalized, current_timeframes)

    return OracleResponse(
        watchlist=normalized,
        timeframes=current_timeframes,
        available_stocks=available_stocks,
        available_categories=available_categories,
        available_timeframes=available_timeframes,
    )


@app.post("/api/oracle/watchlist", response_model=OracleResponse)
async def add_ticker_to_watchlist(
    data: AddToWatchlistRequest,
    user: UserInDB = Depends(require_verified_user),
):
    """Add a single ticker to watchlist."""
    available_stocks, available_categories, available_timeframes = _get_available_for_user(user)

    ticker = data.ticker.upper()
    if ticker not in [s.upper() for s in available_stocks]:
        raise HTTPException(
            400,
            detail=f"Ticker '{ticker}' is not available in your tier. Please upgrade to access more stocks.",
        )

    oracle = add_to_watchlist(user.id, ticker)

    return OracleResponse(
        watchlist=oracle.watchlist,
        timeframes=oracle.timeframes,
        available_stocks=available_stocks,
        available_categories=available_categories,
        available_timeframes=available_timeframes,
    )


@app.delete("/api/oracle/watchlist/{ticker}", response_model=OracleResponse)
async def remove_ticker_from_watchlist(
    ticker: str,
    user: UserInDB = Depends(require_verified_user),
):
    """Remove a ticker from watchlist."""
    available_stocks, available_categories, available_timeframes = _get_available_for_user(user)

    oracle = remove_from_watchlist(user.id, ticker.upper())

    return OracleResponse(
        watchlist=oracle.watchlist,
        timeframes=oracle.timeframes,
        available_stocks=available_stocks,
        available_categories=available_categories,
        available_timeframes=available_timeframes,
    )


@app.put("/api/oracle/timeframes", response_model=OracleResponse)
async def update_preferred_timeframes(
    data: UpdateTimeframesRequest,
    user: UserInDB = Depends(require_verified_user),
):
    """Update preferred timeframes."""
    available_stocks, available_categories, available_timeframes = _get_available_for_user(user)

    # Validate all timeframes are accessible
    invalid = [tf for tf in data.timeframes if tf not in available_timeframes]
    if invalid:
        raise HTTPException(
            400,
            detail=f"Timeframes not available in your tier: {', '.join(invalid)}",
        )

    if not data.timeframes:
        raise HTTPException(400, detail="At least one timeframe is required")

    oracle = update_oracle_timeframes(user.id, data.timeframes)

    return OracleResponse(
        watchlist=oracle.watchlist,
        timeframes=oracle.timeframes,
        available_stocks=available_stocks,
        available_categories=available_categories,
        available_timeframes=available_timeframes,
    )


@app.get("/api/oracle/predictions")
async def get_oracle_predictions(
    user: UserInDB = Depends(require_verified_user),
):
    """Get predictions for all stocks in user's watchlist across their preferred timeframes."""
    oracle = get_user_oracle(user.id)
    available_stocks, available_categories, available_timeframes = _get_available_for_user(user)

    if not oracle or not oracle.watchlist:
        return {"predictions": [], "watchlist": [], "timeframes": available_timeframes[:1]}

    # Filter to valid stocks and timeframes
    valid_watchlist = [
        t for t in oracle.watchlist
        if t.upper() in [s.upper() for s in available_stocks]
    ]
    valid_timeframes = [
        tf for tf in oracle.timeframes
        if tf in available_timeframes
    ]
    if not valid_timeframes:
        valid_timeframes = [available_timeframes[0]] if available_timeframes else ["1d"]

    predictions = []
    for ticker in valid_watchlist:
        for tf in valid_timeframes:
            try:
                prob_up, signal, last_close = predict_for_ticker(ticker, tf)
                predictions.append({
                    "ticker": ticker,
                    "timeframe": tf,
                    "prob_up": prob_up,
                    "signal": signal,
                    "last_close": last_close,
                })
            except Exception as e:
                # Skip failed predictions
                continue

    return {
        "predictions": predictions,
        "watchlist": valid_watchlist,
        "timeframes": valid_timeframes,
    }


# ============================================================
# Analysis Endpoints
# ============================================================

def _get_rate_limit_for_user(user: UserInDB) -> tuple[int, int, int]:
    """Get rate limit info for user. Returns (used, remaining, limit)."""
    tier_config = TIER_CONFIG[user.tier]
    limit = tier_config.get("daily_requests")

    if limit is None:
        return 0, -1, -1  # unlimited

    today = datetime.utcnow().strftime("%Y-%m-%d")
    user_limits = RATE_LIMIT_STORAGE[user.id]

    if user_limits["date"] != today:
        user_limits["date"] = today
        user_limits["count"] = 0

    used = user_limits["count"]
    remaining = limit - used
    return used, remaining, limit


def _increment_rate_limit(user: UserInDB) -> None:
    """Increment rate limit counter for user."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    user_limits = RATE_LIMIT_STORAGE[user.id]

    if user_limits["date"] != today:
        user_limits["date"] = today
        user_limits["count"] = 0

    user_limits["count"] += 1


@app.get("/api/user/features", response_model=UserFeaturesResponse)
async def get_user_features(user: UserInDB = Depends(require_verified_user)):
    """Get user's tier features and limits."""
    tier_config = TIER_CONFIG[user.tier]
    used, remaining, limit = _get_rate_limit_for_user(user)

    return UserFeaturesResponse(
        tier=user.tier.value,
        features={
            "export_csv": tier_config.get("export_csv", False),
            "models": tier_config.get("models", []),
            "tasks": tier_config.get("tasks", []),
        },
        limits={
            "daily_requests": limit if limit != -1 else None,
            "requests_used": used,
            "requests_remaining": remaining if remaining != -1 else None,
            "max_stocks_per_request": tier_config.get("max_stocks_per_request", 5),
            "max_historical_days": tier_config.get("max_historical_days", 30),
        },
    )


@app.get("/api/analyze/models", response_model=ModelsAvailableResponse)
async def get_available_models(user: UserInDB = Depends(require_verified_user)):
    """Get available models for user's tier."""
    tier_config = TIER_CONFIG[user.tier]

    return ModelsAvailableResponse(
        models=tier_config.get("models", ["gradient_boosting", "linear_regression"]),
        tasks=tier_config.get("tasks", ["classifier"]),
        can_use_custom=user.tier == SubscriptionTier.PRO,
    )


@app.get("/api/analyze/universe")
async def get_analysis_universe(user: UserInDB = Depends(require_verified_user)):
    """Get available stocks organized by category for the analysis page."""
    tier_config = TIER_CONFIG[user.tier]
    stocks_limit = tier_config["stocks_limit"]

    if stocks_limit == -1:
        available_stocks = set(UNIVERSE)
    else:
        available_stocks = set(UNIVERSE[:stocks_limit])

    # Filter categories to only include stocks user has access to
    filtered_categories = {}
    for category, stocks in STOCK_CATEGORIES.items():
        available_in_category = [s for s in stocks if s in available_stocks]
        if available_in_category:
            filtered_categories[category] = available_in_category

    return {
        "stocks": list(available_stocks),
        "categories": filtered_categories,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def run_analysis(
    data: AnalyzeRequest,
    user: UserInDB = Depends(require_verified_user),
):
    """Run analysis on selected stocks."""
    tier_config = TIER_CONFIG[user.tier]
    logger.info(f"Analysis request from {user.email}: {len(data.tickers)} tickers, model={data.model}")

    # Check rate limit
    used, remaining, limit = _get_rate_limit_for_user(user)
    if limit != -1 and remaining <= 0:
        logger.warning(f"Rate limit exceeded for {user.email} ({used}/{limit} requests)")
        raise HTTPException(
            429,
            detail="Daily rate limit exceeded. Please upgrade your plan or try again tomorrow.",
        )

    # Validate number of stocks
    max_stocks = tier_config.get("max_stocks_per_request", 5)
    if len(data.tickers) > max_stocks:
        raise HTTPException(
            400,
            detail=f"Maximum {max_stocks} stocks per request for your tier.",
        )

    # Validate model is available
    available_models = tier_config.get("models", [])
    if data.model not in available_models:
        raise HTTPException(
            403,
            detail=f"Model '{data.model}' not available for your tier. Available: {', '.join(available_models)}",
        )

    # Validate task is available
    available_tasks = tier_config.get("tasks", [])
    if data.task not in available_tasks:
        raise HTTPException(
            403,
            detail=f"Task '{data.task}' not available for your tier. Available: {', '.join(available_tasks)}",
        )

    # Validate period based on max_historical_days
    max_days = tier_config.get("max_historical_days", 30)
    period_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
    requested_days = period_days.get(data.period, 30)
    if requested_days > max_days:
        raise HTTPException(
            403,
            detail=f"Period '{data.period}' requires more historical data than your tier allows. Maximum: {max_days} days.",
        )

    # Validate tickers are accessible
    stocks_limit = tier_config["stocks_limit"]
    if stocks_limit == -1:
        available_stocks = UNIVERSE
    else:
        available_stocks = UNIVERSE[:stocks_limit]

    available_upper = [s.upper() for s in available_stocks]
    invalid_tickers = [t for t in data.tickers if t.upper() not in available_upper]
    if invalid_tickers:
        raise HTTPException(
            400,
            detail=f"Tickers not available in your tier: {', '.join(invalid_tickers)}",
        )

    # Run analysis for each ticker
    results = []
    for ticker in data.tickers:
        try:
            prob_up, signal, last_close = predict_for_ticker(ticker.upper(), horizon=data.horizon)

            # For regression task, calculate a mock predicted return
            predicted_return = None
            if data.task == "regressor":
                import random
                predicted_return = (prob_up - 0.5) * 0.1 + random.uniform(-0.02, 0.02)

            results.append(AnalyzeResultItem(
                ticker=ticker.upper(),
                last_close=last_close,
                prob_up=prob_up,
                signal=signal,
                predicted_return=predicted_return,
            ))
        except Exception as e:
            # Add failed result
            results.append(AnalyzeResultItem(
                ticker=ticker.upper(),
                last_close=None,
                prob_up=None,
                signal=None,
                predicted_return=None,
            ))

    # Increment rate limit
    _increment_rate_limit(user)

    return AnalyzeResponse(
        results=results,
        metadata={
            "model": data.model,
            "task": data.task,
            "period": data.period,
            "horizon": data.horizon,
            "analyzed_at": datetime.utcnow().isoformat(),
        },
    )


# ============================================================
# Company Endpoints
# ============================================================

@app.get("/api/companies/{ticker}", response_model=CompanyDetailResponse)
async def get_company_detail(ticker: str):
    """Get company detail including info and recent news."""
    company = get_company(ticker.upper())
    if not company:
        raise HTTPException(404, detail=f"Company '{ticker.upper()}' not found")

    info = get_company_info(ticker.upper())
    news_rows = get_company_news(ticker.upper(), limit=20)

    company_resp = CompanyResponse(**company)

    info_resp = None
    if info:
        info_copy = {k: v for k, v in info.items() if k != "raw_info"}
        info_resp = CompanyInfoResponse(**info_copy)

    news_items = [
        CompanyNewsItem(
            article_id=n.get("article_id"),
            title=n.get("title"),
            publisher=n.get("publisher"),
            link=n.get("link"),
            published_at=n.get("published_at"),
            article_type=n.get("article_type"),
            thumbnail_url=n.get("thumbnail_url"),
            related_tickers=n.get("related_tickers"),
        )
        for n in news_rows
    ]

    return CompanyDetailResponse(company=company_resp, info=info_resp, news=news_items)


@app.get("/api/companies/{ticker}/news", response_model=list[CompanyNewsItem])
async def get_company_news_endpoint(ticker: str, limit: int = 50):
    """Get company news articles."""
    company = get_company(ticker.upper())
    if not company:
        raise HTTPException(404, detail=f"Company '{ticker.upper()}' not found")

    news_rows = get_company_news(ticker.upper(), limit=limit)
    return [
        CompanyNewsItem(
            article_id=n.get("article_id"),
            title=n.get("title"),
            publisher=n.get("publisher"),
            link=n.get("link"),
            published_at=n.get("published_at"),
            article_type=n.get("article_type"),
            thumbnail_url=n.get("thumbnail_url"),
            related_tickers=n.get("related_tickers"),
        )
        for n in news_rows
    ]


# ============================================================
# Static File Serving (React SPA)
# ============================================================

if FRONTEND_DIR.exists():
    # Serve static assets
    if (FRONTEND_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve React SPA - return index.html for all non-API routes."""
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {"message": "Frontend not built. Run 'npm run build' in frontend/ directory."}
