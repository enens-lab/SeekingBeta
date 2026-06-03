"""
Pythia Claude - FastAPI Service
Serves the React frontend, prediction endpoints, and authentication
"""
from __future__ import annotations

import sys
import asyncio
import time
import uuid
import os
import math
import csv
import re
import logging
import json
import base64
import unicodedata
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .logging_config import setup_logging, get_logger

# Initialize logging (may already be initialized by main.py, but safe to call again)
setup_logging()
logger = get_logger("service")
access_logger = get_logger("access")

# Ensure pythia is importable
PYTHIA_PATH = Path(__file__).parent.parent.parent / "pythia"
if PYTHIA_PATH.exists() and str(PYTHIA_PATH) not in sys.path:
    sys.path.insert(0, str(PYTHIA_PATH))

from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
try:
    import stripe
except Exception:  # pragma: no cover - optional import guard for non-billing environments
    stripe = None


from .models import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    TierInfo,
    MessageResponse,
    BetaTesterSignupRequest,
    BetaTesterSignupResponse,
    VerifyEmailRequest,
    ResendVerificationRequest,
    RefreshTokenRequest,
    PasswordResetRequest,
    PasswordResetConfirm,
    ChangePasswordRequest,
    DeleteAccountRequest,
    ErrorResponse,
    SubscriptionTier,
    TIER_CONFIG,
    MODEL_HORIZONS,
    PredictResponse,
    MarketHistoryResponse,
    OracleResponse,
    WatchlistInsightsResponse,
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
    UserPreferencesResponse,
    UpdatePreferencesRequest,
    TrackRecordResponse,
    TrackRecordCurveResponse,
    SportsBoardsResponse,
    SportsBoardCollection,
    BillingCheckoutSessionRequest,
    BillingCheckoutSessionResponse,
    BillingPortalSessionResponse,
    BillingChangeSubscriptionRequest,
    BillingChangeSubscriptionResponse,
    BillingStatusResponse,
    AppleVerifyRequest,
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
    upsert_company,
    upsert_company_info,
    update_user_tier,
    update_user_password,
    delete_user_account,
    list_users_by_tiers,
)
from .price_cache_store import (
    ensure_table as ensure_price_cache_table,
    upsert_last_close as upsert_last_close_pg,
    get_last_close as get_last_close_pg,
    is_enabled as price_cache_enabled,
)
from .compliance_store import (
    ensure_tables as ensure_compliance_tables,
    is_enabled as compliance_store_enabled,
    get_or_create_preferences,
    upsert_preferences,
    set_newsletter_opt_in,
    record_consent_event,
    audit_email_event,
    upsert_suppression,
    disable_newsletter_by_email,
    delete_user_records,
)
from .billing_store import (
    ensure_tables as ensure_billing_tables,
    is_enabled as billing_store_enabled,
    get_state_by_user_id,
    get_state_by_customer_id,
    upsert_customer_state,
    upsert_state_by_customer_id,
    delete_state_by_user_id,
    register_webhook_event,
    mark_webhook_event,
    list_states_with_expired_legacy_grace,
)
from .beta_program_store import (
    ensure_table as ensure_beta_program_table,
    is_enabled as beta_program_store_enabled,
    upsert_beta_tester,
)
from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    generate_verification_token,
    validate_password_strength,
)
from .email_service import send_verification_email, send_welcome_email, send_password_reset_email
from .performance import get_track_record, get_track_record_curve
from .models import UserInDB


def _normalize_ticker(value: str) -> str | None:
    ticker = value.strip().upper()
    if not ticker:
        return None
    for char in ticker:
        if not (char.isalnum() or char in ".-"):
            return None
    return ticker


def _dedupe_tickers(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        ticker = _normalize_ticker(raw)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def _load_local_universe(base_universe: list[str]) -> list[str]:
    csv_tickers: list[str] = []
    candidate_paths = [
        Path(__file__).parent / "universe.csv",              # bundled with prophecy api image
        Path(__file__).parent.parent / "data" / "universe.csv",  # runtime data volume override
    ]
    for csv_path in candidate_paths:
        if not csv_path.exists():
            continue
        import csv

        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if not row:
                    continue
                if row[0].strip().lower() in {"ticker", "symbol", "a"}:
                    continue
                ticker = _normalize_ticker(row[0])
                if ticker:
                    csv_tickers.append(ticker)
        if csv_tickers:
            break

    merged = _dedupe_tickers([*base_universe, *csv_tickers])
    return merged or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def _categorize_local(stocks: list[str]) -> dict[str, list[str]]:
    tech = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "AMD",
        "INTC",
        "AVGO",
        "QCOM",
        "TXN",
        "MU",
        "AMAT",
        "LRCX",
        "CRM",
        "ORCL",
        "ADBE",
        "NOW",
        "INTU",
        "IBM",
        "CSCO",
        "HPQ",
        "DELL",
    ]
    financials = [
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "USB",
        "PNC",
        "BRK-B",
        "AIG",
        "MET",
        "PRU",
        "BLK",
        "SCHW",
        "AXP",
        "V",
        "MA",
        "PYPL",
    ]
    healthcare = [
        "JNJ",
        "PFE",
        "MRK",
        "ABBV",
        "LLY",
        "BMY",
        "AMGN",
        "GILD",
        "REGN",
        "MRNA",
        "UNH",
        "CVS",
        "CI",
        "MDT",
        "ABT",
    ]
    consumer = [
        "HD",
        "LOW",
        "NKE",
        "SBUX",
        "MCD",
        "TGT",
        "TJX",
        "BKNG",
        "MAR",
        "F",
        "GM",
        "LULU",
        "WMT",
        "COST",
        "PG",
        "KO",
        "PEP",
        "PM",
        "MO",
        "CL",
    ]
    energy = ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY", "KMI"]
    industrials = ["CAT", "BA", "UPS", "FDX", "HON", "RTX", "LMT", "GE", "DE", "UNP"]
    etfs = [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "IVV",
        "MDY",
        "XLF",
        "XLK",
        "XLV",
        "XLE",
        "XLI",
        "XLY",
        "XLP",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
        "BND",
        "TLT",
        "IEF",
        "LQD",
        "HYG",
        "GLD",
        "SLV",
        "USO",
        "UNG",
        "EFA",
        "EEM",
        "VWO",
        "FXI",
        "EWJ",
    ]

    categories: dict[str, list[str]] = {}
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

    categorized = set()
    for cat_stocks in categories.values():
        categorized.update(cat_stocks)

    other = [s for s in stocks if s not in categorized]
    if not other:
        return categories

    if len(other) > 200:
        buckets: dict[str, list[str]] = {}
        for symbol in other:
            lead = symbol[0] if symbol else "#"
            key = lead if lead.isalpha() else "#"
            buckets.setdefault(key, []).append(symbol)
        for key in sorted(buckets):
            categories[f"Other ({key})"] = buckets[key]
        return categories

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
    DIVINATION_PATH = Path(__file__).parent.parent.parent / "pythia_divination"
    if DIVINATION_PATH.exists() and str(DIVINATION_PATH) not in sys.path:
        sys.path.insert(0, str(DIVINATION_PATH))
    try:
        from universe import load_universe as shared_load_universe
        from universe import categorize_stocks as shared_categorize_stocks
    except Exception:
        shared_load_universe = None
        shared_categorize_stocks = None

    # Load universe from config.yaml + optional CSV expansion (doesn't require DATABASE_URL)
    divination_config = load_divination_config()
    base_universe = divination_config.get("universe", [])
    if shared_load_universe:
        UNIVERSE = shared_load_universe(base_universe=base_universe)
    else:
        UNIVERSE = _load_local_universe(base_universe)
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
    if shared_categorize_stocks:
        STOCK_CATEGORIES = shared_categorize_stocks(UNIVERSE)
    else:
        STOCK_CATEGORIES = _categorize_local(UNIVERSE)
    UNIVERSE_UPPER = {s.upper() for s in UNIVERSE}

    logger.info(f"Loaded universe with {len(UNIVERSE)} stocks from pythia_divination")
    logger.info(f"Categories: {', '.join(f'{k} ({len(v)})' for k, v in STOCK_CATEGORIES.items())}")
except Exception as e:
    logger.warning(f"Could not load from pythia_divination: {e}")
    logger.info("Running in demo mode with mock data")
    PYTHIA_AVAILABLE = False
    UNIVERSE = _load_local_universe([])
    THRESHOLD = 0.55
    STOCK_CATEGORIES = _categorize_local(UNIVERSE)
    UNIVERSE_UPPER = {s.upper() for s in UNIVERSE}

# Simple in-memory rate limiting (resets on server restart).
# For horizontal scaling, move these counters to Redis or Postgres.
from collections import defaultdict, deque

# Rolling-window request counters for auth + webhooks.
REQUEST_RATE_LIMIT_STORAGE: dict[str, deque[float]] = defaultdict(deque)
# Daily counters for analysis request quotas by user tier.
ANALYSIS_RATE_LIMIT_STORAGE: dict[str, dict] = defaultdict(lambda: {"date": None, "count": 0})


def check_rate_limit(identifier: str, max_requests: int, window_seconds: int) -> tuple[bool, str]:
    """
    Rolling-window limiter.
    Returns (is_allowed, message).
    """
    if max_requests <= 0 or window_seconds <= 0:
        return False, "Rate limit configuration is invalid"

    now_ts = time.time()
    window_start = now_ts - window_seconds
    bucket = REQUEST_RATE_LIMIT_STORAGE[identifier]

    # Drop requests outside the current window.
    while bucket and bucket[0] <= window_start:
        bucket.popleft()

    if len(bucket) >= max_requests:
        retry_after = max(1, int(window_seconds - (now_ts - bucket[0])))
        return False, f"Rate limit exceeded. Try again in {retry_after} seconds"

    bucket.append(now_ts)
    return True, "OK"


def _available_stocks_for_tier(tier_config: dict[str, Any]) -> list[str]:
    """Return deterministic, category-diversified stocks for tier-limited users."""
    stocks_limit = int(tier_config.get("stocks_limit", -1))
    if stocks_limit == -1 or stocks_limit >= len(UNIVERSE):
        return list(UNIVERSE)
    if stocks_limit <= 0:
        return []

    selected: list[str] = []
    selected_upper: set[str] = set()

    def _append(symbol: str) -> None:
        upper = symbol.upper()
        if upper in selected_upper or upper not in UNIVERSE_UPPER:
            return
        selected.append(symbol)
        selected_upper.add(upper)

    primary_buckets: list[deque[str]] = []
    other_buckets: list[deque[str]] = []
    for category, symbols in STOCK_CATEGORIES.items():
        bucket = deque([symbol for symbol in symbols if symbol.upper() in UNIVERSE_UPPER])
        if not bucket:
            continue
        if category.lower().startswith("other"):
            other_buckets.append(bucket)
        else:
            primary_buckets.append(bucket)

    # Round-robin over primary categories first, then "Other" buckets.
    for buckets in (primary_buckets, other_buckets):
        progressed = True
        while len(selected) < stocks_limit and progressed:
            progressed = False
            for bucket in buckets:
                while bucket and bucket[0].upper() in selected_upper:
                    bucket.popleft()
                if not bucket:
                    continue
                _append(bucket.popleft())
                progressed = True
                if len(selected) >= stocks_limit:
                    break

    # Backfill from the raw universe ordering if category lists are exhausted.
    if len(selected) < stocks_limit:
        for symbol in UNIVERSE:
            _append(symbol)
            if len(selected) >= stocks_limit:
                break

    return selected


# FastAPI app
app = FastAPI(
    title="Pythia Claude API",
    description="""
# Pythia Claude - Stock Signal Intelligence Platform

An AI-powered stock prediction and analysis platform providing:
- Real-time stock signal predictions powered by machine learning
- Model-rated stock analysis (5-day and 20-day horizons)
- Personalized watchlists and alerts
- Advanced stock analysis tools

## Authentication
All protected endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <access_token>
```

## Rate Limiting
- Signup: 5 attempts per hour per email
- Login: 10 attempts per hour per email
- Email verification: 20 attempts per hour per token

## Subscription Tiers
- **Free**: 5 stocks, model ratings (5d/20d), basic signals
- **Basic**: 15 stocks, model ratings (5d/20d), email alerts, CSV export
- **Pro**: Unlimited stocks, model ratings (5d/20d), priority features

## Error Responses
All errors follow a standardized format:
```json
{
    "error": "Error type",
    "code": "ERROR_CODE",
    "message": "Detailed error message"
}
```
    """,
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    openapi_tags=[
        {"name": "Authentication", "description": "User signup, login, and token management"},
        {"name": "Community", "description": "Public community and beta program endpoints"},
        {"name": "Preferences", "description": "User settings and preferences"},
        {"name": "Predictions", "description": "Stock price predictions and signals"},
        {"name": "Oracle", "description": "Personal watchlist and alert management"},
        {"name": "Analysis", "description": "Advanced stock analysis tools"},
        {"name": "Companies", "description": "Company information and news"},
        {"name": "Subscriptions", "description": "Subscription tier information"},
        {"name": "Billing", "description": "Stripe billing sessions and status"},
        {"name": "System", "description": "Health checks and system status"},
    ],
)


# Startup validation
@app.on_event("startup")
async def startup_validation():
    """Validate configuration on startup."""
    errors = []
    warnings = []

    # Check environment
    environment = os.getenv("ENVIRONMENT", "development")
    logger.info(f"Starting in {environment} environment")

    # Check critical environment variables based on environment
    if environment == "production":
        email_provider = os.getenv("EMAIL_PROVIDER", "smtp").strip().lower()
        smtp_user = os.getenv("SMTP_USER", "").strip()
        smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
        postmark_token = os.getenv("POSTMARK_SERVER_TOKEN", "").strip()

        if not os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET_KEY") == "your-secret-key":
            errors.append("JWT_SECRET_KEY is not set securely in production")

        if os.getenv("EMAIL_DEV_MODE", "false").lower() == "true":
            errors.append("EMAIL_DEV_MODE should be disabled in production")

        if email_provider == "postmark":
            if not postmark_token and (not smtp_user or not smtp_password):
                warnings.append(
                    "Postmark email configured but no credentials found. "
                    "Set POSTMARK_SERVER_TOKEN or SMTP_USER/SMTP_PASSWORD."
                )
        else:
            if not smtp_user:
                warnings.append("SMTP_USER not configured - email sending will fail")
            if not smtp_password:
                warnings.append("SMTP_PASSWORD not configured - email sending will fail")

    # Check frontend URL
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    if not frontend_url:
        warnings.append("FRONTEND_URL not set - password reset emails may not work properly")

    # Check database connection (if needed)
    logger.info(f"Frontend URL: {frontend_url}")
    logger.info(f"CORS Origins: {ALLOWED_ORIGINS}")

    # Ensure Postgres-backed price cache table for dashboard last_close persistence.
    if price_cache_enabled():
        try:
            ensure_price_cache_table()
        except Exception as e:
            errors.append(f"Postgres price cache initialization failed: {e}")
    else:
        warnings.append("DATABASE_URL not set - dashboard last_close cache disabled")

    # Ensure Postgres-backed compliance tables for consent/suppression/preferences.
    if compliance_store_enabled():
        try:
            ensure_compliance_tables()
        except Exception as e:
            errors.append(f"Compliance store initialization failed: {e}")
    else:
        warnings.append("DATABASE_URL not set - compliance store disabled")

    # Ensure Postgres-backed beta tester applications table.
    if beta_program_store_enabled():
        try:
            ensure_beta_program_table()
        except Exception as e:
            errors.append(f"Beta program store initialization failed: {e}")
    else:
        warnings.append("DATABASE_URL not set - beta program signup storage disabled")

    # Validate Stripe configuration and ensure Postgres-backed billing tables.
    if STRIPE_ENABLED:
        if stripe is None:
            errors.append("stripe package is missing while STRIPE_ENABLED=true")
        if not STRIPE_SECRET_KEY:
            errors.append("STRIPE_SECRET_KEY is required when STRIPE_ENABLED=true")
        if not STRIPE_WEBHOOK_SECRET:
            errors.append("STRIPE_WEBHOOK_SECRET is required when STRIPE_ENABLED=true")
        if not STRIPE_PRICE_BASIC_MONTHLY:
            errors.append("STRIPE_PRICE_BASIC_MONTHLY is required when STRIPE_ENABLED=true")
        if not STRIPE_PRICE_PRO_MONTHLY:
            errors.append("STRIPE_PRICE_PRO_MONTHLY is required when STRIPE_ENABLED=true")
        if STRIPE_LEGACY_GRACE_DAYS < 0:
            errors.append("STRIPE_LEGACY_GRACE_DAYS cannot be negative")
        if not billing_store_enabled():
            errors.append("DATABASE_URL is required when STRIPE_ENABLED=true")

    if billing_store_enabled():
        try:
            ensure_billing_tables()
            if STRIPE_ENABLED:
                _seed_legacy_grace_state_for_existing_paid_users()
                expired_states = list_states_with_expired_legacy_grace()
                for state in expired_states:
                    upsert_customer_state(
                        state["user_id"],
                        state["email"],
                        updates={
                            "subscription_status": "grace_expired",
                            "plan_tier": SubscriptionTier.FREE.value,
                        },
                        source="startup_grace_enforcement",
                    )
                    update_user_tier(state["user_id"], SubscriptionTier.FREE)
        except Exception as e:
            errors.append(f"Billing store initialization failed: {e}")
    elif STRIPE_ENABLED:
        errors.append("Billing store disabled while STRIPE_ENABLED=true")

    if not SES_SNS_ALLOWED_TOPIC_ARNS:
        warnings.append("SES_SNS_ALLOWED_TOPIC_ARNS is empty - SES webhook will reject all topics")

    # Log warnings
    for warning in warnings:
        logger.warning(f"Configuration warning: {warning}")

    # Fail startup if there are errors
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise RuntimeError(f"Configuration errors found: {'; '.join(errors)}")

    logger.info("Startup validation completed successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Application shutting down")


# CORS configuration - lock down to specific origins in production
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost").split(",")
logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)


# Global exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors with standardized format."""
    logger.warning(f"Validation error in {request.url.path}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": [
                {
                    "field": ".".join(str(x) for x in error["loc"][1:]),
                    "message": error["msg"],
                    "type": error["type"],
                }
                for error in exc.errors()
            ]
        }
    )


# Global exception handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with standardized format."""
    logger.warning(f"HTTP {exc.status_code} in {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
        }
    )


# Global exception handler for all other exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions with standardized format."""
    logger.error(f"Unexpected error in {request.url.path}: {exc}", exc_info=True)

    # Don't expose sensitive details in production
    environment = os.getenv("ENVIRONMENT", "development")
    if environment == "production":
        message = "Internal server error"
    else:
        message = str(exc)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "code": "INTERNAL_SERVER_ERROR",
            "message": message,
        }
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all HTTP requests with timing and structured data."""
    start_time = time.time()

    # Generate request ID for tracing
    request_id = str(uuid.uuid4())

    # Get client info
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    query_string = str(request.url.query) if request.url.query else ""

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000

    # Log the request (skip static assets for cleaner logs)
    if not path.startswith("/assets/") and path != "/favicon.ico":
        # Create a LogRecord-like object with extra fields
        extra_fields = {
            "request_id": request_id,
            "duration_ms": duration_ms,
            "status_code": response.status_code,
        }

        # Log with extra context
        message = f"{method} {path} - {response.status_code} ({duration_ms:.1f}ms) from {client_ip}"
        record = access_logger.makeRecord(
            "pythia.access",
            logging.INFO,
            request.url.path,
            0,
            message,
            (),
            None,
            extra=extra_fields
        )
        access_logger.handle(record)

    return response


# Path to React build output
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"
SPORTS_DATA_SOURCE_DIR = Path(__file__).parent.parent / "frontend_data"
SPORTS_FRONTEND_SOURCE_DIR = Path(__file__).parent.parent / "frontend" / "src" / "data"
DIVINATION_API_URL = os.getenv("PYTHIA_API_URL", "http://divination-api:8000").rstrip("/")
PRICE_CACHE_MAX_AGE_HOURS = int(os.getenv("PRICE_CACHE_MAX_AGE_HOURS", "72"))
ALLOW_RANDOM_FALLBACK = os.getenv("ALLOW_RANDOM_FALLBACK", "false").lower() == "true"
DIVINATION_LSTM_TIMEOUT_SECONDS = float(os.getenv("DIVINATION_LSTM_TIMEOUT_SECONDS", "8"))
SPORTS_MLB_BOARDS_TIMEOUT_SECONDS = float(os.getenv("SPORTS_MLB_BOARDS_TIMEOUT_SECONDS", "60"))
TENNIS_UPCOMING_LOOKAHEAD_DAYS = int(os.getenv("TENNIS_UPCOMING_LOOKAHEAD_DAYS", "60"))
TENNIS_UPCOMING_PER_TOUR = {"ATP": 12, "WTA": 12}
TENNIS_ATP_LIVE_RESULTS_URL = "https://stats.tennismylife.org/data/{year}.csv"
TENNIS_LIVE_RESULTS_CACHE_TTL_SECONDS = 60 * 60
_TENNIS_ATP_RESULTS_CACHE: dict[int, tuple[float, list[dict[str, Any]]]] = {}
def _resolve_pga_normalized_dir() -> Path:
    """Locate the PGA schedule data directory.

    Prefer the prophecy-local copy at ``pythia_prophecy/data/sports/pga/normalized``
    (populated by ``make sync-sports-data`` and shipped via the Dockerfile's
    ``COPY data/`` step). Fall back to the cross-repo path for local dev where
    both repos sit side-by-side and the file has not been synced yet.
    """
    local = Path(__file__).resolve().parents[1] / "data" / "sports" / "pga" / "normalized"
    if local.exists():
        return local
    return Path(__file__).resolve().parents[2] / "pythia_divination" / "data" / "sports" / "pga" / "normalized"


PGA_NORMALIZED_DIR = _resolve_pga_normalized_dir()
LSTM_PROXY_DISABLE_LOCAL_FALLBACK = (
    os.getenv("LSTM_PROXY_DISABLE_LOCAL_FALLBACK", "true").lower() == "true"
)
POLICY_VERSION = os.getenv("POLICY_VERSION", "2026-02-27")
STRIPE_ENABLED = os.getenv("STRIPE_ENABLED", "false").lower() == "true"
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
STRIPE_PRICE_BASIC_MONTHLY = os.getenv("STRIPE_PRICE_BASIC_MONTHLY", "").strip()
STRIPE_PRICE_PRO_MONTHLY = os.getenv("STRIPE_PRICE_PRO_MONTHLY", "").strip()
APPLE_IAP_PRODUCT_BASIC_MONTHLY = os.getenv(
    "APPLE_IAP_PRODUCT_BASIC_MONTHLY",
    "ai.seekingbeta.basic.monthly",
).strip()
APPLE_IAP_PRODUCT_PRO_MONTHLY = os.getenv(
    "APPLE_IAP_PRODUCT_PRO_MONTHLY",
    "ai.seekingbeta.pro.monthly",
).strip()
STRIPE_LEGACY_GRACE_DAYS = int(os.getenv("STRIPE_LEGACY_GRACE_DAYS", "30"))
STRIPE_BILLING_PORTAL_RETURN_URL = os.getenv("STRIPE_BILLING_PORTAL_RETURN_URL", "").strip()
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID", "").strip()
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()
ALPACA_TRADING_BASE_URL = os.getenv(
    "ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets"
).rstrip("/")
ALPACA_COMPANY_HYDRATE_COOLDOWN_SECONDS = int(
    os.getenv("ALPACA_COMPANY_HYDRATE_COOLDOWN_SECONDS", "900")
)
COMPANY_AUTO_POPULATE_ON_DEMAND = (
    os.getenv("COMPANY_AUTO_POPULATE_ON_DEMAND", "true").lower() == "true"
)
SES_SNS_ALLOWED_TOPIC_ARNS = {
    arn.strip()
    for arn in os.getenv("SES_SNS_ALLOWED_TOPIC_ARNS", "").split(",")
    if arn.strip()
}
SES_SNS_AUTO_CONFIRM = os.getenv("SES_SNS_AUTO_CONFIRM", "false").lower() == "true"
AUTH_SIGNUP_RATE_LIMIT = int(os.getenv("AUTH_SIGNUP_RATE_LIMIT", "5"))
AUTH_SIGNUP_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_SIGNUP_RATE_WINDOW_SECONDS", "3600"))
AUTH_SIGNUP_IP_RATE_LIMIT = int(os.getenv("AUTH_SIGNUP_IP_RATE_LIMIT", "30"))
AUTH_SIGNUP_IP_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_SIGNUP_IP_RATE_WINDOW_SECONDS", "3600"))
AUTH_LOGIN_RATE_LIMIT = int(os.getenv("AUTH_LOGIN_RATE_LIMIT", "10"))
AUTH_LOGIN_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_LOGIN_RATE_WINDOW_SECONDS", "3600"))
AUTH_LOGIN_IP_RATE_LIMIT = int(os.getenv("AUTH_LOGIN_IP_RATE_LIMIT", "120"))
AUTH_LOGIN_IP_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_LOGIN_IP_RATE_WINDOW_SECONDS", "3600"))
AUTH_VERIFY_RATE_LIMIT = int(os.getenv("AUTH_VERIFY_RATE_LIMIT", "20"))
AUTH_VERIFY_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_VERIFY_RATE_WINDOW_SECONDS", "3600"))
AUTH_RESEND_RATE_LIMIT = int(os.getenv("AUTH_RESEND_RATE_LIMIT", "1"))
AUTH_RESEND_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_RESEND_RATE_WINDOW_SECONDS", "60"))
AUTH_PASSWORD_RESET_RATE_LIMIT = int(os.getenv("AUTH_PASSWORD_RESET_RATE_LIMIT", "5"))
AUTH_PASSWORD_RESET_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_PASSWORD_RESET_RATE_WINDOW_SECONDS", "900"))
AUTH_PASSWORD_RESET_CONFIRM_RATE_LIMIT = int(
    os.getenv("AUTH_PASSWORD_RESET_CONFIRM_RATE_LIMIT", "10")
)
AUTH_PASSWORD_RESET_CONFIRM_RATE_WINDOW_SECONDS = int(
    os.getenv("AUTH_PASSWORD_RESET_CONFIRM_RATE_WINDOW_SECONDS", "3600")
)
AUTH_REFRESH_RATE_LIMIT = int(os.getenv("AUTH_REFRESH_RATE_LIMIT", "60"))
AUTH_REFRESH_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_REFRESH_RATE_WINDOW_SECONDS", "300"))
AUTH_CHANGE_PASSWORD_RATE_LIMIT = int(os.getenv("AUTH_CHANGE_PASSWORD_RATE_LIMIT", "5"))
AUTH_CHANGE_PASSWORD_RATE_WINDOW_SECONDS = int(
    os.getenv("AUTH_CHANGE_PASSWORD_RATE_WINDOW_SECONDS", "3600")
)
AUTH_DELETE_ACCOUNT_RATE_LIMIT = int(os.getenv("AUTH_DELETE_ACCOUNT_RATE_LIMIT", "3"))
AUTH_DELETE_ACCOUNT_RATE_WINDOW_SECONDS = int(
    os.getenv("AUTH_DELETE_ACCOUNT_RATE_WINDOW_SECONDS", "3600")
)
PUBLIC_BETA_SIGNUP_RATE_LIMIT = int(os.getenv("PUBLIC_BETA_SIGNUP_RATE_LIMIT", "3"))
PUBLIC_BETA_SIGNUP_RATE_WINDOW_SECONDS = int(
    os.getenv("PUBLIC_BETA_SIGNUP_RATE_WINDOW_SECONDS", "3600")
)
PUBLIC_BETA_SIGNUP_IP_RATE_LIMIT = int(os.getenv("PUBLIC_BETA_SIGNUP_IP_RATE_LIMIT", "30"))
PUBLIC_BETA_SIGNUP_IP_RATE_WINDOW_SECONDS = int(
    os.getenv("PUBLIC_BETA_SIGNUP_IP_RATE_WINDOW_SECONDS", "3600")
)
SES_WEBHOOK_RATE_LIMIT = int(os.getenv("SES_WEBHOOK_RATE_LIMIT", "240"))
SES_WEBHOOK_RATE_WINDOW_SECONDS = int(os.getenv("SES_WEBHOOK_RATE_WINDOW_SECONDS", "60"))
STRIPE_WEBHOOK_RATE_LIMIT = int(os.getenv("STRIPE_WEBHOOK_RATE_LIMIT", "240"))
STRIPE_WEBHOOK_RATE_WINDOW_SECONDS = int(os.getenv("STRIPE_WEBHOOK_RATE_WINDOW_SECONDS", "60"))
DISCORD_INVITE_URL = os.getenv("DISCORD_INVITE_URL", "https://discord.gg/ckTC8JhWU9").strip()
FINVIZ_API_BASE_URL = os.getenv("FINVIZ_API_BASE_URL", "").strip().rstrip("/")
FINVIZ_API_KEY = os.getenv("FINVIZ_API_KEY", "").strip()
FINVIZ_API_AUTH_HEADER = os.getenv("FINVIZ_API_AUTH_HEADER", "X-API-KEY").strip() or "X-API-KEY"
FINVIZ_API_KEY_PREFIX = os.getenv("FINVIZ_API_KEY_PREFIX", "").strip()
FINVIZ_API_AUTH_QUERY_PARAM = os.getenv("FINVIZ_API_AUTH_QUERY_PARAM", "auth").strip()
FINVIZ_API_QUOTE_PATH_TEMPLATE = os.getenv(
    "FINVIZ_API_QUOTE_PATH_TEMPLATE", "/quote/{ticker}"
).strip() or "/quote/{ticker}"
FINVIZ_CHART_URL_TEMPLATE = os.getenv(
    "FINVIZ_CHART_URL_TEMPLATE",
    "https://finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l",
).strip()
FINVIZ_API_TIMEOUT_SECONDS = float(os.getenv("FINVIZ_API_TIMEOUT_SECONDS", "8"))
FINVIZ_INSIGHTS_CACHE_TTL_SECONDS = int(os.getenv("FINVIZ_INSIGHTS_CACHE_TTL_SECONDS", "120"))

ACTIVE_STRIPE_SUBSCRIPTION_STATUSES = {"active", "trialing", "past_due"}
TERMINAL_STRIPE_SUBSCRIPTION_STATUSES = {"canceled", "unpaid", "incomplete_expired"}

if stripe and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


_ALPACA_COMPANY_HYDRATE_BACKOFF_UNTIL: float = 0.0
_ALPACA_COMPANY_HYDRATE_LAST_LOG: float = 0.0
_FINVIZ_INSIGHTS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _sports_data_path(filename: str) -> Path:
    copied_path = SPORTS_DATA_SOURCE_DIR / filename
    if copied_path.exists():
        return copied_path
    return SPORTS_FRONTEND_SOURCE_DIR / filename


def _load_sports_json(filename: str) -> list[dict[str, Any]]:
    path = _sports_data_path(filename)
    if not path.exists():
        logger.warning("Sports data file not found: %s", path)
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse sports data file %s: %s", path, exc)
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    logger.warning("Sports data file %s did not contain a JSON list", path)
    return []


def _sports_data_updated_at(filenames: list[str]) -> datetime:
    mtimes: list[float] = []
    for filename in filenames:
        path = _sports_data_path(filename)
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    if not mtimes:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc)


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        compact = re.sub(r"\D", "", value)
        if compact:
            try:
                return int(compact)
            except Exception:
                return None
    return None


def _canonical_sports_name(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"^\d{4}\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _event_year(item: dict[str, Any]) -> Optional[int]:
    scheduled = _safe_int(item.get("scheduledDate"))
    if scheduled:
        return scheduled // 10000

    latest = _safe_int(item.get("latestDate"))
    if latest:
        return latest // 10000

    name = str(item.get("name") or "").strip()
    match = re.match(r"^(\d{4})\s+", name)
    if match:
        return int(match.group(1))
    return None


def _runtime_today_key() -> int:
    return int(datetime.now(timezone.utc).strftime("%Y%m%d"))


def _synthetic_next_occurrence_date(date_key: int, today_key: Optional[int] = None) -> Optional[int]:
    if not date_key:
        return None
    try:
        today = datetime.strptime(str(today_key or _runtime_today_key()), "%Y%m%d").date()
        month_day = int(date_key) % 10000
        month = month_day // 100
        day = month_day % 100
        candidate = datetime(today.year, month, day).date()
        if candidate < today:
            candidate = datetime(today.year + 1, month, day).date()
        return int(candidate.strftime("%Y%m%d"))
    except Exception:
        return None


def _build_dynamic_tennis_upcoming(backtests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    today_key = _runtime_today_key()
    today_dt = datetime.strptime(str(today_key), "%Y%m%d").date()
    max_date_key = int((today_dt + timedelta(days=TENNIS_UPCOMING_LOOKAHEAD_DAYS)).strftime("%Y%m%d"))
    per_tour_counts = {tour: 0 for tour in TENNIS_UPCOMING_PER_TOUR}
    seen: set[tuple[str, str]] = set()
    upcoming: list[dict[str, Any]] = []

    sorted_backtests = sorted(
        backtests,
        key=lambda row: (
            _synthetic_next_occurrence_date(_safe_int(row.get("latestDate")) or 0, today_key) or 99999999,
            str(row.get("tour") or ""),
            str(row.get("tournament") or ""),
        ),
    )

    for row in sorted_backtests:
        tour = str(row.get("tour") or "").upper()
        tournament = str(row.get("tournament") or "").strip()
        if not tour or not tournament:
            continue
        scheduled = _synthetic_next_occurrence_date(_safe_int(row.get("latestDate")) or 0, today_key)
        if not scheduled or scheduled < today_key or scheduled > max_date_key:
            continue
        key = (tour, _canonical_sports_name(tournament))
        if key in seen:
            continue
        if per_tour_counts.get(tour, 0) >= TENNIS_UPCOMING_PER_TOUR.get(tour, 10):
            continue
        seen.add(key)
        per_tour_counts[tour] = per_tour_counts.get(tour, 0) + 1
        upcoming.append(
            {
                "id": f"{tour.lower()}-{_canonical_sports_name(tournament).replace(' ', '-')}",
                "name": f"{datetime.strptime(str(scheduled), '%Y%m%d').year} {tournament}",
                "tour": tour,
                "course": str(row.get("surface") or "Unknown"),
                "surface": str(row.get("surface") or "Unknown"),
                "scheduledDate": scheduled,
                "latestDate": scheduled,
                "predictedWinner": row.get("predictedWinner"),
                "predictions": row.get("fullField") or [],
            }
        )

    return upcoming


def _canonical_tennis_event_name(value: Optional[str]) -> str:
    text = _canonical_sports_name(value)
    text = text.replace("monte-carlo", "monte carlo")
    text = re.sub(r"\bmasters\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonical_player_name(value: Optional[str]) -> str:
    text = unicodedata.normalize("NFKD", (value or "").strip())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace(".", "")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def _fetch_live_atp_completed_tournaments(current_year: int) -> list[dict[str, Any]]:
    cached = _TENNIS_ATP_RESULTS_CACHE.get(current_year)
    if cached and (time.time() - cached[0]) < TENNIS_LIVE_RESULTS_CACHE_TTL_SECONDS:
        return cached[1]

    url = TENNIS_ATP_LIVE_RESULTS_URL.format(year=current_year)
    try:
        response = httpx.get(url, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Unable to fetch live ATP results for %s: %s", current_year, exc)
        return cached[1] if cached else []

    today_key = _runtime_today_key()
    events: dict[tuple[str, int], dict[str, Any]] = {}
    reader = csv.DictReader(StringIO(response.text))
    for row in reader:
        if str(row.get("round") or "").strip().upper() != "F":
            continue
        tournament = str(row.get("tourney_name") or "").strip()
        date_key = _safe_int(row.get("tourney_date"))
        winner = str(row.get("winner_name") or "").strip()
        if not tournament or not date_key or not winner or date_key > today_key:
            continue
        canonical = _canonical_tennis_event_name(tournament)
        if not canonical:
            continue
        events[(canonical, date_key)] = {
            "tournament": tournament,
            "canonical": canonical,
            "latestDate": date_key,
            "actualWinner": winner,
            "surface": str(row.get("surface") or "").strip() or None,
        }

    completed = sorted(events.values(), key=lambda item: item["latestDate"], reverse=True)
    _TENNIS_ATP_RESULTS_CACHE[current_year] = (time.time(), completed)
    return completed


def _build_runtime_tennis_backtests(
    upcoming: list[dict[str, Any]],
    backtests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_year = datetime.now(timezone.utc).year
    existing_keys = {
        (
            str(row.get("tour") or "").upper(),
            _canonical_tennis_event_name(str(row.get("tournament") or "")),
            _safe_int(row.get("year")),
        )
        for row in backtests
        if row.get("tournament") and row.get("tour") and row.get("year")
    }

    predictions_by_event: dict[tuple[str, str], dict[str, Any]] = {}
    for item in upcoming:
        if str(item.get("tour") or "").upper() != "ATP":
            continue
        if _event_year(item) != current_year:
            continue
        key = ("ATP", _canonical_tennis_event_name(str(item.get("name") or "")))
        predictions_by_event[key] = item

    runtime_backtests: list[dict[str, Any]] = []
    for result in _fetch_live_atp_completed_tournaments(current_year):
        key = ("ATP", str(result["canonical"]))
        if (*key, current_year) in existing_keys:
            continue
        item = predictions_by_event.get(key)
        if not item:
            continue

        predictions = [dict(entry) for entry in (item.get("predictions") or []) if isinstance(entry, dict)]
        if not predictions:
            continue

        actual_winner = str(result["actualWinner"])
        actual_winner_key = _canonical_player_name(actual_winner)
        predicted_names = [str(entry.get("playerName") or "").strip() for entry in predictions]
        predicted_keys = [_canonical_player_name(name) for name in predicted_names]
        predicted_winner = predicted_names[0] if predicted_names else ""

        if actual_winner_key and predicted_keys and actual_winner_key == predicted_keys[0]:
            hit_status = "Top Pick"
        elif actual_winner_key and actual_winner_key in predicted_keys[:3]:
            hit_status = "Top 3"
        elif actual_winner_key and actual_winner_key in predicted_keys[:5]:
            hit_status = "Top 5"
        else:
            hit_status = "Miss"

        full_field: list[dict[str, Any]] = []
        for index, entry in enumerate(predictions, start=1):
            player_name = str(entry.get("playerName") or "").strip()
            enriched = dict(entry)
            enriched["rank"] = int(entry.get("rank") or index)
            enriched["actualWinner"] = _canonical_player_name(player_name) == actual_winner_key
            full_field.append(enriched)

        runtime_backtests.append(
            {
                "year": current_year,
                "tournament": re.sub(r"^\d{4}\s+", "", str(item.get("name") or "")).strip() or str(result["tournament"]),
                "tour": "ATP",
                "surface": str(item.get("surface") or item.get("course") or result.get("surface") or "Unknown"),
                "predictedWinner": predicted_winner,
                "predictedTop3": predicted_names[:3],
                "predictedTop5": predicted_names[:5],
                "actualWinner": actual_winner,
                "hitStatus": hit_status,
                "prob": float((_parse_float((predictions[0] or {}).get("winProbability")) or 0.0) / 100.0),
                "fullField": full_field,
                "latestDate": int(result["latestDate"]),
                "tournamentId": str(item.get("id") or f"ATP:{current_year}:{result['canonical']}"),
            }
        )

    return runtime_backtests


def _parse_golf_display_end_date(display_date: str, *, current_year: int) -> Optional[int]:
    text = (display_date or "").strip()
    if not text:
        return None
    match = re.match(
        r"^(?P<start_month>[A-Za-z]{3})\s+\d{1,2}\s*-\s*(?:(?P<end_month>[A-Za-z]{3})\s+)?(?P<end_day>\d{1,2})$",
        text,
    )
    if not match:
        return None
    month_token = (match.group("end_month") or match.group("start_month") or "").title()
    month_map = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    month = month_map.get(month_token)
    if month is None:
        return None
    try:
        day = int(match.group("end_day"))
        return int(datetime(current_year, month, day).strftime("%Y%m%d"))
    except Exception:
        return None


def _build_runtime_golf_backtests(
    upcoming: list[dict[str, Any]],
    backtests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_year = datetime.now(timezone.utc).year
    schedule_path = PGA_NORMALIZED_DIR / f"schedule_{current_year}_latest.csv"
    if not schedule_path.exists():
        return []

    existing_keys = {
        (
            str(row.get("tour") or "").upper(),
            _canonical_sports_name(str(row.get("tournament") or "")),
        )
        for row in backtests
        if _safe_int(row.get("year")) == current_year and row.get("tournament") and row.get("tour")
    }
    predictions_by_event: dict[tuple[str, str], dict[str, Any]] = {}
    for item in upcoming:
        if _event_year(item) not in {None, current_year} and _event_year(item) != current_year:
            continue
        tour = str(item.get("tour") or "").upper()
        if tour != "PGA":
            continue
        event_name = str(item.get("original_name") or item.get("name") or "").strip()
        if not event_name:
            continue
        predictions_by_event[(tour, _canonical_sports_name(event_name))] = item

    today_key = _runtime_today_key()
    runtime_backtests: list[dict[str, Any]] = []
    with schedule_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tournament = str(row.get("tournament_name") or "").strip()
            champion = str(row.get("champion_name") or "").strip()
            if not tournament or not champion:
                continue
            latest_date = _parse_golf_display_end_date(str(row.get("display_date") or ""), current_year=current_year)
            if not latest_date or latest_date > today_key:
                continue
            key = ("PGA", _canonical_sports_name(tournament))
            if key in existing_keys:
                continue
            item = predictions_by_event.get(key)
            if not item:
                continue
            predictions = [dict(entry) for entry in (item.get("predictions") or []) if isinstance(entry, dict)]
            if not predictions:
                continue

            predicted_names = [str(entry.get("playerName") or "").strip() for entry in predictions]
            predicted_winner = predicted_names[0] if predicted_names else ""
            if champion == predicted_winner:
                hit_status = "Top Pick"
            elif champion in predicted_names[:3]:
                hit_status = "Top 3"
            elif champion in predicted_names[:5]:
                hit_status = "Top 5"
            else:
                hit_status = "Miss"

            full_field: list[dict[str, Any]] = []
            for index, entry in enumerate(predictions, start=1):
                player_name = str(entry.get("playerName") or "").strip()
                enriched = dict(entry)
                enriched["rank"] = int(entry.get("rank") or index)
                enriched["actualWinner"] = player_name == champion
                full_field.append(enriched)

            runtime_backtests.append(
                {
                    "year": current_year,
                    "tournament": tournament,
                    "tour": "PGA",
                    "predictedWinner": predicted_winner,
                    "predictedTop3": predicted_names[:3],
                    "predictedTop5": predicted_names[:5],
                    "actualWinner": champion,
                    "hitStatus": hit_status,
                    "prob": float((_parse_float((predictions[0] or {}).get("winProbability")) or 0.0) / 100.0),
                    "fullField": full_field,
                    "latestDate": latest_date,
                }
            )

    return runtime_backtests


def _build_mlb_available_dates(upcoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_day: dict[str, int] = {}
    for item in upcoming:
        scheduled = _safe_int(item.get("scheduledDate"))
        if not scheduled:
            continue
        key = datetime.strptime(str(scheduled), "%Y%m%d").strftime("%Y-%m-%d")
        per_day[key] = per_day.get(key, 0) + 1

    available: list[dict[str, Any]] = []
    for key in sorted(per_day.keys()):
        label = datetime.strptime(key, "%Y-%m-%d").strftime("%a, %b %d").replace(" 0", " ")
        available.append({"dateKey": key, "label": label, "gameCount": per_day[key]})
    return available


def _sports_backtest_key(item: dict[str, Any]) -> tuple[str, str, int]:
    tour = str(item.get("tour") or "").upper()
    tournament = _canonical_sports_name(str(item.get("tournament") or item.get("name") or ""))
    date_key = _safe_int(item.get("scheduledDate")) or _safe_int(item.get("latestDate")) or 0
    return tour, tournament, date_key


def _sort_sports_backtests(backtests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        backtests,
        key=lambda item: (
            _safe_int(item.get("latestDate")) or _safe_int(item.get("scheduledDate")) or 0,
            str(item.get("tournament") or item.get("name") or ""),
        ),
        reverse=True,
    )


def _merge_runtime_backtests(
    base_backtests: list[dict[str, Any]],
    runtime_backtests: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in base_backtests:
        merged[_sports_backtest_key(item)] = item
    for item in runtime_backtests or []:
        merged[_sports_backtest_key(item)] = item
    return _sort_sports_backtests(list(merged.values()))


def _build_sports_season_summary(
    backtests: list[dict[str, Any]],
    *,
    current_year: int | None = None,
) -> dict[str, Any] | None:
    target_year = current_year or datetime.now(timezone.utc).year
    season_rows = [row for row in backtests if _safe_int(row.get("year")) == target_year]
    if not season_rows:
        return {
            "year": target_year,
            "sampleSize": 0,
            "topPickHits": 0,
            "topPickAccuracy": None,
            "top3Hits": None,
            "top3Accuracy": None,
            "top5Hits": None,
            "top5Accuracy": None,
        }

    sample_size = len(season_rows)
    top_pick_hits = sum(1 for row in season_rows if str(row.get("hitStatus") or "").strip() == "Top Pick")
    top3_hits = sum(1 for row in season_rows if str(row.get("hitStatus") or "").strip() in {"Top Pick", "Top 3"})
    top5_hits = sum(1 for row in season_rows if str(row.get("hitStatus") or "").strip() in {"Top Pick", "Top 3", "Top 5"})
    has_place_hits = any(str(row.get("hitStatus") or "").strip() in {"Top 3", "Top 5"} for row in season_rows)
    return {
        "year": target_year,
        "sampleSize": sample_size,
        "topPickHits": top_pick_hits,
        "topPickAccuracy": top_pick_hits / sample_size if sample_size else None,
        "top3Hits": top3_hits if has_place_hits else None,
        "top3Accuracy": (top3_hits / sample_size) if has_place_hits and sample_size else None,
        "top5Hits": top5_hits if has_place_hits else None,
        "top5Accuracy": (top5_hits / sample_size) if has_place_hits and sample_size else None,
    }


# Max historical backtests embedded per sport in the API response. The full set
# (mlb alone has ~3,400) ballooned /api/sports/boards to ~60MB / ~19s. The
# season summary is computed from the FULL set first (accuracy unchanged), then
# the embedded list is capped to the most recent N (sorted newest-first, so the
# current season's track record is preserved for the History tab + the iOS
# fallback summary). 0 / negative disables the cap.
SPORTS_BACKTESTS_RESPONSE_CAP = int(os.getenv("SPORTS_BACKTESTS_RESPONSE_CAP", "150"))


def _build_sports_board_collection(
    *,
    upcoming: list[dict[str, Any]],
    backtests: list[dict[str, Any]],
    updated_at: datetime,
    source: str,
    selected_date: str | None = None,
    available_dates: list[dict[str, Any]] | None = None,
) -> SportsBoardCollection:
    # Compute the season summary from the full history BEFORE capping.
    season_summary = _build_sports_season_summary(backtests)
    sorted_backtests = _sort_sports_backtests(backtests)
    if SPORTS_BACKTESTS_RESPONSE_CAP > 0:
        sorted_backtests = sorted_backtests[:SPORTS_BACKTESTS_RESPONSE_CAP]
    return SportsBoardCollection(
        upcoming=upcoming,
        backtests=sorted_backtests,
        updated_at=updated_at,
        source=source,
        selectedDate=selected_date,
        availableDates=available_dates or [],
        seasonSummary=season_summary,
    )


def _build_dated_collection_from_upcoming(
    upcoming: list[dict[str, Any]],
    backtests: list[dict[str, Any]],
    *,
    selected_date: str | None,
    updated_at: datetime,
    source: str,
) -> SportsBoardCollection:
    today_key = _runtime_today_key()
    filtered = sorted(
        [
            item
            for item in upcoming
            if (_safe_int(item.get("scheduledDate")) or 0) >= today_key
        ],
        key=lambda item: (
            _safe_int(item.get("scheduledDate")) or 99999999,
            str(item.get("name") or ""),
        ),
    )
    available_dates = _build_mlb_available_dates(filtered)
    available_keys = {str(item["dateKey"]) for item in available_dates}
    resolved_selected = selected_date if selected_date and selected_date in available_keys else None
    if resolved_selected is None and available_dates:
        resolved_selected = str(available_dates[0]["dateKey"])

    if resolved_selected:
        resolved_date_key = int(datetime.strptime(resolved_selected, "%Y-%m-%d").strftime("%Y%m%d"))
        filtered = [item for item in filtered if (_safe_int(item.get("scheduledDate")) or 0) == resolved_date_key]
    else:
        filtered = []

    return _build_sports_board_collection(
        upcoming=filtered,
        backtests=backtests,
        updated_at=updated_at,
        source=source,
        selected_date=resolved_selected,
        available_dates=available_dates,
    )


ESPN_TENNIS_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis"
TENNIS_LIVE_SCHEDULE_DAYS_AHEAD = 80
TENNIS_LIVE_SCHEDULE_CACHE_TTL_SECONDS = int(os.getenv("TENNIS_LIVE_SCHEDULE_CACHE_TTL_SECONDS", "900"))
TENNIS_LIVE_SCHEDULE_TIMEOUT_SECONDS = float(os.getenv("TENNIS_LIVE_SCHEDULE_TIMEOUT_SECONDS", "12"))
# (timestamp, schedule dict) — mirrors the _TENNIS_ATP_RESULTS_CACHE pattern.
_TENNIS_LIVE_SCHEDULE_CACHE: tuple[float, dict[tuple[str, str], dict[str, Any]]] | None = None


# Normalize differing names for the same event across the static board and ESPN
# (e.g. the women's major is "French Open" in our data, "Roland Garros" on ESPN).
_TENNIS_NAME_ALIASES = {
    "french open": "roland garros",
    "the championships wimbledon": "wimbledon",
    "championships wimbledon": "wimbledon",
    "us open tennis championships": "us open",
    "australian open tennis": "australian open",
}


def _tennis_match_key(tour: str, name: str) -> tuple[str, str]:
    """Match key that ignores a leading year and normalizes major-name aliases
    so "2026 French Open" (static board) lines up with "Roland Garros" (ESPN)."""
    stripped = re.sub(r"^\s*(?:19|20)\d{2}\s+", "", str(name or ""))
    canonical = _canonical_sports_name(stripped)
    canonical = _TENNIS_NAME_ALIASES.get(canonical, canonical)
    return (str(tour or "").upper(), canonical)


def _fetch_espn_tennis_schedule() -> dict[tuple[str, str], dict[str, Any]]:
    """Live ATP + WTA tournament schedule from ESPN, keyed by (TOUR, canonical
    name) -> {name, tour, startKey, endKey, venue}. Cached; never raises."""
    global _TENNIS_LIVE_SCHEDULE_CACHE
    cached = _TENNIS_LIVE_SCHEDULE_CACHE
    if cached is not None and (time.time() - cached[0]) < TENNIS_LIVE_SCHEDULE_CACHE_TTL_SECONDS:
        return cached[1]

    today = datetime.now(timezone.utc).date()
    window = f"{today:%Y%m%d}-{today + timedelta(days=TENNIS_LIVE_SCHEDULE_DAYS_AHEAD):%Y%m%d}"
    schedule: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with httpx.Client(timeout=TENNIS_LIVE_SCHEDULE_TIMEOUT_SECONDS) as client:
            for tour in ("atp", "wta"):
                resp = client.get(
                    f"{ESPN_TENNIS_BASE_URL}/{tour}/scoreboard",
                    params={"dates": window, "limit": 300},
                )
                resp.raise_for_status()
                for event in resp.json().get("events", []) or []:
                    name = str(event.get("name") or "").strip()
                    start_key = _safe_int(str(event.get("date") or "")[:10].replace("-", ""))
                    end_key = _safe_int(str(event.get("endDate") or "")[:10].replace("-", "")) or start_key
                    if not name or not start_key:
                        continue
                    comp = (event.get("competitions") or [{}])[0]
                    venue = (comp.get("venue") or {}).get("fullName")
                    schedule[_tennis_match_key(tour, name)] = {
                        "name": name,
                        "tour": tour.upper(),
                        "startKey": start_key,
                        "endKey": end_key,
                        "venue": venue,
                    }
    except Exception as exc:  # pragma: no cover - network
        logger.warning("Live tennis schedule fetch failed: %s", exc)
        return cached[1] if cached else {}

    _TENNIS_LIVE_SCHEDULE_CACHE = (time.time(), schedule)
    return schedule


def _apply_live_tennis_schedule(upcoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Correct tennis tournament dates from the live ESPN schedule so events
    (especially Grand Slams) track their real windows instead of drifting with
    the static board. Predictions are preserved.

    We only override dates on tournaments already in the curated board; we do
    NOT append ESPN's full calendar, because those entries have no model field
    (empty predictions) and ESPN's naming would duplicate curated events. The
    curated board stays the source of which tournaments show + their fields."""
    schedule = _fetch_espn_tennis_schedule()
    if not schedule:
        return upcoming

    enriched: list[dict[str, Any]] = []
    for item in upcoming:
        key = _tennis_match_key(str(item.get("tour") or ""), str(item.get("name") or ""))
        live = schedule.get(key)
        if live:
            item = dict(item)
            item["scheduledDate"] = live["startKey"]
            item["latestDate"] = live["endKey"]
            if not item.get("course") and live.get("venue"):
                item["course"] = live["venue"]
        enriched.append(item)
    return enriched


# How long a tennis tournament can stay on the "upcoming" board after its start
# date when no explicit end date is present. Grand Slams run ~2 weeks, so a
# fortnight-plus grace keeps an in-progress major (e.g. Roland Garros) visible
# instead of dropping it the day after its first match. Finished events still
# fall off once they appear in completed results (the `completed` dedup below).
TENNIS_UPCOMING_GRACE_DAYS = 16


def _filter_upcoming_tennis(
    upcoming: list[dict[str, Any]],
    backtests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not any(_safe_int(item.get("scheduledDate")) for item in upcoming):
        upcoming = _build_dynamic_tennis_upcoming(backtests)

    today_key = _runtime_today_key()
    grace_cutoff_key = int(
        (datetime.now(timezone.utc).date() - timedelta(days=TENNIS_UPCOMING_GRACE_DAYS)).strftime("%Y%m%d")
    )
    completed = {
        (
            str(row.get("tour") or "").upper(),
            _canonical_sports_name(str(row.get("tournament") or "")),
            _safe_int(row.get("year")),
        )
        for row in backtests
        if row.get("tournament") and row.get("tour") and row.get("year")
    }

    filtered: list[dict[str, Any]] = []
    for item in upcoming:
        scheduled = _safe_int(item.get("scheduledDate"))
        end = _safe_int(item.get("latestDate"))
        # Drop only tournaments that have clearly finished: past their end date,
        # or — when no end date is provided — older than the grace window. This
        # keeps an in-progress multi-week event (e.g. a Grand Slam mid-fortnight)
        # on the board instead of dropping it the day after it starts.
        if end:
            if end < today_key:
                continue
        elif scheduled and scheduled < grace_cutoff_key:
            continue
        item_year = _event_year(item)
        event_name = _canonical_sports_name(str(item.get("name") or ""))
        tour = str(item.get("tour") or "").upper()
        if (tour, event_name, item_year) in completed:
            continue
        filtered.append(item)
    return sorted(
        filtered,
        key=lambda item: (
            _safe_int(item.get("scheduledDate")) or 99999999,
            str(item.get("tour") or ""),
            str(item.get("name") or ""),
        ),
    )


def _filter_upcoming_golf(
    upcoming: list[dict[str, Any]],
    backtests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_year = datetime.now(timezone.utc).year
    completed = {
        (str(row.get("tour") or "").upper(), _canonical_sports_name(str(row.get("tournament") or "")))
        for row in backtests
        if _safe_int(row.get("year")) == current_year
    }
    filtered: list[dict[str, Any]] = []
    for item in upcoming:
        tour = str(item.get("tour") or "").upper()
        event_name = _canonical_sports_name(str(item.get("original_name") or item.get("name") or ""))
        if (tour, event_name) in completed:
            continue
        filtered.append(item)
    return filtered


def _sports_board_collection(
    upcoming_filename: str,
    backtests_filename: str,
    sport: str,
    selected_date: str | None = None,
) -> SportsBoardCollection:
    upcoming = _load_sports_json(upcoming_filename)
    backtests = _load_sports_json(backtests_filename)

    if sport == "tennis":
        upcoming = _apply_live_tennis_schedule(upcoming)
        runtime_backtests = _build_runtime_tennis_backtests(upcoming, backtests)
        backtests = _merge_runtime_backtests(backtests, runtime_backtests)
        upcoming = _filter_upcoming_tennis(upcoming, backtests)
    elif sport == "golf":
        runtime_backtests = _build_runtime_golf_backtests(upcoming, backtests)
        backtests = _merge_runtime_backtests(backtests, runtime_backtests)
        upcoming = _filter_upcoming_golf(upcoming, backtests)

    updated_at = _sports_data_updated_at([upcoming_filename, backtests_filename])
    if sport in {"mlb", "football"}:
        return _build_dated_collection_from_upcoming(
            upcoming,
            backtests,
            selected_date=selected_date,
            updated_at=updated_at,
            source="runtime_filtered_sports_feed",
        )
    return _build_sports_board_collection(
        upcoming=upcoming,
        backtests=backtests,
        updated_at=updated_at,
        source="runtime_filtered_sports_feed",
    )


async def _live_mlb_board_collection(mlb_date: str | None = None) -> Optional[SportsBoardCollection]:
    backtests_filename = "mlb_historical_backtests.json"
    backtests = _load_sports_json(backtests_filename)
    fallback_updated_at = _sports_data_updated_at([backtests_filename])

    try:
        async with httpx.AsyncClient(timeout=SPORTS_MLB_BOARDS_TIMEOUT_SECONDS) as client:
            params = {"mlb_date": mlb_date} if mlb_date else None
            response = await client.get(f"{DIVINATION_API_URL}/api/sports/mlb/boards", params=params)
            response.raise_for_status()
            payload = response.json()
            upcoming = payload.get("upcoming") if isinstance(payload, dict) else None
            if isinstance(upcoming, list) and not upcoming:
                refresh_response = await client.get(
                    f"{DIVINATION_API_URL}/api/sports/mlb/boards",
                    params={"force_refresh": "true", **({"mlb_date": mlb_date} if mlb_date else {})},
                )
                refresh_response.raise_for_status()
                payload = refresh_response.json()
                upcoming = payload.get("upcoming") if isinstance(payload, dict) else None
        runtime_backtests = payload.get("completed") if isinstance(payload, dict) else None
        if not isinstance(runtime_backtests, list):
            runtime_backtests = []
        merged_backtests = _merge_runtime_backtests(backtests, runtime_backtests)
        updated_at_raw = payload.get("updated_at") if isinstance(payload, dict) else None
        if not isinstance(upcoming, list):
            return None
        updated_at = _to_utc_datetime(updated_at_raw) or fallback_updated_at
        return _build_sports_board_collection(
            upcoming=upcoming,
            backtests=merged_backtests,
            updated_at=updated_at,
            source=str(payload.get("source") or "divination_live_mlb_feed"),
            selected_date=str(payload.get("selectedDate")) if payload.get("selectedDate") else None,
            available_dates=payload.get("availableDates") or [],
        )
    except Exception as exc:
        logger.warning("Falling back to cached MLB board feed: %s", exc)
        return None


async def _live_basketball_board_collection(basketball_date: str | None = None) -> Optional[SportsBoardCollection]:
    backtests_filename = "basketball_historical_backtests.json"
    backtests = _load_sports_json(backtests_filename)
    fallback_updated_at = _sports_data_updated_at([backtests_filename])

    try:
        async with httpx.AsyncClient(timeout=SPORTS_MLB_BOARDS_TIMEOUT_SECONDS) as client:
            params = {"basketball_date": basketball_date} if basketball_date else None
            response = await client.get(f"{DIVINATION_API_URL}/api/sports/basketball/boards", params=params)
            response.raise_for_status()
            payload = response.json()
            upcoming = payload.get("upcoming") if isinstance(payload, dict) else None
            if isinstance(upcoming, list) and not upcoming:
                refresh_response = await client.get(
                    f"{DIVINATION_API_URL}/api/sports/basketball/boards",
                    params={"force_refresh": "true", **({"basketball_date": basketball_date} if basketball_date else {})},
                )
                refresh_response.raise_for_status()
                payload = refresh_response.json()
                upcoming = payload.get("upcoming") if isinstance(payload, dict) else None
        runtime_backtests = payload.get("completed") if isinstance(payload, dict) else None
        if not isinstance(runtime_backtests, list):
            runtime_backtests = []
        merged_backtests = _merge_runtime_backtests(backtests, runtime_backtests)
        updated_at_raw = payload.get("updated_at") if isinstance(payload, dict) else None
        if not isinstance(upcoming, list):
            return None
        updated_at = _to_utc_datetime(updated_at_raw) or fallback_updated_at
        return _build_sports_board_collection(
            upcoming=upcoming,
            backtests=merged_backtests,
            updated_at=updated_at,
            source=str(payload.get("source") or "divination_live_basketball_feed"),
            selected_date=str(payload.get("selectedDate")) if payload.get("selectedDate") else None,
            available_dates=payload.get("availableDates") or [],
        )
    except Exception as exc:
        logger.warning("Falling back to cached Basketball board feed: %s", exc)
        return None


async def _live_football_board_collection(football_date: str | None = None) -> Optional[SportsBoardCollection]:
    backtests_filename = "football_historical_backtests.json"
    backtests = _load_sports_json(backtests_filename)
    fallback_updated_at = _sports_data_updated_at([backtests_filename])

    try:
        async with httpx.AsyncClient(timeout=SPORTS_MLB_BOARDS_TIMEOUT_SECONDS) as client:
            params = {"football_date": football_date} if football_date else None
            response = await client.get(f"{DIVINATION_API_URL}/api/sports/football/boards", params=params)
            response.raise_for_status()
            payload = response.json()
            upcoming = payload.get("upcoming") if isinstance(payload, dict) else None
            if isinstance(upcoming, list) and not upcoming:
                refresh_response = await client.get(
                    f"{DIVINATION_API_URL}/api/sports/football/boards",
                    params={"force_refresh": "true", **({"football_date": football_date} if football_date else {})},
                )
                refresh_response.raise_for_status()
                payload = refresh_response.json()
                upcoming = payload.get("upcoming") if isinstance(payload, dict) else None
        runtime_backtests = payload.get("completed") if isinstance(payload, dict) else None
        if not isinstance(runtime_backtests, list):
            runtime_backtests = []
        merged_backtests = _merge_runtime_backtests(backtests, runtime_backtests)
        updated_at_raw = payload.get("updated_at") if isinstance(payload, dict) else None
        if not isinstance(upcoming, list):
            return None
        updated_at = _to_utc_datetime(updated_at_raw) or fallback_updated_at
        return _build_sports_board_collection(
            upcoming=upcoming,
            backtests=merged_backtests,
            updated_at=updated_at,
            source=str(payload.get("source") or "divination_live_football_feed"),
            selected_date=str(payload.get("selectedDate")) if payload.get("selectedDate") else None,
            available_dates=payload.get("availableDates") or [],
        )
    except Exception as exc:
        logger.warning("Falling back to cached Football board feed: %s", exc)
        return None


async def _live_soccer_board_collection(soccer_date: str | None = None) -> Optional[SportsBoardCollection]:
    backtests_filename = "soccer_historical_backtests.json"
    backtests = _load_sports_json(backtests_filename)
    fallback_updated_at = _sports_data_updated_at([backtests_filename])

    try:
        async with httpx.AsyncClient(timeout=SPORTS_MLB_BOARDS_TIMEOUT_SECONDS) as client:
            params = {"soccer_date": soccer_date} if soccer_date else None
            response = await client.get(f"{DIVINATION_API_URL}/api/sports/soccer/boards", params=params)
            response.raise_for_status()
            payload = response.json()
            upcoming = payload.get("upcoming") if isinstance(payload, dict) else None
            if isinstance(upcoming, list) and not upcoming:
                refresh_response = await client.get(
                    f"{DIVINATION_API_URL}/api/sports/soccer/boards",
                    params={"force_refresh": "true", **({"soccer_date": soccer_date} if soccer_date else {})},
                )
                refresh_response.raise_for_status()
                payload = refresh_response.json()
                upcoming = payload.get("upcoming") if isinstance(payload, dict) else None
        runtime_backtests = payload.get("completed") if isinstance(payload, dict) else None
        if not isinstance(runtime_backtests, list):
            runtime_backtests = []
        merged_backtests = _merge_runtime_backtests(backtests, runtime_backtests)
        updated_at_raw = payload.get("updated_at") if isinstance(payload, dict) else None
        if not isinstance(upcoming, list):
            return None
        updated_at = _to_utc_datetime(updated_at_raw) or fallback_updated_at
        return _build_sports_board_collection(
            upcoming=upcoming,
            backtests=merged_backtests,
            updated_at=updated_at,
            source=str(payload.get("source") or "divination_live_soccer_feed"),
            selected_date=str(payload.get("selectedDate")) if payload.get("selectedDate") else None,
            available_dates=payload.get("availableDates") or [],
        )
    except Exception as exc:
        logger.warning("Falling back to cached Soccer board feed: %s", exc)
        return None


async def _live_olympics_board_collection() -> Optional[SportsBoardCollection]:
    backtests_filename = "olympics_historical_backtests.json"
    backtests = _load_sports_json(backtests_filename)
    fallback_updated_at = _sports_data_updated_at([backtests_filename])

    try:
        async with httpx.AsyncClient(timeout=SPORTS_MLB_BOARDS_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{DIVINATION_API_URL}/api/sports/olympics/boards")
            response.raise_for_status()
            payload = response.json()
            upcoming = payload.get("upcoming") if isinstance(payload, dict) else None
        runtime_backtests = payload.get("completed") if isinstance(payload, dict) else None
        if not isinstance(runtime_backtests, list):
            runtime_backtests = []
        merged_backtests = _merge_runtime_backtests(backtests, runtime_backtests)
        if not isinstance(upcoming, list):
            return None
        updated_at = _to_utc_datetime(payload.get("updated_at")) or fallback_updated_at
        return _build_sports_board_collection(
            upcoming=upcoming,
            backtests=merged_backtests,
            updated_at=updated_at,
            source=str(payload.get("source") or "divination_olympics_medal_model"),
            selected_date=None,
            available_dates=[],
        )
    except Exception as exc:
        logger.warning("Falling back to cached Olympics board feed: %s", exc)
        return None


def _extract_client_ip(request: Request) -> Optional[str]:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return None


def _extract_user_agent(request: Request) -> Optional[str]:
    return request.headers.get("user-agent")


def _rate_limit_client_id(request: Optional[Request]) -> str:
    if request is None:
        return "unknown"
    return _extract_client_ip(request) or "unknown"


def _enforce_rate_limit(identifier: str, max_requests: int, window_seconds: int) -> None:
    allowed, message = check_rate_limit(
        identifier=identifier,
        max_requests=max_requests,
        window_seconds=window_seconds,
    )
    if not allowed:
        raise HTTPException(429, detail=message)


def _billing_runtime_enabled() -> bool:
    return billing_store_enabled()


def _stripe_price_for_tier(tier: SubscriptionTier) -> Optional[str]:
    if tier == SubscriptionTier.BASIC:
        return STRIPE_PRICE_BASIC_MONTHLY or None
    if tier == SubscriptionTier.PRO:
        return STRIPE_PRICE_PRO_MONTHLY or None
    return None


def _tier_from_price_id(price_id: Optional[str]) -> Optional[SubscriptionTier]:
    if not price_id:
        return None
    if price_id == STRIPE_PRICE_BASIC_MONTHLY:
        return SubscriptionTier.BASIC
    if price_id == STRIPE_PRICE_PRO_MONTHLY:
        return SubscriptionTier.PRO
    return None


def _tier_from_apple_product_id(product_id: Optional[str]) -> Optional[SubscriptionTier]:
    normalized = str(product_id or "").strip().lower()
    if not normalized:
        return None
    if normalized == APPLE_IAP_PRODUCT_BASIC_MONTHLY.lower():
        return SubscriptionTier.BASIC
    if normalized == APPLE_IAP_PRODUCT_PRO_MONTHLY.lower():
        return SubscriptionTier.PRO
    if "basic" in normalized:
        return SubscriptionTier.BASIC
    if "pro" in normalized:
        return SubscriptionTier.PRO
    return None


def _coerce_subscription_tier(raw_value: Optional[str], fallback: SubscriptionTier) -> SubscriptionTier:
    if not raw_value:
        return fallback
    try:
        return SubscriptionTier(str(raw_value))
    except Exception:
        return fallback


def _to_utc_datetime(value: Optional[object]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.isdigit():
            numeric = int(candidate)
            if numeric > 10_000_000_000:
                return datetime.fromtimestamp(numeric / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(float(numeric), tz=timezone.utc)
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def _subscription_tier_rank(tier: SubscriptionTier) -> int:
    if tier == SubscriptionTier.PRO:
        return 2
    if tier == SubscriptionTier.BASIC:
        return 1
    return 0


def _max_tier(*tiers: Optional[SubscriptionTier]) -> SubscriptionTier:
    resolved = SubscriptionTier.FREE
    for tier in tiers:
        if tier and _subscription_tier_rank(tier) > _subscription_tier_rank(resolved):
            resolved = tier
    return resolved


def _stripe_tier_from_state(state: Optional[dict]) -> Optional[SubscriptionTier]:
    if not state:
        return None
    status = str(state.get("subscription_status") or "none").lower()
    plan_tier = _coerce_subscription_tier(state.get("plan_tier"), SubscriptionTier.FREE)
    if status in ACTIVE_STRIPE_SUBSCRIPTION_STATUSES:
        return plan_tier
    if status == "legacy_grace" and _is_grace_active(state):
        return plan_tier
    return None


def _apple_subscription_active(state: Optional[dict]) -> bool:
    if not state:
        return False
    status = str(state.get("apple_subscription_status") or "").strip().lower()
    if status not in {"active", "trialing", "verified"}:
        return False
    expires_at = _to_utc_datetime(state.get("apple_expires_at"))
    if expires_at is None:
        return True
    return expires_at > datetime.now(timezone.utc)


def _apple_tier_from_state(state: Optional[dict]) -> Optional[SubscriptionTier]:
    if not _apple_subscription_active(state):
        return None
    stored = _coerce_subscription_tier(state.get("apple_tier"), SubscriptionTier.FREE) if state and state.get("apple_tier") else None
    if stored and stored != SubscriptionTier.FREE:
        return stored
    return _tier_from_apple_product_id(state.get("apple_product_id") if state else None)


def _billing_provider_from_tiers(
    stripe_tier: Optional[SubscriptionTier],
    apple_tier: Optional[SubscriptionTier],
) -> str:
    has_stripe = stripe_tier is not None and stripe_tier != SubscriptionTier.FREE
    has_apple = apple_tier is not None and apple_tier != SubscriptionTier.FREE
    if has_stripe and has_apple:
        return "hybrid"
    if has_stripe:
        return "stripe"
    if has_apple:
        return "apple"
    return "none"


def _entitlement_source_from_tiers(
    stripe_tier: Optional[SubscriptionTier],
    apple_tier: Optional[SubscriptionTier],
    user_tier: SubscriptionTier,
) -> Optional[str]:
    provider = _billing_provider_from_tiers(stripe_tier, apple_tier)
    if provider == "hybrid":
        return "hybrid"
    if provider == "stripe":
        return "stripe"
    if provider == "apple":
        return "apple"
    if user_tier != SubscriptionTier.FREE:
        return "manual"
    return None


def _decode_unverified_jws_payload(jws: str) -> dict[str, Any]:
    parts = jws.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload + padding)
        loaded = json.loads(decoded.decode("utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _parse_apple_transaction_blob(blob: str) -> dict[str, Any]:
    candidate = (blob or "").strip()
    if not candidate:
        return {}
    try:
        loaded = json.loads(candidate)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return _decode_unverified_jws_payload(candidate)


def _value_for_keys(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _ensure_billing_state_for_user(user: UserInDB) -> Optional[dict]:
    if not billing_store_enabled():
        return None

    state = get_state_by_user_id(user.id)
    if state:
        return state

    now = datetime.now(timezone.utc)
    updates: dict[str, object] = {}
    source = "seed"

    if STRIPE_ENABLED and user.tier in {SubscriptionTier.BASIC, SubscriptionTier.PRO}:
        updates["plan_tier"] = user.tier.value
        updates["subscription_status"] = "legacy_grace"
        updates["legacy_grace_expires_at"] = now + timedelta(days=STRIPE_LEGACY_GRACE_DAYS)
        source = "legacy_seed"
    else:
        updates["plan_tier"] = user.tier.value
        updates["subscription_status"] = "none"
        source = "seed"

    return upsert_customer_state(user.id, user.email, updates=updates, source=source)


def _is_grace_active(state: Optional[dict], now: Optional[datetime] = None) -> bool:
    if not state:
        return False
    if state.get("subscription_status") != "legacy_grace":
        return False
    expiry = _to_utc_datetime(state.get("legacy_grace_expires_at"))
    if not expiry:
        return False
    current_time = now or datetime.now(timezone.utc)
    return expiry > current_time


def _effective_tier_from_state(state: Optional[dict], fallback_tier: SubscriptionTier) -> SubscriptionTier:
    if not state:
        return fallback_tier

    return _max_tier(_stripe_tier_from_state(state), _apple_tier_from_state(state))


def _sync_user_tier_with_billing(user: UserInDB) -> UserInDB:
    if not _billing_runtime_enabled():
        return user

    state = _ensure_billing_state_for_user(user)
    if not state:
        return user

    desired_tier = _effective_tier_from_state(state, user.tier)
    updates: dict[str, object] = {}
    source: Optional[str] = None

    if state.get("subscription_status") == "legacy_grace" and not _is_grace_active(state):
        updates.update(
            {
                "subscription_status": "grace_expired",
                "plan_tier": SubscriptionTier.FREE.value,
                "legacy_grace_expires_at": state.get("legacy_grace_expires_at"),
            }
        )
        source = "legacy_grace_expired"
        desired_tier = SubscriptionTier.FREE

    if updates:
        state = upsert_customer_state(user.id, user.email, updates=updates, source=source)

    if desired_tier != user.tier:
        update_user_tier(user.id, desired_tier)
        refreshed = get_user_by_id(user.id)
        if refreshed:
            return refreshed

    return user


def _billing_status_payload(user: UserInDB, state: Optional[dict]) -> BillingStatusResponse:
    stripe_tier = _stripe_tier_from_state(state)
    apple_tier = _apple_tier_from_state(state)
    effective_tier = _effective_tier_from_state(state, user.tier)
    plan_tier = _max_tier(stripe_tier, apple_tier, user.tier if not state else None)
    stripe_status = str(state.get("subscription_status") if state else "none")
    apple_status = str(state.get("apple_subscription_status")) if state and state.get("apple_subscription_status") else None
    stripe_expires_at = _to_utc_datetime(state.get("current_period_end")) if state else None
    apple_expires_at = _to_utc_datetime(state.get("apple_expires_at")) if state else None
    billing_provider = _billing_provider_from_tiers(stripe_tier, apple_tier)
    entitlement_source = _entitlement_source_from_tiers(stripe_tier, apple_tier, user.tier)
    return BillingStatusResponse(
        billing_enabled=_billing_runtime_enabled(),
        user_tier=user.tier,
        effective_tier=effective_tier,
        plan_tier=plan_tier,
        subscription_status=stripe_status,
        tier_stripe=stripe_tier,
        tier_apple=apple_tier,
        billing_provider=billing_provider,
        entitlement_source=entitlement_source,
        stripe_status=stripe_status,
        stripe_expires_at=stripe_expires_at,
        apple_product_id=str(state.get("apple_product_id")) if state and state.get("apple_product_id") else None,
        apple_subscription_status=apple_status,
        apple_expires_at=apple_expires_at,
        is_active=effective_tier != SubscriptionTier.FREE or _is_grace_active(state),
        cancel_at_period_end=bool(state.get("cancel_at_period_end")) if state else False,
        current_period_end=stripe_expires_at,
        legacy_grace_expires_at=_to_utc_datetime(state.get("legacy_grace_expires_at")) if state else None,
        stripe_customer_id=str(state.get("stripe_customer_id")) if state and state.get("stripe_customer_id") else None,
        stripe_subscription_id=str(state.get("stripe_subscription_id")) if state and state.get("stripe_subscription_id") else None,
        price_id=str(state.get("price_id")) if state and state.get("price_id") else None,
        grace_active=_is_grace_active(state),
    )


def _seed_legacy_grace_state_for_existing_paid_users() -> None:
    if not (STRIPE_ENABLED and billing_store_enabled()):
        return

    paid_users = list_users_by_tiers([SubscriptionTier.BASIC, SubscriptionTier.PRO])
    now = datetime.now(timezone.utc)
    grace_expiry = now + timedelta(days=STRIPE_LEGACY_GRACE_DAYS)

    for user in paid_users:
        state = get_state_by_user_id(user.id)
        if not state:
            upsert_customer_state(
                user.id,
                user.email,
                updates={
                    "plan_tier": user.tier.value,
                    "subscription_status": "legacy_grace",
                    "legacy_grace_expires_at": grace_expiry,
                },
                source="legacy_seed",
            )
            continue

        status = str(state.get("subscription_status") or "none").lower()
        if status in ACTIVE_STRIPE_SUBSCRIPTION_STATUSES:
            continue
        if status == "legacy_grace" and _is_grace_active(state, now=now):
            continue

        if not state.get("legacy_grace_expires_at"):
            upsert_customer_state(
                user.id,
                user.email,
                updates={
                    "plan_tier": user.tier.value,
                    "subscription_status": "legacy_grace",
                    "legacy_grace_expires_at": grace_expiry,
                },
                source="legacy_seed",
            )


def _apply_subscription_snapshot(
    state: dict,
    subscription_payload: dict,
    source: str,
) -> dict:
    status = str(subscription_payload.get("status") or "none").lower()
    current_period_end = _to_utc_datetime(subscription_payload.get("current_period_end"))
    cancel_at_period_end = bool(subscription_payload.get("cancel_at_period_end", False))

    price_id: Optional[str] = None
    items = subscription_payload.get("items", {})
    if isinstance(items, dict):
        item_data = items.get("data", [])
        if item_data and isinstance(item_data, list):
            first_item = item_data[0] or {}
            if isinstance(first_item, dict):
                price = first_item.get("price", {})
                if isinstance(price, dict):
                    price_id = price.get("id")

    mapped_tier = _tier_from_price_id(price_id)
    updates: dict[str, object] = {
        "subscription_status": status,
        "stripe_subscription_id": subscription_payload.get("id"),
        "price_id": price_id,
        "current_period_end": current_period_end,
        "cancel_at_period_end": cancel_at_period_end,
    }

    if mapped_tier:
        updates["plan_tier"] = mapped_tier.value
    elif status in TERMINAL_STRIPE_SUBSCRIPTION_STATUSES:
        updates["plan_tier"] = SubscriptionTier.FREE.value

    return upsert_customer_state(
        user_id=state["user_id"],
        email=state["email"],
        updates=updates,
        source=source,
    )

def _looks_like_etf(name: Optional[str]) -> bool:
    if not name:
        return False
    upper_name = name.upper()
    return " ETF" in upper_name or upper_name.endswith("ETF")


def _infer_asset_type_from_alpaca(asset: dict) -> str:
    asset_class = str(asset.get("class", "")).lower()
    if asset_class == "crypto":
        return "crypto"
    if asset_class == "us_equity":
        if _looks_like_etf(asset.get("name")):
            return "etf"
        return "stock"
    if asset_class:
        return asset_class
    return "stock"


async def _hydrate_company_from_alpaca(ticker: str) -> Optional[dict]:
    global _ALPACA_COMPANY_HYDRATE_BACKOFF_UNTIL
    global _ALPACA_COMPANY_HYDRATE_LAST_LOG

    if not (ALPACA_KEY_ID and ALPACA_SECRET_KEY):
        return None

    now_ts = time.time()
    if now_ts < _ALPACA_COMPANY_HYDRATE_BACKOFF_UNTIL:
        # Log at most once every 60 seconds while in cooldown to avoid noise.
        if now_ts - _ALPACA_COMPANY_HYDRATE_LAST_LOG >= 60:
            remaining = int(_ALPACA_COMPANY_HYDRATE_BACKOFF_UNTIL - now_ts)
            logger.info(
                "Skipping Alpaca company hydrate for %s during cooldown (%ss remaining)",
                ticker.upper(),
                max(0, remaining),
            )
            _ALPACA_COMPANY_HYDRATE_LAST_LOG = now_ts
        return None

    url = f"{ALPACA_TRADING_BASE_URL}/v2/assets/{ticker.upper()}"
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                _ALPACA_COMPANY_HYDRATE_BACKOFF_UNTIL = (
                    time.time() + max(60, ALPACA_COMPANY_HYDRATE_COOLDOWN_SECONDS)
                )
                _ALPACA_COMPANY_HYDRATE_LAST_LOG = 0.0
                logger.warning(
                    "Alpaca company hydrate rate-limited for %s; entering cooldown for %ss",
                    ticker.upper(),
                    max(60, ALPACA_COMPANY_HYDRATE_COOLDOWN_SECONDS),
                )
                return None
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning(f"Alpaca company hydrate failed for {ticker.upper()}: {exc}")
        return None

    if not isinstance(payload, dict):
        return None

    name = payload.get("name") or ticker.upper()
    asset_type = _infer_asset_type_from_alpaca(payload)
    upsert_company(ticker.upper(), name=name, asset_type=asset_type)

    # Alpaca asset endpoint does not include full fundamentals; store minimal metadata.
    upsert_company_info(
        ticker.upper(),
        {
            "exchange": payload.get("exchange"),
            "country": "United States",
            "description": "Profile auto-populated from Alpaca asset metadata.",
            "raw_info": payload,
        },
    )
    return payload


async def _hydrate_company_from_yfinance(ticker: str) -> Optional[dict]:
    """Best-effort yfinance hydrate for symbols not covered by Alpaca asset metadata."""
    try:
        from .stock_info import fetch_and_store_stock_info
    except Exception as exc:
        logger.warning("Could not import stock_info module for %s: %s", ticker.upper(), exc)
        return None

    try:
        result = await asyncio.to_thread(
            fetch_and_store_stock_info,
            ticker.upper(),
            False,  # skip news on-demand to keep responses responsive
        )
    except Exception as exc:
        logger.warning("yfinance company hydrate failed for %s: %s", ticker.upper(), exc)
        return None

    if not isinstance(result, dict):
        return None
    if result.get("info_stored"):
        return result
    return None


def _persist_company_placeholder_info(ticker: str) -> None:
    """Persist minimal placeholder company + info so UI has non-empty profile payload."""
    upsert_company(
        ticker.upper(),
        name=ticker.upper(),
        asset_type="stock",
    )
    upsert_company_info(
        ticker.upper(),
        {
            "description": (
                "Fundamentals are currently unavailable for this symbol. "
                "Price-based model signals remain available."
            ),
            "exchange": "N/A",
            "country": "N/A",
            "currency": "USD",
            "raw_info": {"provider": "placeholder", "status": "limited_profile"},
        },
    )


async def _ensure_company_profile(ticker: str) -> Tuple[Optional[dict], Optional[dict]]:
    company = get_company(ticker.upper())
    info = get_company_info(ticker.upper())

    if company and info:
        return company, info

    if COMPANY_AUTO_POPULATE_ON_DEMAND:
        await _hydrate_company_from_alpaca(ticker.upper())
        company = get_company(ticker.upper())
        info = get_company_info(ticker.upper())

        if not (company and info):
            await _hydrate_company_from_yfinance(ticker.upper())
            company = get_company(ticker.upper())
            info = get_company_info(ticker.upper())

    if ticker.upper() in UNIVERSE_UPPER and not company:
        upsert_company(ticker.upper(), name=ticker.upper(), asset_type="stock")
        company = get_company(ticker.upper())

    if ticker.upper() in UNIVERSE_UPPER and not info:
        _persist_company_placeholder_info(ticker.upper())
        info = get_company_info(ticker.upper())

    return company, info


def _finviz_enabled() -> bool:
    return bool(FINVIZ_API_BASE_URL and FINVIZ_API_KEY)


def _finviz_chart_url(ticker: str) -> str:
    try:
        return FINVIZ_CHART_URL_TEMPLATE.format(ticker=ticker.upper())
    except Exception:
        return f"https://finviz.com/chart.ashx?t={ticker.upper()}&ty=c&ta=1&p=d&s=l"


def _finviz_headers() -> dict[str, str]:
    if not FINVIZ_API_KEY or FINVIZ_API_AUTH_QUERY_PARAM:
        return {}
    value = f"{FINVIZ_API_KEY_PREFIX}{FINVIZ_API_KEY}" if FINVIZ_API_KEY_PREFIX else FINVIZ_API_KEY
    return {FINVIZ_API_AUTH_HEADER: value}


def _build_finviz_quote_url(ticker: str) -> Optional[str]:
    if not FINVIZ_API_BASE_URL:
        return None
    try:
        path = FINVIZ_API_QUOTE_PATH_TEMPLATE.format(ticker=ticker.upper())
    except Exception:
        path = f"/quote/{ticker.upper()}"

    if path.startswith("http://") or path.startswith("https://"):
        base_url = path
    else:
        normalized_path = path if path.startswith("/") else f"/{path}"
        base_url = f"{FINVIZ_API_BASE_URL}{normalized_path}"

    if not FINVIZ_API_AUTH_QUERY_PARAM or not FINVIZ_API_KEY:
        return base_url

    token_value = (
        f"{FINVIZ_API_KEY_PREFIX}{FINVIZ_API_KEY}" if FINVIZ_API_KEY_PREFIX else FINVIZ_API_KEY
    )
    parsed = urlparse(base_url)
    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_items.setdefault(FINVIZ_API_AUTH_QUERY_PARAM, token_value)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode(query_items),
            parsed.fragment,
        )
    )


def _pick_value(payload: dict[str, Any], keys: list[str]) -> Any:
    if not payload:
        return None
    for key in keys:
        if key in payload:
            return payload.get(key)
    lower_map = {str(k).lower(): v for k, v in payload.items()}
    for key in keys:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None


def _parse_float(value: Any, percent: bool = False) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        compact = raw.replace(",", "")
        match = re.search(r"-?\d+(\.\d+)?", compact)
        if not match:
            return None
        try:
            number = float(match.group(0))
        except Exception:
            return None
        if percent and "%" not in raw and abs(number) <= 1:
            number *= 100.0
    else:
        return None

    if not math.isfinite(number):
        return None
    return number


async def _fetch_finviz_quote(ticker: str) -> Optional[dict[str, Any]]:
    if not _finviz_enabled():
        return None

    symbol = ticker.upper()
    now_ts = time.time()
    cached = _FINVIZ_INSIGHTS_CACHE.get(symbol)
    if cached and now_ts - cached[0] < FINVIZ_INSIGHTS_CACHE_TTL_SECONDS:
        return cached[1]

    url = _build_finviz_quote_url(symbol)
    if not url:
        return None

    try:
        async with httpx.AsyncClient(timeout=FINVIZ_API_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=_finviz_headers())
            if response.status_code in (401, 403):
                logger.warning("Finviz quote access denied for %s: status=%s", symbol, response.status_code)
                return None
            if response.status_code == 404:
                return None
            response.raise_for_status()
            try:
                payload: Any = response.json()
            except Exception:
                payload = None
                text_body = response.text.strip()
                if text_body:
                    try:
                        payload = json.loads(text_body)
                    except Exception:
                        payload = None
                        try:
                            import csv

                            reader = csv.DictReader(StringIO(text_body))
                            payload = next(reader, None)
                        except Exception:
                            payload = None
    except Exception as exc:
        logger.debug("Finviz quote fetch failed for %s: %s", symbol, exc)
        return None

    if isinstance(payload, list):
        if not payload:
            return None
        payload = payload[0]

    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = payload.get("data")

    if not isinstance(payload, dict):
        return None

    _FINVIZ_INSIGHTS_CACHE[symbol] = (now_ts, payload)
    return payload


def _normalize_watchlist_insight(
    ticker: str,
    finviz_quote: Optional[dict[str, Any]],
) -> dict[str, Any]:
    symbol = ticker.upper()
    quote = finviz_quote or {}

    price = _parse_float(_pick_value(quote, ["price", "last", "last_price", "close"]))
    change_pct = _parse_float(
        _pick_value(quote, ["change_pct", "change_percent", "changePercent", "change"]),
        percent=True,
    )
    rsi = _parse_float(_pick_value(quote, ["rsi", "rsi14", "rsi_14"]))
    sma20 = _parse_float(_pick_value(quote, ["sma20", "sma_20"]))
    sma50 = _parse_float(_pick_value(quote, ["sma50", "sma_50"]))
    sma200 = _parse_float(_pick_value(quote, ["sma200", "sma_200"]))
    volume = _parse_float(_pick_value(quote, ["volume", "avg_volume", "avgVolume"]))
    rel_volume = _parse_float(_pick_value(quote, ["rel_volume", "relative_volume", "relativeVolume"]))
    atr = _parse_float(_pick_value(quote, ["atr", "atr14"]))
    support = _parse_float(_pick_value(quote, ["support", "pivot_support"]))
    resistance = _parse_float(_pick_value(quote, ["resistance", "pivot_resistance"]))
    trend_raw = _pick_value(quote, ["trend", "signal", "technical_signal", "pattern"])
    summary_raw = _pick_value(quote, ["summary", "technical_summary", "analysis", "commentary"])
    updated_raw = _pick_value(quote, ["updated_at", "timestamp", "generated_at", "as_of"])

    cached_price = None
    if price is None:
        cached_row = get_last_close_pg(ticker=symbol, timeframe="5d", max_age_hours=168)
        if cached_row and cached_row.get("last_close") is not None:
            cached_price = _parse_float(cached_row.get("last_close"))
            price = cached_price

    summary = str(summary_raw).strip() if isinstance(summary_raw, str) and summary_raw.strip() else None
    if not summary:
        company = get_company(symbol)
        info = get_company_info(symbol)
        sector = info.get("sector") if isinstance(info, dict) else None
        industry = info.get("industry") if isinstance(info, dict) else None
        company_name = company.get("name") if isinstance(company, dict) else symbol
        parts = [p for p in [sector, industry] if isinstance(p, str) and p.strip()]
        if parts:
            summary = f"{company_name}: {' / '.join(parts)}."

    updated_at = (
        str(updated_raw).strip()
        if isinstance(updated_raw, str) and str(updated_raw).strip()
        else datetime.now(timezone.utc).isoformat()
    )

    return {
        "ticker": symbol,
        "chart_url": _finviz_chart_url(symbol),
        "source": "finviz" if finviz_quote else "fallback",
        "updated_at": updated_at,
        "price": price,
        "change_pct": change_pct,
        "rsi": rsi,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "volume": volume,
        "rel_volume": rel_volume,
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "trend": str(trend_raw).strip() if isinstance(trend_raw, str) and str(trend_raw).strip() else None,
        "summary": summary,
    }


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
        if user:
            try:
                user = _sync_user_tier_with_billing(user)
            except Exception as sync_err:
                logger.warning("Billing tier sync failed for user_id=%s: %s", user.id, sync_err)
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

@app.post("/api/auth/signup", response_model=MessageResponse, tags=["Authentication"])
async def signup(data: UserCreate, request: Request):
    """Register a new user account."""
    email = data.email.lower().strip()
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"auth:signup:email:{email}",
        max_requests=AUTH_SIGNUP_RATE_LIMIT,
        window_seconds=AUTH_SIGNUP_RATE_WINDOW_SECONDS,
    )
    _enforce_rate_limit(
        identifier=f"auth:signup:ip:{client_id}",
        max_requests=AUTH_SIGNUP_IP_RATE_LIMIT,
        window_seconds=AUTH_SIGNUP_IP_RATE_WINDOW_SECONDS,
    )

    if not data.accept_terms:
        raise HTTPException(400, detail="You must accept the Terms of Service")

    if not data.accept_privacy:
        raise HTTPException(400, detail="You must accept the Privacy Policy")

    # Validate password strength
    is_valid, message = validate_password_strength(data.password)
    if not is_valid:
        logger.warning(f"Signup attempt with weak password: {message}")
        raise HTTPException(400, detail=message)

    # Check if email already exists
    existing = get_user_by_email(email)
    if existing:
        logger.info(f"Signup attempt with existing email: {email}")
        raise HTTPException(400, detail="Email already registered")

    # Create user
    verification_token = generate_verification_token()
    now = datetime.utcnow()

    user = UserInDB(
        id=str(uuid.uuid4()),
        email=email,
        first_name=data.first_name,
        last_name=data.last_name,
        hashed_password=hash_password(data.password),
        tier=SubscriptionTier.FREE,
        email_verified=False,
        verification_token=verification_token,
        created_at=now,
        updated_at=now,
    )

    create_user(user)
    logger.info(f"New user created: {user.email} (tier={user.tier.value})")

    if billing_store_enabled():
        try:
            upsert_customer_state(
                user.id,
                user.email,
                updates={
                    "plan_tier": SubscriptionTier.FREE.value,
                    "subscription_status": "none",
                },
                source="signup",
            )
        except Exception as e:
            logger.warning("Failed to initialize billing state for %s: %s", user.email, e)

    client_ip = _extract_client_ip(request)
    user_agent = _extract_user_agent(request)
    policy_version = data.policy_version.strip() or POLICY_VERSION

    # Persist consent and default communication preferences in Postgres.
    try:
        record_consent_event(
            user_id=user.id,
            email=user.email,
            policy_version=policy_version,
            consent_type="terms",
            granted=True,
            ip_address=client_ip,
            user_agent=user_agent,
        )
        record_consent_event(
            user_id=user.id,
            email=user.email,
            policy_version=policy_version,
            consent_type="privacy",
            granted=True,
            ip_address=client_ip,
            user_agent=user_agent,
        )
        record_consent_event(
            user_id=user.id,
            email=user.email,
            policy_version=policy_version,
            consent_type="marketing",
            granted=bool(data.marketing_opt_in),
            ip_address=client_ip,
            user_agent=user_agent,
        )
        set_newsletter_opt_in(user.id, user.email, bool(data.marketing_opt_in))
    except Exception as e:
        logger.error("Failed to persist compliance records for signup %s: %s", user.email, e)
        raise HTTPException(500, detail="Failed to persist compliance preferences")

    # Send verification email
    send_verification_email(user.email, user.first_name, verification_token)

    return MessageResponse(
        message="Account created. Please check your email to verify your account."
    )


@app.post("/api/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(data: UserLogin, request: Request):
    """Login with email and password."""
    email = data.email.lower().strip()
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"auth:login:email:{email}",
        max_requests=AUTH_LOGIN_RATE_LIMIT,
        window_seconds=AUTH_LOGIN_RATE_WINDOW_SECONDS,
    )
    _enforce_rate_limit(
        identifier=f"auth:login:ip:{client_id}",
        max_requests=AUTH_LOGIN_IP_RATE_LIMIT,
        window_seconds=AUTH_LOGIN_IP_RATE_WINDOW_SECONDS,
    )

    user = get_user_by_email(email)

    if not user or not verify_password(data.password, user.hashed_password):
        logger.warning(f"Failed login attempt for: {email}")
        raise HTTPException(401, detail="Invalid email or password")

    if not user.email_verified:
        logger.info(f"Blocked login for unverified email: {user.email}")
        raise HTTPException(
            403,
            detail="Email not verified. Please verify your email before logging in.",
        )

    access_token = create_access_token(user.id, user.email, token_type="access")
    refresh_token = create_access_token(user.id, user.email, token_type="refresh")
    logger.info(f"User logged in: {user.email}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
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
async def verify_email(data: VerifyEmailRequest, request: Request):
    """Verify email with token from email link."""
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"auth:verify:token:{data.token}",
        max_requests=AUTH_VERIFY_RATE_LIMIT,
        window_seconds=AUTH_VERIFY_RATE_WINDOW_SECONDS,
    )
    _enforce_rate_limit(
        identifier=f"auth:verify:ip:{client_id}",
        max_requests=AUTH_VERIFY_RATE_LIMIT,
        window_seconds=AUTH_VERIFY_RATE_WINDOW_SECONDS,
    )

    user = get_user_by_verification_token(data.token)

    if not user:
        logger.warning("Invalid verification token used")
        raise HTTPException(400, detail="Invalid or expired verification token")

    verify_user_email(user.id)
    logger.info(f"Email verified for user: {user.email}")

    # Send welcome email
    send_welcome_email(user.email, user.first_name, user.tier.value)

    # Return token so user is logged in
    access_token = create_access_token(user.id, user.email, token_type="access")
    refresh_token = create_access_token(user.id, user.email, token_type="refresh")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
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
async def resend_verification(data: ResendVerificationRequest, request: Request):
    """Resend verification email."""
    email = data.email.lower().strip()
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"auth:resend:email:{email}",
        max_requests=AUTH_RESEND_RATE_LIMIT,
        window_seconds=AUTH_RESEND_RATE_WINDOW_SECONDS,
    )
    _enforce_rate_limit(
        identifier=f"auth:resend:ip:{client_id}",
        max_requests=AUTH_RESEND_RATE_LIMIT * 5,
        window_seconds=AUTH_RESEND_RATE_WINDOW_SECONDS,
    )

    user = get_user_by_email(email)

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


@app.get("/api/email/unsubscribe", response_model=MessageResponse, tags=["Authentication"])
async def email_unsubscribe(token: str):
    """One-click unsubscribe endpoint for marketing communications."""
    payload = decode_access_token(token, expected_type="unsubscribe")
    if not payload:
        raise HTTPException(400, detail="Invalid or expired unsubscribe token")

    user_id = payload.get("sub")
    email = payload.get("email")
    if not user_id or not email:
        raise HTTPException(400, detail="Invalid unsubscribe token payload")

    set_newsletter_opt_in(user_id=user_id, email=email, enabled=False)
    record_consent_event(
        user_id=user_id,
        email=email,
        policy_version=POLICY_VERSION,
        consent_type="marketing",
        granted=False,
        ip_address=None,
        user_agent="one-click-unsubscribe",
    )
    return MessageResponse(message="You have been unsubscribed from marketing emails.")


@app.post("/api/webhooks/ses-sns", response_model=MessageResponse, tags=["System"])
async def ses_sns_webhook(request: Request):
    """Receive SES notifications from SNS and persist suppression/audit records."""
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"webhook:ses-sns:ip:{client_id}",
        max_requests=SES_WEBHOOK_RATE_LIMIT,
        window_seconds=SES_WEBHOOK_RATE_WINDOW_SECONDS,
    )

    try:
        envelope = await request.json()
    except Exception:
        raise HTTPException(400, detail="Invalid JSON payload")

    message_type = request.headers.get("x-amz-sns-message-type") or envelope.get("Type")
    topic_arn = envelope.get("TopicArn", "")
    if not SES_SNS_ALLOWED_TOPIC_ARNS:
        raise HTTPException(503, detail="SES SNS webhook is not configured")
    if topic_arn not in SES_SNS_ALLOWED_TOPIC_ARNS:
        raise HTTPException(403, detail="Topic ARN is not allowed")

    if message_type == "SubscriptionConfirmation":
        subscribe_url = envelope.get("SubscribeURL")
        if SES_SNS_AUTO_CONFIRM and subscribe_url:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.get(subscribe_url)
            except Exception as exc:
                logger.warning("SNS subscription auto-confirm failed: %s", exc)
                raise HTTPException(502, detail="Failed to auto-confirm SNS subscription")
        return MessageResponse(message="SNS subscription confirmation processed")

    if message_type != "Notification":
        return MessageResponse(message=f"Ignored SNS message type: {message_type}")

    raw_message = envelope.get("Message", {})
    if isinstance(raw_message, str):
        try:
            ses_message = json.loads(raw_message)
        except json.JSONDecodeError:
            ses_message = {"raw_message": raw_message}
    elif isinstance(raw_message, dict):
        ses_message = raw_message
    else:
        ses_message = {"raw_message": str(raw_message)}

    event_type = (ses_message.get("notificationType") or ses_message.get("eventType") or "unknown").lower()
    mail = ses_message.get("mail", {}) if isinstance(ses_message.get("mail"), dict) else {}
    provider_message_id = mail.get("messageId")

    suppressed_count = 0
    audited_count = 0

    def _audit(email: Optional[str], evt: str) -> None:
        nonlocal audited_count
        if audit_email_event(
            provider="ses",
            event_type=evt,
            payload=ses_message,
            email=email,
            provider_message_id=provider_message_id,
        ):
            audited_count += 1

    if event_type == "bounce":
        bounce = ses_message.get("bounce", {}) if isinstance(ses_message.get("bounce"), dict) else {}
        recipients = bounce.get("bouncedRecipients", [])
        for recipient in recipients:
            if not isinstance(recipient, dict):
                continue
            email = recipient.get("emailAddress")
            if not email:
                continue
            _audit(email, "bounce")
            upsert_suppression(
                email=email,
                reason=bounce.get("bounceType", "bounce"),
                source="ses-sns",
                provider_event_type="bounce",
                provider_message_id=provider_message_id,
                raw_event=ses_message,
            )
            disable_newsletter_by_email(email)
            suppressed_count += 1
    elif event_type == "complaint":
        complaint = ses_message.get("complaint", {}) if isinstance(ses_message.get("complaint"), dict) else {}
        recipients = complaint.get("complainedRecipients", [])
        for recipient in recipients:
            if not isinstance(recipient, dict):
                continue
            email = recipient.get("emailAddress")
            if not email:
                continue
            _audit(email, "complaint")
            upsert_suppression(
                email=email,
                reason=complaint.get("complaintFeedbackType", "complaint"),
                source="ses-sns",
                provider_event_type="complaint",
                provider_message_id=provider_message_id,
                raw_event=ses_message,
            )
            disable_newsletter_by_email(email)
            suppressed_count += 1
    elif event_type == "delivery":
        delivery = ses_message.get("delivery", {}) if isinstance(ses_message.get("delivery"), dict) else {}
        recipients = delivery.get("recipients", [])
        if isinstance(recipients, list) and recipients:
            for recipient in recipients:
                if recipient:
                    _audit(str(recipient), "delivery")
        else:
            _audit(None, "delivery")
    else:
        _audit(None, event_type)

    logger.info(
        "Processed SES SNS notification type=%s audited=%s suppressed=%s",
        event_type,
        audited_count,
        suppressed_count,
    )
    return MessageResponse(
        message=(
            f"Processed SES notification type={event_type}; "
            f"audited={audited_count}; suppressed={suppressed_count}"
        )
    )


@app.post("/api/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(data: RefreshTokenRequest, request: Request):
    """Refresh access token using refresh token."""
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"auth:refresh:ip:{client_id}",
        max_requests=AUTH_REFRESH_RATE_LIMIT,
        window_seconds=AUTH_REFRESH_RATE_WINDOW_SECONDS,
    )

    payload = decode_access_token(data.refresh_token, expected_type="refresh")

    if not payload:
        logger.warning("Invalid refresh token used")
        raise HTTPException(401, detail="Invalid or expired refresh token")

    user_id = payload.get("sub")
    email = payload.get("email")
    user = get_user_by_id(user_id)

    if not user:
        logger.warning(f"Refresh token for non-existent user: {user_id}")
        raise HTTPException(401, detail="User not found")

    # Create new access token
    new_access_token = create_access_token(user.id, user.email, token_type="access")
    logger.info(f"Token refreshed for user: {user.email}")

    return TokenResponse(
        access_token=new_access_token,
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


@app.post("/api/auth/password-reset", response_model=MessageResponse)
async def request_password_reset(data: PasswordResetRequest, request: Request):
    """Request password reset email."""
    email = data.email.lower().strip()
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"auth:password-reset:email:{email}",
        max_requests=AUTH_PASSWORD_RESET_RATE_LIMIT,
        window_seconds=AUTH_PASSWORD_RESET_RATE_WINDOW_SECONDS,
    )
    _enforce_rate_limit(
        identifier=f"auth:password-reset:ip:{client_id}",
        max_requests=AUTH_PASSWORD_RESET_RATE_LIMIT * 5,
        window_seconds=AUTH_PASSWORD_RESET_RATE_WINDOW_SECONDS,
    )

    user = get_user_by_email(email)

    if not user:
        # Don't reveal if email exists
        return MessageResponse(message="If the email exists, a password reset link has been sent.")

    # Generate password reset token
    reset_token = create_access_token(user.id, user.email, token_type="password_reset")

    # Send password reset email
    send_password_reset_email(user.email, user.first_name, reset_token)
    logger.info(f"Password reset requested for: {user.email}")

    return MessageResponse(message="If the email exists, a password reset link has been sent.")


@app.post("/api/auth/password-reset-confirm", response_model=MessageResponse)
async def confirm_password_reset(data: PasswordResetConfirm, request: Request):
    """Confirm password reset with token and new password."""
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"auth:password-reset-confirm:token:{data.token}",
        max_requests=AUTH_PASSWORD_RESET_CONFIRM_RATE_LIMIT,
        window_seconds=AUTH_PASSWORD_RESET_CONFIRM_RATE_WINDOW_SECONDS,
    )
    _enforce_rate_limit(
        identifier=f"auth:password-reset-confirm:ip:{client_id}",
        max_requests=AUTH_PASSWORD_RESET_CONFIRM_RATE_LIMIT,
        window_seconds=AUTH_PASSWORD_RESET_CONFIRM_RATE_WINDOW_SECONDS,
    )

    payload = decode_access_token(data.token, expected_type="password_reset")

    if not payload:
        logger.warning("Invalid password reset token used")
        raise HTTPException(400, detail="Invalid or expired password reset token")

    user_id = payload.get("sub")
    user = get_user_by_id(user_id)

    if not user:
        logger.warning(f"Password reset for non-existent user: {user_id}")
        raise HTTPException(400, detail="User not found")

    # Validate new password strength
    is_valid, message = validate_password_strength(data.new_password)
    if not is_valid:
        logger.warning(f"Password reset with weak password: {message}")
        raise HTTPException(400, detail=message)

    # Update password in database
    updated = update_user_password(user.id, hash_password(data.new_password))
    if not updated:
        raise HTTPException(500, detail="Failed to update password")

    logger.info(f"Password reset for user: {user.email}")

    return MessageResponse(message="Password reset successfully. You can now login with your new password.")


@app.post("/api/auth/change-password", response_model=MessageResponse, tags=["Authentication"])
async def change_password(
    data: ChangePasswordRequest,
    request: Request,
    user: UserInDB = Depends(require_auth),
):
    """Change password for authenticated user."""
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"auth:change-password:user:{user.id}",
        max_requests=AUTH_CHANGE_PASSWORD_RATE_LIMIT,
        window_seconds=AUTH_CHANGE_PASSWORD_RATE_WINDOW_SECONDS,
    )
    _enforce_rate_limit(
        identifier=f"auth:change-password:ip:{client_id}",
        max_requests=AUTH_CHANGE_PASSWORD_RATE_LIMIT * 5,
        window_seconds=AUTH_CHANGE_PASSWORD_RATE_WINDOW_SECONDS,
    )

    if not verify_password(data.current_password, user.hashed_password):
        raise HTTPException(400, detail="Current password is incorrect")

    if verify_password(data.new_password, user.hashed_password):
        raise HTTPException(400, detail="New password must be different from current password")

    is_valid, message = validate_password_strength(data.new_password)
    if not is_valid:
        raise HTTPException(400, detail=message)

    updated = update_user_password(user.id, hash_password(data.new_password))
    if not updated:
        raise HTTPException(500, detail="Failed to update password")

    logger.info("Password changed for user: %s", user.email)
    return MessageResponse(message="Password updated successfully.")


@app.post("/api/auth/delete-account", response_model=MessageResponse, tags=["Authentication"])
async def delete_account(
    data: DeleteAccountRequest,
    request: Request,
    user: UserInDB = Depends(require_auth),
):
    """Delete authenticated user account and related records."""
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"auth:delete-account:user:{user.id}",
        max_requests=AUTH_DELETE_ACCOUNT_RATE_LIMIT,
        window_seconds=AUTH_DELETE_ACCOUNT_RATE_WINDOW_SECONDS,
    )
    _enforce_rate_limit(
        identifier=f"auth:delete-account:ip:{client_id}",
        max_requests=AUTH_DELETE_ACCOUNT_RATE_LIMIT * 5,
        window_seconds=AUTH_DELETE_ACCOUNT_RATE_WINDOW_SECONDS,
    )

    if data.confirm_text.strip().upper() != "DELETE":
        raise HTTPException(400, detail='Confirmation text must be "DELETE"')

    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(400, detail="Password is incorrect")

    # Best-effort subscription cancellation before deleting local account.
    if STRIPE_ENABLED and stripe and billing_store_enabled():
        try:
            state = get_state_by_user_id(user.id)
            subscription_id = str(state.get("stripe_subscription_id") or "").strip() if state else ""
            if subscription_id:
                try:
                    stripe.Subscription.delete(subscription_id)
                    logger.info("Canceled Stripe subscription before account deletion: %s", subscription_id)
                except Exception as cancel_err:
                    logger.warning(
                        "Stripe subscription cancel failed for user_id=%s subscription_id=%s: %s",
                        user.id,
                        subscription_id,
                        cancel_err,
                    )
        except Exception as billing_err:
            logger.warning("Billing lookup failed before account deletion user_id=%s: %s", user.id, billing_err)

    if compliance_store_enabled():
        try:
            delete_user_records(user.id, user.email)
        except Exception as compliance_err:
            logger.warning("Compliance cleanup failed for user_id=%s: %s", user.id, compliance_err)

    if billing_store_enabled():
        try:
            delete_state_by_user_id(user.id)
        except Exception as billing_err:
            logger.warning("Billing cleanup failed for user_id=%s: %s", user.id, billing_err)

    deleted = delete_user_account(user.id)
    if not deleted:
        raise HTTPException(500, detail="Failed to delete account")

    logger.info("Deleted account for user: %s", user.email)
    return MessageResponse(message="Account deleted successfully.")


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
# User Preferences Endpoints
# ============================================================

@app.get("/api/user/preferences", response_model=UserPreferencesResponse, tags=["Preferences"])
async def get_user_preferences(user: UserInDB = Depends(require_auth)):
    """Get user preferences."""
    prefs = get_or_create_preferences(user.id, user.email)
    return UserPreferencesResponse(
        user_id=user.id,
        **{k: v for k, v in prefs.items() if k != "updated_at"},
        updated_at=prefs["updated_at"],
    )


@app.put("/api/user/preferences", response_model=UserPreferencesResponse, tags=["Preferences"])
async def update_user_preferences(
    data: UpdatePreferencesRequest,
    user: UserInDB = Depends(require_auth),
):
    """Update user preferences."""
    updates = data.model_dump(exclude_none=True)

    # Update with provided values
    if data.theme is not None:
        if data.theme not in ["light", "dark"]:
            raise HTTPException(400, detail="Theme must be 'light' or 'dark'")

    prefs = upsert_preferences(user.id, user.email, updates)
    logger.info(f"Preferences updated for user: {user.email}")

    return UserPreferencesResponse(
        user_id=user.id,
        **{k: v for k, v in prefs.items() if k != "updated_at"},
        updated_at=prefs["updated_at"],
    )


# ============================================================
# Tier Endpoints
# ============================================================

@app.get("/api/tiers", response_model=list[TierInfo], tags=["Subscriptions"])
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
# Community Endpoints
# ============================================================

@app.post("/api/beta-testers/signup", response_model=BetaTesterSignupResponse, tags=["Community"])
async def beta_tester_signup(data: BetaTesterSignupRequest, request: Request):
    """Capture public beta tester applications from landing page."""
    if not data.accept_contact:
        raise HTTPException(400, detail="You must agree to be contacted for beta updates.")

    safe_email = str(data.email).strip().lower()
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"community:beta-signup:email:{safe_email}",
        max_requests=PUBLIC_BETA_SIGNUP_RATE_LIMIT,
        window_seconds=PUBLIC_BETA_SIGNUP_RATE_WINDOW_SECONDS,
    )
    _enforce_rate_limit(
        identifier=f"community:beta-signup:ip:{client_id}",
        max_requests=PUBLIC_BETA_SIGNUP_IP_RATE_LIMIT,
        window_seconds=PUBLIC_BETA_SIGNUP_IP_RATE_WINDOW_SECONDS,
    )

    if not beta_program_store_enabled():
        logger.error("Beta signup attempted while beta program store is disabled")
        raise HTTPException(503, detail="Beta signup is temporarily unavailable.")

    try:
        saved = upsert_beta_tester(
            email=safe_email,
            full_name=data.full_name,
            role=data.role,
            organization=data.organization,
            investing_experience=data.investing_experience,
            testing_focus=data.testing_focus,
            source=data.source or "landing_page",
            ip_address=_extract_client_ip(request),
            user_agent=_extract_user_agent(request),
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        logger.error("Failed to store beta tester signup for %s: %s", safe_email, e)
        raise HTTPException(500, detail="Could not process beta signup right now.")

    logger.info(
        "Beta tester signup stored email=%s created=%s source=%s",
        safe_email,
        saved.get("created", False),
        data.source or "landing_page",
    )

    if saved.get("created"):
        message = "Application received. We will reach out with beta access updates soon."
    else:
        message = "Your beta application was updated. We will follow up with next steps."

    return BetaTesterSignupResponse(
        message=message,
        discord_url=DISCORD_INVITE_URL or "https://discord.gg/ckTC8JhWU9",
    )


# ============================================================
# Billing Endpoints
# ============================================================

def _require_billing_enabled() -> None:
    if not STRIPE_ENABLED:
        raise HTTPException(503, detail="Stripe billing is disabled")
    if stripe is None:
        raise HTTPException(503, detail="Stripe SDK is not installed")
    if not billing_store_enabled():
        raise HTTPException(503, detail="Billing store is unavailable")
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, detail="Stripe secret key is not configured")


def _checkout_redirect_url(result: str) -> str:
    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    base = frontend_url.rstrip("/") if frontend_url else "https://seekingbeta.ai"
    return f"{base}/pricing?checkout={result}"


def _is_stripe_resource_missing_error(exc: Exception) -> bool:
    """True when Stripe reports missing customer/subscription under current API key."""
    code = str(getattr(exc, "code", "") or "").lower()
    if code == "resource_missing":
        return True

    message = str(exc).lower()
    return (
        "resource_missing" in message
        or "no such customer" in message
        or "no such subscription" in message
    )


def _apple_verify_runtime_enabled() -> bool:
    return billing_store_enabled()


def _normalized_text(value: Optional[object]) -> str:
    return str(value or "").strip()


def _validate_apple_purchase_payload(data: AppleVerifyRequest) -> dict[str, Any]:
    parsed = _parse_apple_transaction_blob(data.signed_transaction_info)
    if not parsed:
        return {}

    parsed_transaction_id = _normalized_text(
        _value_for_keys(parsed, ["transactionId", "transaction_id", "id"])
    )
    parsed_original_transaction_id = _normalized_text(
        _value_for_keys(parsed, ["originalTransactionId", "original_transaction_id", "originalID"])
    )
    parsed_product_id = _normalized_text(
        _value_for_keys(parsed, ["productId", "product_id", "productID"])
    )
    parsed_account_token = _normalized_text(
        _value_for_keys(parsed, ["appAccountToken", "app_account_token"])
    )

    if parsed_transaction_id and parsed_transaction_id != data.transaction_id:
        raise HTTPException(400, detail="Apple transaction payload mismatch")
    if parsed_product_id and parsed_product_id != data.product_id:
        raise HTTPException(400, detail="Apple product payload mismatch")
    if (
        parsed_original_transaction_id
        and data.original_transaction_id
        and parsed_original_transaction_id != data.original_transaction_id
    ):
        raise HTTPException(400, detail="Apple original transaction mismatch")
    if parsed_account_token and data.app_account_token and parsed_account_token != data.app_account_token:
        raise HTTPException(400, detail="Apple app account token mismatch")

    return parsed


def _apple_state_updates_from_purchase(
    data: AppleVerifyRequest,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    tier = _tier_from_apple_product_id(data.product_id)
    if not tier:
        raise HTTPException(400, detail="Unsupported Apple product id")

    expires_at = _to_utc_datetime(
        _value_for_keys(parsed, ["expiresDate", "expires_date", "expirationDate", "expiration_date"])
    )
    revocation_date = _to_utc_datetime(
        _value_for_keys(parsed, ["revocationDate", "revocation_date"])
    )
    environment = _normalized_text(_value_for_keys(parsed, ["environment"]))
    subscription_status = "active"
    if revocation_date is not None:
        subscription_status = "revoked"
    elif expires_at is not None and expires_at <= datetime.now(timezone.utc):
        subscription_status = "expired"

    return {
        "apple_tier": tier.value,
        "apple_product_id": data.product_id,
        "apple_transaction_id": data.transaction_id,
        "apple_original_transaction_id": data.original_transaction_id,
        "apple_app_account_token": data.app_account_token,
        "apple_subscription_status": subscription_status,
        "apple_expires_at": expires_at,
        "apple_environment": environment or None,
        "apple_last_verified_at": datetime.now(timezone.utc),
    }


def _reset_stripe_linked_state_for_mode_switch(user: UserInDB, source: str) -> Optional[dict]:
    """Clear Stripe-linked ids so checkout can re-provision customer/subscription cleanly."""
    if not billing_store_enabled():
        return None
    return upsert_customer_state(
        user.id,
        user.email,
        updates={
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
            "price_id": None,
            "current_period_end": None,
            "cancel_at_period_end": False,
            "subscription_status": "none",
            "plan_tier": SubscriptionTier.FREE.value,
        },
        source=source,
    )


def _ensure_stripe_customer(user: UserInDB, state: Optional[dict]) -> str:
    customer_id = state.get("stripe_customer_id") if state else None
    if customer_id:
        customer_id = str(customer_id)
        try:
            stripe.Customer.retrieve(customer_id)
            return customer_id
        except Exception as e:
            if _is_stripe_resource_missing_error(e):
                logger.warning(
                    "Stored Stripe customer %s is invalid under current key for user=%s; resetting and recreating",
                    customer_id,
                    user.email,
                )
                _reset_stripe_linked_state_for_mode_switch(
                    user, source="stripe_customer_missing_reset"
                )
            else:
                logger.warning(
                    "Stripe customer validation failed for %s (user=%s): %s",
                    customer_id,
                    user.email,
                    e,
                )
                return customer_id

    customer = stripe.Customer.create(
        email=user.email,
        name=f"{user.first_name} {user.last_name}".strip(),
        metadata={"user_id": user.id},
    )
    customer_id = str(customer.get("id"))

    upsert_customer_state(
        user.id,
        user.email,
        updates={"stripe_customer_id": customer_id},
        source="stripe_customer_create",
    )
    return customer_id


def _create_checkout_session_for_tier(
    user: UserInDB,
    tier: SubscriptionTier,
    current_state: Optional[dict] = None,
) -> BillingCheckoutSessionResponse:
    if tier not in {SubscriptionTier.BASIC, SubscriptionTier.PRO}:
        raise HTTPException(400, detail="Only paid tiers are supported for checkout")

    price_id = _stripe_price_for_tier(tier)
    if not price_id:
        raise HTTPException(500, detail=f"No Stripe price configured for tier '{tier.value}'")

    state = current_state or _ensure_billing_state_for_user(user)
    customer_id = _ensure_stripe_customer(user, state)

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=_checkout_redirect_url("success"),
            cancel_url=_checkout_redirect_url("cancelled"),
            client_reference_id=user.id,
            metadata={"user_id": user.id, "requested_tier": tier.value},
            allow_promotion_codes=True,
        )
    except Exception as e:
        logger.error("Failed creating Stripe checkout session for %s: %s", user.email, e)
        raise HTTPException(502, detail="Could not create checkout session")

    session_id = str(session.get("id"))
    checkout_url = str(session.get("url") or "")
    if not checkout_url:
        raise HTTPException(502, detail="Stripe checkout session missing URL")

    upsert_customer_state(
        user.id,
        user.email,
        updates={"stripe_customer_id": customer_id},
        source="checkout_session_create",
    )

    return BillingCheckoutSessionResponse(checkout_url=checkout_url, session_id=session_id)


@app.get("/api/billing/status", response_model=BillingStatusResponse, tags=["Billing"])
async def get_billing_status(user: UserInDB = Depends(require_auth)):
    refreshed = _sync_user_tier_with_billing(user)
    state = get_state_by_user_id(refreshed.id) if billing_store_enabled() else None
    return _billing_status_payload(refreshed, state)


@app.post("/api/billing/apple/verify", response_model=BillingStatusResponse, tags=["Billing"])
async def verify_apple_billing_purchase(
    data: AppleVerifyRequest,
    user: UserInDB = Depends(require_verified_user),
):
    if not _apple_verify_runtime_enabled():
        raise HTTPException(503, detail="Apple billing verification is unavailable")

    parsed = _validate_apple_purchase_payload(data)
    state = _ensure_billing_state_for_user(user)
    if not state:
        raise HTTPException(503, detail="Billing store is unavailable")

    upsert_customer_state(
        user.id,
        user.email,
        updates=_apple_state_updates_from_purchase(data, parsed),
        source="apple_purchase_verify",
    )
    refreshed = _sync_user_tier_with_billing(user)
    latest_state = get_state_by_user_id(refreshed.id) if billing_store_enabled() else None
    return _billing_status_payload(refreshed, latest_state)


@app.post(
    "/api/billing/checkout-session",
    response_model=BillingCheckoutSessionResponse,
    tags=["Billing"],
)
async def create_billing_checkout_session(
    data: BillingCheckoutSessionRequest,
    user: UserInDB = Depends(require_verified_user),
):
    _require_billing_enabled()
    return _create_checkout_session_for_tier(user=user, tier=data.tier)


@app.post(
    "/api/billing/portal-session",
    response_model=BillingPortalSessionResponse,
    tags=["Billing"],
)
async def create_billing_portal_session(user: UserInDB = Depends(require_verified_user)):
    _require_billing_enabled()

    state = _ensure_billing_state_for_user(user)
    customer_id = _ensure_stripe_customer(user, state)

    return_url = STRIPE_BILLING_PORTAL_RETURN_URL or _checkout_redirect_url("portal_return")
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
    except Exception as e:
        logger.error("Failed creating Stripe billing portal session for %s: %s", user.email, e)
        raise HTTPException(502, detail="Could not create billing portal session")

    return BillingPortalSessionResponse(portal_url=str(session.get("url")))


def _resolve_subscription_id_for_customer(customer_id: str, state: Optional[dict]) -> Optional[str]:
    subscription_id = str((state or {}).get("stripe_subscription_id") or "").strip()
    if subscription_id:
        try:
            sub = stripe.Subscription.retrieve(subscription_id)
            payload = (
                sub.to_dict_recursive() if hasattr(sub, "to_dict_recursive") else dict(sub)
            )
            status = str(payload.get("status") or "").lower()
            if status in ACTIVE_STRIPE_SUBSCRIPTION_STATUSES:
                return subscription_id
        except Exception as e:
            if _is_stripe_resource_missing_error(e):
                logger.warning(
                    "Stored Stripe subscription %s is invalid under current key for customer=%s; clearing cached id",
                    subscription_id,
                    customer_id,
                )
                if state:
                    upsert_customer_state(
                        state["user_id"],
                        state["email"],
                        updates={
                            "stripe_subscription_id": None,
                            "price_id": None,
                            "current_period_end": None,
                            "cancel_at_period_end": False,
                            "subscription_status": "none",
                            "plan_tier": SubscriptionTier.FREE.value,
                        },
                        source="stripe_subscription_missing_reset",
                    )
            else:
                logger.warning(
                    "Unable to validate stored Stripe subscription %s for customer=%s: %s",
                    subscription_id,
                    customer_id,
                    e,
                )

    try:
        subscriptions = stripe.Subscription.list(customer=customer_id, status="all", limit=25)
        records = subscriptions.get("data", []) if isinstance(subscriptions, dict) else getattr(subscriptions, "data", [])
        for sub in records or []:
            status = str(sub.get("status") or "").lower()
            if status in ACTIVE_STRIPE_SUBSCRIPTION_STATUSES:
                return str(sub.get("id") or "").strip() or None
    except Exception as e:
        logger.warning("Unable to list subscriptions for customer %s: %s", customer_id, e)

    return None


@app.post(
    "/api/billing/change-subscription",
    response_model=BillingChangeSubscriptionResponse,
    tags=["Billing"],
)
async def change_billing_subscription(
    data: BillingChangeSubscriptionRequest,
    user: UserInDB = Depends(require_verified_user),
):
    """Switch a paid subscription between Basic and Pro."""
    _require_billing_enabled()

    if data.tier not in {SubscriptionTier.BASIC, SubscriptionTier.PRO}:
        raise HTTPException(400, detail="Target tier must be basic or pro")

    state = _ensure_billing_state_for_user(user)
    customer_id = _ensure_stripe_customer(user, state)
    target_price_id = _stripe_price_for_tier(data.tier)
    if not target_price_id:
        raise HTTPException(500, detail=f"No Stripe price configured for tier '{data.tier.value}'")

    subscription_id = _resolve_subscription_id_for_customer(customer_id, state)
    if not subscription_id:
        checkout = _create_checkout_session_for_tier(user=user, tier=data.tier, current_state=state)
        return BillingChangeSubscriptionResponse(
            mode="checkout",
            message="No active subscription found. Redirect to checkout to start subscription.",
            checkout_url=checkout.checkout_url,
        )

    try:
        subscription = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
    except Exception as e:
        logger.error("Failed retrieving Stripe subscription %s: %s", subscription_id, e)
        raise HTTPException(502, detail="Could not load current subscription")

    payload = (
        subscription.to_dict_recursive()
        if hasattr(subscription, "to_dict_recursive")
        else dict(subscription)
    )
    status = str(payload.get("status") or "").lower()
    if status not in ACTIVE_STRIPE_SUBSCRIPTION_STATUSES:
        checkout = _create_checkout_session_for_tier(user=user, tier=data.tier, current_state=state)
        return BillingChangeSubscriptionResponse(
            mode="checkout",
            message="Current subscription is not active. Redirect to checkout to re-subscribe.",
            checkout_url=checkout.checkout_url,
        )

    items = payload.get("items", {}).get("data", []) if isinstance(payload.get("items"), dict) else []
    if not items:
        raise HTTPException(500, detail="Subscription has no billable items")

    first_item = items[0] if isinstance(items[0], dict) else {}
    current_price_id = str((first_item.get("price") or {}).get("id") or "")
    if current_price_id == target_price_id:
        return BillingChangeSubscriptionResponse(
            mode="no_op",
            message=f"Subscription is already on {data.tier.value}.",
            checkout_url=None,
        )

    item_id = str(first_item.get("id") or "")
    if not item_id:
        raise HTTPException(500, detail="Subscription item id missing")

    try:
        modified = stripe.Subscription.modify(
            subscription_id,
            items=[{"id": item_id, "price": target_price_id}],
            proration_behavior="create_prorations",
            cancel_at_period_end=False,
        )
        modified_payload = (
            modified.to_dict_recursive()
            if hasattr(modified, "to_dict_recursive")
            else dict(modified)
        )
        _process_subscription_event(modified_payload, source="stripe_change_subscription")
    except Exception as e:
        logger.error("Failed switching subscription %s to %s: %s", subscription_id, data.tier.value, e)
        raise HTTPException(502, detail="Could not change subscription tier")

    return BillingChangeSubscriptionResponse(
        mode="updated",
        message=f"Subscription changed to {data.tier.value}.",
        checkout_url=None,
    )


@app.post("/api/billing/cancel-subscription", response_model=MessageResponse, tags=["Billing"])
async def cancel_billing_subscription(user: UserInDB = Depends(require_verified_user)):
    """Mark active Stripe subscription to cancel at period end."""
    _require_billing_enabled()

    state = _ensure_billing_state_for_user(user)
    customer_id = _ensure_stripe_customer(user, state)
    subscription_id = _resolve_subscription_id_for_customer(customer_id, state)
    if not subscription_id:
        raise HTTPException(400, detail="No active subscription found to cancel")

    try:
        subscription = stripe.Subscription.modify(
            subscription_id,
            cancel_at_period_end=True,
        )
        payload = (
            subscription.to_dict_recursive()
            if hasattr(subscription, "to_dict_recursive")
            else dict(subscription)
        )
        _process_subscription_event(payload, source="stripe_cancel_at_period_end")
    except Exception as e:
        logger.error("Failed to cancel subscription for %s: %s", user.email, e)
        raise HTTPException(502, detail="Could not cancel subscription")

    refreshed_state = get_state_by_user_id(user.id)
    current_period_end = _to_utc_datetime(refreshed_state.get("current_period_end") if refreshed_state else None)
    if current_period_end:
        return MessageResponse(
            message=f"Subscription will cancel at period end ({current_period_end.isoformat()})."
        )
    return MessageResponse(message="Subscription cancellation scheduled at period end.")


def _process_subscription_event(subscription_payload: dict, source: str) -> None:
    customer_id = str(subscription_payload.get("customer") or "").strip()
    if not customer_id:
        return

    state = get_state_by_customer_id(customer_id)
    if not state:
        metadata = subscription_payload.get("metadata") or {}
        user_id = metadata.get("user_id")
        if not user_id:
            logger.warning("Stripe subscription event ignored: no mapped customer state for %s", customer_id)
            return
        user = get_user_by_id(str(user_id))
        if not user:
            logger.warning("Stripe subscription event ignored: user not found for user_id=%s", user_id)
            return
        state = upsert_customer_state(
            user.id,
            user.email,
            updates={"stripe_customer_id": customer_id},
            source="stripe_subscription_user_link",
        )

    updated = _apply_subscription_snapshot(state, subscription_payload, source=source)
    user = get_user_by_id(updated["user_id"])
    if user:
        _sync_user_tier_with_billing(user)


def _process_checkout_completed(event_payload: dict) -> None:
    customer_id = str(event_payload.get("customer") or "").strip()
    subscription_id = str(event_payload.get("subscription") or "").strip()
    metadata = event_payload.get("metadata") or {}
    user_id = str(event_payload.get("client_reference_id") or metadata.get("user_id") or "").strip()

    state: Optional[dict] = None
    user: Optional[UserInDB] = None
    if user_id:
        user = get_user_by_id(user_id)
        if user:
            state = upsert_customer_state(
                user.id,
                user.email,
                updates={"stripe_customer_id": customer_id or None},
                source="checkout_completed_link",
            )

    if not state and customer_id:
        state = get_state_by_customer_id(customer_id)
        if state:
            user = get_user_by_id(state["user_id"])

    if not state:
        logger.warning(
            "Stripe checkout.session.completed ignored: cannot map customer=%s user_id=%s",
            customer_id,
            user_id,
        )
        return

    if subscription_id:
        try:
            subscription = stripe.Subscription.retrieve(
                subscription_id,
                expand=["items.data.price"],
            )
            payload = (
                subscription.to_dict_recursive()
                if hasattr(subscription, "to_dict_recursive")
                else dict(subscription)
            )
            _process_subscription_event(payload, source="stripe_checkout_completed")
            return
        except Exception as e:
            logger.warning(
                "Stripe checkout completed subscription fetch failed for %s: %s",
                subscription_id,
                e,
            )

    requested_tier = metadata.get("requested_tier")
    requested_tier_enum = _coerce_subscription_tier(requested_tier, SubscriptionTier.FREE)
    updates: dict[str, object] = {
        "stripe_customer_id": customer_id or state.get("stripe_customer_id"),
        "stripe_subscription_id": subscription_id or state.get("stripe_subscription_id"),
    }
    if requested_tier_enum in {SubscriptionTier.BASIC, SubscriptionTier.PRO}:
        updates["plan_tier"] = requested_tier_enum.value
        updates["subscription_status"] = "active"
    updated = upsert_customer_state(state["user_id"], state["email"], updates=updates, source="stripe_checkout_completed")
    if user is None:
        user = get_user_by_id(updated["user_id"])
    if user:
        _sync_user_tier_with_billing(user)


def _process_invoice_payment_failed(event_payload: dict) -> None:
    customer_id = str(event_payload.get("customer") or "").strip()
    if not customer_id:
        return
    updated = upsert_state_by_customer_id(
        customer_id,
        updates={
            "subscription_status": "past_due",
            "stripe_subscription_id": event_payload.get("subscription"),
        },
        source="stripe_invoice_payment_failed",
    )
    if not updated:
        return
    user = get_user_by_id(updated["user_id"])
    if user:
        _sync_user_tier_with_billing(user)


@app.post("/api/webhooks/stripe", response_model=MessageResponse, tags=["System"])
async def stripe_webhook(request: Request):
    _require_billing_enabled()
    client_id = _rate_limit_client_id(request)
    _enforce_rate_limit(
        identifier=f"webhook:stripe:ip:{client_id}",
        max_requests=STRIPE_WEBHOOK_RATE_LIMIT,
        window_seconds=STRIPE_WEBHOOK_RATE_WINDOW_SECONDS,
    )

    sig_header = request.headers.get("stripe-signature")
    if not sig_header:
        raise HTTPException(403, detail="Missing Stripe signature header")

    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logger.warning("Invalid Stripe webhook signature: %s", e)
        raise HTTPException(403, detail="Invalid Stripe webhook signature")

    event_payload = (
        event.to_dict_recursive() if hasattr(event, "to_dict_recursive") else dict(event)
    )

    event_id = str(event_payload.get("id") or "")
    event_type = str(event_payload.get("type") or "").lower()
    if not event_id:
        raise HTTPException(400, detail="Missing Stripe event id")

    inserted = register_webhook_event(event_id, event_type, event_payload)
    if not inserted:
        return MessageResponse(message=f"Duplicate Stripe event ignored: {event_id}")

    try:
        event_data = event_payload.get("data", {}).get("object", {})
        if event_type == "checkout.session.completed":
            _process_checkout_completed(event_data)
        elif event_type in {"customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"}:
            _process_subscription_event(event_data, source=f"stripe_{event_type.replace('.', '_')}")
        elif event_type == "invoice.payment_failed":
            _process_invoice_payment_failed(event_data)
        else:
            logger.info("Unhandled Stripe event type=%s (recorded only)", event_type)

        mark_webhook_event(event_id, "success")
        return MessageResponse(message=f"Processed Stripe event type={event_type}")
    except Exception as e:
        logger.error("Stripe webhook processing failed for event %s: %s", event_id, e, exc_info=True)
        mark_webhook_event(event_id, "error", error_message=str(e))
        raise HTTPException(500, detail="Stripe webhook processing failed")


# ============================================================
# Prediction Endpoints
# ============================================================

SUPPORTED_HORIZON_ALIASES: dict[str, str] = {
    "5d": "5d",
    "5day": "5d",
    "5days": "5d",
    "5 days": "5d",
    "20d": "20d",
    "20day": "20d",
    "20days": "20d",
    "20 days": "20d",
}

MODEL_HORIZON_REQUIREMENTS: dict[str, str] = {
    "lstm_5d": "5d",
    "lstm_jackpot": "20d",
}


def _canonical_model_horizon(h: str | None) -> str | None:
    if h is None:
        return None
    return SUPPORTED_HORIZON_ALIASES.get(h.strip().lower())


def _require_supported_horizon(h: str | None) -> str:
    canonical = _canonical_model_horizon(h)
    if canonical:
        return canonical
    supported = ", ".join(MODEL_HORIZONS)
    raise HTTPException(400, detail=f"Unsupported horizon '{h}'. Allowed horizons: {supported}.")


def _normalize_horizon(h: str) -> str:
    """Normalize supported prediction horizons to data timeframe."""
    # Current production models are trained for 5d/20d horizons using daily bars.
    canonical = _canonical_model_horizon(h) or "5d"
    if canonical not in {"5d", "20d"}:
        return "1Day"
    return "1Day"


def _horizon_to_display(h: str) -> str:
    """Convert horizon to canonical display format."""
    canonical = _canonical_model_horizon(h)
    return canonical or "5d"


def _sanitize_probability(value: Any, default: float = 0.5) -> float:
    """Normalize probability to finite [0,1], accepting either 0-1 or 0-100."""
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(prob):
        return default

    if prob > 1.0 and prob <= 100.0:
        prob = prob / 100.0

    if prob < 0.0:
        return 0.0
    if prob > 1.0:
        return 1.0
    return prob


def _sanitize_last_close(value: Any, default: float = 0.0) -> float:
    """Normalize last_close to finite non-negative float."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(price):
        return default
    if price < 0.0:
        return default
    return price


def predict_for_ticker(ticker: str, horizon: str = "5d"):
    """Generate prediction for a single ticker."""
    if not PYTHIA_AVAILABLE:
        if not ALLOW_RANDOM_FALLBACK:
            raise RuntimeError(
                "Local ML models unavailable and random fallback disabled."
            )
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
    prob_up = _sanitize_probability(MODEL.predict_proba(Xs)[:, 1][-1], default=0.5)
    last_close = _sanitize_last_close(raw["Close"].iloc[-1], default=0.0)
    signal = "buy" if prob_up >= THRESHOLD else ("sell" if prob_up <= 1 - THRESHOLD else "hold")

    return prob_up, signal, last_close


def _cache_last_close_for_dashboard(
    ticker: str,
    horizon: str,
    last_close: float,
    source: str,
) -> None:
    """Persist last_close for dashboard fallback use."""
    if last_close is None:
        return
    try:
        value = float(last_close)
        if value <= 0:
            return
        upsert_last_close_pg(
            ticker=ticker.upper(),
            timeframe=_horizon_to_display(horizon),
            last_close=value,
            source=source,
        )
    except Exception as exc:
        logger.debug("Failed to cache last_close for %s (%s): %s", ticker, horizon, exc)


def _cached_price_fallback_response(
    ticker: str,
    horizon: str,
    reason: str,
) -> Optional[PredictResponse]:
    """Return neutral prediction with cached last_close when live prediction fails."""
    try:
        cached = get_last_close_pg(
            ticker=ticker.upper(),
            timeframe=_horizon_to_display(horizon),
            max_age_hours=PRICE_CACHE_MAX_AGE_HOURS,
        )
    except Exception as exc:
        logger.debug("Failed reading cached price for %s (%s): %s", ticker, horizon, exc)
        return None

    if not cached:
        return None

    logger.warning(
        "Using cached last_close for %s (%s) due to live prediction failure: %s",
        ticker.upper(),
        horizon,
        reason,
    )
    return PredictResponse(
        ticker=ticker.upper(),
        horizon=_horizon_to_display(horizon),
        prob_up=0.5,
        signal="hold",
        last_close=float(cached["last_close"]),
    )


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

    allowed_stocks = _available_stocks_for_tier(tier_config)
    allowed_stocks_upper = {symbol.upper() for symbol in allowed_stocks}
    return ticker.upper() in allowed_stocks_upper


@app.get("/predict/{ticker}", response_model=PredictResponse, tags=["Predictions"])
async def predict(
    ticker: str,
    horizon: str = "5d",
    model: Optional[str] = None,
    user: Optional[UserInDB] = Depends(get_current_user),
):
    """Get prediction for a single ticker.

    Supports optional model parameter to use specific ML model:
    - gradient_boosting (default)
    - random_forest
    - linear_regression
    - lstm
    """
    user_email = user.email if user else "anonymous"
    canonical_horizon = _require_supported_horizon(horizon)

    # Check tier access (but allow anonymous users basic access)
    if user and not check_tier_access(user, ticker, canonical_horizon):
        tier_name = user.tier.value
        logger.info(
            f"Prediction access denied: {ticker} ({canonical_horizon}) for {user_email} (tier={tier_name})"
        )
        raise HTTPException(
            403,
            detail=f"Your {tier_name} tier doesn't include access to this ticker/horizon. Please upgrade.",
        )

    try:
        # Use divination backend for model-specific predictions
        if model and model.lower() in ["gradient_boosting", "random_forest", "linear_regression", "lstm"]:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{DIVINATION_API_URL}/predict/{ticker.upper()}",
                        params={
                            "horizon": canonical_horizon,
                            "model": model.lower(),
                            "task": "classifier",
                        },
                    )
                    if response.status_code == 200:
                        data = response.json()
                        result = PredictResponse(
                            ticker=data.get("ticker", ticker.upper()),
                            horizon=_horizon_to_display(str(data.get("horizon", canonical_horizon))),
                            prob_up=_sanitize_probability(data.get("prob_up"), default=0.5),
                            signal=data.get("signal", "hold"),
                            last_close=_sanitize_last_close(data.get("last_close"), default=0.0),
                        )
                        _cache_last_close_for_dashboard(
                            ticker=ticker.upper(),
                            horizon=canonical_horizon,
                            last_close=result.last_close,
                            source=f"divination:{model.lower()}",
                        )
                        return result
            except Exception as e:
                logger.warning(f"Failed to fetch {model} prediction from divination: {e}, using fallback")

        # Default: use local prediction (gradient boosting)
        prob_up, signal, last_close = predict_for_ticker(ticker.upper(), horizon=canonical_horizon)
        logger.debug(
            f"Prediction: {ticker.upper()} ({canonical_horizon}) = {signal} ({prob_up:.2%}) for {user_email}"
        )
        result = PredictResponse(
            ticker=ticker.upper(),
            horizon=canonical_horizon,
            prob_up=_sanitize_probability(prob_up, default=0.5),
            signal=signal,
            last_close=_sanitize_last_close(last_close, default=0.0),
        )
        if PYTHIA_AVAILABLE:
            _cache_last_close_for_dashboard(
                ticker=ticker.upper(),
                horizon=canonical_horizon,
                last_close=result.last_close,
                source="local:model",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {ticker}: {e}")
        fallback = _cached_price_fallback_response(
            ticker=ticker,
            horizon=canonical_horizon,
            reason=str(e),
        )
        if fallback:
            return fallback
        raise HTTPException(400, detail=str(e))


@app.get("/predict/lstm_5d/{ticker}", response_model=PredictResponse)
async def predict_lstm_5d(ticker: str):
    """Proxy LSTM 5-Day predictions from divination backend, or use fallback."""
    upstream_error = "unknown upstream error"
    try:
        async with httpx.AsyncClient(timeout=DIVINATION_LSTM_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{DIVINATION_API_URL}/predict/lstm_5d/{ticker}")
            if response.status_code == 200:
                data = response.json()
                # Transform LSTM response to match PredictResponse schema
                result = PredictResponse(
                    ticker=data.get("ticker", ticker),
                    horizon=_horizon_to_display(str(data.get("horizon", "5d"))),
                    prob_up=_sanitize_probability(
                        data.get("prob_up", data.get("probability")),
                        default=0.5,
                    ),
                    signal=data.get("signal", "hold"),
                    last_close=_sanitize_last_close(data.get("last_close"), default=0.0),
                )
                _cache_last_close_for_dashboard(
                    ticker=ticker.upper(),
                    horizon="5d",
                    last_close=result.last_close,
                    source="divination:lstm_5d",
                )
                return result
            try:
                detail = response.json().get("detail") or response.text.strip()
            except Exception:
                detail = response.text.strip() or "no error body"
            upstream_error = f"status {response.status_code}: {detail}"
    except Exception as e:
        upstream_error = str(e)
        logger.debug(f"LSTM 5D prediction from divination failed: {e}, evaluating fallback")

    fallback = _cached_price_fallback_response(ticker=ticker, horizon="5d", reason=upstream_error)
    if fallback:
        return fallback

    if LSTM_PROXY_DISABLE_LOCAL_FALLBACK:
        raise HTTPException(503, detail=f"LSTM 5D upstream unavailable: {upstream_error}")

    if not PYTHIA_AVAILABLE and not ALLOW_RANDOM_FALLBACK:
        raise HTTPException(503, detail=f"LSTM 5D upstream unavailable: {upstream_error}")

    # Fallback to local prediction
    try:
        prob_up, signal, last_close = predict_for_ticker(ticker.upper(), horizon="5d")
        result = PredictResponse(
            ticker=ticker.upper(),
            horizon="5d",
            prob_up=_sanitize_probability(prob_up, default=0.5),
            signal=signal,
            last_close=_sanitize_last_close(last_close, default=0.0),
        )
        if PYTHIA_AVAILABLE or ALLOW_RANDOM_FALLBACK:
            _cache_last_close_for_dashboard(
                ticker=ticker.upper(),
                horizon="5d",
                last_close=result.last_close,
                source="local:lstm_5d_fallback" if PYTHIA_AVAILABLE else "demo:lstm_5d_fallback",
            )
        return result
    except Exception as e:
        logger.error(f"LSTM 5D prediction error for {ticker}: {e}")
        fallback = _cached_price_fallback_response(ticker=ticker, horizon="5d", reason=str(e))
        if fallback:
            return fallback
        raise HTTPException(503, detail=f"LSTM 5D unavailable: {e}")




@app.get("/predict/lstm_jackpot/{ticker}", response_model=PredictResponse)
async def predict_lstm_jackpot(ticker: str):
    """Proxy LSTM Jackpot predictions from divination backend, or use fallback."""
    upstream_error = "unknown upstream error"
    try:
        async with httpx.AsyncClient(timeout=DIVINATION_LSTM_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{DIVINATION_API_URL}/predict/lstm_jackpot/{ticker}")
            if response.status_code == 200:
                data = response.json()
                # Transform LSTM response to match PredictResponse schema
                result = PredictResponse(
                    ticker=data.get("ticker", ticker),
                    horizon=_horizon_to_display(str(data.get("horizon", "20d"))),
                    prob_up=_sanitize_probability(
                        data.get("prob_up", data.get("probability")),
                        default=0.5,
                    ),
                    signal=data.get("signal", "hold"),
                    last_close=_sanitize_last_close(data.get("last_close"), default=0.0),
                )
                _cache_last_close_for_dashboard(
                    ticker=ticker.upper(),
                    horizon="20d",
                    last_close=result.last_close,
                    source="divination:lstm_jackpot",
                )
                return result
            try:
                detail = response.json().get("detail") or response.text.strip()
            except Exception:
                detail = response.text.strip() or "no error body"
            upstream_error = f"status {response.status_code}: {detail}"
    except Exception as e:
        upstream_error = str(e)
        logger.debug(f"LSTM Jackpot prediction from divination failed: {e}, evaluating fallback")

    fallback = _cached_price_fallback_response(ticker=ticker, horizon="20d", reason=upstream_error)
    if fallback:
        return fallback

    if LSTM_PROXY_DISABLE_LOCAL_FALLBACK:
        raise HTTPException(503, detail=f"LSTM Jackpot upstream unavailable: {upstream_error}")

    if not PYTHIA_AVAILABLE and not ALLOW_RANDOM_FALLBACK:
        raise HTTPException(503, detail=f"LSTM Jackpot upstream unavailable: {upstream_error}")

    # Fallback to local prediction
    try:
        prob_up, signal, last_close = predict_for_ticker(ticker.upper(), horizon="20d")
        result = PredictResponse(
            ticker=ticker.upper(),
            horizon="20d",
            prob_up=_sanitize_probability(prob_up, default=0.5),
            signal=signal,
            last_close=_sanitize_last_close(last_close, default=0.0),
        )
        if PYTHIA_AVAILABLE or ALLOW_RANDOM_FALLBACK:
            _cache_last_close_for_dashboard(
                ticker=ticker.upper(),
                horizon="20d",
                last_close=result.last_close,
                source="local:lstm_jackpot_fallback" if PYTHIA_AVAILABLE else "demo:lstm_jackpot_fallback",
            )
        return result
    except Exception as e:
        logger.error(f"LSTM Jackpot prediction error for {ticker}: {e}")
        fallback = _cached_price_fallback_response(ticker=ticker, horizon="20d", reason=str(e))
        if fallback:
            return fallback
        raise HTTPException(503, detail=f"LSTM Jackpot unavailable: {e}")


@app.get("/api/market/history/{ticker}", response_model=MarketHistoryResponse, tags=["Market"])
async def market_history(
    ticker: str,
    interval: str = Query(default="1d"),
    limit: int = Query(default=252, ge=60, le=1000),
):
    """Proxy mobile-ready daily OHLCV history from divination while preserving one public base URL."""
    try:
        async with httpx.AsyncClient(timeout=DIVINATION_LSTM_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f"{DIVINATION_API_URL}/api/market/history/{ticker.upper()}",
                params={"interval": interval, "limit": limit},
            )
            if response.status_code == 200:
                return MarketHistoryResponse.model_validate(response.json())
            try:
                detail = response.json().get("detail") or response.text.strip()
            except Exception:
                detail = response.text.strip() or "no error body"
            raise HTTPException(response.status_code, detail=detail)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Market history proxy error for %s: %s", ticker, exc)
        raise HTTPException(503, detail=f"Market history unavailable: {exc}") from exc



@app.get("/api/universe")
async def get_universe(user: Optional[UserInDB] = Depends(get_current_user)):
    """Get the list of tickers available to the user based on their tier."""
    if not user:
        tier_config = TIER_CONFIG[SubscriptionTier.FREE]
    else:
        tier_config = TIER_CONFIG[user.tier]

    stocks_limit = tier_config["stocks_limit"]
    available = _available_stocks_for_tier(tier_config)

    return {
        "universe": available,
        "total_available": len(UNIVERSE),
        "your_limit": stocks_limit,
        "timeframes": tier_config["timeframes"],
    }


@app.get("/healthz", tags=["System"])
def healthz():
    """Health check endpoint with system status."""
    return JSONResponse({
        "ok": True,
        "message": "ok",
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "version": "0.1.0",
        "universe": UNIVERSE,
        "data_source": DATA_SOURCE if PYTHIA_AVAILABLE else "demo",
        "ml_models_available": PYTHIA_AVAILABLE,
    })


@app.get("/api/status", tags=["System"])
def status():
    """Get API status and configuration."""
    return JSONResponse({
        "service": "pythia_prophecy",
        "version": "0.1.0",
        "status": "operational",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "auth": "operational",
            "predictions": "operational" if PYTHIA_AVAILABLE else "degraded",
            "database": "operational",
            "email": "operational",
        },
        "limits": {
            "universe_size": len(UNIVERSE),
            "prediction_threshold": THRESHOLD,
        }
    })


@app.get("/api/performance/track-record", response_model=TrackRecordResponse, tags=["Performance"])
def performance_track_record(model: str = Query("lstm_5d")):
    """Public track record summary for investor/user transparency."""
    return get_track_record(model=model)


@app.get("/api/performance/curve", response_model=TrackRecordCurveResponse, tags=["Performance"])
def performance_curve(model: str = Query("lstm_5d")):
    """Public model-vs-benchmark curve for homepage visualization."""
    return get_track_record_curve(model=model)


_SPORTS_BOARDS_ALL = ("golf", "tennis", "basketball", "mlb", "football", "soccer", "olympics")


def _empty_sports_board_collection() -> SportsBoardCollection:
    return SportsBoardCollection(
        upcoming=[],
        backtests=[],
        updated_at=datetime.now(timezone.utc),
        source="not_requested",
        selectedDate=None,
        availableDates=[],
        seasonSummary=None,
    )


def _parse_sports_filter(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    requested: set[str] = set()
    for token in raw.split(","):
        normalized = token.strip().lower()
        if not normalized:
            continue
        if normalized in _SPORTS_BOARDS_ALL:
            requested.add(normalized)
    return requested or None


@app.get("/api/sports/boards", response_model=SportsBoardsResponse, tags=["Sports"])
async def sports_boards(
    mlb_date: str | None = Query(None),
    basketball_date: str | None = Query(None),
    football_date: str | None = Query(None),
    soccer_date: str | None = Query(None),
    sports: str | None = Query(
        None,
        description=(
            "Optional comma-separated list of sports to compute "
            "(golf, tennis, basketball, mlb, football, soccer). "
            "Omit to return all sports (legacy behavior). "
            "Sports not requested are returned as empty placeholder collections "
            "so the response shape is unchanged for older clients."
        ),
    ),
):
    """Runtime-filtered sports boards for public marketing and dashboard surfaces.

    Supports per-sport pagination via ``?sports=`` so clients (web tab, iOS sport
    switcher) only pay for inference on the sport(s) they intend to render. When
    ``sports`` is omitted the endpoint returns every sport for backward compat.
    Live sport feeds (basketball, mlb, football) are fetched concurrently.
    """
    requested = _parse_sports_filter(sports)
    wants = (lambda key: True) if requested is None else (lambda key: key in requested)

    def _golf() -> SportsBoardCollection:
        return _sports_board_collection(
            upcoming_filename="upcoming_tournaments.json",
            backtests_filename="historical_backtests.json",
            sport="golf",
        )

    def _tennis() -> SportsBoardCollection:
        return _sports_board_collection(
            upcoming_filename="wta_upcoming_tournaments.json",
            backtests_filename="wta_historical_backtests.json",
            sport="tennis",
        )

    golf = _golf() if wants("golf") else _empty_sports_board_collection()
    tennis = _tennis() if wants("tennis") else _empty_sports_board_collection()

    async def _basketball_or_none() -> Optional[SportsBoardCollection]:
        if not wants("basketball"):
            return None
        return await _live_basketball_board_collection(basketball_date=basketball_date)

    async def _mlb_or_none() -> Optional[SportsBoardCollection]:
        if not wants("mlb"):
            return None
        return await _live_mlb_board_collection(mlb_date=mlb_date)

    async def _football_or_none() -> Optional[SportsBoardCollection]:
        if not wants("football"):
            return None
        return await _live_football_board_collection(football_date=football_date)

    async def _soccer_or_none() -> Optional[SportsBoardCollection]:
        if not wants("soccer"):
            return None
        return await _live_soccer_board_collection(soccer_date=soccer_date)

    async def _olympics_or_none() -> Optional[SportsBoardCollection]:
        if not wants("olympics"):
            return None
        return await _live_olympics_board_collection()

    basketball_live, mlb_live, football_live, soccer_live, olympics_live = await asyncio.gather(
        _basketball_or_none(),
        _mlb_or_none(),
        _football_or_none(),
        _soccer_or_none(),
        _olympics_or_none(),
    )

    if not wants("basketball"):
        basketball = _empty_sports_board_collection()
    elif basketball_live is not None:
        basketball = basketball_live
    else:
        basketball = _sports_board_collection(
            upcoming_filename="basketball_upcoming_tournaments.json",
            backtests_filename="basketball_historical_backtests.json",
            sport="basketball",
            selected_date=basketball_date,
        )

    if not wants("mlb"):
        mlb = _empty_sports_board_collection()
    elif mlb_live is not None:
        mlb = mlb_live
    else:
        mlb = _sports_board_collection(
            upcoming_filename="mlb_upcoming_tournaments.json",
            backtests_filename="mlb_historical_backtests.json",
            sport="mlb",
            selected_date=mlb_date,
        )

    if not wants("football"):
        football = _empty_sports_board_collection()
    elif football_live is not None:
        football = football_live
    else:
        football = _sports_board_collection(
            upcoming_filename="football_upcoming_tournaments.json",
            backtests_filename="football_historical_backtests.json",
            sport="football",
            selected_date=football_date,
        )

    if not wants("soccer"):
        soccer = _empty_sports_board_collection()
    elif soccer_live is not None:
        soccer = soccer_live
    else:
        soccer = _sports_board_collection(
            upcoming_filename="soccer_upcoming_tournaments.json",
            backtests_filename="soccer_historical_backtests.json",
            sport="soccer",
            selected_date=soccer_date,
        )

    if not wants("olympics"):
        olympics = _empty_sports_board_collection()
    elif olympics_live is not None:
        olympics = olympics_live
    else:
        olympics = _sports_board_collection(
            upcoming_filename="olympics_upcoming_tournaments.json",
            backtests_filename="olympics_historical_backtests.json",
            sport="olympics",
        )

    return SportsBoardsResponse(
        golf=golf,
        tennis=tennis,
        basketball=basketball,
        mlb=mlb,
        football=football,
        soccer=soccer,
        olympics=olympics,
    )


# ============================================================
# Oracle (Watchlist) Endpoints
# ============================================================

def _get_available_for_user(user: UserInDB) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Get available stocks, categories, and timeframes for user's tier."""
    tier_config = TIER_CONFIG[user.tier]
    available_stocks = _available_stocks_for_tier(tier_config)

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
            valid_timeframes = [available_timeframes[0]] if available_timeframes else ["5d"]
    else:
        valid_watchlist = []
        valid_timeframes = [available_timeframes[0]] if available_timeframes else ["5d"]

    return OracleResponse(
        watchlist=valid_watchlist,
        timeframes=valid_timeframes,
        available_stocks=available_stocks,
        available_categories=available_categories,
        available_timeframes=available_timeframes,
    )


@app.get("/api/oracle/watchlist-insights", response_model=WatchlistInsightsResponse)
async def get_watchlist_insights(
    limit: int = Query(20, ge=1, le=100),
    user: UserInDB = Depends(require_verified_user),
):
    """Return watchlist insights with optional Finviz enrichment."""
    oracle = get_user_oracle(user.id)
    available_stocks, _, _ = _get_available_for_user(user)
    available_set = {symbol.upper() for symbol in available_stocks}

    if not oracle or not oracle.watchlist:
        return WatchlistInsightsResponse(
            watchlist=[],
            finviz_enabled=_finviz_enabled(),
            insights=[],
        )

    scoped_watchlist: list[str] = []
    seen: set[str] = set()
    for ticker in oracle.watchlist:
        normalized = ticker.upper()
        if normalized in available_set and normalized not in seen:
            seen.add(normalized)
            scoped_watchlist.append(normalized)

    if limit > 0:
        scoped_watchlist = scoped_watchlist[:limit]

    insights: list[dict[str, Any]] = []
    if _finviz_enabled() and scoped_watchlist:
        quote_tasks = [_fetch_finviz_quote(ticker) for ticker in scoped_watchlist]
        quote_results = await asyncio.gather(*quote_tasks, return_exceptions=True)
        for ticker, result in zip(scoped_watchlist, quote_results):
            quote_payload: Optional[dict[str, Any]]
            if isinstance(result, Exception):
                quote_payload = None
            else:
                quote_payload = result
            insights.append(_normalize_watchlist_insight(ticker, quote_payload))
    else:
        for ticker in scoped_watchlist:
            insights.append(_normalize_watchlist_insight(ticker, None))

    return WatchlistInsightsResponse(
        watchlist=scoped_watchlist,
        finviz_enabled=_finviz_enabled(),
        insights=insights,
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
    current_timeframes = oracle.timeframes if oracle else ["5d"]

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
        valid_timeframes = [available_timeframes[0]] if available_timeframes else ["5d"]

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
    user_limits = ANALYSIS_RATE_LIMIT_STORAGE[user.id]

    if user_limits["date"] != today:
        user_limits["date"] = today
        user_limits["count"] = 0

    used = user_limits["count"]
    remaining = limit - used
    return used, remaining, limit


def _increment_rate_limit(user: UserInDB) -> None:
    """Increment rate limit counter for user."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    user_limits = ANALYSIS_RATE_LIMIT_STORAGE[user.id]

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
        models=tier_config.get("models", ["lstm_5d", "lstm_jackpot"]),
        tasks=tier_config.get("tasks", ["classifier"]),
        can_use_custom=user.tier == SubscriptionTier.PRO,
    )


@app.get("/api/analyze/universe")
async def get_analysis_universe(user: UserInDB = Depends(require_verified_user)):
    """Get available stocks organized by category for the analysis page."""
    tier_config = TIER_CONFIG[user.tier]
    available_stocks = _available_stocks_for_tier(tier_config)

    available_set = set(available_stocks)

    # Filter categories to only include stocks user has access to
    filtered_categories = {}
    for category, stocks in STOCK_CATEGORIES.items():
        available_in_category = [s for s in stocks if s in available_set]
        if available_in_category:
            filtered_categories[category] = available_in_category

    return {
        "stocks": available_stocks,
        "categories": filtered_categories,
    }


def _analysis_error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        detail: str | object | None = None
        try:
            payload = exc.response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail") or payload.get("message") or payload.get("error")
            else:
                detail = payload
        except Exception:
            detail = None
        if detail is None:
            detail = exc.response.text.strip() or str(exc)
        return f"upstream status {status}: {detail}"
    if isinstance(exc, httpx.RequestError):
        return f"upstream request error: {exc}"
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc)


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

    canonical_horizon = _require_supported_horizon(data.horizon)
    required_horizon = MODEL_HORIZON_REQUIREMENTS.get(data.model)
    if required_horizon and canonical_horizon != required_horizon:
        raise HTTPException(
            400,
            detail=f"Model '{data.model}' only supports horizon '{required_horizon}'.",
        )

    # Validate tickers are accessible
    available_stocks = _available_stocks_for_tier(tier_config)

    available_upper = {s.upper() for s in available_stocks}
    invalid_tickers = [t for t in data.tickers if t.upper() not in available_upper]
    if invalid_tickers:
        raise HTTPException(
            400,
            detail=f"Tickers not available in your tier: {', '.join(invalid_tickers)}",
        )

    # Run analysis for each ticker against divination so selected model/task are respected.
    results = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for ticker in data.tickers:
            try:
                response = await client.get(
                    f"{DIVINATION_API_URL}/predict/{ticker.upper()}",
                    params={
                        "horizon": canonical_horizon,
                        "model": data.model,
                        "task": data.task,
                        "period": data.period,
                    },
                )
                response.raise_for_status()
                payload = response.json()

                results.append(AnalyzeResultItem(
                    ticker=ticker.upper(),
                    last_close=_sanitize_last_close(payload.get("last_close"), default=0.0),
                    prob_up=_sanitize_probability(
                        payload.get("prob_up", payload.get("probability")),
                        default=0.5,
                    ),
                    signal=payload.get("signal"),
                    predicted_return=payload.get("predicted_return"),
                    error=None,
                ))
            except Exception as e:
                error_detail = _analysis_error_detail(e)
                logger.warning(f"Analysis prediction failed for {ticker.upper()}: {error_detail}")
                results.append(AnalyzeResultItem(
                    ticker=ticker.upper(),
                    last_close=None,
                    prob_up=None,
                    signal=None,
                    predicted_return=None,
                    error=error_detail,
                ))

    success_count = sum(1 for row in results if row.error is None)
    failed_rows = [row for row in results if row.error]

    if success_count == 0:
        preview = "; ".join(
            f"{row.ticker}: {row.error}" for row in failed_rows[:3]
        )
        raise HTTPException(
            503,
            detail=f"Analysis failed for all selected tickers. {preview}",
        )

    # Increment rate limit only when at least one ticker succeeds.
    _increment_rate_limit(user)

    return AnalyzeResponse(
        results=results,
        metadata={
            "model": data.model,
            "task": data.task,
            "period": data.period,
            "horizon": canonical_horizon,
            "analyzed_at": datetime.utcnow().isoformat(),
            "requested": len(data.tickers),
            "successful": success_count,
            "failed": len(failed_rows),
        },
    )


# ============================================================
# Company Endpoints
# ============================================================

@app.get("/api/companies/{ticker}", response_model=CompanyDetailResponse)
async def get_company_detail(ticker: str):
    """Get company detail including info and recent news."""
    normalized = _normalize_ticker(ticker)
    if not normalized:
        raise HTTPException(400, detail="Invalid ticker")

    company, info = await _ensure_company_profile(normalized)
    if not company:
        if normalized not in UNIVERSE_UPPER:
            raise HTTPException(404, detail=f"Company '{normalized}' not found")
        company = {
            "ticker": normalized,
            "name": normalized,
            "asset_type": "stock",
            "created_at": None,
            "updated_at": None,
        }

    news_rows = get_company_news(normalized, limit=20)

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
    normalized = _normalize_ticker(ticker)
    if not normalized:
        raise HTTPException(400, detail="Invalid ticker")

    company, _ = await _ensure_company_profile(normalized)
    if not company and normalized not in UNIVERSE_UPPER:
        raise HTTPException(404, detail=f"Company '{normalized}' not found")

    news_rows = get_company_news(normalized, limit=limit)
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
