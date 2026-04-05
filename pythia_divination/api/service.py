"""Pythia Signals API - Multi-model ML Backend."""

import asyncio
import contextlib
import json
import logging
import os
import time
import yaml
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from threading import Lock
import pandas as pd

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Body, Depends, Form
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

from typing import Dict, Tuple, List, Optional, Any

from data.fetch import fetch_ohlcv, fetch_panel, fetch_ohlcv_for_lstm
from features.technical import make_features, get_feature_columns
from features.sequences import create_inference_sequence
from models.utils import load_artifacts, model_exists
from models.registry import MODEL_REGISTRY, get_model, get_artifacts_path, get_trained_models
from models.base import ModelConfig, ModelTask
from api.schemas import (
    IngestRequest,
    PredictResponse,
    AllocateRequest,
    AllocateResponse,
    TradePlan,
    SimRequest,
    SimResponse,
    ModelInfo,
    ModelPrediction,
    MultiModelPredictRequest,
    MultiModelPredictResponse,
    LoginResponse,
    UserResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeResultItem,
    UserFeaturesResponse,
    UniverseResponse,
    ModelsAvailableResponse,
)
from api.auth import (
    get_current_user,
    get_optional_user,
    authenticate_user,
    create_access_token,
    check_rate_limit,
    increment_user_rate_limit,
    hash_password,
    User,
    TEST_ACCOUNTS,
    JWT_EXPIRATION_HOURS,
)
from api.tiers import get_tier_config, TIER_CONFIGS, UserTier
from storage.postgres import create_user
from api.analytics import router as analytics_router
from backtest.simulate import equal_weight_topk

from broker.alpaca import (
    get_positions as alp_positions,
    get_open_orders as alp_orders,
    account as alp_account,
)

from strategy.metrics import snapshot_positions, compute_and_store_metrics
from strategy.runner import run_once

from storage.postgres import init_db, list_metrics
from storage.market_data_store import (
    ensure_tables as ensure_market_data_tables,
    get_lstm_prediction as get_lstm_prediction_db,
    get_ohlcv_dataframe as get_ohlcv_dataframe_db,
    is_enabled as market_data_store_enabled,
    list_lstm_predictions as list_lstm_predictions_db,
    select_stale_tickers as select_stale_tickers_db,
    upsert_lstm_prediction as upsert_lstm_prediction_db,
    upsert_ohlcv_dataframe as upsert_ohlcv_dataframe_db,
)
from config.settings import settings

# Default model for backward compatibility
DEFAULT_MODEL = "gradient_boosting"
DEFAULT_TASK = "classifier"

# Model cache: {(model_type, task): model_instance}
LOADED_MODELS: Dict[Tuple[str, str], any] = {}

# Legacy global model (for backward compatibility)
try:
    MODEL, SCALER, FEATS, _ = load_artifacts()
except (FileNotFoundError, ValueError):
    MODEL = SCALER = FEATS = None
UNIVERSE = settings.universe
DATA_SOURCE = settings.data_source
CFG_LOOKBACK = settings.lookback_download
THRESHOLD = settings.threshold
BROKER_HEALTHCHECK_ENABLED = os.getenv("ALPACA_BROKER_HEALTHCHECK_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

EXTRA_FEATS: Dict[Tuple[str, str], Dict[str, float]] = {}

# Lightweight in-memory caches to reduce repeated market-data provider calls.
# Primary target: homepage Magnificent 7 LSTM endpoints.
HOME_CACHE_TICKERS = {
    t.strip().upper()
    for t in os.getenv("HOME_CACHE_TICKERS", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META").split(",")
    if t.strip()
}
CACHE_DEFAULT_MODELS = os.getenv("HOME_CACHE_MODELS", "lstm_5d,lstm_jackpot")
HOME_CACHE_MODELS = tuple(m.strip() for m in CACHE_DEFAULT_MODELS.split(",") if m.strip())
CACHE_ALL_LSTM = os.getenv("CACHE_ALL_LSTM_PREDICTIONS", "false").lower() == "true"
LSTM_RESPONSE_CACHE_TTL = int(os.getenv("LSTM_RESPONSE_CACHE_TTL_SECONDS", "900"))
LSTM_STALE_CACHE_TTL = int(os.getenv("LSTM_STALE_CACHE_TTL_SECONDS", "3600"))
LSTM_RAW_DATA_CACHE_TTL = int(os.getenv("LSTM_RAW_DATA_CACHE_TTL_SECONDS", "600"))
LSTM_RAW_DATA_STALE_TTL = int(os.getenv("LSTM_RAW_DATA_STALE_TTL_SECONDS", "86400"))
LSTM_CACHE_MAX_ITEMS = int(os.getenv("LSTM_CACHE_MAX_ITEMS", "512"))
LSTM_SINGLEFLIGHT_WAIT_SECONDS = float(os.getenv("LSTM_SINGLEFLIGHT_WAIT_SECONDS", "3.0"))
HOME_CACHE_WARM_ENABLED = os.getenv("HOME_CACHE_WARM_ENABLED", "true").lower() == "true"
HOME_CACHE_WARM_ON_STARTUP = os.getenv("HOME_CACHE_WARM_ON_STARTUP", "true").lower() == "true"
HOME_CACHE_WARM_INTERVAL_SECONDS = int(os.getenv("HOME_CACHE_WARM_INTERVAL_SECONDS", "900"))
HOME_CACHE_SNAPSHOT_PATH = Path(os.getenv("HOME_CACHE_SNAPSHOT_PATH", "/app/artifacts/home_cache_snapshot.json"))
HOMEPAGE_CACHE_ONLY = os.getenv("HOMEPAGE_CACHE_ONLY", "true").lower() == "true"
LSTM_SERVE_CACHE_ONLY = os.getenv("LSTM_SERVE_CACHE_ONLY", "false").lower() == "true"
LSTM_DB_MARKET_DATA_ENABLED = os.getenv("LSTM_DB_MARKET_DATA_ENABLED", "true").lower() == "true"
LSTM_ALLOW_LIVE_DATA_FETCH = os.getenv("LSTM_ALLOW_LIVE_DATA_FETCH", "true").lower() == "true"
MARKET_DATA_SYNC_ENABLED = os.getenv("MARKET_DATA_SYNC_ENABLED", "false").lower() == "true"
MARKET_DATA_SYNC_INTERVAL_SECONDS = int(os.getenv("MARKET_DATA_SYNC_INTERVAL_SECONDS", "86400"))
MARKET_DATA_SYNC_BATCH_SIZE = int(os.getenv("MARKET_DATA_SYNC_BATCH_SIZE", "250"))
MARKET_DATA_SYNC_MAX_AGE_DAYS = int(os.getenv("MARKET_DATA_SYNC_MAX_AGE_DAYS", "1"))
MARKET_DATA_SYNC_LOOKBACK_DAYS = int(os.getenv("MARKET_DATA_SYNC_LOOKBACK_DAYS", "520"))
MARKET_DATA_SYNC_TICKERS_ENV = os.getenv("MARKET_DATA_SYNC_TICKERS", "")
MARKET_DATA_SYNC_SOURCE = os.getenv("MARKET_DATA_SYNC_SOURCE", DATA_SOURCE).strip().lower() or DATA_SOURCE

_LSTM_RESPONSE_CACHE: Dict[Tuple[str, str, str], Tuple[float, dict]] = {}
_LSTM_RAW_DATA_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
_LSTM_SINGLEFLIGHT_LOCKS: Dict[Tuple[str, str, str], Lock] = {}
_LSTM_CACHE_LOCK = Lock()
_HOME_CACHE_WARMER_TASK: Optional[asyncio.Task] = None
_MARKET_DATA_SYNC_TASK: Optional[asyncio.Task] = None
_MLB_UPCOMING_BOARDS_CACHE: Dict[str, Tuple[float, dict[str, Any]]] = {}
_MLB_UPCOMING_BOARDS_LOCK = Lock()
MLB_UPCOMING_CACHE_TTL_SECONDS = int(os.getenv("MLB_UPCOMING_CACHE_TTL_SECONDS", "900"))
_BASKETBALL_UPCOMING_BOARDS_CACHE: Dict[str, Tuple[float, dict[str, Any]]] = {}
_BASKETBALL_UPCOMING_BOARDS_LOCK = Lock()
BASKETBALL_UPCOMING_CACHE_TTL_SECONDS = int(os.getenv("BASKETBALL_UPCOMING_CACHE_TTL_SECONDS", "900"))

LSTM_MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "lstm_5d": {
        "description": "5-Day Consistency Model (Production LSTM)",
        "horizon": "5 days",
        "target_return": ">2%",
        "high_cutoff": 0.60,
        "medium_cutoff": 0.40,
        "high_signal": "buy",
        "medium_signal": "hold",
        "low_signal": "sell",
        "high_reco": "Strong buy signal ({prob:.1f}% probability of >2% gain in 5 days)",
        "medium_reco": "Neutral signal ({prob:.1f}% probability)",
        "low_reco": "Weak signal ({prob:.1f}% probability, consider avoiding)",
    },
    "lstm_jackpot": {
        "description": "20-Day Jackpot Model (High-Return Hunter)",
        "horizon": "20 days",
        "target_return": ">20%",
        "high_cutoff": 0.55,
        "medium_cutoff": 0.45,
        "high_signal": "strong_buy",
        "medium_signal": "hold",
        "low_signal": "avoid",
        "high_reco": "High-conviction jackpot opportunity ({prob:.1f}% probability of >20% gain in 20 days)",
        "medium_reco": "Moderate signal ({prob:.1f}% probability, monitor for entry)",
        "low_reco": "Low probability ({prob:.1f}%), not a jackpot candidate",
    },
}


def _load_live_mlb_upcoming_payload(force_refresh: bool = False, mlb_date: str | None = None) -> dict[str, Any]:
    global _MLB_UPCOMING_BOARDS_CACHE

    now_ts = time.time()
    cache_key = mlb_date or "__default__"
    with _MLB_UPCOMING_BOARDS_LOCK:
        if (
            not force_refresh
            and cache_key in _MLB_UPCOMING_BOARDS_CACHE
            and now_ts - _MLB_UPCOMING_BOARDS_CACHE[cache_key][0] <= MLB_UPCOMING_CACHE_TTL_SECONDS
        ):
            return dict(_MLB_UPCOMING_BOARDS_CACHE[cache_key][1])

    from scripts.export_mlb_frontend_data import build_live_upcoming_payload

    payload = build_live_upcoming_payload(selected_date=mlb_date)
    boards = payload.get("upcoming", [])

    with _MLB_UPCOMING_BOARDS_LOCK:
        if boards or cache_key not in _MLB_UPCOMING_BOARDS_CACHE:
            _MLB_UPCOMING_BOARDS_CACHE[cache_key] = (now_ts, payload)
        elif _MLB_UPCOMING_BOARDS_CACHE.get(cache_key, ({}, {}))[1].get("upcoming"):
            return dict(_MLB_UPCOMING_BOARDS_CACHE[cache_key][1])
    return dict(payload)


def _load_live_basketball_upcoming_payload(force_refresh: bool = False, basketball_date: str | None = None) -> dict[str, Any]:
    global _BASKETBALL_UPCOMING_BOARDS_CACHE

    now_ts = time.time()
    cache_key = basketball_date or "__default__"
    with _BASKETBALL_UPCOMING_BOARDS_LOCK:
        if (
            not force_refresh
            and cache_key in _BASKETBALL_UPCOMING_BOARDS_CACHE
            and now_ts - _BASKETBALL_UPCOMING_BOARDS_CACHE[cache_key][0] <= BASKETBALL_UPCOMING_CACHE_TTL_SECONDS
        ):
            return dict(_BASKETBALL_UPCOMING_BOARDS_CACHE[cache_key][1])

    from scripts.export_basketball_frontend_data import build_live_upcoming_payload

    payload = build_live_upcoming_payload(selected_date=basketball_date)
    boards = payload.get("upcoming", [])

    with _BASKETBALL_UPCOMING_BOARDS_LOCK:
        if boards or cache_key not in _BASKETBALL_UPCOMING_BOARDS_CACHE:
            _BASKETBALL_UPCOMING_BOARDS_CACHE[cache_key] = (now_ts, payload)
        elif _BASKETBALL_UPCOMING_BOARDS_CACHE.get(cache_key, ({}, {}))[1].get("upcoming"):
            return dict(_BASKETBALL_UPCOMING_BOARDS_CACHE[cache_key][1])
    return dict(payload)


def _should_cache_ticker(ticker: str) -> bool:
    return CACHE_ALL_LSTM or ticker.upper() in HOME_CACHE_TICKERS


def _cache_key(model_name: str, ticker: str) -> Tuple[str, str, str]:
    return (DATA_SOURCE.lower(), model_name, ticker.upper())


def _cache_get_entry(model_name: str, ticker: str) -> Optional[Tuple[float, dict]]:
    key = _cache_key(model_name, ticker)
    with _LSTM_CACHE_LOCK:
        entry = _LSTM_RESPONSE_CACHE.get(key)
        if not entry:
            return None
        ts, payload = entry
        return ts, dict(payload)


def _cache_get(model_name: str, ticker: str, max_age_seconds: int) -> Optional[dict]:
    now = time.time()
    entry = _cache_get_entry(model_name, ticker)
    if entry:
        ts, payload = entry
        if now - ts <= max_age_seconds:
            return payload

    if not market_data_store_enabled():
        return None
    try:
        payload = get_lstm_prediction_db(
            data_source=DATA_SOURCE,
            model_name=model_name,
            ticker=ticker,
            max_age_seconds=max_age_seconds,
        )
    except Exception as exc:
        logger.debug("DB prediction cache read failed for %s/%s: %s", model_name, ticker, exc)
        return None
    if not payload:
        return None
    with _LSTM_CACHE_LOCK:
        _LSTM_RESPONSE_CACHE[_cache_key(model_name, ticker)] = (now, dict(payload))
    return dict(payload)


def _cache_set(model_name: str, ticker: str, payload: dict) -> None:
    key = _cache_key(model_name, ticker)
    now = time.time()
    cached_payload = dict(payload)
    with _LSTM_CACHE_LOCK:
        _LSTM_RESPONSE_CACHE[key] = (now, cached_payload)
        if len(_LSTM_RESPONSE_CACHE) > LSTM_CACHE_MAX_ITEMS:
            oldest = min(_LSTM_RESPONSE_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _LSTM_RESPONSE_CACHE.pop(oldest, None)
    if not market_data_store_enabled():
        return
    try:
        upsert_lstm_prediction_db(
            data_source=DATA_SOURCE,
            model_name=model_name,
            ticker=ticker,
            payload=cached_payload,
        )
    except Exception as exc:
        logger.debug("DB prediction cache write failed for %s/%s: %s", model_name, ticker, exc)


def _singleflight_lock(model_name: str, ticker: str) -> Lock:
    key = _cache_key(model_name, ticker)
    with _LSTM_CACHE_LOCK:
        lock = _LSTM_SINGLEFLIGHT_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _LSTM_SINGLEFLIGHT_LOCKS[key] = lock
        return lock


def _load_home_cache_snapshot() -> None:
    if not HOME_CACHE_SNAPSHOT_PATH.exists():
        return
    try:
        payload = json.loads(HOME_CACHE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        loaded = 0
        with _LSTM_CACHE_LOCK:
            for raw_key, value in payload.items():
                if not isinstance(raw_key, str) or not isinstance(value, dict):
                    continue
                parts = raw_key.split("|", 2)
                if len(parts) != 3:
                    continue
                ts = float(value.get("ts", 0))
                data = value.get("payload")
                if ts <= 0 or not isinstance(data, dict):
                    continue
                _LSTM_RESPONSE_CACHE[(parts[0], parts[1], parts[2])] = (ts, data)
                loaded += 1
        if loaded:
            logger.info("Loaded %d cached LSTM predictions from snapshot", loaded)
    except Exception as exc:
        logger.warning("Failed loading cache snapshot at %s: %s", HOME_CACHE_SNAPSHOT_PATH, exc)


def _save_home_cache_snapshot() -> None:
    try:
        serialized: Dict[str, dict] = {}
        with _LSTM_CACHE_LOCK:
            for key, (ts, payload) in _LSTM_RESPONSE_CACHE.items():
                serialized["|".join(key)] = {"ts": ts, "payload": payload}
        HOME_CACHE_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        HOME_CACHE_SNAPSHOT_PATH.write_text(
            json.dumps(serialized, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed writing cache snapshot at %s: %s", HOME_CACHE_SNAPSHOT_PATH, exc)


def _get_cached_lstm_raw_data(ticker: str) -> pd.DataFrame:
    """Return cached OHLCV raw data for LSTM endpoints to avoid duplicate provider hits."""
    cache_key = (DATA_SOURCE.lower(), ticker.upper())
    now = time.time()
    stale_df: Optional[pd.DataFrame] = None
    with _LSTM_CACHE_LOCK:
        entry = _LSTM_RAW_DATA_CACHE.get(cache_key)
        if entry:
            ts, df = entry
            if now - ts <= LSTM_RAW_DATA_CACHE_TTL:
                return df.copy()
            if now - ts <= LSTM_RAW_DATA_STALE_TTL:
                stale_df = df.copy()

    if LSTM_DB_MARKET_DATA_ENABLED and market_data_store_enabled():
        try:
            db_df = get_ohlcv_dataframe_db(ticker=ticker, max_rows=max(500, 120))
            if len(db_df) >= 120:
                with _LSTM_CACHE_LOCK:
                    _LSTM_RAW_DATA_CACHE[cache_key] = (now, db_df.copy())
                return db_df
            if stale_df is None and len(db_df) > 0:
                stale_df = db_df.copy()
        except Exception as exc:
            logger.debug("DB OHLCV read failed for %s: %s", ticker, exc)

    if not LSTM_ALLOW_LIVE_DATA_FETCH:
        if stale_df is not None and len(stale_df) >= 120:
            logger.warning("Using stale OHLCV cache for %s with live fetch disabled", ticker)
            return stale_df
        raise RuntimeError(f"Live OHLCV fetch disabled and no usable DB/cache data for {ticker}")

    try:
        raw = fetch_ohlcv_for_lstm(ticker, sequence_length=60, data_source=DATA_SOURCE)
    except Exception:
        if stale_df is not None and len(stale_df) >= 120:
            logger.warning(
                "Using stale OHLCV cache for %s after provider fetch failure (stale age <= %ss)",
                ticker,
                LSTM_RAW_DATA_STALE_TTL,
            )
            return stale_df
        raise

    if LSTM_DB_MARKET_DATA_ENABLED and market_data_store_enabled():
        try:
            upsert_ohlcv_dataframe_db(
                ticker=ticker,
                frame=raw.tail(max(500, 120)),
                source=f"live:{DATA_SOURCE}",
            )
        except Exception as exc:
            logger.debug("DB OHLCV write failed for %s: %s", ticker, exc)

    with _LSTM_CACHE_LOCK:
        _LSTM_RAW_DATA_CACHE[cache_key] = (now, raw.copy())
        if len(_LSTM_RAW_DATA_CACHE) > LSTM_CACHE_MAX_ITEMS:
            oldest = min(_LSTM_RAW_DATA_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _LSTM_RAW_DATA_CACHE.pop(oldest, None)
    return raw


def load_model(model_type: str, task: str):
    """Load a model from artifacts (with caching)."""
    key = (model_type, task)

    if key not in LOADED_MODELS:
        task_enum = ModelTask.CLASSIFICATION if task == "classifier" else ModelTask.REGRESSION
        artifacts_path = get_artifacts_path(model_type, task_enum)

        if not artifacts_path.exists():
            raise ValueError(f"No artifacts found for {model_type}/{task}")

        config = ModelConfig(model_type=model_type, task=task_enum, hyperparams={})
        model = get_model(model_type, config)
        model.load(artifacts_path)
        LOADED_MODELS[key] = model

    return LOADED_MODELS[key]


app = FastAPI(title="Pythia Signals API (Multi-Model)")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Include analytics router
app.include_router(analytics_router)


def _configured_home_models() -> Tuple[str, ...]:
    models = []
    for model_name in HOME_CACHE_MODELS:
        if model_name not in LSTM_MODEL_METADATA:
            logger.warning("Ignoring unknown HOME_CACHE model: %s", model_name)
            continue
        models.append(model_name)
    return tuple(models)


def _configured_market_sync_tickers() -> Tuple[str, ...]:
    if MARKET_DATA_SYNC_TICKERS_ENV.strip():
        return tuple(
            t.strip().upper()
            for t in MARKET_DATA_SYNC_TICKERS_ENV.split(",")
            if t.strip()
        )
    # Default to full configured universe; batch size and interval control throughput.
    return tuple(str(t).upper() for t in UNIVERSE)


def _hydrate_prediction_cache_from_db() -> int:
    if not market_data_store_enabled():
        return 0
    try:
        rows = list_lstm_predictions_db(
            data_source=DATA_SOURCE,
            model_names=_configured_home_models(),
            tickers=tuple(HOME_CACHE_TICKERS),
            max_age_seconds=LSTM_STALE_CACHE_TTL,
        )
    except Exception as exc:
        logger.warning("Failed loading LSTM prediction cache from DB: %s", exc)
        return 0

    now = time.time()
    loaded = 0
    with _LSTM_CACHE_LOCK:
        for (model_name, ticker), payload in rows.items():
            _LSTM_RESPONSE_CACHE[_cache_key(model_name, ticker)] = (now, dict(payload))
            loaded += 1
    if loaded:
        logger.info("Hydrated %d LSTM prediction entries from DB cache", loaded)
    return loaded


def _sync_market_data_once(limit: Optional[int] = None) -> Dict[str, Any]:
    tickers = _configured_market_sync_tickers()
    batch_size = max(1, limit or MARKET_DATA_SYNC_BATCH_SIZE)
    candidates = select_stale_tickers_db(
        tickers=tickers,
        max_age_days=MARKET_DATA_SYNC_MAX_AGE_DAYS,
        limit=batch_size,
    )
    summary: Dict[str, Any] = {
        "source": MARKET_DATA_SYNC_SOURCE,
        "total_configured": len(tickers),
        "requested": len(candidates),
        "synced": 0,
        "failed": 0,
        "errors": [],
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    if not candidates:
        return summary

    start = (datetime.now(timezone.utc) - pd.Timedelta(days=MARKET_DATA_SYNC_LOOKBACK_DAYS)).date().isoformat()

    for ticker in candidates:
        try:
            frame = fetch_ohlcv(
                ticker=ticker,
                start=start,
                data_source=MARKET_DATA_SYNC_SOURCE,
                timeframe="1Day",
                retries=1,
            )
            rows = upsert_ohlcv_dataframe_db(
                ticker=ticker,
                frame=frame,
                source=f"sync:{MARKET_DATA_SYNC_SOURCE}",
            )
            summary["synced"] += 1
            if rows == 0:
                summary["failed"] += 1
                if len(summary["errors"]) < 20:
                    summary["errors"].append({"ticker": ticker, "error": "no rows persisted"})
        except Exception as exc:
            summary["failed"] += 1
            if len(summary["errors"]) < 20:
                summary["errors"].append({"ticker": ticker, "error": str(exc)})
    return summary


def _warm_home_cache_once(force_refresh: bool = False) -> Dict[str, Any]:
    tickers = sorted(HOME_CACHE_TICKERS)
    models = _configured_home_models()
    summary: Dict[str, Any] = {
        "tickers": len(tickers),
        "models": len(models),
        "success": 0,
        "failed": 0,
        "errors": [],
        "force_refresh": force_refresh,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }

    for model_name in models:
        for ticker in tickers:
            try:
                _predict_lstm_cached(model_name, ticker, force_refresh=force_refresh)
                summary["success"] += 1
            except Exception as exc:
                summary["failed"] += 1
                if len(summary["errors"]) < 20:
                    summary["errors"].append({"model": model_name, "ticker": ticker, "error": str(exc)})

    _save_home_cache_snapshot()
    return summary


async def _home_cache_warmer_loop() -> None:
    if HOME_CACHE_WARM_ON_STARTUP:
        summary = await run_in_threadpool(_warm_home_cache_once, False)
        logger.info("Initial homepage cache warm complete: %s", summary)

    while True:
        await asyncio.sleep(max(30, HOME_CACHE_WARM_INTERVAL_SECONDS))
        summary = await run_in_threadpool(_warm_home_cache_once, True)
        logger.info("Periodic homepage cache warm complete: %s", summary)


async def _market_data_sync_loop() -> None:
    if not MARKET_DATA_SYNC_ENABLED:
        return

    summary = await run_in_threadpool(_sync_market_data_once)
    logger.info("Initial market-data sync complete: %s", summary)

    while True:
        await asyncio.sleep(max(300, MARKET_DATA_SYNC_INTERVAL_SECONDS))
        summary = await run_in_threadpool(_sync_market_data_once)
        logger.info("Periodic market-data sync complete: %s", summary)


def _start_home_cache_warmer() -> None:
    global _HOME_CACHE_WARMER_TASK
    if not HOME_CACHE_WARM_ENABLED:
        logger.info("Homepage cache warmer disabled via HOME_CACHE_WARM_ENABLED=false")
        return
    if _HOME_CACHE_WARMER_TASK is not None and not _HOME_CACHE_WARMER_TASK.done():
        return
    _HOME_CACHE_WARMER_TASK = asyncio.create_task(_home_cache_warmer_loop())


def _start_market_data_sync() -> None:
    global _MARKET_DATA_SYNC_TASK
    if not MARKET_DATA_SYNC_ENABLED:
        logger.info("Market-data sync disabled via MARKET_DATA_SYNC_ENABLED=false")
        return
    if not market_data_store_enabled():
        logger.warning("Market-data sync requested but DB store is disabled")
        return
    if _MARKET_DATA_SYNC_TASK is not None and not _MARKET_DATA_SYNC_TASK.done():
        return
    _MARKET_DATA_SYNC_TASK = asyncio.create_task(_market_data_sync_loop())


@app.on_event("startup")
async def _startup():
    # Temporarily disable database for LSTM testing
    try:
        await init_db()
        # Seed test accounts
        await seed_test_accounts()
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")

    if market_data_store_enabled():
        try:
            ensure_market_data_tables()
        except Exception as exc:
            logger.warning("Failed to initialize market-data tables: %s", exc)

    _load_home_cache_snapshot()
    _hydrate_prediction_cache_from_db()
    _start_home_cache_warmer()
    _start_market_data_sync()


@app.on_event("shutdown")
async def _shutdown():
    global _HOME_CACHE_WARMER_TASK, _MARKET_DATA_SYNC_TASK
    if _HOME_CACHE_WARMER_TASK is not None:
        _HOME_CACHE_WARMER_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _HOME_CACHE_WARMER_TASK
        _HOME_CACHE_WARMER_TASK = None
    if _MARKET_DATA_SYNC_TASK is not None:
        _MARKET_DATA_SYNC_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _MARKET_DATA_SYNC_TASK
        _MARKET_DATA_SYNC_TASK = None
    _save_home_cache_snapshot()


async def seed_test_accounts():
    """Seed test user accounts on startup."""
    for account in TEST_ACCOUNTS:
        try:
            password_hash = hash_password(account["password"])
            await create_user(account["username"], password_hash, account["tier"])
        except Exception:
            pass  # Account may already exist


def latest_ohlcv(ticker: str):
    return fetch_ohlcv(
        ticker,
        period=CFG_LOOKBACK,
        data_source=DATA_SOURCE,
    )


def _normalize_horizon(h: str) -> str:
    """
    Normalize horizon tokens to a canonical timeframe string.
    """
    if not h:
        return "1Day"
    h = h.strip().lower()
    table = {
        "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
        "1h": "1Hour", "60m": "1Hour", "4h": "4Hour", "240m": "4Hour",
        "1d": "1Day", "1day": "1Day", "2d": "2Day", "2day": "2Day",
        "3d": "3Day", "3day": "3Day",
        "1w": "1Week", "1wk": "1Week", "1week": "1Week",
        "1mo": "1Month", "1month": "1Month",
    }
    return table.get(h, "1Day")


def _fallback_prediction_from_features(feat: pd.DataFrame, last_close: float, task: str, model_type: str):
    """Generate lightweight fallback predictions from recent features."""
    last = feat.iloc[-1]
    base = float(last.get("ret_1d", 0.0)) * 4.0 + float(last.get("ret_5d", 0.0)) * 1.5
    # Deterministic per-model offset so rows are not identical in the UI.
    offset = ((sum(ord(c) for c in model_type) % 7) - 3) * 0.01
    prob_up = max(0.05, min(0.95, 0.5 + base + offset))

    if task == "classifier":
        signal = "buy" if prob_up >= THRESHOLD else ("sell" if prob_up <= 1 - THRESHOLD else "hold")
        return prob_up, signal, last_close, None

    predicted_return = float(last.get("ret_1d", 0.0))
    return None, None, last_close, predicted_return


def _lstm_signal_and_recommendation(model_name: str, prob_up: float) -> Tuple[str, str]:
    meta = LSTM_MODEL_METADATA[model_name]
    prob_pct = prob_up * 100.0
    if prob_up >= float(meta["high_cutoff"]):
        signal = str(meta["high_signal"])
        recommendation = str(meta["high_reco"]).format(prob=prob_pct)
    elif prob_up >= float(meta["medium_cutoff"]):
        signal = str(meta["medium_signal"])
        recommendation = str(meta["medium_reco"]).format(prob=prob_pct)
    else:
        signal = str(meta["low_signal"])
        recommendation = str(meta["low_reco"]).format(prob=prob_pct)
    return signal, recommendation


def _prepare_lstm_inference(model_name: str, ticker: str) -> Dict[str, Any]:
    if model_name not in LSTM_MODEL_METADATA:
        raise HTTPException(404, detail=f"Unknown LSTM model: {model_name}")

    ticker = ticker.upper()
    meta = LSTM_MODEL_METADATA[model_name]

    raw = _get_cached_lstm_raw_data(ticker)
    model = load_model(model_name, "classifier")

    try:
        feat = model.compute_features(raw, ticker=ticker)
    except TypeError:
        feat = model.compute_features(raw)
    feat_cols = list(feat.columns)

    if len(feat) < 60:
        raise HTTPException(400, f"Not enough data after feature engineering (need 60 samples, got {len(feat)})")

    X = feat[feat_cols].values
    scaler = model.get_scaler()
    X_scaled = scaler.transform(X) if scaler else X
    X_seq = create_inference_sequence(X_scaled, 60)

    result = model.predict(X_seq)
    prob_up = float(result.prob_up)
    last_close = float(raw["Close"].iloc[-1])
    signal, recommendation = _lstm_signal_and_recommendation(model_name, prob_up)

    return {
        "ticker": ticker,
        "model_name": model_name,
        "meta": meta,
        "raw": raw,
        "model": model,
        "feat": feat,
        "feature_cols": feat_cols,
        "X_seq": X_seq,
        "prob_up": prob_up,
        "last_close": last_close,
        "signal": signal,
        "recommendation": recommendation,
        "sentiment": feat.attrs.get("sentiment", {}),
    }


def _compute_integrated_gradients_tf(tf_model: Any, x_seq: Any, steps: int = 24) -> Any:
    import tensorflow as tf

    steps = max(8, min(int(steps), 256))
    x = tf.convert_to_tensor(x_seq, dtype=tf.float32)
    baseline = tf.zeros_like(x)
    alphas = tf.linspace(0.0, 1.0, steps + 1)

    grad_acc = tf.zeros_like(x)
    for alpha in alphas:
        x_step = baseline + alpha * (x - baseline)
        with tf.GradientTape() as tape:
            tape.watch(x_step)
            preds = tf_model(x_step, training=False)
            target = tf.reshape(preds, (-1,))[0]
        grads = tape.gradient(target, x_step)
        if grads is None:
            grads = tf.zeros_like(x_step)
        grad_acc += grads

    avg_grads = grad_acc / tf.cast(steps + 1, tf.float32)
    integrated_grads = (x - baseline) * avg_grads
    return integrated_grads.numpy()[0]


def _build_lstm_attribution(
    prepared: Dict[str, Any],
    method: str = "integrated_gradients",
    steps: int = 24,
    top_k: int = 5,
) -> Dict[str, Any]:
    import numpy as np

    method_norm = method.strip().lower()
    if method_norm in {"timeshap", "time_shap"}:
        raise HTTPException(
            501,
            detail="TimeSHAP is not enabled yet in this runtime. Use method=integrated_gradients.",
        )
    if method_norm not in {"integrated_gradients", "ig"}:
        raise HTTPException(400, detail=f"Unsupported attribution method: {method}")

    model = prepared["model"]
    tf_model = getattr(model, "tf_model", None)
    if tf_model is None:
        raise HTTPException(
            501,
            detail="Integrated gradients currently support TensorFlow .keras LSTM artifacts only.",
        )

    ig = _compute_integrated_gradients_tf(tf_model, prepared["X_seq"], steps=steps)
    feature_cols = list(prepared["feature_cols"])
    signed = ig.sum(axis=0)
    magnitude = np.abs(ig).sum(axis=0)

    k = max(1, min(int(top_k), len(feature_cols)))
    ranked_idx = np.argsort(magnitude)[::-1][:k]

    top_drivers = []
    for idx in ranked_idx:
        top_drivers.append({
            "feature": feature_cols[int(idx)],
            "signed_contribution": float(signed[int(idx)]),
            "magnitude": float(magnitude[int(idx)]),
            "direction": "positive" if signed[int(idx)] >= 0 else "negative",
        })

    positive_sorted = [d for d in top_drivers if d["signed_contribution"] >= 0]
    negative_sorted = [d for d in top_drivers if d["signed_contribution"] < 0]
    positive_sorted.sort(key=lambda d: d["signed_contribution"], reverse=True)
    negative_sorted.sort(key=lambda d: d["signed_contribution"])

    summary = []
    if top_drivers:
        strongest = top_drivers[0]
        summary.append(
            f"Strongest local driver: {strongest['feature']} ({strongest['direction']}, magnitude {strongest['magnitude']:.6f})."
        )
    if positive_sorted:
        summary.append(
            "Top positive contributors: " + ", ".join(d["feature"] for d in positive_sorted[:3]) + "."
        )
    if negative_sorted:
        summary.append(
            "Top negative contributors: " + ", ".join(d["feature"] for d in negative_sorted[:3]) + "."
        )

    return {
        "method": "integrated_gradients",
        "baseline": "zeros_scaled_space",
        "steps": max(8, min(int(steps), 256)),
        "sequence_length": int(ig.shape[0]),
        "feature_count": int(ig.shape[1]),
        "top_k": k,
        "top_drivers": top_drivers,
        "top_positive_drivers": positive_sorted[:k],
        "top_negative_drivers": negative_sorted[:k],
        "summary": summary,
    }


def _compute_lstm_attribution(
    model_name: str,
    ticker: str,
    method: str = "integrated_gradients",
    steps: int = 24,
    top_k: int = 5,
) -> Dict[str, Any]:
    prepared = _prepare_lstm_inference(model_name, ticker)
    attribution = _build_lstm_attribution(prepared, method=method, steps=steps, top_k=top_k)
    return {
        "ticker": prepared["ticker"],
        "model": model_name,
        "description": str(prepared["meta"]["description"]),
        "horizon": str(prepared["meta"]["horizon"]),
        "target_return": str(prepared["meta"]["target_return"]),
        "probability": round(prepared["prob_up"] * 100, 2),
        "signal": prepared["signal"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": DATA_SOURCE,
        "attribution": attribution,
    }


def _compute_lstm_prediction(
    model_name: str,
    ticker: str,
    with_attribution: bool = False,
    attribution_method: str = "integrated_gradients",
    attribution_steps: int = 24,
    attribution_top_k: int = 5,
) -> dict:
    prepared = _prepare_lstm_inference(model_name, ticker)
    meta = prepared["meta"]
    sentiment = prepared["sentiment"]
    response = {
        "ticker": prepared["ticker"],
        "model": model_name,
        "description": str(meta["description"]),
        "horizon": str(meta["horizon"]),
        "target_return": str(meta["target_return"]),
        "probability": round(prepared["prob_up"] * 100, 2),
        "signal": prepared["signal"],
        "last_close": prepared["last_close"],
        "recommendation": prepared["recommendation"],
        "sentiment_score": round(float(sentiment.get("score", 0.0)), 4),
        "sentiment_articles": int(sentiment.get("num_articles", 0)),
        "sentiment_source": sentiment.get("source", "default"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": DATA_SOURCE,
    }
    if with_attribution:
        response["attribution"] = _build_lstm_attribution(
            prepared,
            method=attribution_method,
            steps=attribution_steps,
            top_k=attribution_top_k,
        )
    return response


def _predict_lstm_cached(model_name: str, ticker: str, force_refresh: bool = False) -> dict:
    ticker = ticker.upper()
    use_cache = _should_cache_ticker(ticker) or LSTM_SERVE_CACHE_ONLY

    if use_cache and not force_refresh:
        cached = _cache_get(model_name, ticker, LSTM_RESPONSE_CACHE_TTL)
        if cached:
            cached["cache_status"] = "hit"
            return cached
        if LSTM_SERVE_CACHE_ONLY:
            stale = _cache_get(model_name, ticker, LSTM_STALE_CACHE_TTL)
            if stale:
                stale["cache_status"] = "stale_cache_only"
                stale["warning"] = "Serving stale cached prediction while live refresh is disabled"
                return stale
            raise HTTPException(503, detail=f"Cached prediction unavailable for {model_name}/{ticker}")

    lock = _singleflight_lock(model_name, ticker)
    acquired = lock.acquire(timeout=LSTM_SINGLEFLIGHT_WAIT_SECONDS)
    if not acquired:
        if use_cache:
            stale = _cache_get(model_name, ticker, LSTM_STALE_CACHE_TTL)
            if stale:
                stale["cache_status"] = "stale_inflight"
                stale["warning"] = "Refresh in progress; returned cached prediction"
                return stale
        raise HTTPException(503, detail="Prediction refresh in progress. Please retry shortly.")

    try:
        if use_cache and not force_refresh:
            cached = _cache_get(model_name, ticker, LSTM_RESPONSE_CACHE_TTL)
            if cached:
                cached["cache_status"] = "hit"
                return cached

        response = _compute_lstm_prediction(model_name, ticker)
        response["cache_status"] = "miss" if not force_refresh else "refreshed"
        if use_cache:
            _cache_set(model_name, ticker, response)
        return response
    except HTTPException as exc:
        if use_cache:
            stale = _cache_get(model_name, ticker, LSTM_STALE_CACHE_TTL)
            if stale:
                stale["cache_status"] = "stale"
                stale["warning"] = "Returning stale cached result due live data fetch failure"
                stale["last_error"] = str(exc.detail)
                return stale
        raise
    except Exception as exc:
        if use_cache:
            stale = _cache_get(model_name, ticker, LSTM_STALE_CACHE_TTL)
            if stale:
                stale["cache_status"] = "stale"
                stale["warning"] = "Returning stale cached result due live data fetch failure"
                stale["last_error"] = str(exc)
                return stale
        raise HTTPException(400, detail=str(exc))
    finally:
        lock.release()


def _lookback_days_from_period(period: Optional[str]) -> Optional[int]:
    if not period:
        return None

    normalized = str(period).strip().upper()
    if not normalized:
        return None

    explicit = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "5Y": 1825,
    }
    if normalized in explicit:
        return explicit[normalized]

    digits = "".join(ch for ch in normalized if ch.isdigit())
    if not digits:
        return None

    value = int(digits)
    if normalized.endswith("Y"):
        return value * 365
    if normalized.endswith("M"):
        return value * 30
    if normalized.endswith("W"):
        return value * 7
    return value


def _prediction_lookback_period(requested_period: Optional[str]) -> str:
    configured_days = _lookback_days_from_period(CFG_LOOKBACK) or 120
    requested_days = _lookback_days_from_period(requested_period) or configured_days
    lookback_days = max(configured_days, requested_days)
    return f"{lookback_days}d"


def predict_for_ticker(
    ticker: str,
    horizon: str = "1d",
    model_type: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK,
    period: Optional[str] = None,
):
    """
    Generate prediction for a ticker using specified model.

    Args:
        ticker: Stock ticker symbol
        horizon: Prediction horizon
        model_type: Model to use
        task: "classifier" or "regressor"

    Returns:
        Tuple of (prob_up/predicted_return, signal, last_close)
    """
    tf = _normalize_horizon(horizon)
    if DATA_SOURCE.lower() == "stooq" and tf != "1Day":
        raise HTTPException(400, detail=f"Horizon '{horizon}' not supported for provider 'stooq' (daily only).")

    # Fetch data
    raw = fetch_ohlcv(
        ticker,
        period=_prediction_lookback_period(period),
        data_source=DATA_SOURCE,
        timeframe=tf,
    )

    task_enum = ModelTask.CLASSIFICATION if task == "classifier" else ModelTask.REGRESSION
    include_regression = task_enum == ModelTask.REGRESSION
    feat = make_features(raw, include_regression_target=include_regression)

    # Merge in extra features
    for (dt, tkr), extra in list(EXTRA_FEATS.items()):
        if tkr == ticker and dt in feat.index.strftime("%Y-%m-%d").tolist():
            for k, v in extra.items():
                feat.loc[feat.index.strftime("%Y-%m-%d") == dt, k] = v

    feat_cols = get_feature_columns()
    last_close = float(raw["Close"].iloc[-1])

    # Try to load specified model
    try:
        model = load_model(model_type, task)
    except Exception:
        # Fall back to legacy model for backward compatibility
        if model_type == DEFAULT_MODEL and task == DEFAULT_TASK:
            if MODEL is not None and SCALER is not None and FEATS is not None:
                X = feat[FEATS].values
                Xs = SCALER.transform(X)
                prob_up = float(MODEL.predict_proba(Xs)[:, 1][-1])
                signal = "buy" if prob_up >= THRESHOLD else ("sell" if prob_up <= 1 - THRESHOLD else "hold")
                return prob_up, signal, last_close, None

        logger.warning("Model load failed for %s/%s; using feature-based fallback", model_type, task)
        return _fallback_prediction_from_features(feat, last_close, task, model_type)

    # Prepare features for the model
    # If the model provides its own feature computation (LSTM wrappers), use it.
    if hasattr(model, "compute_features"):
        try:
            try:
                feat = model.compute_features(raw, ticker=ticker.upper())
            except TypeError:
                feat = model.compute_features(raw)
            feat_cols = list(feat.columns)
        except Exception as exc:
            raise HTTPException(400, detail=f"Model feature computation failed: {exc}")

    X = feat[feat_cols].values
    scaler = None
    if hasattr(model, "get_scaler"):
        scaler = model.get_scaler()

    if model.requires_sequences:
        seq_len = model.config.sequence_length
        if len(X) < seq_len:
            raise HTTPException(400, f"Not enough data for LSTM (need {seq_len} samples, got {len(X)})")
        X_scaled = scaler.transform(X) if scaler else X
        X_seq = create_inference_sequence(X_scaled, seq_len)
        result = model.predict(X_seq)
    else:
        X_scaled = scaler.transform(X) if scaler else X
        result = model.predict(X_scaled)

    if task == "classifier":
        prob_up = result.prob_up
        signal = result.signal
        predicted_return = None
    else:
        prob_up = None
        signal = None
        predicted_return = result.predicted_return

    return prob_up, signal, last_close, predicted_return


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/", status_code=307)


@app.get("/models")
def list_models():
    """List all available models and their training status."""
    return get_trained_models()


@app.get("/predict/homepage")
def predict_homepage(
    force_refresh: bool = False,
    tickers: Optional[str] = None,
    models: Optional[str] = None,
):
    """Batch endpoint for homepage cards to avoid one-request-per-card traffic spikes."""
    requested_tickers = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers
        else sorted(HOME_CACHE_TICKERS)
    )
    requested_models = (
        [m.strip() for m in models.split(",") if m.strip()]
        if models
        else list(_configured_home_models())
    )

    invalid_models = [m for m in requested_models if m not in LSTM_MODEL_METADATA]
    if invalid_models:
        raise HTTPException(400, detail=f"Unsupported model(s): {', '.join(invalid_models)}")

    cache_only_mode = HOMEPAGE_CACHE_ONLY and not force_refresh
    rows = []
    for model_name in requested_models:
        model_predictions = []
        failures = []
        for ticker in requested_tickers:
            if cache_only_mode:
                cached = _cache_get(model_name, ticker, LSTM_RESPONSE_CACHE_TTL)
                if cached:
                    cached["cache_status"] = "hit"
                    model_predictions.append(cached)
                    continue
                stale = _cache_get(model_name, ticker, LSTM_STALE_CACHE_TTL)
                if stale:
                    stale["cache_status"] = "stale_cache_only"
                    stale["warning"] = "Returned stale cache entry (homepage cache-only mode)"
                    model_predictions.append(stale)
                    continue
                failures.append({"ticker": ticker, "error": "cache-miss"})
                continue

            try:
                model_predictions.append(
                    _predict_lstm_cached(model_name, ticker, force_refresh=force_refresh)
                )
            except Exception as exc:
                failures.append({"ticker": ticker, "error": str(exc)})

        rows.append({
            "model": model_name,
            "predictions": model_predictions,
            "requested_tickers": len(requested_tickers),
            "successful_predictions": len(model_predictions),
            "failures": failures,
        })

    return {
        "available": any(row["successful_predictions"] > 0 for row in rows),
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data_source": DATA_SOURCE,
        "cache_only": cache_only_mode,
        "rows": rows,
    }


@app.get("/predict/{ticker}", response_model=PredictResponse)
def predict(
    ticker: str,
    horizon: str = "1d",
    model: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK,
    period: Optional[str] = None,
):
    """
    Get prediction from a specific model.

    Args:
        ticker: Stock ticker symbol
        horizon: Prediction horizon (1d, 4h, 1w, etc.)
        model: Model to use (gradient_boosting, linear_regression, random_forest, lstm)
        task: Task type (classifier, regressor)
    """
    try:
        prob_up, signal, last_close, predicted_return = predict_for_ticker(
            ticker.upper(),
            horizon=horizon,
            model_type=model,
            task=task,
            period=period,
        )
        return PredictResponse(
            ticker=ticker.upper(),
            horizon=horizon,
            prob_up=prob_up,
            signal=signal,
            last_close=last_close,
            predicted_return=predicted_return,
            model=model,
            task=task
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, detail=str(e))


@app.get("/predict/lstm_5d/{ticker}")
def predict_lstm_5d(
    ticker: str,
    with_attribution: bool = False,
    attribution_method: str = "integrated_gradients",
    attribution_steps: int = 24,
    attribution_top_k: int = 5,
):
    base = _predict_lstm_cached("lstm_5d", ticker)
    if not with_attribution:
        return base
    response = dict(base)
    response["attribution"] = _compute_lstm_attribution(
        "lstm_5d",
        ticker,
        method=attribution_method,
        steps=attribution_steps,
        top_k=attribution_top_k,
    )["attribution"]
    return response


@app.get("/predict/lstm_jackpot/{ticker}")
def predict_lstm_jackpot(
    ticker: str,
    with_attribution: bool = False,
    attribution_method: str = "integrated_gradients",
    attribution_steps: int = 24,
    attribution_top_k: int = 5,
):
    base = _predict_lstm_cached("lstm_jackpot", ticker)
    if not with_attribution:
        return base
    response = dict(base)
    response["attribution"] = _compute_lstm_attribution(
        "lstm_jackpot",
        ticker,
        method=attribution_method,
        steps=attribution_steps,
        top_k=attribution_top_k,
    )["attribution"]
    return response


@app.get("/predict/lstm/{model_name}/{ticker}/attribution")
def predict_lstm_attribution(
    model_name: str,
    ticker: str,
    method: str = "integrated_gradients",
    steps: int = 24,
    top_k: int = 5,
):
    return _compute_lstm_attribution(
        model_name=model_name,
        ticker=ticker,
        method=method,
        steps=steps,
        top_k=top_k,
    )


@app.get("/predict/cache/status")
def prediction_cache_status():
    """Operational visibility for cached homepage prediction entries."""
    now = time.time()
    entries = []
    with _LSTM_CACHE_LOCK:
        snapshot = list(_LSTM_RESPONSE_CACHE.items())
    for key, (ts, payload) in snapshot:
        source, model_name, ticker = key
        entries.append({
            "source": source,
            "model": model_name,
            "ticker": ticker,
            "age_seconds": round(now - ts, 2),
            "signal": payload.get("signal"),
            "probability": payload.get("probability"),
            "generated_at": payload.get("generated_at"),
        })

    entries.sort(key=lambda row: (row["model"], row["ticker"]))
    return {
        "count": len(entries),
        "fresh_ttl_seconds": LSTM_RESPONSE_CACHE_TTL,
        "stale_ttl_seconds": LSTM_STALE_CACHE_TTL,
        "warm_interval_seconds": HOME_CACHE_WARM_INTERVAL_SECONDS,
        "homepage_cache_only": HOMEPAGE_CACHE_ONLY,
        "lstm_cache_only": LSTM_SERVE_CACHE_ONLY,
        "db_market_data_enabled": LSTM_DB_MARKET_DATA_ENABLED and market_data_store_enabled(),
        "live_fetch_enabled": LSTM_ALLOW_LIVE_DATA_FETCH,
        "entries": entries,
    }


@app.post("/predict/cache/warm")
async def warm_prediction_cache(force_refresh: bool = True):
    """Manual operational endpoint for warming homepage prediction cache."""
    summary = await run_in_threadpool(_warm_home_cache_once, force_refresh)
    return summary


@app.get("/ops/market-data/status")
def market_data_status():
    tickers = _configured_market_sync_tickers()
    return {
        "enabled": MARKET_DATA_SYNC_ENABLED,
        "source": MARKET_DATA_SYNC_SOURCE,
        "interval_seconds": MARKET_DATA_SYNC_INTERVAL_SECONDS,
        "batch_size": MARKET_DATA_SYNC_BATCH_SIZE,
        "max_age_days": MARKET_DATA_SYNC_MAX_AGE_DAYS,
        "lookback_days": MARKET_DATA_SYNC_LOOKBACK_DAYS,
        "configured_tickers": len(tickers),
        "db_enabled": market_data_store_enabled(),
    }


@app.post("/ops/market-data/sync")
async def sync_market_data(limit: Optional[int] = None):
    """Manual market-data sync operation (DB upsert of daily OHLCV)."""
    if not market_data_store_enabled():
        raise HTTPException(503, detail="Market-data DB store is disabled (DATABASE_URL missing)")
    summary = await run_in_threadpool(_sync_market_data_once, limit)
    return summary


@app.post("/predict/multi", response_model=MultiModelPredictResponse)
def predict_multi(req: MultiModelPredictRequest):
    """
    Get predictions from multiple models.

    Returns predictions from all specified models along with a consensus signal.
    """
    ticker = req.ticker.upper()
    models = req.models or list(MODEL_REGISTRY.keys())
    tasks = req.tasks or ["classifier"]

    predictions = []
    last_close = None

    for model_type in models:
        for task in tasks:
            try:
                prob_up, signal, lc, predicted_return = predict_for_ticker(
                    ticker,
                    horizon=req.horizon,
                    model_type=model_type,
                    task=task
                )
                if last_close is None:
                    last_close = lc

                predictions.append(ModelPrediction(
                    model=model_type,
                    task=task,
                    prob_up=prob_up,
                    signal=signal,
                    predicted_return=predicted_return,
                ))
            except Exception:
                # Skip models that fail
                continue

    # Compute consensus signal (majority vote for classifiers)
    consensus_signal = None
    if predictions:
        signals = [p.signal for p in predictions if p.signal]
        if signals:
            consensus_signal = Counter(signals).most_common(1)[0][0]

    return MultiModelPredictResponse(
        ticker=ticker,
        last_close=last_close or 0.0,
        predictions=predictions,
        consensus_signal=consensus_signal
    )


@app.post("/data/ingest")
def ingest(req: IngestRequest):
    for row in req.rows:
        EXTRA_FEATS[(row.date, row.ticker.upper())] = row.features
    return {"status": "ok", "count": len(req.rows)}


@app.post("/portfolio/allocate", response_model=AllocateResponse)
def allocate(req: AllocateRequest):
    tickers = [t.upper() for t in (req.tickers or UNIVERSE)]
    preds = []
    for t in tickers:
        p, _, px, _ = predict_for_ticker(t)
        preds.append((t, p, px))
    preds.sort(key=lambda x: x[1], reverse=True)
    chosen = [x for x in preds if x[1] >= req.threshold][:req.k_top]

    plan = []
    used = 0.0
    if chosen:
        each = req.budget / len(chosen)
        for t, p, px in chosen:
            shares = int(each // px)
            if shares > 0:
                plan.append(TradePlan(ticker=t, shares=shares, reference_price=px))
                used += shares * px
    return AllocateResponse(budget=req.budget, used=round(used, 2), plan=plan)


@app.post("/simulate/dry-run", response_model=SimResponse)
def simulate(req: SimRequest):
    """
    Robust dry-run backtesting simulation.
    """
    from backtest.simulate import equal_weight_topk

    tickers = UNIVERSE
    ohlcvs = {}
    for t in tickers:
        try:
            df = fetch_ohlcv(t, start=req.start, data_source=DATA_SOURCE)
            need = ["Open", "High", "Low", "Close", "Volume"]
            if not set(need).issubset(df.columns):
                raise ValueError(f"{t}: missing columns, got {list(df.columns)}")
            ohlcvs[t] = df.sort_index()
        except Exception:
            continue

    if not ohlcvs:
        raise HTTPException(400, detail="No OHLCV data available for any tickers.")

    closes = pd.concat({t: df["Close"] for t, df in ohlcvs.items()}, axis=1)
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index().dropna(how="all")

    if closes.empty:
        raise HTTPException(400, detail="Aligned close-price frame is empty.")

    prob_wide = pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)

    for t, df in ohlcvs.items():
        try:
            feat = make_features(df)
            feat = feat.loc[feat.index.intersection(closes.index)]
            if feat.empty:
                continue
            Xs = SCALER.transform(feat[FEATS].values)
            prob = MODEL.predict_proba(Xs)[:, 1]
            prob_wide.loc[feat.index, t] = prob
        except Exception:
            continue

    prob_wide = prob_wide.dropna(how="all")
    if prob_wide.empty:
        raise HTTPException(400, detail="No predictions could be generated.")

    aligned_closes = closes.loc[prob_wide.index]
    stats = equal_weight_topk(prob_wide, aligned_closes, req.budget, req.k_top, req.threshold)

    return SimResponse(**stats)


@app.post("/paper/run")
async def paper_run(
    universe: List[str] = Body(default=None),
    budget: float = Body(..., gt=0),
    threshold: float = Body(default=0.55),
):
    u = universe or UNIVERSE
    res = await run_in_threadpool(run_once, u, threshold, budget, DATA_SOURCE, True)
    return res


@app.get("/paper/positions")
async def paper_positions():
    try:
        return await run_in_threadpool(alp_positions)
    except Exception as e:
        raise HTTPException(400, detail=str(e))


@app.get("/paper/orders")
async def paper_orders():
    try:
        return await run_in_threadpool(alp_orders)
    except Exception as e:
        raise HTTPException(400, detail=str(e))


@app.get("/paper/metrics")
async def paper_metrics(limit: int = 200):
    try:
        items = await list_metrics(limit=limit)
        return {"items": items}
    except Exception as e:
        raise HTTPException(400, detail=str(e))


@app.post("/paper/snapshot")
async def paper_snapshot(notes: Optional[str] = None):
    """Save portfolio snapshot and compute metrics."""
    try:
        acct = await run_in_threadpool(alp_account)
        poss = await run_in_threadpool(alp_positions)
        ts = await snapshot_positions(poss)
        metrics = await compute_and_store_metrics(acct, poss, DATA_SOURCE, notes=notes)
        metrics["snapshot_ts"] = ts
        return metrics
    except Exception as e:
        raise HTTPException(400, detail=str(e))


@app.get("/healthz")
async def healthz():
    db_ok = True
    broker_ok = True
    broker_msg = "ok"
    try:
        await init_db()
    except Exception as e:
        db_ok = False
        broker_msg = f"db_error: {e}"

    # Broker checks are optional and only needed when paper/live broker routes are in active use.
    if settings.data_source.lower() == "alpaca" and BROKER_HEALTHCHECK_ENABLED:
        try:
            await run_in_threadpool(alp_account)
        except Exception as e:
            broker_ok = False
            broker_msg = f"broker_error: {e}"

    return JSONResponse({
        "ok": db_ok and broker_ok,
        "db_ok": db_ok,
        "broker_ok": broker_ok,
        "message": broker_msg if (not db_ok or not broker_ok) else "ok",
        "data_source": settings.data_source,
        "broker_healthcheck_enabled": BROKER_HEALTHCHECK_ENABLED,
        "universe_size": len(settings.universe),
        "available_models": list(MODEL_REGISTRY.keys()),
    })


@app.get("/api/sports/mlb/boards")
async def api_live_mlb_boards(force_refresh: bool = False, mlb_date: str | None = None):
    try:
        return await run_in_threadpool(_load_live_mlb_upcoming_payload, force_refresh, mlb_date)
    except Exception as exc:
        logger.error("Live MLB board generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Live MLB boards are temporarily unavailable")


@app.get("/api/sports/basketball/boards")
async def api_live_basketball_boards(force_refresh: bool = False, basketball_date: str | None = None):
    try:
        return await run_in_threadpool(_load_live_basketball_upcoming_payload, force_refresh, basketball_date)
    except Exception as exc:
        logger.error("Live Basketball board generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Live Basketball boards are temporarily unavailable")


# ---------- Auth Endpoints ----------

@app.post("/auth/login", response_model=LoginResponse)
async def login(username: str = Form(...), password: str = Form(...)):
    """Authenticate user and return JWT token."""
    user = await authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    access_token = create_access_token(user["id"], user["username"], user["tier"])

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user=UserResponse(
            id=user["id"],
            username=user["username"],
            tier=user["tier"],
        ),
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user info."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        tier=current_user.tier,
    )


# ---------- Analyze Endpoints ----------

@app.get("/api/universe", response_model=UniverseResponse)
async def get_universe(current_user: User = Depends(get_current_user)):
    """Get available stocks based on user's tier."""
    tier_config = current_user.tier_config
    stocks = list(UNIVERSE)

    # Categorize stocks (simplified - based on common patterns)
    categories = categorize_stocks(stocks)

    return UniverseResponse(
        stocks=stocks,
        categories=categories,
        total=len(stocks),
    )


def categorize_stocks(stocks: List[str]) -> Dict[str, List[str]]:
    """Categorize stocks by sector (simplified)."""
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


@app.get("/api/analyze/models", response_model=ModelsAvailableResponse)
async def get_available_models(current_user: User = Depends(get_current_user)):
    """Get models available to the user based on tier."""
    tier_config = current_user.tier_config

    return ModelsAvailableResponse(
        models=tier_config.allowed_models,
        tasks=tier_config.allowed_tasks,
        can_use_custom=("custom_models" in tier_config.premium_features),
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def run_analysis(
    req: AnalyzeRequest,
    current_user: User = Depends(get_current_user)
):
    """Run analysis on selected stocks with selected model."""
    tier_config = current_user.tier_config

    # Validate tier limits
    if len(req.tickers) > tier_config.max_stocks_per_request:
        raise HTTPException(
            400,
            f"Too many stocks. {current_user.tier} tier allows max {tier_config.max_stocks_per_request} stocks."
        )

    if req.model not in tier_config.allowed_models:
        raise HTTPException(
            403,
            f"Model '{req.model}' not available for {current_user.tier} tier."
        )

    if req.task not in tier_config.allowed_tasks:
        raise HTTPException(
            403,
            f"Task '{req.task}' not available for {current_user.tier} tier."
        )

    # Check rate limit
    await check_rate_limit(current_user)

    # Run predictions
    results = []
    for ticker in req.tickers:
        try:
            prob_up, signal, last_close, predicted_return = predict_for_ticker(
                ticker.upper(),
                horizon=req.horizon,
                model_type=req.model,
                task=req.task,
                period=req.period,
            )
            results.append(AnalyzeResultItem(
                ticker=ticker.upper(),
                last_close=last_close,
                prob_up=prob_up,
                signal=signal,
                predicted_return=predicted_return,
            ))
        except Exception as e:
            # Include failed ticker with null values
            results.append(AnalyzeResultItem(
                ticker=ticker.upper(),
                last_close=0.0,
                prob_up=None,
                signal=None,
                predicted_return=None,
            ))

    # Increment rate limit
    await increment_user_rate_limit(current_user)

    return AnalyzeResponse(
        results=results,
        metadata={
            "model": req.model,
            "task": req.task,
            "period": req.period,
            "horizon": req.horizon,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "ticker_count": len(results),
        },
    )


@app.get("/api/user/features", response_model=UserFeaturesResponse)
async def get_user_features(current_user: User = Depends(get_current_user)):
    """Get UI features available to user based on tier."""
    tier_config = current_user.tier_config
    from storage.postgres import get_rate_limit

    requests_used = await get_rate_limit(current_user.id)

    # Build features dict
    all_features = [
        "advanced_charts", "export_csv", "multi_model_comparison",
        "custom_models", "api_access", "priority_support"
    ]
    features = {f: f in tier_config.premium_features for f in all_features}

    # Build limits dict
    limits = {
        "daily_requests": tier_config.daily_rate_limit,
        "requests_used": requests_used,
        "requests_remaining": (
            tier_config.daily_rate_limit - requests_used
            if tier_config.daily_rate_limit else None
        ),
        "max_stocks_per_request": tier_config.max_stocks_per_request,
        "max_historical_days": tier_config.max_historical_period_days,
    }

    return UserFeaturesResponse(
        tier=current_user.tier,
        features=features,
        limits=limits,
    )
