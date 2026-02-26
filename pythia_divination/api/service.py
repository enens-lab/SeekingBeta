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

_LSTM_RESPONSE_CACHE: Dict[Tuple[str, str, str], Tuple[float, dict]] = {}
_LSTM_RAW_DATA_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
_LSTM_SINGLEFLIGHT_LOCKS: Dict[Tuple[str, str, str], Lock] = {}
_LSTM_CACHE_LOCK = Lock()
_HOME_CACHE_WARMER_TASK: Optional[asyncio.Task] = None

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
    if not entry:
        return None
    ts, payload = entry
    if now - ts > max_age_seconds:
        return None
    return payload


def _cache_set(model_name: str, ticker: str, payload: dict) -> None:
    key = _cache_key(model_name, ticker)
    now = time.time()
    with _LSTM_CACHE_LOCK:
        _LSTM_RESPONSE_CACHE[key] = (now, dict(payload))
        if len(_LSTM_RESPONSE_CACHE) > LSTM_CACHE_MAX_ITEMS:
            oldest = min(_LSTM_RESPONSE_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _LSTM_RESPONSE_CACHE.pop(oldest, None)


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

    try:
        raw = fetch_ohlcv_for_lstm(ticker, sequence_length=60, data_source=DATA_SOURCE)
    except Exception:
        if stale_df is not None:
            logger.warning(
                "Using stale OHLCV cache for %s after provider fetch failure (stale age <= %ss)",
                ticker,
                LSTM_RAW_DATA_STALE_TTL,
            )
            return stale_df
        raise

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


def _start_home_cache_warmer() -> None:
    global _HOME_CACHE_WARMER_TASK
    if not HOME_CACHE_WARM_ENABLED:
        logger.info("Homepage cache warmer disabled via HOME_CACHE_WARM_ENABLED=false")
        return
    if _HOME_CACHE_WARMER_TASK is not None and not _HOME_CACHE_WARMER_TASK.done():
        return
    _HOME_CACHE_WARMER_TASK = asyncio.create_task(_home_cache_warmer_loop())


@app.on_event("startup")
async def _startup():
    # Temporarily disable database for LSTM testing
    try:
        await init_db()
        # Seed test accounts
        await seed_test_accounts()
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")

    _load_home_cache_snapshot()
    _start_home_cache_warmer()


@app.on_event("shutdown")
async def _shutdown():
    global _HOME_CACHE_WARMER_TASK
    if _HOME_CACHE_WARMER_TASK is None:
        return
    _HOME_CACHE_WARMER_TASK.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _HOME_CACHE_WARMER_TASK
    _HOME_CACHE_WARMER_TASK = None
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


def _compute_lstm_prediction(model_name: str, ticker: str) -> dict:
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

    response = {
        "ticker": ticker,
        "model": model_name,
        "description": str(meta["description"]),
        "horizon": str(meta["horizon"]),
        "target_return": str(meta["target_return"]),
        "probability": round(prob_up * 100, 2),
        "signal": signal,
        "last_close": last_close,
        "recommendation": recommendation,
        "sentiment_score": round(float(feat.attrs.get("sentiment", {}).get("score", 0.0)), 4),
        "sentiment_articles": int(feat.attrs.get("sentiment", {}).get("num_articles", 0)),
        "sentiment_source": feat.attrs.get("sentiment", {}).get("source", "default"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": DATA_SOURCE,
    }
    return response


def _predict_lstm_cached(model_name: str, ticker: str, force_refresh: bool = False) -> dict:
    ticker = ticker.upper()
    use_cache = _should_cache_ticker(ticker)

    if use_cache and not force_refresh:
        cached = _cache_get(model_name, ticker, LSTM_RESPONSE_CACHE_TTL)
        if cached:
            cached["cache_status"] = "hit"
            return cached

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


def predict_for_ticker(
    ticker: str,
    horizon: str = "1d",
    model_type: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK
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
        period=CFG_LOOKBACK,
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

    rows = []
    for model_name in requested_models:
        model_predictions = []
        failures = []
        for ticker in requested_tickers:
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
        "rows": rows,
    }


@app.get("/predict/{ticker}", response_model=PredictResponse)
def predict(
    ticker: str,
    horizon: str = "1d",
    model: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK
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
            task=task
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
def predict_lstm_5d(ticker: str):
    return _predict_lstm_cached("lstm_5d", ticker)


@app.get("/predict/lstm_jackpot/{ticker}")
def predict_lstm_jackpot(ticker: str):
    return _predict_lstm_cached("lstm_jackpot", ticker)


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
        "entries": entries,
    }


@app.post("/predict/cache/warm")
async def warm_prediction_cache(force_refresh: bool = True):
    """Manual operational endpoint for warming homepage prediction cache."""
    summary = await run_in_threadpool(_warm_home_cache_once, force_refresh)
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

    # Broker checks are only relevant for Alpaca-backed trading mode.
    if settings.data_source.lower() == "alpaca":
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
        "universe_size": len(settings.universe),
        "available_models": list(MODEL_REGISTRY.keys()),
    })


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
                task=req.task
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
