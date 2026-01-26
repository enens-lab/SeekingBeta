"""Pythia Signals API - Multi-model ML Backend."""

import yaml
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import pandas as pd

from fastapi import FastAPI, HTTPException, Body, Depends, Form
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

from typing import Dict, Tuple, List, Optional

from data.fetch import fetch_ohlcv, fetch_panel
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
MODEL, SCALER, FEATS, _ = load_artifacts()
UNIVERSE = settings.universe
DATA_SOURCE = settings.data_source
CFG_LOOKBACK = settings.lookback_download
THRESHOLD = settings.threshold

EXTRA_FEATS: Dict[Tuple[str, str], Dict[str, float]] = {}


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


@app.on_event("startup")
async def _startup():
    await init_db()
    # Seed test accounts
    await seed_test_accounts()


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
    except (ValueError, FileNotFoundError):
        # Fall back to legacy model for backward compatibility
        if model_type == DEFAULT_MODEL and task == DEFAULT_TASK:
            X = feat[FEATS].values
            Xs = SCALER.transform(X)
            prob_up = float(MODEL.predict_proba(Xs)[:, 1][-1])
            signal = "buy" if prob_up >= THRESHOLD else ("sell" if prob_up <= 1 - THRESHOLD else "hold")
            return prob_up, signal, last_close, None
        raise HTTPException(404, f"Model not found: {model_type}/{task}")

    # Prepare features for the model
    X = feat[feat_cols].values
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
async def paper_snapshot(notes: str | None = None):
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
