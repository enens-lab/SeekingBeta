"""Batch stock-prediction sweep: run the torch options LSTMs for a ticker universe
and write ONE JSON artifact that pythia_prophecy serves instead of calling
divination live.

Why: divination on the web box existed for two things -- on-demand LSTM inference
and the in-process live sports slates. Both are batch-shaped (daily bars, daily
option chains). Serving them from a 5 GB always-on container on a $70/mo box was
the cost; this script is the replacement: it runs on the RunPod sports worker once
a day (job type `stock_predictions`, see runpod/sports/handler.py), the worker
uploads the output to s3://<bucket>/frontend-boards/stock_predictions_latest.json,
and ops/refresh_stock_predictions_runpod.sh pulls it onto the box where prophecy
reads it from a bind mount (api/stock_predictions_store.py). No torch, tensorflow
or divination container on the web box at all.

Parity: per ticker this is EXACTLY api/service.py::_compute_lstm_quant_prediction --
same fetch (data.fetch.fetch_ohlcv_for_lstm, 400d), same 48-feature frame
(models.quant.feature_engine, live options ffilled across the window, fear_index
20.0, externals zero-filled, no dropna), same by-hand StandardScaler, same net, same
signal thresholds and recommendation text -- so the numbers users see do not change
with the move. The one deliberate difference: features are engineered ONCE per
ticker and every model runs on that frame (lstm_5d and lstm_jackpot share the
identical 48-column schema and scaler), so the option chain is fetched once, not
once per model.

Usage (cwd = pythia_divination):
  python scripts/export_stock_predictions.py --universe core --output /tmp/stock_predictions_latest.json
  python scripts/export_stock_predictions.py --tickers AAPL,MSFT --models lstm_5d --workers 2
  python scripts/export_stock_predictions.py --universe core --tickers-file /tmp/extra.txt

  --universe core   = config.yaml curated list (~170) + the homepage tickers + any
                      --tickers/--tickers-file extras (watchlists, tier lists).
  --universe full   = everything in data/universe.csv (~6,700; hours of Yahoo I/O,
                      see the cost note in runpod/sports/README.md).
  --universe none   = only the explicit --tickers/--tickers-file.

Needs no Postgres (every storage helper no-ops without DATABASE_URL; a dummy is
stubbed below so config.settings imports) and no Alpaca (defaults to Yahoo when no
Alpaca key is present). Exit 0 when at least one prediction succeeded, 2 when none did.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

# config.settings (imported by data.fetch) requires DATABASE_URL and refuses
# DATA_SOURCE=alpaca without keys. The sweep touches neither the DB nor Alpaca
# unless the caller provides them, so stub the first and downgrade the second.
if not os.getenv("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://unused:unused@127.0.0.1:5432/none"
if os.getenv("DATA_SOURCE", "").lower() == "alpaca" and not os.getenv("ALPACA_KEY_ID"):
    os.environ["DATA_SOURCE"] = "yahoo"
os.environ.setdefault("DATA_SOURCE", "yahoo")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("export_stock_predictions")

SEQ_LEN = 60
SCHEMA_VERSION = 1
OUTPUT_NAME = "stock_predictions_latest.json"
DEFAULT_MODELS = ("lstm_5d", "lstm_jackpot", "lstm_quant")
DEFAULT_HOMEPAGE_TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META")

# Copied from api/service.py (LSTM_MODEL_METADATA descriptions + _QUANT_RECO) so this
# script does not import the FastAPI app. Keep the two in sync: these strings are
# what the web cards, iOS and Android render.
MODEL_DESCRIPTIONS = {
    "lstm_5d": "5-Day Options-Enriched LSTM (48 features incl. live options flow)",
    "lstm_jackpot": "20-Day Options-Enriched Jackpot LSTM (48 features incl. live options flow)",
    "lstm_quant": "Options-Enriched Quant LSTM (48-feature, live options flow)",
}
RECO_TEXT = {
    "strong_buy": "Strong buy ({prob:.1f}% probability of >{tgt:.0f}% gain in {h}d; options flow confirms)",
    "buy": "Buy ({prob:.1f}% probability of >{tgt:.0f}% gain in {h}d)",
    "hold": "Neutral ({prob:.1f}% probability, hold)",
    "avoid": "Weak signal ({prob:.1f}% probability, consider avoiding)",
}

_FORWARD_LOCK = threading.Lock()


# ── universe ──────────────────────────────────────────────────────────────────

_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


def _clean_tickers(values) -> List[str]:
    """Upper-case, de-duplicate, and drop anything that is not ticker-shaped (a
    stray log line in a --tickers-file must not turn into six failed fetches)."""
    out: List[str] = []
    seen = set()
    for raw in values or []:
        t = str(raw or "").strip().upper()
        if not t or t in seen or not _TICKER_RE.match(t):
            continue
        seen.add(t)
        out.append(t)
    return out


def resolve_universe(mode: str, extra: List[str], homepage: List[str]) -> List[str]:
    """core: config.yaml curated list + homepage + extras. full: data/universe.csv (+ the
    same). none: extras + homepage only. Order is stable so partial runs are deterministic."""
    tickers: List[str] = []
    mode = (mode or "core").lower()
    if mode in ("core", "full"):
        try:
            import yaml
            from universe import load_universe, read_universe_csv, DEFAULT_UNIVERSE_CSV_PATH

            cfg = yaml.safe_load((DIV_ROOT / "config.yaml").read_text()) or {}
            base = [str(t) for t in (cfg.get("universe") or [])]
            if mode == "full":
                tickers = load_universe(base_universe=base)
            else:
                tickers = base or read_universe_csv(DEFAULT_UNIVERSE_CSV_PATH)[:200]
        except Exception as exc:  # noqa: BLE001
            logger.warning("universe load failed (%s); continuing with explicit tickers only", exc)
    return _clean_tickers([*homepage, *extra, *tickers])


# ── inference ─────────────────────────────────────────────────────────────────

def _load_models(model_names: List[str]) -> Dict[str, Dict[str, Any]]:
    from models.quant.lstm_quant import load_quant_model

    states = {}
    for name in model_names:
        states[name] = load_quant_model(name)
    cols = {tuple(s["feature_cols"]) for s in states.values()}
    if len(cols) != 1:
        raise RuntimeError("models disagree on feature columns; cannot share one feature frame")
    return states


def _ohlcv_frame(ticker: str, data_source: str) -> pd.DataFrame:
    from data.fetch import fetch_ohlcv_for_lstm

    raw = fetch_ohlcv_for_lstm(ticker, sequence_length=SEQ_LEN, retries=3, data_source=data_source)
    if raw is None or len(raw) < SEQ_LEN:
        raise ValueError(f"Not enough data (need {SEQ_LEN} bars, got {0 if raw is None else len(raw)})")
    # Same adaptation as _compute_lstm_quant_prediction: DatetimeIndex + capitalized
    # OHLCV -> predict_quant's lowercase schema.
    return pd.DataFrame({
        "date": pd.to_datetime(raw.index),
        "open": raw["Open"].to_numpy(dtype=float),
        "high": raw["High"].to_numpy(dtype=float),
        "low": raw["Low"].to_numpy(dtype=float),
        "close": raw["Close"].to_numpy(dtype=float),
        "volume": raw["Volume"].to_numpy(dtype=float),
    })


def _signal(prob: float) -> str:
    # models.quant.lstm_quant._signal thresholds, restated to avoid a private import.
    if prob >= 0.60:
        return "strong_buy"
    if prob >= 0.55:
        return "buy"
    if prob >= 0.45:
        return "hold"
    return "avoid"


def predict_ticker(ticker: str, states: Dict[str, Dict[str, Any]], *, data_source: str,
                   options_source: str) -> Dict[str, Dict[str, Any]]:
    """All requested models for one ticker. Returns {model: prediction_dict}. Raises on
    a data failure (caller records it under `failures`)."""
    import torch
    from models.quant.feature_engine import engineer_quant_features
    from models.quant.options_features import FEATURE_COLUMNS as OPTION_COLS

    ohlcv = _ohlcv_frame(ticker, data_source)
    last_close = float(ohlcv["close"].iloc[-1])
    any_state = next(iter(states.values()))
    feat = engineer_quant_features(ohlcv, any_state["feature_cols"], symbol=ticker,
                                   options_source=options_source)
    if len(feat) < SEQ_LEN:
        raise ValueError(f"need >= {SEQ_LEN} feature rows, got {len(feat)}")
    opt_cols = [c for c in OPTION_COLS if c in feat.columns]
    options_live = bool(np.abs(feat.iloc[-1][opt_cols].to_numpy(dtype=np.float64)).sum() > 0) if opt_cols else False
    recent = feat.iloc[-SEQ_LEN:].to_numpy(dtype=np.float64)

    generated_at = datetime.now(timezone.utc).isoformat()
    out: Dict[str, Dict[str, Any]] = {}
    for name, state in states.items():
        X = ((recent - state["scaler_mean"]) / state["scaler_scale"]).astype(np.float32)
        X = X.reshape(1, SEQ_LEN, len(state["feature_cols"]))
        with _FORWARD_LOCK, torch.no_grad():
            logit = state["model"](torch.from_numpy(X)).item()
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        prob_pct = round(prob * 100, 2)
        signal = _signal(round(prob, 4))
        horizon = int(state["horizon"])
        target_pct = round(float(state["target_return"]) * 100, 1)
        out[name] = {
            "ticker": ticker,
            "model": name,
            "description": MODEL_DESCRIPTIONS.get(name, name),
            "horizon": f"{horizon} days",
            "target_return": f">{target_pct:.0f}%",
            "probability": prob_pct,
            "signal": signal,
            "last_close": last_close,
            "recommendation": RECO_TEXT.get(signal, "{prob:.1f}% probability").format(
                prob=prob_pct, tgt=target_pct, h=horizon),
            "options_source": options_source,
            "options_live": options_live,
            "generated_at": generated_at,
            "data_source": data_source,
        }
    return out


# ── sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(tickers: List[str], model_names: List[str], *, data_source: str, options_source: str,
              workers: int, deadline_seconds: float, log_every: int = 25) -> Dict[str, Any]:
    started = time.time()
    states = _load_models(model_names)
    predictions: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in model_names}
    failures: List[Dict[str, str]] = []
    truncated = False
    done = 0

    def _budget_left() -> bool:
        return deadline_seconds <= 0 or (time.time() - started) < deadline_seconds

    # Submit in bounded waves so a deadline stops us from queueing hours of work: the
    # RunPod endpoint kills the container at its execution timeout and we would lose
    # everything computed so far instead of publishing a partial (but valid) file.
    wave = max(workers * 4, 8)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, len(tickers), wave):
            if not _budget_left():
                truncated = True
                logger.warning("deadline reached after %d/%d tickers; publishing partial sweep", done, len(tickers))
                break
            batch = tickers[start:start + wave]
            futures = {pool.submit(predict_ticker, t, states, data_source=data_source,
                                   options_source=options_source): t for t in batch}
            for fut in as_completed(futures):
                t = futures[fut]
                done += 1
                try:
                    for name, pred in fut.result().items():
                        predictions[name][t] = pred
                except Exception as exc:  # noqa: BLE001
                    failures.append({"ticker": t, "error": str(exc)[:300]})
                if done % log_every == 0 or done == len(tickers):
                    ok = len(predictions[model_names[0]])
                    logger.info("progress %d/%d ok=%d failed=%d elapsed=%.0fs", done, len(tickers), ok,
                                len(failures), time.time() - started)

    requested = len(tickers)
    succeeded = len(predictions[model_names[0]])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "producer": os.getenv("STOCK_PREDICTIONS_PRODUCER") or socket.gethostname(),
        "data_source": data_source,
        "options_source": options_source,
        "models": model_names,
        "requested": requested,
        "succeeded": succeeded,
        "failed": len(failures),
        "truncated": truncated,
        "duration_seconds": round(time.time() - started, 1),
        "predictions": predictions,
        "failures": failures,
    }


def write_atomic(payload: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(tmp, output)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default="core", choices=["core", "full", "none"])
    ap.add_argument("--tickers", default="", help="comma-separated extra tickers")
    ap.add_argument("--tickers-file", default="", help="file with one ticker per line (extra tickers)")
    ap.add_argument("--homepage-tickers", default=os.getenv("HOME_CACHE_TICKERS", ",".join(DEFAULT_HOMEPAGE_TICKERS)))
    ap.add_argument("--models", default=os.getenv("STOCK_PREDICTIONS_MODELS", ",".join(DEFAULT_MODELS)))
    ap.add_argument("--output", default=str(DIV_ROOT.parent / "pythia_prophecy" / "frontend" / "src" / "data" / OUTPUT_NAME))
    ap.add_argument("--data-source", default=os.getenv("DATA_SOURCE", "yahoo"))
    ap.add_argument("--options-source", default=os.getenv("QUANT_OPTIONS_SOURCE", "yfinance"))
    ap.add_argument("--workers", type=int, default=int(os.getenv("STOCK_PREDICTIONS_WORKERS", "4")))
    ap.add_argument("--deadline-seconds", type=float, default=float(os.getenv("STOCK_PREDICTIONS_DEADLINE_SEC", "1500")))
    ap.add_argument("--limit", type=int, default=0, help="only the first N tickers (smoke tests)")
    args = ap.parse_args(argv)

    extra = _clean_tickers(args.tickers.split(","))
    if args.tickers_file:
        extra += _clean_tickers(Path(args.tickers_file).read_text().split())
    homepage = _clean_tickers(args.homepage_tickers.split(","))
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        ap.error("no models requested")

    tickers = resolve_universe(args.universe, extra, homepage)
    if args.limit > 0:
        tickers = tickers[:args.limit]
    if not tickers:
        ap.error("empty universe")

    import torch
    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))

    logger.info("sweep: %d tickers x %s (universe=%s, data=%s, options=%s, workers=%d, deadline=%.0fs)",
                len(tickers), model_names, args.universe, args.data_source, args.options_source,
                args.workers, args.deadline_seconds)
    payload = run_sweep(tickers, model_names, data_source=args.data_source,
                        options_source=args.options_source, workers=args.workers,
                        deadline_seconds=args.deadline_seconds)
    payload["universe"] = args.universe
    payload["homepage_tickers"] = homepage

    output = Path(args.output)
    write_atomic(payload, output)
    logger.info("wrote %s: %d/%d tickers ok, %d failed, truncated=%s, %.0fs", output,
                payload["succeeded"], payload["requested"], payload["failed"], payload["truncated"],
                payload["duration_seconds"])
    missing_home = [t for t in homepage if t not in payload["predictions"][model_names[0]]]
    if missing_home:
        logger.warning("homepage tickers missing from sweep: %s", missing_home)
    return 0 if payload["succeeded"] > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
