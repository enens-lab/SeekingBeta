"""Precomputed stock predictions (the daily RunPod LSTM sweep), served by prophecy.

Replaces the live divination proxies. The sweep (pythia_divination/scripts/
export_stock_predictions.py, run by the RunPod worker) publishes ONE file,
stock_predictions_latest.json, to S3; ops/refresh_stock_predictions_runpod.sh
validates it and drops it at STOCK_PREDICTIONS_PATH on the box, which docker-compose
bind-mounts read-only into this container. No rebuild, no restart: the file is
re-read whenever its mtime changes.

Shape of the file (schema_version 1):
  {"generated_at": iso, "data_source": "yahoo", "models": [...], "homepage_tickers": [...],
   "predictions": {"<model>": {"<TICKER>": {<exactly divination's per-ticker dict>}}},
   "failures": [{"ticker","error"}], ...}

Every per-ticker dict keeps divination's exact keys (probability as a 0-100 percent,
signal, last_close, recommendation, options_*, generated_at, data_source) because the
web client, iOS and Android all parse that dict loosely and were never changed for
the move. This module only ADDS `cache_status` (hit | stale) and, when stale, a
`warning`, mirroring what divination's cache-only mode emitted.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("/app/data/predictions/stock_predictions_latest.json")
_DEV_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions" / "stock_predictions_latest.json"

STOCK_PREDICTIONS_PATH = Path(os.getenv("STOCK_PREDICTIONS_PATH") or (_DEFAULT_PATH if _DEFAULT_PATH.parent.exists() else _DEV_PATH))
# A sweep runs after each weekday close (~22:15 UTC). Friday's file must serve through
# Monday morning (~60h) and a long weekend (~84h); beyond that, refuse rather than
# quietly show week-old signals as current.
STOCK_PREDICTIONS_MAX_AGE_HOURS = float(os.getenv("STOCK_PREDICTIONS_MAX_AGE_HOURS", "96"))
# Older than this (but under the max) is served with cache_status="stale" + a warning,
# the same contract divination's stale-cache responses had.
STOCK_PREDICTIONS_STALE_AFTER_HOURS = float(os.getenv("STOCK_PREDICTIONS_STALE_AFTER_HOURS", "30"))
MODEL_ALIASES = {"lstm_quant": "lstm_5d"}  # byte-identical model; the sweep may omit the alias


class StockPredictionsStore:
    """mtime-cached loader. Thread-safe; cheap to call per request."""

    def __init__(self, path: Path = STOCK_PREDICTIONS_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime: Optional[float] = None
        self._payload: Optional[Dict[str, Any]] = None
        self._generated_at: Optional[datetime] = None

    # ── loading ───────────────────────────────────────────────────────────

    def _load_locked(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            if self._payload is not None:
                logger.warning("stock predictions file disappeared: %s", self.path)
            self._payload, self._mtime, self._generated_at = None, None, None
            return
        if self._mtime == mtime and self._payload is not None:
            return
        try:
            payload = json.loads(self.path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.error("stock predictions file unreadable (%s): %s", self.path, exc)
            return  # keep serving the previous good payload
        preds = payload.get("predictions") if isinstance(payload, dict) else None
        if not isinstance(preds, dict):
            logger.error("stock predictions file has no 'predictions' map: %s", self.path)
            return
        generated_at = _parse_ts(payload.get("generated_at"))
        self._payload, self._mtime, self._generated_at = payload, mtime, generated_at
        logger.info("stock predictions loaded: %s models, %s tickers, generated_at=%s",
                    len(preds), len(next(iter(preds.values()), {}) or {}), payload.get("generated_at"))

    def payload(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._load_locked()
            return self._payload

    # ── freshness ─────────────────────────────────────────────────────────

    def age_hours(self, now: Optional[datetime] = None) -> Optional[float]:
        with self._lock:
            self._load_locked()
            if self._generated_at is None:
                return None
            now = now or datetime.now(timezone.utc)
            return max(0.0, (now - self._generated_at).total_seconds() / 3600.0)

    def available(self) -> bool:
        age = self.age_hours()
        return age is not None and age <= STOCK_PREDICTIONS_MAX_AGE_HOURS

    def _freshness(self, age: Optional[float]) -> Dict[str, Any]:
        if age is not None and age > STOCK_PREDICTIONS_STALE_AFTER_HOURS:
            return {"cache_status": "stale",
                    "warning": f"Prediction is {age:.0f}h old (next sweep runs after the next market close)"}
        return {"cache_status": "hit"}

    # ── lookups ───────────────────────────────────────────────────────────

    def models(self) -> List[str]:
        p = self.payload()
        return sorted((p or {}).get("predictions", {}).keys()) if p else []

    def get(self, model: str, ticker: str) -> Optional[Dict[str, Any]]:
        """One prediction dict (divination shape + cache_status) or None when the ticker
        is not in the sweep or the whole file is too old to serve."""
        p = self.payload()
        if not p:
            return None
        age = self.age_hours()
        if age is None or age > STOCK_PREDICTIONS_MAX_AGE_HOURS:
            return None
        preds = p.get("predictions") or {}
        table = preds.get(model)
        if table is None and model in MODEL_ALIASES:
            table = preds.get(MODEL_ALIASES[model])
        if not isinstance(table, dict):
            return None
        row = table.get(str(ticker).strip().upper())
        if not isinstance(row, dict):
            return None
        out = dict(row)
        out["model"] = model  # alias requests report the name they asked for
        out.update(self._freshness(age))
        return out

    def homepage(self, tickers: Optional[List[str]] = None, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """divination's /predict/homepage response shape, built from the file."""
        p = self.payload()
        age = self.age_hours()
        usable = bool(p) and age is not None and age <= STOCK_PREDICTIONS_MAX_AGE_HOURS
        requested_tickers = [t.strip().upper() for t in (tickers or (p or {}).get("homepage_tickers") or []) if t.strip()]
        requested_models = list(models or (p or {}).get("models") or [])
        rows = []
        for model in requested_models:
            preds, failures = [], []
            for t in requested_tickers:
                row = self.get(model, t) if usable else None
                if row:
                    preds.append(row)
                else:
                    failures.append({"ticker": t, "error": "cache-miss" if usable else "predictions unavailable"})
            rows.append({"model": model, "predictions": preds, "requested_tickers": len(requested_tickers),
                         "successful_predictions": len(preds), "failures": failures})
        return {
            "available": any(r["successful_predictions"] > 0 for r in rows),
            "as_of": (p or {}).get("generated_at") if usable else datetime.now(timezone.utc).isoformat(),
            "data_source": (p or {}).get("data_source", "unknown"),
            "cache_only": True,
            "source": "precomputed",
            "age_hours": round(age, 1) if age is not None else None,
            "rows": rows,
        }

    def status(self) -> Dict[str, Any]:
        p = self.payload()
        age = self.age_hours()
        preds = (p or {}).get("predictions") or {}
        return {
            "path": str(self.path),
            "loaded": p is not None,
            "available": self.available(),
            "generated_at": (p or {}).get("generated_at"),
            "age_hours": round(age, 1) if age is not None else None,
            "max_age_hours": STOCK_PREDICTIONS_MAX_AGE_HOURS,
            "models": sorted(preds.keys()),
            "tickers": len(next(iter(preds.values()), {}) or {}) if preds else 0,
            "producer": (p or {}).get("producer"),
            "truncated": (p or {}).get("truncated"),
            "failed": (p or {}).get("failed"),
        }


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


STORE = StockPredictionsStore()
