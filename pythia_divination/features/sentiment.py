"""Lightweight sentiment features for LSTM inference.

This module fetches recent headlines from Finviz and computes a bounded
sentiment score in [-1, 1]. It is intentionally dependency-light and uses
safe fallbacks so inference never fails if sentiment is unavailable.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dep fallback
    BeautifulSoup = None


FINVIZ_URL = "https://finviz.com/quote.ashx?t={ticker}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

POSITIVE_TERMS = {
    "beat",
    "beats",
    "bullish",
    "buy",
    "upside",
    "surge",
    "gain",
    "gains",
    "growth",
    "strong",
    "record",
    "upgrade",
    "outperform",
    "expands",
    "profit",
    "profits",
    "rally",
}

NEGATIVE_TERMS = {
    "miss",
    "misses",
    "bearish",
    "sell",
    "downside",
    "drop",
    "loss",
    "losses",
    "weak",
    "downgrade",
    "underperform",
    "lawsuit",
    "probe",
    "cuts",
    "decline",
    "fall",
    "warning",
}

WORD_RE = re.compile(r"[a-zA-Z]{2,}")


@dataclass(frozen=True)
class SentimentSnapshot:
    score: float
    num_articles: int
    source: str
    ts_unix: float


_CACHE: dict[str, SentimentSnapshot] = {}


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _clip_score(score: float) -> float:
    if score > 1.0:
        return 1.0
    if score < -1.0:
        return -1.0
    return score


def _headline_score(text: str) -> float:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_TERMS)
    neg = sum(1 for w in words if w in NEGATIVE_TERMS)
    raw = (pos - neg) / max(1, pos + neg)
    return _clip_score(raw)


def _fetch_finviz_headlines(ticker: str, max_headlines: int, timeout_sec: float) -> list[str]:
    if BeautifulSoup is None:
        return []

    url = FINVIZ_URL.format(ticker=ticker.upper())
    resp = requests.get(url, headers=HEADERS, timeout=timeout_sec)
    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    news_table = soup.find(id="news-table")
    if news_table is None:
        return []

    headlines: list[str] = []
    for row in news_table.find_all("tr"):
        link = row.find("a")
        if link is None:
            continue
        headline = link.get_text(" ", strip=True)
        if headline:
            headlines.append(headline)
        if len(headlines) >= max_headlines:
            break
    return headlines


def get_sentiment_snapshot(ticker: str) -> SentimentSnapshot:
    """Return cached sentiment snapshot for ticker with safe fallback."""
    enabled = _env_bool("SENTIMENT_ENABLED", True)
    ttl_sec = int(os.getenv("SENTIMENT_CACHE_TTL_SEC", "900"))
    max_headlines = int(os.getenv("SENTIMENT_MAX_HEADLINES", "10"))
    timeout_sec = float(os.getenv("SENTIMENT_TIMEOUT_SEC", "3.0"))

    now = time.time()
    if not enabled:
        return SentimentSnapshot(score=0.0, num_articles=0, source="disabled", ts_unix=now)

    key = ticker.upper()
    cached = _CACHE.get(key)
    if cached and (now - cached.ts_unix) <= ttl_sec:
        return cached

    try:
        headlines = _fetch_finviz_headlines(key, max_headlines=max_headlines, timeout_sec=timeout_sec)
        if not headlines:
            snap = SentimentSnapshot(score=0.0, num_articles=0, source="finviz-empty", ts_unix=now)
            _CACHE[key] = snap
            return snap

        per_headline = [_headline_score(h) for h in headlines]
        score = _clip_score(sum(per_headline) / max(1, len(per_headline)))
        snap = SentimentSnapshot(score=score, num_articles=len(headlines), source="finviz-lexicon", ts_unix=now)
        _CACHE[key] = snap
        return snap
    except Exception:
        snap = SentimentSnapshot(score=0.0, num_articles=0, source="error", ts_unix=now)
        _CACHE[key] = snap
        return snap

