"""Rank a ticker's recent headlines by how well they explain the stock's move.

Yahoo's news feed for a symbol is noisy: listicles, crypto pieces and ETF
comparisons that merely tag the ticker. The board cards want the two or three
headlines that actually bear on the stock, so we rerank the recent ones with a
Voyage AI reranker served through MongoDB Atlas's model API
(https://ai.mongodb.com/v1/rerank, key prefix "al-").

Cheap by construction: one rerank call per (ticker, set of articles), cached in
process for NEWS_RANK_CACHE_SECONDS, so a homepage full of flips costs a handful
of calls a day. Without a key, or if the API fails, we fall back to the newest
headlines and say so (`ranked: false`), so the cards never break on it.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .logging_config import get_logger

logger = get_logger("news_ranker")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "").strip()
VOYAGE_API_BASE = os.getenv("VOYAGE_API_BASE", "https://ai.mongodb.com/v1").rstrip("/")
VOYAGE_RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")
NEWS_RANK_MAX_AGE_DAYS = float(os.getenv("NEWS_RANK_MAX_AGE_DAYS", "7"))
NEWS_RANK_MIN_SCORE = float(os.getenv("NEWS_RANK_MIN_SCORE", "0.4"))
NEWS_RANK_CACHE_SECONDS = float(os.getenv("NEWS_RANK_CACHE_SECONDS", "21600"))
NEWS_RANK_TIMEOUT_SECONDS = float(os.getenv("NEWS_RANK_TIMEOUT_SECONDS", "6"))

_cache: Dict[Tuple[str, Tuple[str, ...], int], Tuple[float, Dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def recent_articles(articles: List[dict], max_age_days: float = NEWS_RANK_MAX_AGE_DAYS,
                    now: Optional[datetime] = None) -> List[dict]:
    """Articles published within max_age_days, newest first, duplicate titles dropped."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
    seen, out = set(), []
    for art in articles:
        title = (art.get("title") or "").strip()
        ts = _parse_ts(art.get("published_at"))
        if not title or ts is None or ts < cutoff:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({**art, "_ts": ts})
    out.sort(key=lambda a: a["_ts"], reverse=True)
    return out


def _item(art: dict, score: Optional[float]) -> Dict[str, Any]:
    return {
        "title": art.get("title"),
        "publisher": art.get("publisher"),
        "link": art.get("link"),
        "published_at": art["_ts"].isoformat(),
        "score": round(score, 3) if score is not None else None,
    }


def rank_query(ticker: str, company_name: Optional[str]) -> str:
    name = (company_name or "").strip()
    who = f"{name} ({ticker})" if name and name.upper() != ticker.upper() else ticker
    return f"News explaining why {who} stock is moving: earnings, guidance, analyst calls, deals, regulation, products"


async def _rerank(query: str, documents: List[str], top_k: int) -> List[Tuple[int, float]]:
    async with httpx.AsyncClient(timeout=NEWS_RANK_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            f"{VOYAGE_API_BASE}/rerank",
            headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
            json={"query": query, "documents": documents, "model": VOYAGE_RERANK_MODEL,
                  "top_k": top_k, "truncation": True},
        )
        resp.raise_for_status()
        return [(int(r["index"]), float(r["relevance_score"])) for r in resp.json().get("data", [])]


async def top_headlines(ticker: str, company_name: Optional[str], articles: List[dict],
                        k: int = 3) -> Dict[str, Any]:
    """{ticker, ranked, model, items[{title, publisher, link, published_at, score}]}.

    ranked=True: items are the reranker's picks above NEWS_RANK_MIN_SCORE (may be
    fewer than k, or none, when nothing recent is really about the stock).
    ranked=False: no key / API error, items are simply the newest k headlines.
    """
    ticker = ticker.upper()
    k = max(1, min(int(k), 5))
    recent = recent_articles(articles)
    base = {"ticker": ticker, "max_age_days": NEWS_RANK_MAX_AGE_DAYS}
    if not recent:
        return {**base, "ranked": False, "model": None, "items": []}

    cache_key = (ticker, tuple(a.get("article_id") or a["title"] for a in recent), k)
    with _cache_lock:
        hit = _cache.get(cache_key)
        if hit and time.monotonic() - hit[0] < NEWS_RANK_CACHE_SECONDS:
            return hit[1]

    result: Dict[str, Any]
    if VOYAGE_API_KEY:
        try:
            pairs = await _rerank(rank_query(ticker, company_name), [a["title"] for a in recent], top_k=k)
            items = [_item(recent[i], s) for i, s in pairs if s >= NEWS_RANK_MIN_SCORE and 0 <= i < len(recent)]
            result = {**base, "ranked": True, "model": VOYAGE_RERANK_MODEL, "items": items}
        except Exception as exc:  # noqa: BLE001 - never break the card over ranking
            logger.warning("headline rerank failed for %s: %s", ticker, exc)
            return {**base, "ranked": False, "model": None, "items": [_item(a, None) for a in recent[:k]]}
    else:
        result = {**base, "ranked": False, "model": None, "items": [_item(a, None) for a in recent[:k]]}

    with _cache_lock:
        if len(_cache) > 2000:
            _cache.clear()
        _cache[cache_key] = (time.monotonic(), result)
    return result
