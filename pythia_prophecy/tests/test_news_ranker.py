"""Checks for api/news_ranker.py (board-card headlines reranked via the Atlas model API).

Plain script like the other tests here:  python tests/test_news_ranker.py
The rerank call is stubbed; no network.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import news_ranker as nr  # noqa: E402

CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg)


NOW = datetime.now(timezone.utc)


def art(i, title, days_ago):
    return {"article_id": f"id{i}", "title": title, "publisher": "Pub", "link": f"https://x/{i}",
            "published_at": (NOW - timedelta(days=days_ago)).isoformat()}


ARTICLES = [
    art(1, "Is XRP the Best Cryptocurrency You Can Buy Right Now?", 0.1),
    art(2, "Nvidia stock drops 6% on report tied to export probe", 0.2),
    art(3, "Nvidia stock drops 6% on report tied to export probe", 0.3),  # duplicate title
    art(4, "Goldman calls chip stocks oversold, names Nvidia a buy", 1),
    art(5, "Nvidia beats on earnings, guides above consensus", 30),  # too old
]


def test_recent_filters_and_dedupes():
    recent = nr.recent_articles(ARTICLES, max_age_days=7, now=NOW)
    titles = [a["title"] for a in recent]
    check(len(recent) == 3, f"expected 3 recent unique, got {titles}")
    check(titles[0].startswith("Is XRP"), "newest first")
    check(not any("earnings" in t for t in titles), "30-day-old article dropped")


def test_ranked_picks_above_threshold():
    nr._cache.clear()
    nr.VOYAGE_API_KEY = "test"
    calls = []

    async def fake(query, docs, top_k):
        calls.append((query, docs, top_k))
        # recent order: XRP(0), drop(1), Goldman(2)
        return [(1, 0.82), (2, 0.66), (0, 0.21)]

    nr._rerank = fake
    out = asyncio.run(nr.top_headlines("nvda", "NVIDIA Corporation", ARTICLES, k=3))
    check(out["ranked"] is True and out["model"] == nr.VOYAGE_RERANK_MODEL, f"ranked result: {out}")
    check([i["title"][:7] for i in out["items"]] == ["Nvidia ", "Goldman"], f"low-score XRP filtered: {out['items']}")
    check("NVIDIA Corporation (NVDA)" in calls[0][0], f"query names the company: {calls[0][0]}")
    # cached: same articles -> no second call
    asyncio.run(nr.top_headlines("NVDA", "NVIDIA Corporation", ARTICLES, k=3))
    check(len(calls) == 1, "second identical request served from cache")


def test_fallback_on_error_and_without_key():
    nr._cache.clear()
    nr.VOYAGE_API_KEY = "test"

    async def boom(*a, **k):
        raise RuntimeError("503")

    nr._rerank = boom
    out = asyncio.run(nr.top_headlines("NVDA", None, ARTICLES, k=2))
    check(out["ranked"] is False and len(out["items"]) == 2, f"newest-first fallback on error: {out}")
    check(not nr._cache, "errors are not cached")

    nr.VOYAGE_API_KEY = ""
    out = asyncio.run(nr.top_headlines("NVDA", None, ARTICLES, k=3))
    check(out["ranked"] is False and len(out["items"]) == 3, "no key -> newest 3")


def test_no_recent_news():
    out = asyncio.run(nr.top_headlines("NVDA", None, [art(9, "Old news", 40)], k=3))
    check(out["items"] == [] and out["ranked"] is False, "nothing recent -> empty")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"news_ranker: {CHECKS} checks passed")
