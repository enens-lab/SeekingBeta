"""Append-only log of published market picks, and grading from that log only.

Why a log: the existing history boards are rebuilt at serve time by running the
CURRENT model over completed games. That is fine for a win-probability demo but
cannot honestly grade a spread or total pick: the line moves, the model changes,
and hindsight leaks in. A market pick is graded against exactly what was shown to
users before the game.

Layout (written by the RunPod worker at every bake, synced to/from S3 under
sports-data/market_picks/<sport>/ -- a prefix the worker may already write):

    data/sports/market_picks/<sport>/picks_<YYYYMMDDTHHMMSSZ>.jsonl

One file per bake, never rewritten; one JSON object per pick:
    {marketId, boardId, sport, type, period, side, label, line, modelLine,
     modelProbability, outcomeProbabilities, market{...}, publishedAt,
     gameId, gameDate (YYYYMMDD), gameStart (ISO, optional), homeTeam, awayTeam,
     season, modelVersion}

Grading picks, per marketId, the LAST snapshot published before the game started
(gameStart if known, else 12:00 UTC on gameDate -- the daily bake runs at 07:10 UTC,
before any US or European slate), joins the final score, and grades with
sports/markets.py::grade_pick. Summaries are per (type, season) with half results
counted half, pushes excluded from the win rate, units at the stated price.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Collection, Any, Callable, Iterable, Optional

from . import markets as mk

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "sports" / "market_picks"
STANDARD_DECIMAL = 1.0 + 100.0 / 110.0  # -110, the conventional price when none is logged
BREAK_EVEN_STANDARD = 110.0 / 210.0     # 0.5238
# A 50% rate has a standard error of ~9 pts at n=30 and ~5 pts at n=100: below 100
# graded picks we show the W-L-P record but not a percentage or ROI.
MIN_GRADED_FOR_RATE = int(os.getenv("MARKET_SUMMARY_MIN_GRADED", "100"))
# Sports whose events start at varied times of day: a pick is only gradable with an
# explicit gameStart, because a noon fallback can be after an early match began.
REQUIRE_GAME_START = {"tennis", "golf", "soccer"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── writing ───────────────────────────────────────────────────────────────────

def picks_from_boards(sport: str, boards: Iterable[dict], *, season_of: Callable[[int], str],
                      model_version: str = "", published_at: Optional[str] = None) -> list[dict]:
    """Flatten the `markets` blocks of upcoming boards into log records."""
    published_at = published_at or _utc_now().isoformat()
    out = []
    for board in boards:
        game_date = board.get("scheduledDate")
        for m in board.get("markets") or []:
            out.append({
                **{k: m.get(k) for k in ("marketId", "type", "period", "side", "label", "line", "modelLine",
                                          "modelProbability", "outcomeProbabilities", "market", "confidenceTier")},
                "boardId": board.get("id"),
                "sport": sport,
                "publishedAt": m.get("publishedAt") or published_at,
                "gameId": board.get("gameId") or board.get("id"),
                "gameDate": game_date,
                "gameStart": board.get("gameStart"),
                "homeTeam": board.get("homeTeam"),
                "awayTeam": board.get("awayTeam"),
                "season": season_of(int(game_date)) if game_date else None,
                "modelVersion": model_version,
            })
    return out


def write_snapshot(sport: str, records: list[dict], *, root: Path = DEFAULT_ROOT, now: Optional[datetime] = None) -> Optional[Path]:
    """Write one immutable snapshot file; returns its path (None when nothing to log).
    Refuses to overwrite: a second bake in the same second gets a suffix."""
    if not records:
        return None
    directory = Path(root) / sport
    directory.mkdir(parents=True, exist_ok=True)
    base = f"picks_{_stamp(now or _utc_now())}"
    path = directory / f"{base}.jsonl"
    n = 1
    while path.exists():
        path = directory / f"{base}_{n}.jsonl"; n += 1
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, separators=(",", ":"), default=str) + "\n" for r in records))
    os.replace(tmp, path)
    logger.info("market pick log: wrote %d picks to %s", len(records), path)
    return path


# ── reading + selecting the pick users saw ───────────────────────────────────

def load_snapshots(sport: str, *, root: Path = DEFAULT_ROOT) -> list[dict]:
    directory = Path(root) / sport
    records: list[dict] = []
    for path in sorted(directory.glob("picks_*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("market pick log: skipping corrupt line in %s", path.name)
    return records


def _game_cutoff(record: dict) -> Optional[datetime]:
    start = record.get("gameStart")
    if not start and str(record.get("sport") or "").lower() in REQUIRE_GAME_START:
        return None
    if start:
        try:
            ts = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    gd = record.get("gameDate")
    if not gd:
        return None
    try:
        day = datetime.strptime(str(int(gd)), "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return day + timedelta(hours=12)


def pregame_picks(records: Iterable[dict]) -> dict[str, dict]:
    """marketId -> the last snapshot published strictly before the game cutoff.

    The cutoff is the EARLIEST start any snapshot reported for that game: if a game
    is moved earlier, older snapshots must not keep the later cutoff and become
    gradable after the real first pitch."""
    records = list(records)
    earliest: dict[str, datetime] = {}
    for r in records:
        c = _game_cutoff(r)
        gid = str(r.get("gameId") or r.get("boardId") or "")
        if c is not None and gid and (gid not in earliest or c < earliest[gid]):
            earliest[gid] = c
    chosen: dict[str, dict] = {}
    for r in records:
        mid = r.get("marketId")
        cutoff = earliest.get(str(r.get("gameId") or r.get("boardId") or "")) or _game_cutoff(r)
        try:
            published = datetime.fromisoformat(str(r.get("publishedAt")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if not mid or cutoff is None or published >= cutoff:
            continue
        prev = chosen.get(mid)
        if prev is None or published > prev["_published"]:
            chosen[mid] = {**r, "_published": published}
    for v in chosen.values():
        v.pop("_published", None)
    return chosen


# ── grading + summaries ───────────────────────────────────────────────────────

# Market types that need a regulation-length game. MLB: a game shortened by weather
# ("Completed Early") is official for the moneyline but run lines and totals are
# void (standard sportsbook rule).
_NEEDS_FULL_GAME = {"run_line", "spread", "total", "team_total_home", "team_total_away"}


def _normalize_result(value: Any) -> Optional[dict]:
    """Accept (home, away) tuples or dicts {home, away, status?, regulationHome?,
    regulationAway?}. status: final | postponed | cancelled | shortened."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    home, away = value
    return {"home": home, "away": away, "status": "final"}


def grade_picks(picks: dict[str, dict], results: dict[str, Any]) -> list[dict]:
    """Grade pre-game picks whose game has a result. `results` maps gameId ->
    (home_score, away_score) or a dict with status and regulation scores.

    * postponed / cancelled -> void (stake returned).
    * shortened (MLB "Completed Early") -> run line, spread, totals void; moneyline graded.
    * regulation scores, when present, grade soccer and any period="reg_time" pick
      (a World Cup knockout's 1X2 / totals settle on 90 minutes, not extra time).
    * a type the grader does not support is kept as "ungradable" (logged loudly),
      never silently voided: a losing pick must not vanish from the record.
    Picks without any result yet stay ungraded."""
    graded = []
    for mid, p in picks.items():
        res = _normalize_result(results.get(str(p.get("gameId"))))
        if res is None:
            continue
        status = str(res.get("status") or "final").lower()
        price = (p.get("market") or {}).get("decimalOdds")
        home, away = res.get("home"), res.get("away")
        if (p.get("period") == "reg_time" or str(p.get("sport") or "") == "soccer") and res.get("regulationHome") is not None:
            home, away = res["regulationHome"], res["regulationAway"]
        if status in ("postponed", "cancelled", "canceled") or (status == "shortened" and p.get("type") in _NEEDS_FULL_GAME):
            result, unit = "void", 0.0
        else:
            try:
                result, unit = mk.grade_pick(p["type"], p["side"], p.get("line"), home, away,
                                             decimal_odds=float(price or STANDARD_DECIMAL))
            except (ValueError, KeyError, TypeError) as exc:
                logger.error("market pick %s (%s) is UNGRADABLE: %s", mid, p.get("type"), exc)
                result, unit = "ungradable", None
        # Units only exist at a real, logged price. A model-only market (no line
        # snapshot) gets a result but no invented -110 return.
        graded.append({**p, "result": result, "unitReturn": unit if (price and unit is not None) else None,
                       "homeScore": res.get("home"), "awayScore": res.get("away")})
    return graded


def summarize(graded: Iterable[dict], *, record_types: Collection[str] = ()) -> list[dict]:
    """Per (type, season) summary.

    A W-L record only means something when each pick was a one-sided choice the
    product actually made (a lean). Types listed in `record_types` get recordKind
    "record". Every other type gets recordKind "calibration": wins/losses/rate/units/
    ROI are None (both sides of a market, a fixed side, or several pooled lines would
    make the hit rate a product of the pick rule, not skill), and the row carries
    hits vs expectedHits plus Brier against the bucket's base rate instead.

    * wins/losses count half results as 0.5; pushes are excluded from the win rate.
    * Units and ROI use only PRICED picks (a logged market price). ROI denominator =
      units staked on priced picks that were not pushes (a half result stakes 0.5 on
      the decided half). Unpriced (model-only) picks contribute W-L-P but no units.
    * marketBaselineWinRate = mean de-vigged market probability of the picked side, i.e.
      the hit rate the market itself expected for these exact picks.
    * Brier is on the binary 'picked side won' outcome for full win/loss grades;
      market Brier uses the logged de-vigged probability."""
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for g in graded:
        key = (str(g.get("type")), str(g.get("season") or "unknown"))
        b = buckets.setdefault(key, {"type": key[0], "season": key[1], "graded": 0, "wins": 0.0, "losses": 0.0, "pushes": 0,
                                     "voids": 0, "ungradable": 0,
                                     "units": 0.0, "stake": 0.0, "priced": 0, "brier": [], "mbrier": [],
                                     "prices": [], "implied": [], "expected": 0.0, "outcomes": []})
        r = g["result"]
        if r in ("void", "ungradable"):
            b["voids" if r == "void" else "ungradable"] += 1
            continue
        b["graded"] += 1
        b["wins"] += {"win": 1, "half_win": 0.5}.get(r, 0)
        b["losses"] += {"loss": 1, "half_loss": 0.5}.get(r, 0)
        b["pushes"] += 1 if r == "push" else 0
        price = (g.get("market") or {}).get("decimalOdds")
        if price and g.get("unitReturn") is not None:
            b["priced"] += 1
            b["units"] += float(g["unitReturn"])
            b["stake"] += 0.0 if r == "push" else (0.5 if r in ("half_win", "half_loss") else 1.0)
            b["prices"].append(float(price))
        implied = (g.get("market") or {}).get("impliedProbability")
        if implied is not None:
            b["implied"].append(float(implied))
        p = g.get("modelProbability")
        if p is not None and r != "push":
            b["expected"] += float(p)
        if r in ("win", "loss"):
            y = 1.0 if r == "win" else 0.0
            b["outcomes"].append(y)
            if p is not None:
                b["brier"].append((float(p) - y) ** 2)
            mp = (g.get("market") or {}).get("impliedProbability")
            if mp is not None:
                b["mbrier"].append((float(mp) - y) ** 2)
    record_set = set(record_types)
    out = []
    for b in buckets.values():
        decided = b["wins"] + b["losses"]
        priced = b["priced"] > 0
        avg_price = sum(b["prices"]) / len(b["prices"]) if b["prices"] else None
        is_record = b["type"] in record_set
        ys = b["outcomes"]
        base = sum(ys) / len(ys) if ys else None
        out.append({
            "type": b["type"], "season": b["season"], "graded": b["graded"],
            "recordKind": "record" if is_record else "calibration",
            "wins": b["wins"] if is_record else None, "losses": b["losses"] if is_record else None,
            "pushes": b["pushes"],
            "voids": b["voids"], "ungradable": b["ungradable"],
            # Hide the rate on small samples: a 7-3 start is not a record.
            "winRateExPush": round(b["wins"] / decided, 4) if is_record and decided >= MIN_GRADED_FOR_RATE else None,
            "breakEvenRate": round(1.0 / avg_price, 4) if is_record and avg_price else None,
            "unitsAtStatedPrice": round(b["units"], 2) if is_record and priced else None,
            "roi": round(b["units"] / b["stake"], 4) if is_record and priced and b["stake"] >= MIN_GRADED_FOR_RATE else None,
            "marketBaselineWinRate": round(sum(b["implied"]) / len(b["implied"]), 4) if b["implied"] else None,
            "hits": b["wins"],
            "expectedHits": round(b["expected"], 2),
            "brier": round(sum(b["brier"]) / len(b["brier"]), 4) if b["brier"] else None,
            "baseRateBrier": round(base * (1 - base), 4) if base is not None else None,
            "marketBrier": round(sum(b["mbrier"]) / len(b["mbrier"]), 4) if b["mbrier"] else None,
        })
    return sorted(out, key=lambda s: (s["season"], s["type"]), reverse=True)


# ── season keys ───────────────────────────────────────────────────────────────

def season_cross_year(start_month: int) -> Callable[[int], str]:
    """Season label for leagues that span the new year (NFL Sep->Feb: start_month=8;
    NBA Oct->Jun: 9; European soccer Aug->May: 7). Returns e.g. '2025-26'."""
    def _season(date_key: int) -> str:
        y, m = date_key // 10000, (date_key // 100) % 100
        start = y if m >= start_month else y - 1
        return f"{start}-{str(start + 1)[-2:]}"
    return _season


def season_calendar(date_key: int) -> str:
    """Single-calendar-year seasons (MLB, WNBA, tennis, golf)."""
    return str(date_key // 10000)
