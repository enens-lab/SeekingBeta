"""Weekly "Receipts" digest (MARKET_ROADMAP.md Wave 1.3).

The trust play: publish the graded record every week, losses included, with
sample sizes attached to every number. Research finding driving this: the
sports-pick market has a documented transparency crisis (reviewers estimate ~9
of 10 handicappers inflate records), so honestly reporting a bad week is the
rarest and most-praised behavior in the category.

Honesty rules baked into this module, not left to the copywriter:

* Only genuinely graded boards inside the window are counted. A sport whose
  grading pipeline has not caught up simply reports no week, never a zero-loss
  week (an empty record must never read as a perfect record).
* "Placed" finishes (Top 3 / Top 5 in ranked fields) count as LOSSES on the
  headline W-L and are disclosed separately. Only an outright hit is a win.
* Every season figure carries its sample size and a "graded through" date, so a
  stale pipeline is visible to the reader instead of being hidden by an average.
* The stock record is labeled with its own as-of date and is never presented as
  a weekly result.

Pure/offline: assembly takes already-fetched payloads so tests never hit the
network. Shares board helpers with the Daily Brief.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any, Optional

from .daily_brief import (
    SPORT_LABELS,
    SPORT_ORDER,
    _board_headline,
    _board_pick,
    _fmt_pct,
    _get,
)

_HIT = "Top Pick"
_PLACED = {"Top 3", "Top 5"}


def _date_key(day) -> int:
    return day.year * 10000 + day.month * 100 + day.day


def week_window(now: Optional[datetime] = None) -> tuple[int, int, str, str]:
    """(start_key, end_key, start_iso, end_iso) for the 7 days ending yesterday.

    Yesterday is the last day that can be fully graded, so a Monday send covers
    the completed week behind it.
    """
    now = now or datetime.now(timezone.utc)
    end = now.date() - timedelta(days=1)
    start = end - timedelta(days=6)
    return _date_key(start), _date_key(end), start.isoformat(), end.isoformat()


def _board_date(board: Any) -> int:
    for name in ("scheduledDate", "latestDate"):
        value = _get(board, name)
        if value:
            return int(value)
    return 0


def _season_block(season: Any) -> Optional[dict]:
    """Season-to-date accuracy, only when it rests on a real sample."""
    if season is None:
        return None
    sample = _get(season, "sampleSize") or 0
    if not sample:
        return None
    return {
        "sample_size": sample,
        "top_pick_hits": _get(season, "topPickHits") or 0,
        "top_pick_accuracy": _get(season, "topPickAccuracy"),
        "top3_accuracy": _get(season, "top3Accuracy"),
        "top5_accuracy": _get(season, "top5Accuracy"),
        "year": _get(season, "year"),
    }


def summarize_week(boards_response: Any, start_key: int, end_key: int) -> list[dict]:
    """Per-sport weekly + season records. Sports with neither are omitted."""
    out: list[dict] = []
    for sport in SPORT_ORDER:
        coll = _get(boards_response, sport)
        if coll is None:
            continue

        backtests = _get(coll, "backtests") or []
        wins = losses = placed = 0
        graded: list[dict] = []
        newest_graded = 0

        for board in backtests:
            key = _board_date(board)
            if key:
                newest_graded = max(newest_graded, key)
            if not key or not (start_key <= key <= end_key):
                continue
            status = str(_get(board, "hitStatus") or "").strip()
            if status == _HIT:
                wins += 1
            else:
                # Placed finishes are disclosed, but they are not wins.
                losses += 1
                if status in _PLACED:
                    placed += 1
            graded.append(
                {
                    "event": _board_headline(board)
                    if _get(board, "homeTeam")
                    else str(_get(board, "tournament") or ""),
                    "pick": _get(board, "predictedWinner"),
                    "actual": _get(board, "actualWinner"),
                    "hitStatus": status or "Miss",
                    "date_key": key,
                }
            )

        season = _season_block(_get(coll, "seasonSummary"))
        if not graded and season is None:
            continue

        graded.sort(key=lambda row: row["date_key"])
        out.append(
            {
                "sport": sport,
                "label": SPORT_LABELS.get(sport, sport.title()),
                "week": {
                    "wins": wins,
                    "losses": losses,
                    "placed": placed,
                    "count": len(graded),
                    "details": graded,
                },
                "season": season,
                "graded_through": newest_graded or None,
            }
        )
    return out


def pick_teaser(boards_response: Any, today_key: int) -> tuple[Optional[dict], int]:
    """(teaser, remaining_boards). One fully revealed upcoming pick plus a count
    of everything else live right now.

    The count is deliberately a plain board count today. Once the Pro top-list
    ships (Wave 1.5) this becomes the locked-pick count; until then, claiming
    picks are "locked" would be false.
    """
    candidates: list[tuple[float, dict]] = []
    total = 0
    for sport in SPORT_ORDER:
        coll = _get(boards_response, sport)
        if coll is None:
            continue
        for board in (_get(coll, "upcoming") or []):
            key = _get(board, "scheduledDate") or 0
            state = str(_get(board, "eventState") or "")
            if key and key < today_key and state != "live":
                continue
            total += 1
            pick, prob = _board_pick(board)
            if not pick:
                continue
            candidates.append(
                (
                    float(prob or 0.0),
                    {
                        "sport": sport,
                        "label": SPORT_LABELS.get(sport, sport.title()),
                        "event": _board_headline(board),
                        "pick": pick,
                        "probability": prob,
                    },
                )
            )
    if not candidates:
        return None, total
    # Highest-confidence upcoming pick, revealed in full.
    candidates.sort(key=lambda row: row[0], reverse=True)
    teaser = candidates[0][1]
    return teaser, max(total - 1, 0)


def summarize_stock_record(track_record: Any) -> Optional[dict]:
    if not track_record or not _get(track_record, "available"):
        return None
    summary = _get(track_record, "summary")
    if summary is None:
        return None
    hit_rate = _get(summary, "hit_rate")
    sample = _get(summary, "sample_size") or 0
    if hit_rate is None or not sample:
        return None
    return {
        "hit_rate": hit_rate,
        "sample_size": sample,
        "as_of": _get(summary, "as_of"),
        "total_return_net": _get(summary, "total_return_net"),
        "benchmark_return": _get(summary, "benchmark_return"),
    }


def build_receipts(
    boards_response: Any,
    track_record: Any,
    now: Optional[datetime] = None,
) -> dict:
    now = now or datetime.now(timezone.utc)
    start_key, end_key, start_iso, end_iso = week_window(now)
    today_key = _date_key(now.date())

    sports = summarize_week(boards_response, start_key, end_key)
    teaser, remaining = pick_teaser(boards_response, today_key)
    stocks = summarize_stock_record(track_record)

    wins = sum(s["week"]["wins"] for s in sports)
    losses = sum(s["week"]["losses"] for s in sports)
    graded_count = sum(s["week"]["count"] for s in sports)

    return {
        "week_start": start_iso,
        "week_end": end_iso,
        "sports": sports,
        "week_record": {"wins": wins, "losses": losses, "graded": graded_count},
        "stocks": stocks,
        "teaser": teaser,
        "remaining_boards": remaining,
    }


def has_content(receipts: dict) -> bool:
    """Never send an empty "here are our receipts" email: it reads as either a
    broken pipeline or a hidden record, and both cost more trust than skipping."""
    if receipts["week_record"]["graded"] > 0:
        return True
    return any(s.get("season") for s in receipts.get("sports", []))


# ============================================================
# Rendering
# ============================================================

def _fmt_date_key(key: Optional[int]) -> str:
    if not key:
        return ""
    text = str(key)
    if len(text) != 8:
        return ""
    try:
        return datetime.strptime(text, "%Y%m%d").strftime("%b %-d")
    except ValueError:
        return ""


def receipts_subject(receipts: dict) -> str:
    record = receipts["week_record"]
    if record["graded"]:
        return (
            f"Your receipts: we went {record['wins']}-{record['losses']} "
            f"on {record['graded']} graded boards"
        )
    return "Your receipts: season records, losses included"


def _season_sentence(season: dict) -> str:
    bits = [
        f"top pick {season['top_pick_hits']}/{season['sample_size']} "
        f"({_fmt_pct(season['top_pick_accuracy'])})"
    ]
    if season.get("top3_accuracy") is not None:
        bits.append(f"top 3 {_fmt_pct(season['top3_accuracy'])}")
    if season.get("top5_accuracy") is not None:
        bits.append(f"top 5 {_fmt_pct(season['top5_accuracy'])}")
    return " · ".join(bits)


def render_receipts_text(receipts: dict, frontend_url: str) -> str:
    record = receipts["week_record"]
    lines = [
        f"SeekingBeta.AI Receipts — {receipts['week_start']} to {receipts['week_end']}",
        "",
    ]
    if record["graded"]:
        lines.append(
            f"LAST WEEK: {record['wins']}-{record['losses']} across {record['graded']} graded boards."
        )
    else:
        lines.append("LAST WEEK: no boards finished grading in this window.")
    lines.append("")

    for sport in receipts["sports"]:
        week = sport["week"]
        header = f"{sport['label'].upper()}: "
        if week["count"]:
            header += f"{week['wins']}-{week['losses']} this week"
            if week["placed"]:
                header += f" ({week['placed']} placed top 3/5, counted as losses)"
        else:
            header += "nothing graded this week"
        lines.append(header)

        if sport["season"]:
            through = _fmt_date_key(sport["graded_through"])
            suffix = f", graded through {through}" if through else ""
            lines.append(f"  Season: {_season_sentence(sport['season'])}{suffix}")

        for row in week["details"][:5]:
            actual = f" (actual: {row['actual']})" if row.get("actual") else ""
            lines.append(
                f"  {_fmt_date_key(row['date_key'])}: {row['event']} — "
                f"{row['pick'] or '—'} -> {row['hitStatus']}{actual}"
            )
        if week["count"] > 5:
            lines.append(f"  ...and {week['count'] - 5} more, all in the public record.")
        lines.append("")

    stocks = receipts.get("stocks")
    if stocks:
        as_of = f" (as of {stocks['as_of']})" if stocks.get("as_of") else ""
        lines.append(
            f"STOCKS: {_fmt_pct(stocks['hit_rate'])} hit rate over "
            f"{stocks['sample_size']} graded signals, net of costs{as_of}."
        )
        lines.append("  This is the cumulative record, not a weekly number.")
        lines.append("")

    teaser = receipts.get("teaser")
    if teaser:
        prob = f" ({_fmt_pct(teaser['probability'])})" if teaser.get("probability") is not None else ""
        lines.append(
            f"ON THE BOARD NOW — {teaser['label']}: {teaser['event']}, "
            f"model likes {teaser['pick']}{prob}."
        )
        if receipts.get("remaining_boards"):
            lines.append(f"  {receipts['remaining_boards']} more boards are live: {frontend_url}/sports")
        lines.append("")

    lines.append(f"Full record: {frontend_url}/track-record  ·  Method: {frontend_url}/methodology")
    lines.append(
        "Every number above carries its sample size. We publish losses the same "
        "way we publish wins. Research only: we don't place trades or bets."
    )
    return "\n".join(lines)


def render_receipts_html_body(receipts: dict, frontend_url: str) -> str:
    record = receipts["week_record"]
    parts: list[str] = []

    if record["graded"]:
        parts.append(
            f"<p><strong>Last week: {record['wins']}-{record['losses']}</strong> "
            f"across {record['graded']} graded boards "
            f"({escape(receipts['week_start'])} to {escape(receipts['week_end'])}).</p>"
        )
    else:
        parts.append(
            f'<p><strong>Last week:</strong> no boards finished grading between '
            f"{escape(receipts['week_start'])} and {escape(receipts['week_end'])}.</p>"
        )

    for sport in receipts["sports"]:
        week = sport["week"]
        if week["count"]:
            headline = f"{week['wins']}-{week['losses']} this week"
            if week["placed"]:
                headline += f" ({week['placed']} placed top 3/5, counted as losses)"
        else:
            headline = "nothing graded this week"
        parts.append(f"<p><strong>{escape(sport['label'])}</strong>: {escape(headline)}</p>")

        if sport["season"]:
            through = _fmt_date_key(sport["graded_through"])
            suffix = f", graded through {through}" if through else ""
            parts.append(
                f'<p class="muted">Season: {escape(_season_sentence(sport["season"]))}'
                f"{escape(suffix)}</p>"
            )

        items = []
        for row in week["details"][:5]:
            actual = f" (actual: {escape(str(row['actual']))})" if row.get("actual") else ""
            items.append(
                f"<li>{escape(_fmt_date_key(row['date_key']))}: {escape(str(row['event']))} — "
                f"{escape(str(row['pick'] or '—'))} → <strong>{escape(row['hitStatus'])}</strong>{actual}</li>"
            )
        if items:
            parts.append(f"<ul>{''.join(items)}</ul>")
        if week["count"] > 5:
            parts.append(
                f'<p class="muted">…and {week["count"] - 5} more, all in the public record.</p>'
            )

    stocks = receipts.get("stocks")
    if stocks:
        as_of = f" (as of {escape(str(stocks['as_of']))})" if stocks.get("as_of") else ""
        parts.append(
            f"<p><strong>Stocks</strong>: {_fmt_pct(stocks['hit_rate'])} hit rate over "
            f"{stocks['sample_size']} graded signals, net of costs{as_of}. "
            '<span class="muted">Cumulative record, not a weekly number.</span></p>'
        )

    teaser = receipts.get("teaser")
    if teaser:
        prob = f" ({_fmt_pct(teaser['probability'])})" if teaser.get("probability") is not None else ""
        extra = (
            f' {receipts["remaining_boards"]} more boards are live now.'
            if receipts.get("remaining_boards")
            else ""
        )
        parts.append(
            f"<p><strong>On the board now</strong> — {escape(teaser['label'])}: "
            f"{escape(str(teaser['event']))}, model likes "
            f"<strong>{escape(str(teaser['pick']))}</strong>{prob}.{escape(extra)}</p>"
        )

    parts.append(
        '<p class="muted">Every number above carries its sample size. We publish '
        "losses the same way we publish wins. Research only: we don&rsquo;t place "
        "trades or bets.</p>"
    )
    return "\n".join(parts)
