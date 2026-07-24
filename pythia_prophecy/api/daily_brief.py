"""Daily Brief assembly + rendering (Wave 1.2 of MARKET_ROADMAP.md).

One fixed-format morning surface: "N stock signals · boards posted · yesterday's
picks went W-L." The same brief dict powers the public /api/brief/today endpoint
(app/web landing surfaces) and the daily digest email (scripts/send_daily_brief.py).

Everything here is pure/offline — assembly takes already-fetched payloads, so
tests never touch the network. Section order is FIXED by design (Morning Brew
pattern: same time, same format, sub-10-second scan).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any, Optional

SPORT_ORDER = ["golf", "tennis", "basketball", "mlb", "football", "soccer", "olympics"]
SPORT_LABELS = {
    "golf": "Golf",
    "tennis": "Tennis",
    "basketball": "Basketball",
    "mlb": "Baseball",
    "football": "Football",
    "soccer": "Soccer",
    "olympics": "Olympics",
}

# hitStatus taxonomy from the season summaries: "Top Pick" = outright hit,
# "Top 3"/"Top 5" = placed (golf/tennis fields), anything else = miss.
_HIT = "Top Pick"
_PLACED = {"Top 3", "Top 5"}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def utc_date_keys(now: Optional[datetime] = None) -> tuple[int, int, str]:
    """(today_key, yesterday_key, iso_date) in UTC, keys as yyyymmdd ints."""
    now = now or datetime.now(timezone.utc)
    today = now.date()
    yesterday = today - timedelta(days=1)
    to_key = lambda d: d.year * 10000 + d.month * 100 + d.day  # noqa: E731
    return to_key(today), to_key(yesterday), today.isoformat()


def _board_headline(board: Any) -> str:
    away, home = _get(board, "awayTeam"), _get(board, "homeTeam")
    if away and home:
        return f"{away} @ {home}"
    return str(_get(board, "name") or _get(board, "tournament") or "Board")


def _board_pick(board: Any) -> tuple[Optional[str], Optional[float]]:
    """(pick, probability) — top prediction if present, else predictedWinner."""
    predictions = _get(board, "predictions") or []
    if predictions:
        top = predictions[0]
        return _get(top, "playerName"), _get(top, "winProbability")
    pick = _get(board, "predictedWinner")
    prob = _get(board, "homeWinProbability")
    if pick and prob is not None:
        # For team sports the probability belongs to the predicted side.
        if pick == _get(board, "awayTeam"):
            prob = _get(board, "awayWinProbability")
    return pick, prob


def summarize_sports(boards_response: Any, today_key: int, yesterday_key: int) -> list[dict]:
    """Per-sport brief rows from a SportsBoardsResponse (model or dict)."""
    out: list[dict] = []
    for sport in SPORT_ORDER:
        coll = _get(boards_response, sport)
        if coll is None:
            continue

        today_rows = []
        for board in (_get(coll, "upcoming") or []):
            scheduled = _get(board, "scheduledDate") or 0
            state = str(_get(board, "eventState") or "")
            if scheduled == today_key or state == "live":
                pick, prob = _board_pick(board)
                today_rows.append(
                    {"event": _board_headline(board), "pick": pick, "probability": prob}
                )

        wins = losses = placed = 0
        graded = []
        for board in (_get(coll, "backtests") or []):
            scheduled = _get(board, "scheduledDate") or _get(board, "latestDate") or 0
            if scheduled != yesterday_key:
                continue
            status = str(_get(board, "hitStatus") or "").strip()
            if status == _HIT:
                wins += 1
            elif status in _PLACED:
                placed += 1
                losses += 1  # honest scoring: placed is not a win
            else:
                losses += 1
            graded.append(
                {
                    "event": _board_headline(board) if _get(board, "homeTeam") else str(_get(board, "tournament") or ""),
                    "pick": _get(board, "predictedWinner"),
                    "actual": _get(board, "actualWinner"),
                    "hitStatus": status or "Miss",
                }
            )

        season = _get(coll, "seasonSummary")
        accuracy = _get(season, "topPickAccuracy") if season else None

        if today_rows or graded:
            out.append(
                {
                    "sport": sport,
                    "label": SPORT_LABELS.get(sport, sport.title()),
                    "today": today_rows[:3],
                    "today_total": len(today_rows),
                    "yesterday": {"wins": wins, "losses": losses, "placed": placed, "details": graded[:3]},
                    "season_top_pick_accuracy": accuracy,
                }
            )
    return out


def summarize_stocks(homepage_payload: Any, track_record: Any, model: str = "lstm_5d") -> Optional[dict]:
    """Signal counts + strongest names from divination's cached homepage batch,
    plus the public track-record summary. Either input may be None/unavailable."""
    signals = []
    counts = {"buy": 0, "hold": 0, "sell": 0}
    if homepage_payload and _get(homepage_payload, "available"):
        for row in _get(homepage_payload, "rows") or []:
            if _get(row, "model") != model:
                continue
            for pred in _get(row, "predictions") or []:
                signal = str(_get(pred, "signal") or "hold").lower()
                counts[signal] = counts.get(signal, 0) + 1
                signals.append(
                    {
                        "ticker": _get(pred, "ticker"),
                        "signal": signal,
                        "prob_up": _get(pred, "prob_up"),
                    }
                )
    # Strongest conviction first (distance from coin-flip), tolerant of nulls.
    signals.sort(key=lambda s: abs((s.get("prob_up") or 0.5) - 0.5), reverse=True)

    record = None
    summary = _get(track_record, "summary") if track_record and _get(track_record, "available") else None
    if summary is not None:
        record = {
            "hit_rate": _get(summary, "hit_rate"),
            "sample_size": _get(summary, "sample_size"),
        }

    if not signals and record is None:
        return None
    return {"model": model, "counts": counts, "top_signals": signals[:3], "track_record": record}


def build_brief(
    boards_response: Any,
    homepage_payload: Any,
    track_record: Any,
    now: Optional[datetime] = None,
) -> dict:
    today_key, yesterday_key, iso_date = utc_date_keys(now)
    sports = summarize_sports(boards_response, today_key, yesterday_key)
    stocks = summarize_stocks(homepage_payload, track_record)
    total_wins = sum(s["yesterday"]["wins"] for s in sports)
    total_losses = sum(s["yesterday"]["losses"] for s in sports)
    return {
        "date": iso_date,
        "date_key": today_key,
        "stocks": stocks,
        "sports": sports,
        "yesterday_record": {"wins": total_wins, "losses": total_losses},
    }


# ============================================================
# Rendering (fixed format — same order every day)
# ============================================================

def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return ""
    value = value * 100 if value <= 1 else value
    return f"{value:.0f}%"


def brief_subject(brief: dict) -> str:
    bits = []
    stocks = brief.get("stocks")
    if stocks and stocks["counts"]["buy"] + stocks["counts"]["sell"] > 0:
        bits.append(f"{stocks['counts']['buy'] + stocks['counts']['sell']} stock signals")
    live_sports = [s["label"] for s in brief.get("sports", []) if s["today_total"]]
    if live_sports:
        bits.append(f"{', '.join(live_sports[:3])} boards live")
    record = brief.get("yesterday_record") or {}
    if (record.get("wins", 0) + record.get("losses", 0)) > 0:
        bits.append(f"yesterday {record['wins']}-{record['losses']}")
    detail = " · ".join(bits) if bits else "your boards are ready"
    return f"Daily Brief — {detail}"


def render_brief_text(brief: dict, frontend_url: str) -> str:
    lines = [f"SeekingBeta.AI Daily Brief — {brief['date']}", ""]
    stocks = brief.get("stocks")
    if stocks:
        counts = stocks["counts"]
        lines.append(f"STOCKS ({stocks['model']}): {counts['buy']} buy · {counts['hold']} hold · {counts['sell']} sell")
        for sig in stocks["top_signals"]:
            lines.append(f"  {sig['ticker']}: {sig['signal'].upper()} ({_fmt_pct(sig['prob_up'])} up)")
        record = stocks.get("track_record")
        if record and record.get("hit_rate") is not None:
            lines.append(
                f"  Live record: {_fmt_pct(record['hit_rate'])} hit rate over {record['sample_size']} graded signals (net of costs)"
            )
        lines.append("")
    for sport in brief.get("sports", []):
        yesterday = sport["yesterday"]
        parts = []
        if sport["today_total"]:
            parts.append(f"{sport['today_total']} board(s) today")
        if yesterday["wins"] + yesterday["losses"] > 0:
            parts.append(f"yesterday {yesterday['wins']}-{yesterday['losses']}")
        lines.append(f"{sport['label'].upper()}: {' · '.join(parts)}")
        for row in sport["today"]:
            prob = f" ({_fmt_pct(row['probability'])})" if row.get("probability") is not None else ""
            lines.append(f"  Today: {row['event']} — pick {row['pick']}{prob}")
        for row in yesterday["details"]:
            actual = f" (actual: {row['actual']})" if row.get("actual") else ""
            lines.append(f"  Graded: {row['event']} — {row['pick']} -> {row['hitStatus']}{actual}")
        lines.append("")
    lines.append(f"Boards: {frontend_url}/sports  ·  Dashboard: {frontend_url}/dashboard")
    lines.append("Research only. We don't place trades or bets. Losses are published like wins.")
    return "\n".join(lines)


def render_brief_html_body(brief: dict, frontend_url: str) -> str:
    """Inner HTML for email_service._render_email_html's card."""
    parts: list[str] = []
    stocks = brief.get("stocks")
    if stocks:
        counts = stocks["counts"]
        parts.append(
            f"<p><strong>Stocks</strong> ({escape(stocks['model'])}): "
            f"{counts['buy']} buy · {counts['hold']} hold · {counts['sell']} sell</p>"
        )
        if stocks["top_signals"]:
            rows = "".join(
                f"<li>{escape(str(s['ticker']))}: <strong>{escape(s['signal'].upper())}</strong>"
                f" ({_fmt_pct(s['prob_up'])} up)</li>"
                for s in stocks["top_signals"]
            )
            parts.append(f"<ul>{rows}</ul>")
        record = stocks.get("track_record")
        if record and record.get("hit_rate") is not None:
            parts.append(
                f'<p class="muted">Live record: {_fmt_pct(record["hit_rate"])} hit rate over '
                f"{record['sample_size']} graded signals, net of costs.</p>"
            )
    for sport in brief.get("sports", []):
        yesterday = sport["yesterday"]
        header_bits = []
        if sport["today_total"]:
            header_bits.append(f"{sport['today_total']} board(s) today")
        if yesterday["wins"] + yesterday["losses"] > 0:
            header_bits.append(f"yesterday {yesterday['wins']}-{yesterday['losses']}")
        parts.append(f"<p><strong>{escape(sport['label'])}</strong>: {escape(' · '.join(header_bits))}</p>")
        items = []
        for row in sport["today"]:
            prob = f" ({_fmt_pct(row['probability'])})" if row.get("probability") is not None else ""
            items.append(f"<li>Today: {escape(str(row['event']))} — pick {escape(str(row['pick']))}{prob}</li>")
        for row in yesterday["details"]:
            actual = f" (actual: {escape(str(row['actual']))})" if row.get("actual") else ""
            items.append(
                f"<li>Graded: {escape(str(row['event']))} — {escape(str(row['pick'] or '—'))} → "
                f"<strong>{escape(row['hitStatus'])}</strong>{actual}</li>"
            )
        if items:
            parts.append(f"<ul>{''.join(items)}</ul>")
    parts.append(
        '<p class="muted">Research only. We don&rsquo;t place trades or bets. '
        "Losses are published like wins.</p>"
    )
    return "\n".join(parts)
