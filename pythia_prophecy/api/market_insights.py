"""Plain-language "how our market numbers held up" rows for the sports History tab.

The server writes the finished sentences so web, iOS and Android show identical,
jargon-free wording (no log-loss, Brier, calibration or MAE) and the wording can be
tuned without an app release. Every row says whether it is a simulated backtest or
the live record, and how many games it rests on.

Honesty rules (MARKET_ROADMAP.md §2e):
- No win rate for two-sided or fixed-side markets: those land near 50% (or at the
  pick rule's base rate) by construction. Live rows compare how many picks our
  probabilities expected to win with how many did.
- A record with wins and losses only for one-sided picks ("record" summary rows).
- Comparisons are against simple guesses and the betting market, never "beat".
"""
from __future__ import annotations

from typing import Any, Optional

LIVE_MIN_GRADED = 20        # below this a live row is noise, so it is not shown
LIVE_EARLY_GRADED = 100     # below this the row says it is early

SIMULATED = "simulated"
LIVE = "live"

INSIGHTS_NOTE = (
    "Simulated rows re-run the model on past games it was not trained on. Live rows are "
    "graded against the line shown when each pick was published."
)

MARKET_TITLES = {
    "moneyline": "Moneyline",
    "spread": "Point spread",
    "alt_spread": "Alternate spreads",
    "run_line": "Run line",
    "total": "Total (over/under)",
    "alt_total": "Alternate totals",
    "team_total_home": "Home team total",
    "team_total_away": "Away team total",
    "asian_handicap": "Handicap",
    "btts": "Both teams to score",
    "double_chance": "Double chance",
    "draw_no_bet": "Draw no bet",
    "correct_score": "Exact score",
    "1x2": "Match result",
}


def _row(title: str, headline: str, detail: str, basis: str, comparison: Optional[str] = None) -> dict[str, Any]:
    return {"title": title, "headline": headline, "comparison": comparison, "detail": detail, "basis": basis}


def _num(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # drop NaN


def _games(n: Any) -> str:
    count = int(_num(n) or 0)
    return f"{count:,} {'game' if count == 1 else 'games'}"


def _matches(n: Any) -> str:
    count = int(_num(n) or 0)
    return f"{count:,} {'match' if count == 1 else 'matches'}"


def _closeness(model: Any, base: Any, market: Any = None) -> Optional[str]:
    """Compare probability quality (lower log-loss = closer to the results) in words."""
    model, base, market = _num(model), _num(base), _num(market)
    if model is None or base is None:
        return None
    vs_base = (
        "closer to the results than the league average" if model < base
        else "not closer to the results than the league average"
    )
    if market is None:
        return f"Our probabilities were {vs_base}"
    vs_market = (
        "and closer than the betting market" if model < market
        else "but not as close as the betting market"
    )
    return f"Our probabilities were {vs_base}, {vs_market}"


# -- simulated (walk-forward) rows -------------------------------------------------

def basketball_rows(record: Any, league: Optional[str], season: Optional[str]) -> list[dict[str, Any]]:
    leagues = (record or {}).get("leagues") if isinstance(record, dict) else None
    if not isinstance(leagues, dict):
        return []
    key = (league or "").lower()
    lines = (leagues.get(key) or {}).get("walkForwardLines") or {}
    regular = {k: v for k, v in lines.items() if k != "postseason" and isinstance(v, dict)}
    if not regular:
        return []
    season_key = season if season in regular else sorted(regular)[-1]
    row = regular[season_key]
    detail = f"{key.upper()} {season_key} · {_games(row.get('n'))} · simulated"
    out = []
    margin = _num(row.get("marginMae"))
    if margin is not None:
        out.append(_row("Point spread", f"Our fair spread was off by {margin:.1f} points per game on average",
                        detail, SIMULATED))
    total = _num(row.get("totalMae"))
    if total is not None:
        out.append(_row("Total points", f"Our fair total was off by {total:.1f} points per game on average",
                        detail, SIMULATED))
    coverage = _num(row.get("marginCoverage80"))
    if coverage is not None:
        out.append(_row("Final-margin range",
                        f"The final margin landed inside our 80% range in {coverage:.0%} of games",
                        detail, SIMULATED, comparison="A well-sized range lands close to 80%"))
    return out


def soccer_rows(record: Any, season: Optional[str]) -> list[dict[str, Any]]:
    rows = (record or {}).get("allLeagues") if isinstance(record, dict) else None
    if not isinstance(rows, list) or not rows:
        return []
    row = next((r for r in rows if str(r.get("season")) == str(season)), None) or rows[-1]
    detail = f"Top 5 leagues {row.get('season')} · {_matches(row.get('n'))} · simulated"
    out = []
    ou = row.get("overUnder25") or {}
    text = _closeness(ou.get("modelLogLoss"), ou.get("baseRateLogLoss"), ou.get("marketLogLoss"))
    if text:
        out.append(_row("Over/under 2.5 goals", text, detail, SIMULATED))
    btts = row.get("btts") or {}
    text = _closeness(btts.get("modelLogLoss"), btts.get("baseRateLogLoss"))
    if text:
        out.append(_row("Both teams to score", text, detail, SIMULATED))
    top3, naive = _num(row.get("correctScoreTop3HitRate")), _num(row.get("naiveCorrectScoreTop3HitRate"))
    if top3 is not None:
        out.append(_row("Exact score", f"The real score was one of our top 3 in {top3:.0%} of matches", detail,
                        SIMULATED,
                        comparison=f"Always guessing 1-0, 1-1 and 2-1: {naive:.0%}" if naive is not None else None))
    goals = row.get("meanGoals") or {}
    predicted, actual = _num(goals.get("predicted")), _num(goals.get("actual"))
    if predicted is not None and actual is not None:
        out.append(_row("Goals per match", f"We expected {predicted:.1f} goals per match; there were {actual:.1f}",
                        detail, SIMULATED))
    return out


# -- live rows (the publish-time pick log) -----------------------------------------

def live_rows(summary: Any, season: Optional[str]) -> list[dict[str, Any]]:
    if not isinstance(summary, list):
        return []
    rows = [r for r in summary if isinstance(r, dict)]
    if season is not None and any(str(r.get("season")) == str(season) for r in rows):
        rows = [r for r in rows if str(r.get("season")) == str(season)]
    out = []
    for r in sorted(rows, key=lambda r: list(MARKET_TITLES).index(r.get("type")) if r.get("type") in MARKET_TITLES else 99):
        kind = r.get("type")
        graded = int(_num(r.get("graded")) or 0)
        if kind not in MARKET_TITLES or graded < LIVE_MIN_GRADED:
            continue
        decided = graded - int(_num(r.get("pushes")) or 0)
        early = " (early)" if graded < LIVE_EARLY_GRADED else ""
        detail = f"{r.get('season')} · {graded:,} graded{early} · live"
        if r.get("recordKind") == "record" and _num(r.get("wins")) is not None:
            wins, losses = _num(r.get("wins")) or 0, _num(r.get("losses")) or 0
            headline = f"Won {wins:g} of {wins + losses:g}"
            rate, even = _num(r.get("winRateExPush")), _num(r.get("breakEvenRate"))
            comparison = None
            if rate is not None and even is not None:
                comparison = f"{rate:.1%} won; {even:.1%} needed to break even at these prices"
            out.append(_row(MARKET_TITLES[kind], headline, detail, LIVE, comparison=comparison))
            continue
        expected, hits = _num(r.get("expectedHits")), _num(r.get("hits"))
        if expected is None or hits is None or decided <= 0:
            continue
        out.append(_row(MARKET_TITLES[kind],
                        f"Our probabilities said about {expected:.0f} of {decided:,} would win; {hits:g} did",
                        detail, LIVE))
    return out


def build_market_insights(
    sport: str,
    *,
    summary: Any,
    record: Any,
    league: Optional[str] = None,
    season: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Rows for one sport's History tab: simulated market checks first, then live."""
    rows: list[dict[str, Any]] = []
    if sport == "basketball":
        rows += basketball_rows(record, league, season)
    elif sport == "soccer":
        rows += soccer_rows(record, season)
    rows += live_rows(summary, season)
    return rows
