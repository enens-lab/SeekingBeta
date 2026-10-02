"""NFL markets centered on the posted nflverse line: moneyline, spread, total, team totals.

Measured facts this module is built around (season walk-forward 2022-2025, nflverse):
  * No model beats the closing market: de-vigged moneyline Brier 0.2105 vs our best
    model view ~0.22. So the HEADLINE win probability is the de-vigged moneyline, and
    every spread/total probability is centered on the posted line and its price.
  * NFL margins pile up on key numbers: the home margin is exactly +3 in 7.4% of games
    and -3 in 7.5% (a discretized normal says ~2.7%), +7 in 4.0%. A normal therefore
    understates pushes at whole lines by a factor of 2-3. The margin distribution here
    is a discretized normal with symmetric KEY-NUMBER WEIGHTS on |margin| (raked on
    prior seasons so the weighted mixture reproduces the observed |margin| frequencies),
    shifted so P(home covers the posted line, pushes excluded) equals the de-vigged
    spread price. Walk-forward it reproduces the +/-3 and +/-7 margin masses within
    0.7 pp and gives an alt-line ladder Brier of 0.1945 (normal: 0.1958).
  * Totals: a discretized normal (sigma ~13, fit on prior seasons), re-centered on the
    de-vigged over price. Ladder Brier 0.2022; key-number weights did not help.
  * Team totals: (total +/- spread) / 2 with x.5 lines. They carry no information
    beyond the posted lines (Brier equals a constant), so they are calibration-only.

Nothing here is a pick: market_pick(..., show_edge=False) always, and confidence tiers
are dropped (no sport-specific calibrated cut points exist; never on a market-anchored
probability). The ridge "model fair spread" appears only as modelLine.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np

from sports import market_log
from sports import markets as mk

SPORT_KEY = "football"
LINE_SOURCE = "nflverse"
ATTRIBUTION = "Lines: nflverse (CC-BY-4.0)"
SEASON_OF = market_log.season_cross_year(8)

MARGIN_LO, MARGIN_HI = -90, 90
KEY_MAX = 40                      # key-number weights for |margin| 0..40, 1.0 beyond
LADDER_OFFSETS = (-7, -3, 3, 7)   # alt lines relative to the posted line
DEFAULT_SPREAD_SIGMA = 12.7
DEFAULT_TOTAL_SIGMA = 13.0
DEFAULT_TEAM_TOTAL_SIGMA = 9.1


# ── odds ──────────────────────────────────────────────────────────────────────

def _finite(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def devig_two_way(american_a: Any, american_b: Any) -> Optional[tuple[float, float]]:
    """Proportional de-vig of a two-way American price pair; None if either is missing."""
    a, b = _finite(american_a), _finite(american_b)
    if a is None or b is None or a == 0 or b == 0:
        return None
    pa, pb = mk.devig([mk.american_to_decimal(a), mk.american_to_decimal(b)])
    return float(pa), float(pb)


def market_snapshot(*, line: Optional[float], american: Any, implied: Optional[float],
                    captured_at: Optional[str]) -> dict[str, Any]:
    """The SportsMarketPrice block: the sportsbook price as posted, frozen at publish."""
    a = _finite(american)
    return {
        "line": line,
        "americanOdds": int(a) if a is not None and a != 0 else None,
        "decimalOdds": round(mk.american_to_decimal(a), 4) if a is not None and a != 0 else None,
        "impliedProbability": round(float(implied), 4) if implied is not None else None,
        "source": LINE_SOURCE,
        "capturedAt": captured_at,
    }


# ── margin distribution with key numbers ──────────────────────────────────────

def margin_values() -> np.ndarray:
    return np.arange(MARGIN_LO, MARGIN_HI + 1, dtype=float)


def _base_pmf(center: float, sigma: float) -> np.ndarray:
    _, p = mk.integer_normal_pmf(center, sigma, MARGIN_LO, MARGIN_HI)
    return p


def _weight_vector(key_weights: Optional[Iterable[float]]) -> np.ndarray:
    values = np.abs(margin_values()).astype(int)
    if key_weights is None:
        return np.ones(len(values))
    w = np.asarray(list(key_weights), float)
    out = np.ones(len(values))
    inside = values < len(w)
    out[inside] = w[values[inside]]
    return out


def margin_pmf(center: float, sigma: float, key_weights: Optional[Iterable[float]]) -> np.ndarray:
    """P(home margin = m) on margin_values(): discretized normal x key-number weights."""
    p = _base_pmf(center, sigma) * _weight_vector(key_weights)
    return p / p.sum()


def fit_key_weights(margins: Iterable[float], centers: Iterable[float], sigma: float, *,
                    k_max: int = KEY_MAX, iters: int = 30, alpha: float = 2.0) -> list[float]:
    """Rake symmetric weights w(|m|) so the mixture of weighted pmfs centered on each
    game's line reproduces the observed |margin| frequencies (alpha = additive smoothing)."""
    m = np.abs(np.asarray(list(margins), float)).astype(int)
    c = np.asarray(list(centers), float)
    obs = np.array([(m == k).sum() for k in range(k_max + 1)], float)
    base = np.stack([_base_pmf(ci, sigma) for ci in c])
    absv = np.abs(margin_values()).astype(int)
    w = np.ones(k_max + 1)
    for _ in range(iters):
        wv = np.ones(len(absv))
        inside = absv <= k_max
        wv[inside] = w[absv[inside]]
        p = base * wv
        p /= p.sum(axis=1, keepdims=True)
        exp = np.array([p[:, absv == k].sum() for k in range(k_max + 1)])
        w = w * (obs + alpha) / (exp + alpha)
    return [round(float(x), 5) for x in w]


def side_prices(values: np.ndarray, probs: np.ndarray, side: str, line: float) -> mk.LinePrices:
    """Outcome probabilities for `side` at `line` (that side's view) from the HOME
    margin pmf, via the shared markets.spread_prices_normal(key_number_pmf=...)."""
    vals = values if side == "home" else -values
    mean = float(np.dot(vals, probs))
    # The shared helper shifts the pmf by round(expected - mean); passing the pmf's own
    # mean makes that shift exactly 0, so the distribution is used as given.
    return mk.spread_prices_normal(mean, 1.0, float(line), key_number_pmf=(vals, probs))


def solve_margin_center(spread_line: float, target_home_cover: float, sigma: float,
                        key_weights: Optional[Iterable[float]], *, span: float = 6.0) -> float:
    """Center such that P(home covers -spread_line, pushes excluded) == target."""
    weights = list(key_weights) if key_weights is not None else None
    values = margin_values()
    lo, hi = spread_line - span, spread_line + span
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        p = side_prices(values, margin_pmf(mid, sigma, weights), "home", -spread_line).win_ex_push
        if p < target_home_cover:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def home_margin_distribution(spread_line: float, home_spread_odds: Any, away_spread_odds: Any,
                             ladder: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """(values, probs, target) for the home margin, anchored on the posted spread price.
    nflverse spread_line is positive when the HOME team is favoured."""
    devig = devig_two_way(home_spread_odds, away_spread_odds)
    target = devig[0] if devig else 0.5
    sigma = float(ladder.get("spread_sigma") or DEFAULT_SPREAD_SIGMA)
    weights = ladder.get("key_weights")
    center = solve_margin_center(float(spread_line), target, sigma, weights)
    return margin_values(), margin_pmf(center, sigma, weights), target


def solve_total_center(total_line: float, target_over: float, sigma: float, *, span: float = 8.0) -> float:
    lo, hi = total_line - span, total_line + span
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        p = mk.total_prices_normal(mid, sigma, total_line, "over").win_ex_push
        if p < target_over:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def total_center(total_line: float, over_odds: Any, under_odds: Any, ladder: dict[str, Any]) -> tuple[float, float]:
    devig = devig_two_way(over_odds, under_odds)
    target = devig[0] if devig else 0.5
    sigma = float(ladder.get("total_sigma") or DEFAULT_TOTAL_SIGMA)
    return solve_total_center(float(total_line), target, sigma), target


def implied_team_totals(spread_line: float, total_line: float) -> tuple[float, float]:
    """(home, away) implied points: (total + spread)/2 and (total - spread)/2, with the
    nflverse convention spread_line > 0 = home favoured."""
    return (float(total_line) + float(spread_line)) / 2.0, (float(total_line) - float(spread_line)) / 2.0


def fit_ladder_params(margins: Iterable[float], spread_lines: Iterable[float], totals: Iterable[float],
                      total_lines: Iterable[float], home_points: Iterable[float], away_points: Iterable[float]) -> dict[str, Any]:
    """Fit the ladder shape on completed games (all arrays aligned, lines present)."""
    m = np.asarray(list(margins), float); s = np.asarray(list(spread_lines), float)
    t = np.asarray(list(totals), float); tl = np.asarray(list(total_lines), float)
    hp = np.asarray(list(home_points), float); ap = np.asarray(list(away_points), float)
    spread_sigma = float(np.std(m - s, ddof=1))
    total_sigma = float(np.std(t - tl, ddof=1))
    ht, at = (tl + s) / 2.0, (tl - s) / 2.0
    team_sigma = float(np.std(np.r_[hp - ht, ap - at], ddof=1))
    return {
        "spread_sigma": round(spread_sigma, 4),
        "key_weights": fit_key_weights(m, s, spread_sigma),
        "total_sigma": round(total_sigma, 4),
        "team_total_sigma": round(team_sigma, 4),
        "n_games": int(len(m)),
    }


# ── board markets ─────────────────────────────────────────────────────────────

def _fmt_line(line: float) -> str:
    if abs(line) < 1e-9:
        return "PK"
    text = f"{line:+.1f}"
    return text[:-2] if text.endswith(".0") else text


def _is_whole(line: float) -> bool:
    return abs(line - round(line)) < 1e-9


def _half_up(x: float) -> float:
    return round(float(x) * 2.0) / 2.0


def _finalize(entry: dict, *, selection: Optional[str], basis: str, model_version: str,
              attribution: Optional[str], push_ok: bool) -> dict:
    if selection:
        entry["marketId"] = f"{entry['marketId']}:{selection}"
    entry["basis"] = basis
    entry["modelVersion"] = model_version
    if attribution:
        entry["attribution"] = attribution
    # No calibrated cut points exist for these markets (and tiers are never shown on
    # market-anchored probabilities), so no confidence tier is published.
    entry.pop("confidenceTier", None)
    entry.pop("edge", None)
    if not push_ok:
        entry.pop("pushProbability", None)
        outcome = entry.get("outcomeProbabilities")
        if outcome and outcome.get("push", 0) > 0:
            entry.pop("outcomeProbabilities", None)
    return entry


def build_board_markets(*, board_id: str, game: dict[str, Any], ladder: dict[str, Any],
                        model: Optional[dict[str, Any]], published_at: str, captured_at: Optional[str],
                        model_version: str) -> list[dict[str, Any]]:
    """markets[] for one upcoming game.

    game: home_team, away_team, spread_line, total_line, home/away_moneyline,
          home/away_spread_odds, over_odds, under_odds (nflverse columns; NaN = not posted)
    model: optional {home_win_probability, margin, total} from the model view
    ladder: fitted ladder params (+ "push_validated" / "total_push_validated" gates)
    """
    home, away = str(game.get("home_team")), str(game.get("away_team"))
    out: list[dict[str, Any]] = []
    spread_push_ok = bool(ladder.get("push_validated"))
    total_push_ok = bool(ladder.get("total_push_validated"))

    def add(entry: dict, *, selection: Optional[str], basis: str, attribution: Optional[str], push_ok: bool) -> None:
        out.append(_finalize(entry, selection=selection, basis=basis, model_version=model_version,
                             attribution=attribution, push_ok=push_ok))

    # 1) moneyline: the MODEL VIEW (the headline itself is the de-vigged line, on the board).
    ml = devig_two_way(game.get("home_moneyline"), game.get("away_moneyline"))
    if model and model.get("home_win_probability") is not None:
        p_home = float(model["home_win_probability"])
        side = "home" if p_home >= 0.5 else "away"
        p_side = p_home if side == "home" else 1.0 - p_home
        team = home if side == "home" else away
        snapshot = None
        if ml:
            snapshot = market_snapshot(line=None, american=game.get(f"{side}_moneyline"),
                                       implied=ml[0] if side == "home" else ml[1], captured_at=captured_at)
        entry = mk.market_pick(board_id=board_id, market_type="moneyline", side=side, label=f"Model view: {team}",
                               prices=mk.LinePrices(win=p_side, half_win=0.0, push=0.0, half_loss=0.0, loss=1.0 - p_side),
                               line=None, model_line=None, market=snapshot, published_at=published_at, show_edge=False)
        add(entry, selection=None, basis="model", attribution=ATTRIBUTION if snapshot else None, push_ok=True)

    spread_line = _finite(game.get("spread_line"))
    total_line = _finite(game.get("total_line"))
    model_margin = _finite((model or {}).get("margin"))
    model_total = _finite((model or {}).get("total"))

    # 2) spread at the posted line (both sides) + alt ladder for the favourite.
    if spread_line is not None:
        values, probs, target = home_margin_distribution(spread_line, game.get("home_spread_odds"),
                                                         game.get("away_spread_odds"), ladder)
        spread_devig = devig_two_way(game.get("home_spread_odds"), game.get("away_spread_odds"))
        for side, team, line in (("home", home, 0.0 - spread_line), ("away", away, spread_line + 0.0)):
            # A posted whole line is always shown; without a validated push model its
            # push numbers are stripped in _finalize (cover probability stays ex-push).
            prices = side_prices(values, probs, side, line)
            implied = None
            if spread_devig:
                implied = spread_devig[0] if side == "home" else spread_devig[1]
            snapshot = market_snapshot(line=line, american=game.get(f"{side}_spread_odds"), implied=implied,
                                       captured_at=captured_at)
            fair = None
            if model_margin is not None:
                fair = _half_up(-model_margin if side == "home" else model_margin)
            entry = mk.market_pick(board_id=board_id, market_type="spread", side=side, label=f"{team} {_fmt_line(line)}",
                                   prices=prices, line=line, model_line=fair, market=snapshot,
                                   published_at=published_at, show_edge=False)
            add(entry, selection=side, basis="market", attribution=ATTRIBUTION, push_ok=spread_push_ok)
        if spread_line > 0:
            fav_side, fav_team = "home", home
        elif spread_line < 0:
            fav_side, fav_team = "away", away
        else:
            fav_side = "home" if (ml is None or ml[0] >= ml[1]) else "away"
            fav_team = home if fav_side == "home" else away
        fav_line = -abs(spread_line)
        for k in LADDER_OFFSETS:
            line = fav_line + k
            if _is_whole(line) and not spread_push_ok:
                continue  # no push-dependent rungs until the push model is validated
            prices = side_prices(values, probs, fav_side, line)
            entry = mk.market_pick(board_id=board_id, market_type="spread", side=fav_side,
                                   label=f"{fav_team} {_fmt_line(line)} (alt)", prices=prices, line=line,
                                   model_line=None, market=None, published_at=published_at, show_edge=False)
            add(entry, selection=f"{fav_side}:alt{k:+d}", basis="market_implied", attribution=ATTRIBUTION,
                push_ok=spread_push_ok)

    # 3) total at the posted line (over/under) + alt-total ladder (over).
    if total_line is not None:
        center, _ = total_center(total_line, game.get("over_odds"), game.get("under_odds"), ladder)
        sigma = float(ladder.get("total_sigma") or DEFAULT_TOTAL_SIGMA)
        total_devig = devig_two_way(game.get("over_odds"), game.get("under_odds"))
        for side in ("over", "under"):
            prices = mk.total_prices_normal(center, sigma, total_line, side)
            implied = None
            if total_devig:
                implied = total_devig[0] if side == "over" else total_devig[1]
            snapshot = market_snapshot(line=total_line, american=game.get(f"{side}_odds"), implied=implied,
                                       captured_at=captured_at)
            entry = mk.market_pick(board_id=board_id, market_type="total", side=side,
                                   label=f"{side.title()} {_fmt_line(total_line).lstrip('+')}", prices=prices,
                                   line=total_line, model_line=_half_up(model_total) if model_total is not None else None,
                                   market=snapshot, published_at=published_at, show_edge=False)
            add(entry, selection=side, basis="market", attribution=ATTRIBUTION, push_ok=total_push_ok)
        for k in LADDER_OFFSETS:
            line = total_line + k
            if _is_whole(line) and not total_push_ok:
                continue
            prices = mk.total_prices_normal(center, sigma, line, "over")
            entry = mk.market_pick(board_id=board_id, market_type="total", side="over",
                                   label=f"Over {_fmt_line(line).lstrip('+')} (alt)", prices=prices, line=line,
                                   model_line=None, market=None, published_at=published_at, show_edge=False)
            add(entry, selection=f"over:alt{k:+d}", basis="market_implied", attribution=ATTRIBUTION, push_ok=total_push_ok)

    # 4) team totals from the posted spread + total, x.5 lines (no push).
    if spread_line is not None and total_line is not None:
        tt_sigma = float(ladder.get("team_total_sigma") or DEFAULT_TEAM_TOTAL_SIGMA)
        for side, team, implied_pts in zip(("home", "away"), (home, away), implied_team_totals(spread_line, total_line)):
            line = math.floor(implied_pts) + 0.5
            market_type = f"team_total_{side}"
            for ou in ("over", "under"):
                prices = mk.total_prices_normal(implied_pts, tt_sigma, line, ou)
                # The implied points are the MARKET's number, not our model's, so they
                # are not published as modelLine (the x.5 line sits next to them).
                entry = mk.market_pick(board_id=board_id, market_type=market_type, side=ou,
                                       label=f"{team} team total {ou.title()} {line:g}", prices=prices, line=line,
                                       model_line=None, market=None, published_at=published_at,
                                       show_edge=False)
                add(entry, selection=ou, basis="market_implied", attribution=ATTRIBUTION, push_ok=True)
    return out


# ── summaries of the logged record ────────────────────────────────────────────

def restore_labels(record: dict[str, Any]) -> dict[str, Any]:
    """Fill basis / attribution on a logged NFL record when the log did not keep them
    (market_log.picks_from_boards copies a fixed key list). Deterministic from the
    marketId layout build_board_markets writes; values already present are kept."""
    mid = str(record.get("marketId") or "")
    rtype = str(record.get("type") or "")
    if not record.get("basis"):
        if rtype == "moneyline":
            record["basis"] = "model"
        elif ":alt" in mid or rtype.startswith("team_total_"):
            record["basis"] = "market_implied"
        elif rtype in ("spread", "total"):
            record["basis"] = "market"
    if not record.get("attribution") and (record.get("market") or record.get("basis") in ("market", "market_implied")):
        record["attribution"] = ATTRIBUTION
    return record

def summary_type(record: dict[str, Any]) -> Optional[str]:
    """Summary bucket for one graded record, or None to leave it out of the summary.

    Calibration-type markets are logged for both sides (that is what users saw) but
    the two sides of one line mirror each other exactly (home win = away loss), so the
    summary keeps one canonical side and separates alt-ladder rungs from posted lines:
      moneyline (model view) | spread (home side, posted line) | alt_spread | total
      (over, posted line) | alt_total | team_total_home / team_total_away (over)."""
    mid = str(record.get("marketId") or "")
    rtype = str(record.get("type") or "")
    side = str(record.get("side") or "")
    is_alt = ":alt" in mid
    if rtype == "moneyline":
        return "moneyline"
    if rtype == "spread":
        if is_alt:
            return "alt_spread"
        return "spread" if side == "home" else None
    if rtype == "total":
        if is_alt:
            return "alt_total"
        return "total" if side == "over" else None
    if rtype.startswith("team_total_"):
        return rtype if side == "over" else None
    return None


def summarize_graded(graded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for g in graded:
        bucket = summary_type(g)
        if bucket:
            rows.append({**g, "type": bucket})
    return market_log.summarize(rows)
