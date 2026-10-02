"""Shared betting-market math for the sports boards: pricing, grading, de-vig.

Every sport's exporter turns its model output into the same `markets` block
(see pythia_prophecy/api/models.py::SportsMarketPick). This module holds the
sport-agnostic parts so spreads, totals and handicaps are priced and graded the
same way everywhere:

  * Pricing from a MARGIN or TOTAL distribution (NFL/NBA/MLB style): the model
    predicts a mean and we use a normal residual with an empirically fitted sigma.
    Whole lines carry real push mass (NFL lands exactly on 3 about 8.5% of the
    time at a 3-point spread, measured on nflverse 2020-2026), so for integer-valued
    sports the probabilities are computed on the discrete integer outcomes, not by
    pretending the margin is continuous.
  * Pricing from a SCORE MATRIX (soccer: Dixon-Coles / Poisson P[home=i, away=j]):
    exact probabilities for 1X2, double chance, draw-no-bet, BTTS, totals and Asian
    handicaps at any line, including quarter lines (split stake: half on each
    adjacent half/whole line).
  * Grading with the five-way outcome win / half_win / push / half_loss / loss
    (21% of measured closing Asian-handicap grades are not a plain win or loss), and
    unit returns at the stated decimal price.
  * Odds helpers: American <-> decimal, implied probability, proportional de-vig.

Line conventions: a spread/handicap `line` is always from the PICKED side's point of
view (home -3.5 means home must win by 4+; away +3.5 means away may lose by up to 3).
Totals use the posted total; side is "over" or "under".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

OUTCOMES = ("win", "half_win", "push", "half_loss", "loss")
_EPS = 1e-9


# ── odds helpers ──────────────────────────────────────────────────────────────

def american_to_decimal(american: float) -> float:
    a = float(american)
    if a == 0:
        raise ValueError("American odds cannot be 0")
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def decimal_to_american(decimal_odds: float) -> int:
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError("decimal odds must be > 1")
    return int(round((d - 1.0) * 100.0)) if d >= 2.0 else int(round(-100.0 / (d - 1.0)))


def implied_probability(decimal_odds: float) -> float:
    return 1.0 / float(decimal_odds)


def devig(decimal_odds: Sequence[float]) -> list[float]:
    """Proportional (multiplicative) de-vig: normalized implied probabilities.
    Simple and standard; adequate for two- and three-way markets at retail margins."""
    raw = [1.0 / float(d) for d in decimal_odds]
    total = sum(raw)
    return [r / total for r in raw]


def probability_to_fair_american(p: float) -> Optional[int]:
    if p <= 0 or p >= 1:
        return None
    return decimal_to_american(1.0 / p)


# ── grading ───────────────────────────────────────────────────────────────────

def _split_lines(line: float) -> list[float]:
    """Quarter lines (x.25 / x.75) split the stake across the two adjacent lines."""
    q = (abs(line) * 4) % 2
    if abs(q - 1) < _EPS:
        return [line - 0.25, line + 0.25]
    return [line]


def _leg(value: float) -> float:
    return 1.0 if value > _EPS else (-1.0 if value < -_EPS else 0.0)


def grade_pick(market_type: str, side: str, line: Optional[float], home_score: float, away_score: float,
               decimal_odds: float = 1.0 + 100.0 / 110.0) -> tuple[str, float]:
    """(result, unit_return) for a 1-unit stake at `decimal_odds`.

    market_type: moneyline | spread | asian_handicap | total | team_total_home | team_total_away
                 | btts | 1x2 | double_chance | draw_no_bet | correct_score
    side: home | away | draw | over | under | yes | no | home_draw | away_draw | home_away
          | "<home>-<away>" for correct_score (e.g. "1-0")
    Raises ValueError for a type it cannot grade; callers must keep such picks as
    "ungradable" rather than dropping or voiding them (losing picks must not vanish).
    """
    h, a = float(home_score), float(away_score)
    win_ret = decimal_odds - 1.0
    if market_type in ("spread", "asian_handicap", "run_line", "puck_line", "set_handicap", "game_handicap"):
        margin = (h - a) if side == "home" else (a - h)
        r = sum(_leg(margin + l) for l in _split_lines(float(line))) / len(_split_lines(float(line)))
    elif market_type in ("total", "team_total_home", "team_total_away"):
        value = h + a if market_type == "total" else (h if market_type == "team_total_home" else a)
        legs = _split_lines(float(line))
        r = sum(_leg((value - l) if side == "over" else (l - value)) for l in legs) / len(legs)
    elif market_type in ("moneyline", "1x2"):
        outcome = "home" if h > a else ("away" if a > h else "draw")
        if outcome == "draw" and market_type == "moneyline":
            return "push", 0.0  # two-way moneyline with a tie (NFL OT tie): stake returned
        r = 1.0 if outcome == side else -1.0
    elif market_type == "draw_no_bet":
        if h == a:
            return "push", 0.0
        r = 1.0 if (("home" if h > a else "away") == side) else -1.0
    elif market_type == "double_chance":
        outcome = "home" if h > a else ("away" if a > h else "draw")
        r = 1.0 if outcome in side.split("_") else -1.0
    elif market_type == "btts":
        both = h > 0 and a > 0
        r = 1.0 if (both == (side == "yes")) else -1.0
    elif market_type == "correct_score":
        try:
            want_h, want_a = (int(x) for x in str(side).split("-", 1))
        except ValueError as exc:
            raise ValueError(f"correct_score side must be 'H-A', got {side!r}") from exc
        r = 1.0 if (int(h) == want_h and int(a) == want_a) else -1.0
    else:
        raise ValueError(f"unknown market type {market_type!r}")
    label = {1.0: "win", 0.5: "half_win", 0.0: "push", -0.5: "half_loss", -1.0: "loss"}[round(r * 2) / 2]
    unit = {"win": win_ret, "half_win": win_ret / 2, "push": 0.0, "half_loss": -0.5, "loss": -1.0}[label]
    return label, round(unit, 4)


# ── pricing from a discrete margin/total distribution ─────────────────────────

@dataclass(frozen=True)
class LinePrices:
    """Probabilities of the five grading outcomes for one side at one line, plus the
    push-excluded win probability that the boards display."""
    win: float
    half_win: float
    push: float
    half_loss: float
    loss: float

    @property
    def win_ex_push(self) -> float:
        """P(win) with pushes excluded and half results counted half: the number a
        bettor compares against the break-even rate."""
        w = self.win + 0.5 * self.half_win
        l = self.loss + 0.5 * self.half_loss
        return w / (w + l) if (w + l) > 0 else 0.5

    @property
    def expected_units(self) -> float:
        """Expected return per unit at a standard -110 price (for ranking only)."""
        d = 1.0 + 100.0 / 110.0
        return self.win * (d - 1) + self.half_win * (d - 1) / 2 - self.half_loss * 0.5 - self.loss

    def as_dict(self) -> dict:
        return {k: round(getattr(self, k), 4) for k in ("win", "half_win", "push", "half_loss", "loss")}


def _prices_from_pmf(values: np.ndarray, probs: np.ndarray, line: float) -> LinePrices:
    """Outcome probabilities for 'value + line' where value has a discrete pmf.
    Used for spreads (value = picked side's margin) and, with sign flips, totals."""
    out = {"win": 0.0, "half_win": 0.0, "push": 0.0, "half_loss": 0.0, "loss": 0.0}
    legs = _split_lines(line)
    leg_results = np.stack([np.sign(np.round((values + l) * 4) / 4) for l in legs])  # +1/0/-1 per leg
    r = leg_results.mean(axis=0)
    for score, key in ((1.0, "win"), (0.5, "half_win"), (0.0, "push"), (-0.5, "half_loss"), (-1.0, "loss")):
        out[key] = float(probs[np.isclose(r, score)].sum())
    total = sum(out.values()) or 1.0
    return LinePrices(**{k: v / total for k, v in out.items()})


def integer_normal_pmf(mean: float, sigma: float, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Discretized normal on integers [lo, hi]: P(k) = Phi(k+0.5) - Phi(k-0.5)."""
    ks = np.arange(lo, hi + 1, dtype=float)
    z_hi = (ks + 0.5 - mean) / sigma
    z_lo = (ks - 0.5 - mean) / sigma
    cdf = lambda z: 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    p = cdf(z_hi) - cdf(z_lo)
    return ks, p / p.sum()


def spread_prices_normal(expected_margin: float, sigma: float, line: float, *, integer_scores: bool = True,
                         key_number_pmf: Optional[tuple[np.ndarray, np.ndarray]] = None) -> LinePrices:
    """Outcome probabilities for the side whose expected margin is `expected_margin`
    (positive = that side wins by that much) at spread `line` from that side's view.

    `key_number_pmf`, when given, replaces the normal shape with an empirical margin
    distribution (values, probs) shifted to the model mean -- NFL margins pile up on
    3 and 7 and a plain normal underprices pushes there."""
    if key_number_pmf is not None:
        values, probs = key_number_pmf
        values = values + round(expected_margin - float(np.dot(values, probs)))
        return _prices_from_pmf(values, probs, line)
    if integer_scores:
        span = int(max(60, 6 * sigma))
        values, probs = integer_normal_pmf(expected_margin, sigma, int(expected_margin) - span, int(expected_margin) + span)
        return _prices_from_pmf(values, probs, line)
    values = np.linspace(expected_margin - 6 * sigma, expected_margin + 6 * sigma, 4001)
    probs = np.exp(-0.5 * ((values - expected_margin) / sigma) ** 2)
    return _prices_from_pmf(values, probs / probs.sum(), line)


def total_prices_normal(expected_total: float, sigma: float, line: float, side: str = "over", *,
                        integer_scores: bool = True) -> LinePrices:
    """Over/under outcome probabilities for a total with mean `expected_total`."""
    if integer_scores:
        span = int(max(60, 6 * sigma))
        values, probs = integer_normal_pmf(expected_total, sigma, max(0, int(expected_total) - span), int(expected_total) + span)
    else:
        values = np.linspace(max(0.0, expected_total - 6 * sigma), expected_total + 6 * sigma, 4001)
        probs = np.exp(-0.5 * ((values - expected_total) / sigma) ** 2)
        probs = probs / probs.sum()
    # over at L <=> (total - L) > 0 ; under at L <=> (L - total) > 0
    signed = (values if side == "over" else -values)
    return _prices_from_pmf(signed, probs, -line if side == "over" else line)


def fair_line_from_mean(expected: float, half_point: bool = True) -> float:
    """The posted-style line closest to the model mean (x.5 by default so it has no push)."""
    return math.floor(expected) + 0.5 if half_point else round(expected * 2) / 2


# ── pricing from a score matrix (soccer, hockey-style low scoring) ───────────

def score_matrix_markets(matrix: np.ndarray) -> dict[str, float]:
    """Headline probabilities from P[home_goals=i, away_goals=j]."""
    m = np.asarray(matrix, dtype=float)
    m = m / m.sum()
    i, j = np.indices(m.shape)
    p_home = float(m[i > j].sum()); p_draw = float(m[i == j].sum()); p_away = float(m[i < j].sum())
    btts = float(m[(i > 0) & (j > 0)].sum())
    return {
        "home": p_home, "draw": p_draw, "away": p_away,
        "home_draw": p_home + p_draw, "away_draw": p_away + p_draw, "home_away": p_home + p_away,
        "dnb_home": p_home / (p_home + p_away) if (p_home + p_away) > 0 else 0.5,
        "btts_yes": btts, "btts_no": 1.0 - btts,
        "exp_home_goals": float((m.sum(axis=1) * np.arange(m.shape[0])).sum()),
        "exp_away_goals": float((m.sum(axis=0) * np.arange(m.shape[1])).sum()),
    }


def score_matrix_handicap(matrix: np.ndarray, side: str, line: float) -> LinePrices:
    """Asian handicap outcome probabilities for `side` (home|away) at `line` (that
    side's view, e.g. home -0.75), quarter lines included."""
    m = np.asarray(matrix, dtype=float); m = m / m.sum()
    i, j = np.indices(m.shape)
    margin = (i - j) if side == "home" else (j - i)
    return _prices_from_pmf(margin.ravel().astype(float), m.ravel(), line)


def score_matrix_total(matrix: np.ndarray, side: str, line: float) -> LinePrices:
    m = np.asarray(matrix, dtype=float); m = m / m.sum()
    i, j = np.indices(m.shape)
    tot = (i + j).ravel().astype(float)
    signed = tot if side == "over" else -tot
    return _prices_from_pmf(signed, m.ravel(), -line if side == "over" else line)


def correct_scores(matrix: np.ndarray, top: int = 3) -> list[tuple[int, int, float]]:
    m = np.asarray(matrix, dtype=float); m = m / m.sum()
    flat = [(int(a), int(b), float(m[a, b])) for a in range(m.shape[0]) for b in range(m.shape[1])]
    return sorted(flat, key=lambda t: -t[2])[:top]


# ── board helpers ─────────────────────────────────────────────────────────────

def confidence_tier(p_win_ex_push: float, calibrated_edges: Iterable[tuple[float, str]]) -> str:
    """Coarse tier from the push-excluded win probability using sport- and
    market-specific cut points derived from held-out calibration. There are no
    defaults on purpose: generic cuts labelled every heavy moneyline favourite
    "high", which reads as a recommendation the record does not support."""
    cuts = list(calibrated_edges)
    for threshold, tier in sorted(cuts, key=lambda t: -t[0]):
        if p_win_ex_push >= threshold:
            return tier
    return "low"


def market_pick(*, board_id: str, market_type: str, side: str, label: str, prices: LinePrices,
                line: Optional[float], model_line: Optional[float], period: str = "full_game",
                market: Optional[dict] = None, published_at: Optional[str] = None,
                show_edge: bool = False, tier_cuts: Optional[Iterable[tuple[float, str]]] = None) -> dict:
    """One `markets[]` entry in the API schema. `show_edge` gates the vs-market
    comparison: only market types whose logged out-of-sample record beats the
    de-vigged closing line may display an edge (Apple 1.1.6/2.3.1, Play misleading
    claims). The model probability and fair line are always shown."""
    p = prices.win_ex_push
    entry = {
        "marketId": f"{board_id}:{market_type}:{period}",
        "type": market_type,
        "period": period,
        "label": label,
        "side": side,
        "line": line,
        "modelLine": model_line,
        "modelProbability": round(p, 4),
        "pushProbability": round(prices.push, 4) if prices.push > 0.0005 else None,
        "outcomeProbabilities": prices.as_dict(),
        # Tiers only with calibrated, market-specific cut points; never by default.
        "confidenceTier": confidence_tier(p, tier_cuts) if tier_cuts else None,
        "publishedAt": published_at,
    }
    if market:
        entry["market"] = market
        implied = market.get("impliedProbability")
        if show_edge and implied is not None:
            entry["edge"] = round(p - float(implied), 4)
    return {k: v for k, v in entry.items() if v is not None}
