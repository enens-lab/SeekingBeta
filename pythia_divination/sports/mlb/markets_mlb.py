"""MLB `markets` block (run line, game total, team totals) from the NB runs model.

Every market here is a MODEL VIEW (basis "model", market null): we have no MLB line
feed yet, and on 2025 closing lines the market beat the model on the moneyline (Brier
0.2419 vs 0.2425, same 2,379 games) and, in the analyst prototype, on totals (0.2482 vs
0.2505). So these are calibrated probabilities with fixed, documented sides -- never a
"pick" chosen by the model, never an edge, never a confidence tier:

  * run_line          BOTH -1.5 lines, the model favourite's first:
                        home -1.5 (its "loss" outcome is away +1.5) and
                        away -1.5 (its "loss" outcome is home +1.5).
                      The favourite's entry is the standard posted run line; the other is
                      the underdog -1.5 alternate. Each side keeps its own stable marketId
                      (<boardId>:run_line:full_game:<side>) so a favourite flipping between
                      daily bakes never withdraws a logged pick.
  * total             Over at the model fair line: the x.5 line whose P(over) is closest
                      to 50%; modelLine is the projected total runs.
  * team_total_home / team_total_away
                      Over at each team's fair x.5 line; modelLine is projected runs.

Fixed sides keep the logged record honest: a side's win rate is the observed frequency
of the stated outcome (about 40% for the favourite -1.5, 31% for the underdog -1.5), to
be read against the mean modelProbability (calibration), not a hand-picked "lean" that
would show a 60%+ "record" that is only the base rate of the +1.5 side.

Walk-forward calibration of the served run lines (monthly refit, regular season;
details in model_params.json evaluation): Brier below the constant in 2024, 2025 and
2026 for the favourite -1.5, the underdog -1.5 and both pooled, largest quintile gap
<= 0.05 on each of those.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sports import markets as mk
from sports.mlb import pregame_model as pm

PERIOD = "full_game"
RUN_LINE = 1.5


def fair_half_line(pmf: np.ndarray) -> float:
    """The x.5 line whose P(value > line) is closest to 0.5 (no push possible)."""
    sf = 1.0 - np.cumsum(pmf)  # sf[k] = P(X > k) = P(X > k + 0.5)
    k = int(np.argmin(np.abs(sf - 0.5)))
    return k + 0.5


def _over_prices(pmf: np.ndarray, line: float) -> mk.LinePrices:
    values = np.arange(len(pmf), dtype=float)
    over = float(pmf[values > line].sum())
    under = float(pmf[values < line].sum())
    push = max(0.0, 1.0 - over - under)
    return mk.LinePrices(win=over, half_win=0.0, push=push, half_loss=0.0, loss=under)


def run_line_market_id(board_id: str, side: str) -> str:
    return f"{board_id}:run_line:{PERIOD}:{side}"


def mlb_market_picks(*, board_id: str, home_label: str, away_label: str, p_home: float,
                     mu_home: float, mu_away: float, nb_r: float, published_at: Optional[str],
                     model_version: str) -> list[dict]:
    """The markets[] entries for one game (all basis "model", market null)."""
    if any(v is None or not np.isfinite(v) for v in (p_home, mu_home, mu_away, nb_r)):
        return []
    m = pm.score_matrix(float(mu_home), float(mu_away), float(nb_r))
    home_pmf, away_pmf = m.sum(axis=1), m.sum(axis=0)
    exp_home = float((home_pmf * np.arange(len(home_pmf))).sum())
    exp_away = float((away_pmf * np.arange(len(away_pmf))).sum())
    total_pmf = np.bincount((np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))).ravel(), weights=m.ravel())

    labels = {"home": home_label, "away": away_label}
    expected = {"home": exp_home, "away": exp_away}
    favourite = "home" if p_home >= 0.5 else "away"
    entries = []
    for side in (favourite, "away" if favourite == "home" else "home"):
        other = "away" if side == "home" else "home"
        entry = mk.market_pick(board_id=board_id, market_type="run_line", side=side,
                               label=f"{labels[side]} -{RUN_LINE:g}",
                               prices=mk.score_matrix_handicap(m, side, -RUN_LINE), line=-RUN_LINE,
                               # fair handicap from this side's view: minus its expected margin,
                               # on the half-point grid lines are quoted in
                               model_line=mk.fair_line_from_mean(expected[other] - expected[side], half_point=False) + 0.0,
                               period=PERIOD, published_at=published_at)
        entry["marketId"] = run_line_market_id(board_id, side)
        entries.append(entry)
    total_line = fair_half_line(total_pmf)
    entries.append(mk.market_pick(board_id=board_id, market_type="total", side="over", label=f"Over {total_line:g}",
                                  prices=mk.score_matrix_total(m, "over", total_line), line=total_line,
                                  # the line IS the model's 50/50 line; the (skewed) mean would contradict it
                                  model_line=None, period=PERIOD, published_at=published_at))
    for side, pmf, label, exp in (("home", home_pmf, home_label, exp_home), ("away", away_pmf, away_label, exp_away)):
        line = fair_half_line(pmf)
        entries.append(mk.market_pick(board_id=board_id, market_type=f"team_total_{side}", side="over",
                                      label=f"{label} Over {line:g}", prices=_over_prices(pmf, line), line=line,
                                      model_line=None, period=PERIOD, published_at=published_at))
    for entry in entries:
        # No calibrated, market-specific tier cut points exist for MLB: never a tier.
        entry.pop("confidenceTier", None)
        entry["basis"] = "model"
        entry["modelVersion"] = model_version
    return entries
