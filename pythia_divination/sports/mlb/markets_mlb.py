"""MLB `markets` block (run line, game total, team totals) from the NB runs model.

Every market here is a MODEL VIEW (basis "model", market null): we have no MLB line
feed yet, and on 2025 closing lines the market beat the model on the moneyline (Brier
0.2419 vs 0.2425, same 2,379 games) and, in the analyst prototype, on totals (0.2482 vs
0.2505). So these are calibrated probabilities with a fixed, documented side -- never a
"pick" chosen by the model, never an edge:

  * run_line          the favourite -1.5 (favourite = the win model's favourite). This is
                      the standard posted run line; it is home -1.5 / away +1.5 whenever
                      the home team is favoured. The other side's probability is the
                      entry's outcomeProbabilities["loss"].
  * total             Over at the model fair line: the x.5 line whose P(over) is closest
                      to 50%; modelLine is the projected total runs.
  * team_total_home / team_total_away
                      Over at each team's fair x.5 line; modelLine is projected runs.

A fixed side keeps the logged record honest: its win rate is the observed frequency
of the stated outcome (about 40% for favourite -1.5), to be read against the mean
modelProbability (calibration), not a hand-picked "lean" that would show a 60%+
"record" that is only the base rate of the +1.5 side.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sports import markets as mk
from sports.mlb import pregame_model as pm

PERIOD = "full_game"


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

    fav = "home" if p_home >= 0.5 else "away"
    fav_label = home_label if fav == "home" else away_label
    fav_margin = (exp_home - exp_away) if fav == "home" else (exp_away - exp_home)
    entries = [
        mk.market_pick(board_id=board_id, market_type="run_line", side=fav, label=f"{fav_label} -1.5",
                       prices=mk.score_matrix_handicap(m, fav, -1.5), line=-1.5,
                       model_line=round(-fav_margin, 1), period=PERIOD, published_at=published_at),
    ]
    total_line = fair_half_line(total_pmf)
    entries.append(mk.market_pick(board_id=board_id, market_type="total", side="over", label=f"Over {total_line:g}",
                                  prices=mk.score_matrix_total(m, "over", total_line), line=total_line,
                                  model_line=round(exp_home + exp_away, 1), period=PERIOD, published_at=published_at))
    for side, pmf, label, exp in (("home", home_pmf, home_label, exp_home), ("away", away_pmf, away_label, exp_away)):
        line = fair_half_line(pmf)
        entries.append(mk.market_pick(board_id=board_id, market_type=f"team_total_{side}", side="over",
                                      label=f"{label} Over {line:g}", prices=_over_prices(pmf, line), line=line,
                                      model_line=round(exp, 1), period=PERIOD, published_at=published_at))
    for entry in entries:
        entry["basis"] = "model"
        entry["modelVersion"] = model_version
    return entries
