"""Elo with margin of victory for NBA/WNBA (FiveThirtyEight-style).

Pregame ratings only: a game's own result never feeds its own rating, so the
pregame columns are leak-free by construction and can be used both as model
features and as the cold-start fallback when box-score history is thin.

Parameters (tuned by the analyst on early seasons only -- NBA 2022-24, WNBA 2024 --
and stored in sports/basketball/model_params.json): K=16 NBA / 30 WNBA, home
advantage 60 Elo points (none on neutral courts), 0.75 carry-over toward 1500
between seasons, and the 538 MOV multiplier
    ((|margin| + 3) ** 0.8) / (7.5 + 0.006 * winner_elo_edge)
which damps the update when a heavy favourite wins big (autocorrelation fix).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    k: float = 16.0
    hca: float = 60.0
    carry: float = 0.75
    init: float = 1500.0
    mov: bool = True

    @classmethod
    def from_dict(cls, payload: dict) -> "EloParams":
        return cls(**{k: payload[k] for k in ("k", "hca", "carry", "init", "mov") if k in payload})


def elo_win_probability(diff: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(diff, dtype=float) / 400.0))


def mov_multiplier(margin: float, winner_edge: float) -> float:
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * winner_edge)


def run_elo(results: pd.DataFrame, params: EloParams) -> tuple[pd.DataFrame, dict[int, float]]:
    """Pregame Elo for every row of a ONE-league results table (finals and scheduled).

    Rows are processed in tip order; finals update the ratings, scheduled games only
    read them (with the new-season carry-over applied, so an opening-night board
    already shows regressed ratings). Returns (per-game frame, final ratings by team)."""
    ordered = results.sort_values(["tip_utc", "game_id"])
    elo: dict[int, float] = {}
    last_season: dict[int, str] = {}
    rows = []
    for r in ordered.itertuples(index=False):
        h, a, season = int(r.home_team_id), int(r.away_team_id), str(r.season)
        for team in (h, a):
            if team not in elo:
                elo[team] = params.init
            elif last_season.get(team) != season:
                elo[team] = params.init + params.carry * (elo[team] - params.init)
                last_season[team] = season
        eh, ea = elo[h], elo[a]
        diff = eh - ea + (0.0 if bool(r.neutral) else params.hca)
        p = float(elo_win_probability(diff))
        rows.append((r.game_id, eh, ea, diff, p))
        if not bool(r.final):
            continue
        margin = float(r.home_score) - float(r.away_score)
        home_won = 1.0 if margin > 0 else 0.0
        mult = mov_multiplier(margin, diff if home_won else -diff) if params.mov else 1.0
        delta = params.k * mult * (home_won - p)
        elo[h] = eh + delta
        elo[a] = ea - delta
        last_season[h] = last_season[a] = season
    frame = pd.DataFrame(rows, columns=["game_id", "elo_home", "elo_away", "elo_diff", "p_elo"])
    return frame, dict(elo)
