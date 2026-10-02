"""Opponent-adjusted offense/defense point ratings (recency-weighted ridge on scores).

Each team gets an offense rating O (points scored above average) and a defense
rating D (points allowed above average); a game is predicted as
    home_pts = mu + hca/2 + O[home] + D[away]
    away_pts = mu - hca/2 + O[away] + D[home]
so the model yields home points, away points, margin and total at once (team
totals come for free). Games are weighted by recency (half-life in days) and
prior-season games are down-weighted; the team effects are shrunk toward zero by
an L2 penalty so a team with few games is predicted near league average.

The model is refit for each game day using only games from EARLIER local dates,
so every prediction is out of sample. Walk-forward measured on NBA 2024-25 +
2025-26 (n=2,374): margin MAE 11.14, total MAE 14.87; closing market 10.69 / 14.30.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr


@dataclass(frozen=True)
class RatingsParams:
    half_life_days: float = 45.0
    l2: float = 8.0
    prev_season_weight: float = 0.5
    window_days: int = 400
    min_games: int = 30

    @classmethod
    def from_dict(cls, payload: dict) -> "RatingsParams":
        keys = ("half_life_days", "l2", "prev_season_weight", "window_days", "min_games")
        return cls(**{k: payload[k] for k in keys if k in payload})


@dataclass
class FittedRatings:
    mu: float
    hca: float
    offense: dict[int, float]
    defense: dict[int, float]
    n_games: int

    def predict(self, home_team_id: int, away_team_id: int, neutral: bool = False) -> tuple[float, float]:
        hv = 0.0 if neutral else 0.5 * self.hca
        o, d = self.offense, self.defense
        home = self.mu + hv + o.get(int(home_team_id), 0.0) + d.get(int(away_team_id), 0.0)
        away = self.mu - hv + o.get(int(away_team_id), 0.0) + d.get(int(home_team_id), 0.0)
        return float(home), float(away)


def fit_ratings(history: pd.DataFrame, as_of_local_date: pd.Timestamp, current_season: str | None,
                params: RatingsParams) -> FittedRatings | None:
    """Fit on FINAL games with local_date < as_of (and within the window)."""
    day = pd.Timestamp(as_of_local_date).normalize()
    hist = history.loc[history["final"] & (history["local_date"] < day)
                       & (history["local_date"] >= day - pd.Timedelta(days=params.window_days))]
    if len(hist) < params.min_games:
        return None
    teams = sorted(set(hist["home_team_id"].astype(int)) | set(hist["away_team_id"].astype(int)))
    index = {team: i for i, team in enumerate(teams)}
    n_teams, n = len(teams), len(hist)
    age = (day - hist["local_date"]).dt.days.to_numpy(dtype=float)
    weight = 0.5 ** (age / params.half_life_days)
    if current_season is not None:
        weight = weight * np.where(hist["season"].astype(str).to_numpy() == str(current_season), 1.0, params.prev_season_weight)
    sw = np.sqrt(np.concatenate([weight, weight]))
    hi = hist["home_team_id"].astype(int).map(index).to_numpy()
    ai = hist["away_team_id"].astype(int).map(index).to_numpy()
    neutral = hist["neutral"].fillna(False).astype(bool).to_numpy()
    half = np.where(neutral, 0.0, 0.5)
    rows = np.concatenate([np.repeat(np.arange(n), 4), np.repeat(np.arange(n, 2 * n), 4)])
    cols = np.concatenate([
        np.column_stack([np.zeros(n, int), np.ones(n, int), 2 + hi, 2 + n_teams + ai]).ravel(),
        np.column_stack([np.zeros(n, int), np.ones(n, int), 2 + ai, 2 + n_teams + hi]).ravel(),
    ])
    vals = np.concatenate([
        np.column_stack([np.ones(n), half, np.ones(n), np.ones(n)]).ravel(),
        np.column_stack([np.ones(n), -half, np.ones(n), np.ones(n)]).ravel(),
    ])
    design = csr_matrix((vals * np.repeat(sw, 4), (rows, cols)), shape=(2 * n, 2 + 2 * n_teams))
    target = np.concatenate([hist["home_score"].to_numpy(float), hist["away_score"].to_numpy(float)]) * sw
    penalty = csr_matrix((np.full(2 * n_teams, np.sqrt(params.l2)), (np.arange(2 * n_teams), 2 + np.arange(2 * n_teams))),
                         shape=(2 * n_teams, 2 + 2 * n_teams))
    solution = lsqr(vstack([design, penalty]), np.concatenate([target, np.zeros(2 * n_teams)]), atol=1e-10, btol=1e-10)[0]
    mu, hca = float(solution[0]), float(solution[1])
    offense = {team: float(solution[2 + i]) for team, i in index.items()}
    defense = {team: float(solution[2 + n_teams + i]) for team, i in index.items()}
    return FittedRatings(mu=mu, hca=hca, offense=offense, defense=defense, n_games=n)


def predict_points(results: pd.DataFrame, params: RatingsParams, *, only_ids: set[str] | None = None) -> pd.DataFrame:
    """Out-of-sample home/away points for every game of a ONE-league results table:
    one refit per local date, each on games strictly before that date.

    `only_ids` limits the work to the dates of those games (e.g. the upcoming slate)."""
    frame = results.sort_values(["local_date", "tip_utc", "game_id"])
    target = frame if only_ids is None else frame.loc[frame["game_id"].isin(only_ids)]
    out = []
    for day, games in target.groupby("local_date", sort=True):
        season = str(games["season"].iloc[0])
        fitted = fit_ratings(frame, day, season, params)
        if fitted is None:
            continue
        for g in games.itertuples(index=False):
            home, away = fitted.predict(g.home_team_id, g.away_team_id, bool(g.neutral))
            out.append((g.game_id, home, away))
    pts = pd.DataFrame(out, columns=["game_id", "r_home_pts", "r_away_pts"])
    pts["r_margin"] = pts["r_home_pts"] - pts["r_away_pts"]
    pts["r_total"] = pts["r_home_pts"] + pts["r_away_pts"]
    return pts
