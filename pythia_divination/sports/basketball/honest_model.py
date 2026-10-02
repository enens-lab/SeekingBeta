"""Honest NBA/WNBA models: what the basketball boards serve, and how they are graded.

Win probability (P1-4): L1-regularized logistic regression over the box-score form
features (team rolling stats, rotation, expected availability) PLUS Elo with margin
of victory and true rest / back-to-back from local tip dates. It replaces the
torch MLP / HistGradientBoosting pair, whose walk-forward log-loss was worse than a
coin flip (NBA HGB 0.785 on 2024-25; served WNBA torch 0.766 on 2026).

Spread / total (P1-12, model only): opponent-adjusted offense/defense ratings
(sports/basketball/ratings.py) blended 50/50 with a ridge on Elo + rest for the
margin; ratings alone for the total; team totals = (total +/- margin) / 2. Each is
a normal distribution whose sigma is the rolling standard deviation of the last
400 OUT-OF-SAMPLE residuals.

One feature builder serves training, the walk-forward history and the upcoming
boards, so there is no train/serve skew: the rotation features that used to be
NaN at serve time are attached on every path, every feature is "as of strictly
before tip", and form features reset each season.

Honest history: every graded game is predicted by a model trained only on games
from EARLIER local dates (14-day walk-forward blocks); each record carries that
model's training cutoff. Nothing dated on or before a cutoff is ever graded by
the model trained through it.

Calibration: the raw win probability goes through a Platt map (logit q = a + b *
logit p) fitted ONLY on earlier out-of-sample predictions (the walk-forward
predictions of previous blocks, warm-up seasons included) and shrunk toward the
identity, so with little evidence it changes nothing. The "training cutoff" of a
graded game covers both the model's training games and the calibrator's games.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sports.basketball import feature_engineering as fe
from sports.basketball import rotation_features as rf
from sports.basketball.backfill_history import canonical_game_id
from sports.basketball.constants import normalize_season_label, season_start_year_from_label
from sports.basketball.elo import EloParams, run_elo
from sports.basketball.ratings import RatingsParams, predict_points
from sports.basketball.results import add_rest_features, build_results, season_record_before

logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).with_name("model_params.json")

# A win probability within this distance of 0.5 is NO pick. Boards show whole
# percentages, so 0.4950 < p < 0.5050 reads as 50% / 50%; grading such a game as a
# hit or a miss would score a pick the user never saw. Used by win_metrics, the
# published walk-forward history and the moneyline market alike.
NO_PICK_BAND = 0.005


def is_pick(p: Any) -> Any:
    """True where the probability is a pick (|p - 0.5| >= NO_PICK_BAND)."""
    return np.abs(np.asarray(p, dtype=float) - 0.5) >= NO_PICK_BAND

_BOX_STAT_COLUMNS = [
    "points", "assists", "rebounds_total", "rebounds_offensive", "rebounds_defensive", "turnovers", "steals",
    "blocks", "fouls_personal", "field_goals_attempted", "free_throws_attempted", "field_goal_pct",
    "three_point_pct", "free_throw_pct", "bench_points", "fast_break_points", "paint_points",
    "second_chance_points", "true_shooting_pct", "effective_fg_pct",
]
_FEATURE_PREFIXES = (
    "home_team_", "away_team_", "matchup_diff_",
    "home_rotation_", "away_rotation_", "rotation_diff_",
    "home_availability_", "away_availability_", "availability_diff_",
)
_ROTATION_PREFIXES = ("home_rotation_", "away_rotation_", "rotation_diff_",
                      "home_availability_", "away_availability_", "availability_diff_")


def load_params(path: Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or PARAMS_PATH).read_text())


def model_version(params: dict[str, Any]) -> str:
    return str(params.get("version") or "bb-honest")


# ── data assembly ─────────────────────────────────────────────────────────────

@dataclass
class LeagueTables:
    """Raw inputs for one league. `schedules` are schedule-shaped frames in rising
    priority (S3 tables first, live schedule / fresh boxscores last)."""
    league: str
    schedules: list[pd.DataFrame]
    details: pd.DataFrame
    players: pd.DataFrame


def build_game_frame(tables: LeagueTables, params: dict[str, Any], *, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Results + rest + Elo + season records + labels: one row per competitive game."""
    league = tables.league
    lp = params["leagues"][league]
    results = build_results(league, tables.schedules, today=today)
    if results.empty:
        return results
    results = add_rest_features(results, cap=int(params.get("rest_cap_days", 4)))
    results["is_neutral"] = results["neutral"].astype(float)
    elo_frame, _ = run_elo(results, EloParams.from_dict(lp["elo"]))
    frame = results.merge(elo_frame, on="game_id", how="left")
    frame = frame.merge(season_record_before(results), on="game_id", how="left")
    frame["home_win"] = np.where(frame["final"], (frame["home_score"] > frame["away_score"]).astype(float), np.nan)
    frame["margin"] = frame["home_score"] - frame["away_score"]
    frame["total"] = frame["home_score"] + frame["away_score"]
    return frame.sort_values(["tip_utc", "game_id"]).reset_index(drop=True)


def _games_for_feature_code(frame: pd.DataFrame) -> pd.DataFrame:
    """The column layout sports/basketball/feature_engineering + rotation_features expect.
    official_date = naive-UTC tip time on EVERY table, so the strict-before as-of joins
    compare like with like (the old mix of UTC tips and local midnights dropped the
    previous evening's game for next-day games)."""
    games = pd.DataFrame({
        "league": frame["league"].values,
        "game_id": frame["game_id"].values,
        "official_date": pd.to_datetime(frame["tip_utc"]).values,
        "season_display": frame["season"].astype(str).values,
        "home_team_id": frame["home_team_id"].values,
        "away_team_id": frame["away_team_id"].values,
        "home_team_name": frame["home_name"].values,
        "away_team_name": frame["away_name"].values,
        "home_team_tricode": frame["home_tricode"].values,
        "away_team_tricode": frame["away_tricode"].values,
        "final": frame["final"].values,
    })
    games["home_team_key"] = games["league"].astype(str) + ":" + games["home_team_id"].astype(str)
    games["away_team_key"] = games["league"].astype(str) + ":" + games["away_team_id"].astype(str)
    won = frame["home_win"].values
    games["home_is_winner"] = won
    games["away_is_winner"] = np.where(np.isnan(won), np.nan, 1.0 - won)
    return games


def _prepare_details(details: pd.DataFrame) -> pd.DataFrame:
    if details is None or details.empty:
        return pd.DataFrame(columns=["game_id"])
    d = details.copy()
    d["game_id"] = d["game_id"].map(canonical_game_id)
    keep = ["game_id"] + [f"{side}_{stat}" for side in ("home", "away") for stat in _BOX_STAT_COLUMNS
                          if f"{side}_{stat}" in d.columns]
    return d[keep].dropna(subset=["game_id"]).drop_duplicates("game_id", keep="last")


def _prepare_players(players: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Player rows of FINAL competitive games, stamped with the canonical tip time and season."""
    if players is None or players.empty:
        return pd.DataFrame()
    p = players.copy()
    p["game_id"] = p["game_id"].map(canonical_game_id)
    finals = frame.loc[frame["final"], ["game_id", "tip_utc", "season"]]
    p = p.merge(finals, on="game_id", how="inner")
    p["game_date_time_utc"] = pd.to_datetime(p["tip_utc"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    p["season_display"] = p["season"].astype(str)
    p = p.drop(columns=["tip_utc", "season"])
    return p.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")


@dataclass
class FeatureBundle:
    features: pd.DataFrame            # one row per game_id: form / rotation / availability features
    player_logs: pd.DataFrame         # for the projected lineups on upcoming boards
    timings: dict[str, float] = field(default_factory=dict)


def build_features(frame: pd.DataFrame, tables: LeagueTables, *, include_rotation: bool = True) -> FeatureBundle:
    timings: dict[str, float] = {}
    t0 = time.time()
    games = _games_for_feature_code(frame)
    details = _prepare_details(tables.details)
    completed = games.loc[games["final"]].merge(details, on="game_id", how="inner")
    team_logs = fe.build_team_game_logs(completed)
    feats = fe.attach_pregame_team_features(games, team_logs)
    timings["team"] = time.time() - t0
    player_logs = pd.DataFrame()
    players = _prepare_players(tables.players, frame)
    if not players.empty:
        t1 = time.time()
        player_logs = rf.build_player_game_logs(players)
        if include_rotation:
            rotation_logs = rf.build_rotation_game_logs(players)
            feats = rf.attach_pregame_rotation_features(feats, rotation_logs)
            expected = rf.build_expected_rotation_game_logs(games, player_logs)
            feats = rf.attach_expected_rotation_features(feats, expected)
        timings["rotation"] = time.time() - t1
    feats = fe.add_matchup_differentials(feats)
    feats = feats.drop_duplicates("game_id").set_index("game_id")
    return FeatureBundle(features=feats, player_logs=player_logs, timings=timings)


def select_feature_columns(features: pd.DataFrame, params: dict[str, Any]) -> list[str]:
    """Box-score form features by prefix (numeric only), minus identifiers: team ids
    (incl. the *_detail ids that used to leak in as numbers), keys, names, and any
    rest column (rest comes from results.py instead)."""
    include_rotation = bool(params["win_model"].get("include_rotation_features", True))
    cols = []
    for column in features.columns:
        if not column.startswith(_FEATURE_PREFIXES):
            continue
        if not include_rotation and column.startswith(_ROTATION_PREFIXES):
            continue
        lowered = column.lower()
        if (lowered.endswith(("_id", "_key", "_name", "_tricode", "_detail")) or "_id_" in lowered
                or "days_rest" in lowered or "rest_advantage" in lowered or lowered.endswith(("_wins", "_losses"))):
            continue
        if features[column].dtype.kind not in "iufb":
            continue
        cols.append(column)
    return cols


def assemble_design(frame: pd.DataFrame, bundle: FeatureBundle, params: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """frame (results/labels) + box features + extras, aligned by game_id."""
    box_cols = select_feature_columns(bundle.features, params)
    box = bundle.features.reindex(frame["game_id"].values)[box_cols].reset_index(drop=True)
    box.index = frame.index
    extras = list(params["win_model"]["extra_features"])
    design = pd.concat([frame, box], axis=1)
    feature_cols = [c for c in box_cols if c in design.columns] + [c for c in extras if c in design.columns]
    return design, feature_cols


# ── estimators ────────────────────────────────────────────────────────────────

ELO_LOGIT_SCALE = math.log(10.0) / 400.0   # Elo points -> log-odds


class OffsetL1Logistic:
    """L1-regularized logistic regression on top of a FIXED Elo prior.

        logit P(home win) = elo_diff * ln(10)/400  +  b  +  X . w

    The Elo log-odds enter as an offset (slope fixed at 1, never shrunk), the
    intercept b is unpenalized (it re-learns home advantage), and the standardized
    box-score / rest features get an L1 penalty, so with little data the model
    stays at the Elo prior and moves away from it only where the features earn it.
    A plain L1 fit shrinks the Elo coefficient along with 400+ noisy box features;
    on WNBA 2025-26 walk-forward that was worse than raw Elo (Brier 0.2157 vs 0.2103).

    `C` follows scikit-learn's convention (penalty = ||w||_1 / (C * n) on the mean
    log-loss), so a given C regularizes small samples more. Solved with FISTA.
    Median imputation and standardization are fit on the training rows only."""

    def __init__(self, C: float = 0.05, offset_col: str = "elo_diff", penalty_factors: dict[str, float] | None = None,
                 max_iter: int = 5000, tol: float = 1e-6) -> None:
        self.C = float(C)
        self.offset_col = offset_col
        self.penalty_factors = dict(penalty_factors or {})
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _matrix(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        off = pd.to_numeric(X[self.offset_col], errors="coerce").fillna(0.0).to_numpy(float) * ELO_LOGIT_SCALE
        M = X[self.columns_].to_numpy(float)
        M = np.where(np.isnan(M), self.medians_, M)
        return (M - self.means_) / self.stds_, off

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OffsetL1Logistic":
        y = np.asarray(y, dtype=float)
        self.columns_ = [c for c in X.columns if c != self.offset_col]
        raw = X[self.columns_].to_numpy(float)
        with np.errstate(all="ignore"):
            med = np.nanmedian(raw, axis=0) if raw.size else np.zeros(0)
        self.medians_ = np.where(np.isnan(med), 0.0, med)
        filled = np.where(np.isnan(raw), self.medians_, raw)
        self.means_ = filled.mean(axis=0) if len(filled) else np.zeros(len(self.columns_))
        std = filled.std(axis=0) if len(filled) else np.ones(len(self.columns_))
        self.stds_ = np.where(std > 1e-12, std, 1.0)
        Z, off = self._matrix(X)
        n, p = Z.shape
        lam = 1.0 / (self.C * n)
        phi = np.array([float(self.penalty_factors.get(c, 1.0)) for c in self.columns_])
        # Lipschitz constant of the mean log-loss gradient w.r.t. [w, b]: 0.25 * ||[Z, 1]||^2 / n
        A = np.hstack([Z, np.ones((n, 1))])
        v = np.ones(p + 1) / math.sqrt(p + 1)
        for _ in range(30):
            v = A.T @ (A @ v)
            v /= np.linalg.norm(v) or 1.0
        lip = 0.25 * float(np.linalg.norm(A @ v) ** 2) / n + 1e-9
        w = np.zeros(p)
        b = float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)) - off.mean())
        yw, yb, t = w.copy(), b, 1.0
        for _ in range(self.max_iter):
            z = off + Z @ yw + yb
            g = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))) - y
            gw = Z.T @ g / n
            gb = float(g.mean())
            step_w = yw - gw / lip
            w_new = np.sign(step_w) * np.maximum(np.abs(step_w) - lam * phi / lip, 0.0)
            b_new = yb - gb / lip
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            yw = w_new + ((t - 1.0) / t_new) * (w_new - w)
            yb = b_new + ((t - 1.0) / t_new) * (b_new - b)
            delta = max(float(np.max(np.abs(w_new - w))) if p else 0.0, abs(b_new - b))
            w, b, t = w_new, b_new, t_new
            if delta < self.tol:
                break
        self.coef_ = w
        self.intercept_ = b
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        Z, off = self._matrix(X)
        return off + Z @ self.coef_ + self.intercept_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-np.clip(self.decision_function(X), -30, 30)))
        return np.column_stack([1.0 - p, p])

    def nonzero_features(self) -> dict[str, float]:
        return {c: float(v) for c, v in zip(self.columns_, self.coef_) if v != 0.0}


class EloWinModel:
    """Serve the pregame Elo-MOV probability itself: nothing is fitted, so the
    walk-forward and the served path only add the shared shrunk-Platt calibration.
    Used where the fitted model does not beat Elo on paired Brier (review rule:
    prefer Elo when the model-minus-Elo CI includes 0)."""

    input_columns = ["p_elo"]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EloWinModel":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(pd.to_numeric(X["p_elo"], errors="coerce").fillna(0.5).to_numpy(float), 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - p, p])


def win_model_config(params: dict[str, Any], league: str | None = None) -> dict[str, Any]:
    """Global win_model block with the league's `win_model` override merged on top."""
    wm = dict(params["win_model"])
    if league:
        wm.update((params.get("leagues", {}).get(league) or {}).get("win_model") or {})
    return wm


def served_win_model(params: dict[str, Any], league: str) -> str:
    """Short label of the win model a league serves (boards' predictionSource)."""
    kind = win_model_config(params, league).get("type", "logistic_l1")
    return "elo_mov_calibrated" if kind == "elo" else "elo_l1_logistic"


def win_estimator(params: dict[str, Any], league: str | None = None) -> Any:
    wm = win_model_config(params, league)
    if wm.get("type") == "elo":
        return EloWinModel()
    if wm.get("type", "logistic_l1") == "elo_offset_l1":
        return OffsetL1Logistic(C=float(wm.get("C", 0.05)), offset_col="elo_diff",
                                penalty_factors={c: float(wm.get("extra_penalty_factor", 1.0)) for c in wm["extra_features"]})
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l1", C=float(wm.get("C", 0.05)),
                                     solver="liblinear", max_iter=3000, random_state=42)),
    ])


def margin_estimator(params: dict[str, Any]) -> Pipeline:
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.asarray(params["margin_model"]["ridge_alphas"], dtype=float))),
    ])


def _xy(design: pd.DataFrame, rows: pd.Index, cols: list[str]) -> pd.DataFrame:
    return design.loc[rows, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fit_win_model(design: pd.DataFrame, train_rows: pd.Index, feature_cols: list[str], params: dict[str, Any],
                  league: str | None = None) -> Any:
    model = win_estimator(params, league)
    cols = list(getattr(model, "input_columns", None) or feature_cols)
    model.fit(_xy(design, train_rows, cols), design.loc[train_rows, "home_win"].astype(int).values)
    model.serving_columns_ = cols
    return model


def predict_win(model: Any, design: pd.DataFrame, rows: pd.Index) -> np.ndarray:
    """Raw (uncalibrated) home-win probability of a model from fit_win_model."""
    return model.predict_proba(_xy(design, rows, model.serving_columns_))[:, 1]


class ShrunkPlatt:
    """Platt recalibration logit q = a + b * logit p, fitted by penalized maximum
    likelihood on earlier OUT-OF-SAMPLE predictions with a Gaussian prior centred on
    the identity map (a = 0, b = 1):

        minimize  sum_i logloss(y_i, sigmoid(a + b * logit p_i)) + s * (a^2 + (b - 1)^2)

    `s` (prior_strength) is in units of summed log-loss, i.e. roughly "how many
    games of evidence it takes to move the map". Below `min_n` predictions the map
    stays the identity. Only the newest `window` predictions are used, so the map
    follows the model as its training set grows."""

    def __init__(self, prior_strength: float = 20.0, min_n: int = 200, window: int = 1500) -> None:
        self.prior_strength = float(prior_strength)
        self.min_n = int(min_n)
        self.window = int(window)
        self.a = 0.0
        self.b = 1.0
        self.n = 0

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "ShrunkPlatt":
        cfg = params.get("calibration") or {}
        return cls(prior_strength=float(cfg.get("prior_strength", 20.0)), min_n=int(cfg.get("min_oos", 200)),
                   window=int(cfg.get("window", 1500)))

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, p: np.ndarray, y: np.ndarray) -> "ShrunkPlatt":
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = ~np.isnan(p) & ~np.isnan(y)
        p, y = p[ok][-self.window:], y[ok][-self.window:]
        self.a, self.b, self.n = 0.0, 1.0, int(len(y))
        if len(y) < self.min_n:
            return self
        x = self._logit(p)
        X = np.column_stack([np.ones(len(x)), x])
        theta = np.array([0.0, 1.0])
        prior = np.array([0.0, 1.0])
        s = self.prior_strength
        for _ in range(100):
            q = 1.0 / (1.0 + np.exp(-np.clip(X @ theta, -30, 30)))
            grad = X.T @ (q - y) + 2.0 * s * (theta - prior)
            hess = X.T @ (X * (q * (1 - q))[:, None]) + 2.0 * s * np.eye(2)
            step = np.linalg.solve(hess, grad)
            theta = theta - step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.a, self.b = float(theta[0]), float(theta[1])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        z = self.a + self.b * self._logit(p)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def as_dict(self) -> dict[str, float]:
        return {"a": round(self.a, 4), "b": round(self.b, 4), "n": self.n}


def fit_margin_model(design: pd.DataFrame, train_rows: pd.Index, params: dict[str, Any]) -> Pipeline:
    cols = list(params["margin_model"]["elo_rest_features"])
    model = margin_estimator(params)
    model.fit(_xy(design, train_rows, cols), design.loc[train_rows, "margin"].astype(float).values)
    return model


# ── walk-forward + serving ────────────────────────────────────────────────────

def _season_order(frame: pd.DataFrame) -> list[str]:
    return list(frame.sort_values("local_date")["season"].astype(str).drop_duplicates())


def _seasons_from(frame: pd.DataFrame, first: str | None) -> list[str]:
    """Seasons present in `frame` from `first` on (labels sort chronologically:
    '2024-25' < '2025-26', '2025' < '2026')."""
    order = _season_order(frame)
    if first is None:
        return order
    return [s for s in order if s >= str(first)]


def load_league_tables(normalized_dir: Path, league: str, *, extra_schedules: list[pd.DataFrame] | None = None,
                       extra_details: list[pd.DataFrame] | None = None,
                       extra_players: list[pd.DataFrame] | None = None) -> LeagueTables:
    """Normalized (S3-synced) tables for one league plus any fresher frames."""
    from sports.basketball.backfill_history import read_table

    schedule = read_table(normalized_dir, "schedule", league)
    details = read_table(normalized_dir, "game_details", league)
    players = read_table(normalized_dir, "player_boxscores", league)
    details = pd.concat([details, *(extra_details or [])], ignore_index=True) if extra_details else details
    players = pd.concat([players, *(extra_players or [])], ignore_index=True) if extra_players else players
    if "game_id" in details.columns:
        details["game_id"] = details["game_id"].map(canonical_game_id)
        details = details.drop_duplicates("game_id", keep="last")
    if "game_id" in players.columns:
        players["game_id"] = players["game_id"].map(canonical_game_id)
    return LeagueTables(league=league, schedules=[schedule, *(extra_schedules or [])], details=details, players=players)


def season_label_for_date(league: str, date: pd.Timestamp) -> str:
    """The league season a calendar date belongs to: NBA seasons run September ->
    August ('2026-27' from 2026-09-01, as market_log.season_cross_year(9)); WNBA
    seasons are calendar years ('2026')."""
    d = pd.Timestamp(date)
    start = (d.year if d.month >= 9 else d.year - 1) if league == "nba" else d.year
    return normalize_season_label(league, start)


def data_coverage(tables: LeagueTables, params: dict[str, Any], *, today: pd.Timestamp) -> dict[str, Any]:
    """Missing-season guard for the published record (verifier finding 1).

    Required seasons: every season from the league's ``margin_warmup_from_season``
    through the current one. The current season (``season_label_for_date(today)``)
    counts once it has started: a final competitive game of it before today, or
    today on/after its ``season_started_by_mmdd`` (by then it has surely begun, so
    zero games means the data is missing). Until then the previous season is the
    current one.

    A league is degraded when a required season has no final game, when a COMPLETE
    season (every required one except the in-progress season) has fewer final games
    than ``data_guard.min_season_share`` (0.6) x the median of the complete seasons,
    or when fewer than ``data_guard.min_boxscore_share`` (0.6) of a season's final
    games have team box scores / player box scores (the model's form features).
    Degraded leagues keep their previously published record and get no upcoming
    boards (exporter)."""
    league = tables.league
    lp = params["leagues"][league]
    guard = params.get("data_guard") or {}
    min_share = float(guard.get("min_season_share", 0.6))
    min_box_share = float(guard.get("min_boxscore_share", 0.6))
    today_local = pd.Timestamp(today).normalize()
    results = build_results(league, tables.schedules, today=today_local)
    finals = results.loc[results["final"] & (results["local_date"] < today_local)] if not results.empty else results
    current = season_label_for_date(league, today_local)
    current_start = season_start_year_from_label(league, current)
    started_by = lp.get("season_started_by_mmdd")
    started = bool((finals["season"].astype(str) == current).any()) if not finals.empty else False
    if not started and started_by:
        started = today_local >= pd.Timestamp(f"{current_start}-{started_by}")
    last_start = current_start if started else current_start - 1
    first = lp.get("margin_warmup_from_season")
    first_start = season_start_year_from_label(league, first) if first else last_start
    required = [normalize_season_label(league, y) for y in range(first_start, last_start + 1)]
    in_progress = current if started else None

    def game_ids(frame: pd.DataFrame) -> set[str]:
        if frame is None or frame.empty or "game_id" not in frame.columns:
            return set()
        return set(frame["game_id"].dropna().map(canonical_game_id).dropna())

    with_box, with_players = game_ids(tables.details), game_ids(tables.players)
    seasons: dict[str, dict[str, int]] = {}
    for season in required:
        ids = finals.loc[finals["season"].astype(str) == season, "game_id"] if not finals.empty else pd.Series(dtype=object)
        seasons[season] = {"games": int(len(ids)), "teamBoxscores": int(ids.isin(with_box).sum()),
                           "playerBoxscores": int(ids.isin(with_players).sum())}
    problems: list[str] = []
    for season, c in seasons.items():
        if c["games"] == 0:
            problems.append(f"{season}: no final games (season missing from the tables)")
    complete = [s for s in required if s != in_progress and seasons[s]["games"] > 0]
    median = float(np.median([seasons[s]["games"] for s in complete])) if complete else None
    min_games = min_share * median if median is not None else None
    for season in complete:
        if seasons[season]["games"] < min_games:
            problems.append(f"{season}: {seasons[season]['games']} final games < {min_games:.0f} "
                            f"({min_share:.0%} of the {median:.0f}-game median of complete seasons)")
    for season, c in seasons.items():
        if c["games"] == 0:
            continue
        for key in ("teamBoxscores", "playerBoxscores"):
            if c[key] < min_box_share * c["games"]:
                problems.append(f"{season}: {key} for {c[key]} of {c['games']} final games "
                                f"(< {min_box_share:.0%})")
    return {"league": league, "ok": not problems, "problems": problems, "required": required,
            "inProgress": in_progress, "medianCompleteGames": median,
            "minGames": round(min_games, 1) if min_games is not None else None, "seasons": seasons}


def walk_forward(design: pd.DataFrame, feature_cols: list[str], params: dict[str, Any], league: str, *,
                 win_seasons: list[str], margin_seasons: list[str]) -> pd.DataFrame:
    """14-day expanding-window walk-forward over FINAL games.

    For every block [a, a+14d) of a test season (block edges on local dates, anchored
    at the season's first game) the models are fit on final games with local_date < a
    (all earlier seasons included) and predict the block's final games. Returns one
    row per predicted game with its block start and training cutoff."""
    lp = params["leagues"][league]
    block = pd.Timedelta(days=int(params["walk_forward"]["block_days"]))
    min_train = int(lp.get("min_train_games", 100))
    finals = design.loc[design["final"]]
    out = []
    # Earlier OOS raw win predictions (with outcomes) for the Platt calibrator.
    oos_dates: list[pd.Series] = []
    oos_raw: list[np.ndarray] = []
    oos_y: list[np.ndarray] = []
    for season in sorted(set(win_seasons) | set(margin_seasons), key=_season_order(design).index):
        sd = finals.loc[finals["season"].astype(str) == season]
        if sd.empty:
            continue
        start = sd["local_date"].min()
        end = sd["local_date"].max() + pd.Timedelta(days=1)
        a = start
        while a < end:
            b = a + block
            test = sd.index[(sd["local_date"] >= a) & (sd["local_date"] < b)]
            train = finals.index[finals["local_date"] < a]
            if len(test) and len(train) >= min_train:
                part = pd.DataFrame({"game_id": design.loc[test, "game_id"].values}, index=test)
                part["season"] = season
                part["block_start"] = a
                cutoff = finals.loc[train, "local_date"].max()
                part["n_train"] = len(train)
                if season in win_seasons:
                    model = fit_win_model(design, train, feature_cols, params, league)
                    raw = predict_win(model, design, test)
                    calibrator = ShrunkPlatt.from_params(params)
                    if oos_raw:
                        calibrator.fit(np.concatenate(oos_raw), np.concatenate(oos_y))
                        # every calibration game is dated before this block too
                        cutoff = max(cutoff, max(d.max() for d in oos_dates))
                    part["p_raw"] = raw
                    part["p_home"] = calibrator.transform(raw)
                    part["calib_a"], part["calib_b"], part["calib_n"] = calibrator.a, calibrator.b, calibrator.n
                    oos_dates.append(design.loc[test, "local_date"])
                    oos_raw.append(raw)
                    oos_y.append(design.loc[test, "home_win"].to_numpy(float))
                if season in margin_seasons:
                    mm = fit_margin_model(design, train, params)
                    part["m_elorest"] = mm.predict(_xy(design, test, list(params["margin_model"]["elo_rest_features"])))
                part["train_cutoff"] = cutoff
                out.append(part)
            a = b
    if not out:
        return pd.DataFrame(columns=["game_id", "season", "block_start", "train_cutoff", "n_train", "p_raw", "p_home",
                                     "m_elorest"])
    return pd.concat(out).sort_index()


def combine_margin_total(design: pd.DataFrame, preds: pd.DataFrame, points: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Blend ratings + Elo/rest margin, ratings total, team points."""
    w = float(params["margin_model"].get("ratings_weight", 0.5))
    out = preds.merge(points, on="game_id", how="left")
    out["margin_pred"] = np.where(out["r_margin"].notna(), w * out["r_margin"] + (1 - w) * out["m_elorest"], out["m_elorest"])
    out["total_pred"] = out["r_total"]
    out["home_pts_pred"] = (out["total_pred"] + out["margin_pred"]) / 2.0
    out["away_pts_pred"] = (out["total_pred"] - out["margin_pred"]) / 2.0
    return out


def rolling_oos_sigma(residuals: pd.DataFrame, dates: pd.Series, *, window: int, min_n: int,
                      default: float) -> np.ndarray:
    """sigma for each game = std of the last `window` OOS residuals of games on EARLIER
    local dates (>= min_n of them), else `default`."""
    order = np.argsort(dates.values, kind="stable")
    res = residuals.values[order]
    d = dates.values[order]
    sig = np.full(len(res), float(default))
    # games of the same date see only residuals of strictly earlier dates
    unique_dates, first_idx = np.unique(d, return_index=True)
    for date_value, start in zip(unique_dates, first_idx):
        prior = res[:start]
        prior = prior[~np.isnan(prior)]
        if len(prior) >= min_n:
            sig_value = float(np.std(prior[-window:], ddof=1))
            sig[(d == date_value)] = sig_value
    out = np.empty_like(sig)
    out[order] = sig
    return out


def current_sigma(residuals: pd.Series, *, window: int, min_n: int, default: float) -> float:
    r = residuals.dropna().values
    return float(np.std(r[-window:], ddof=1)) if len(r) >= min_n else float(default)


@dataclass
class LeagueRun:
    league: str
    design: pd.DataFrame
    feature_cols: list[str]
    bundle: FeatureBundle
    history: pd.DataFrame            # walk-forward predictions joined to results (finals)
    upcoming: pd.DataFrame           # predictions for scheduled games
    trained_through: pd.Timestamp | None
    sigmas: dict[str, float]
    timings: dict[str, float]
    calibration: dict[str, float] = field(default_factory=dict)   # served Platt map (a, b, n)


def run_league(tables: LeagueTables, params: dict[str, Any], *, today: pd.Timestamp,
               upcoming_ids: set[str] | None = None, with_history: bool = True) -> LeagueRun | None:
    """Everything the exporter needs for one league, from the raw tables."""
    league = tables.league
    lp = params["leagues"][league]
    timings: dict[str, float] = {}
    t0 = time.time()
    frame = build_game_frame(tables, params, today=today)
    if frame.empty:
        return None
    if upcoming_ids is not None:
        # Rest and Elo above already used the whole schedule; features are only needed
        # for completed games and the slate being baked.
        frame = frame.loc[frame["final"] | frame["game_id"].isin(upcoming_ids)].reset_index(drop=True)
    timings["results_elo"] = time.time() - t0
    t1 = time.time()
    bundle = build_features(frame, tables, include_rotation=bool(params["win_model"].get("include_rotation_features", True)))
    timings.update({f"features_{k}": v for k, v in bundle.timings.items()})
    timings["features"] = time.time() - t1
    design, feature_cols = assemble_design(frame, bundle, params)

    today_local = pd.Timestamp(today).normalize()
    finals = design.loc[design["final"] & (design["local_date"] < today_local)]
    scheduled = design.loc[~design["final"]]
    if upcoming_ids is not None:
        scheduled = scheduled.loc[scheduled["game_id"].isin(upcoming_ids)]

    rp = RatingsParams.from_dict(lp["ratings"])
    t2 = time.time()
    history = pd.DataFrame()
    sig_cfg = params["sigma"]
    defaults = lp["sigma_defaults"]
    if with_history:
        # Warm-up seasons (whose games tuned Elo / C / ratings) are predicted out of sample
        # too -- they seed the Platt calibrator and the rolling sigma -- but only seasons
        # from `history_from_season` on are published or scored.
        warmup_seasons = _seasons_from(finals, lp.get("margin_warmup_from_season"))
        preds = walk_forward(design.loc[design["final"] & (design["local_date"] < today_local)], feature_cols, params,
                             league, win_seasons=warmup_seasons, margin_seasons=warmup_seasons)
        points = predict_points(design, rp, only_ids=set(preds["game_id"])) if not preds.empty else pd.DataFrame(
            columns=["game_id", "r_home_pts", "r_away_pts", "r_margin", "r_total"])
        history = combine_margin_total(design, preds, points, params)
        history = history.merge(design[["game_id", "local_date", "margin", "total", "home_score", "away_score", "home_win",
                                        "p_elo"]], on="game_id", how="left")
        history["resid_margin"] = history["margin"] - history["margin_pred"]
        history["resid_total"] = history["total"] - history["total_pred"]
        history["resid_home_pts"] = history["home_score"] - history["home_pts_pred"]
        history["resid_away_pts"] = history["away_score"] - history["away_pts_pred"]
        history = history.sort_values(["local_date", "game_id"]).reset_index(drop=True)
        published = set(_seasons_from(finals, lp.get("history_from_season")))
        history["published"] = history["season"].astype(str).isin(published)
        team_resid = pd.concat([history[["local_date", "resid_home_pts"]].rename(columns={"resid_home_pts": "r"}),
                                history[["local_date", "resid_away_pts"]].rename(columns={"resid_away_pts": "r"})])
        for name, col, default in (("margin", "resid_margin", defaults["margin"]), ("total", "resid_total", defaults["total"])):
            history[f"sigma_{name}"] = rolling_oos_sigma(history[col], history["local_date"], window=int(sig_cfg["window"]),
                                                         min_n=int(sig_cfg["min_residuals"]), default=float(default))
    timings["walk_forward"] = time.time() - t2

    # Served models: everything final before today.
    t3 = time.time()
    upcoming = pd.DataFrame()
    trained_through = finals["local_date"].max() if not finals.empty else None
    sigmas = {
        "margin": current_sigma(history["resid_margin"], window=int(sig_cfg["window"]), min_n=int(sig_cfg["min_residuals"]),
                                default=float(defaults["margin"])) if not history.empty else float(defaults["margin"]),
        "total": current_sigma(history["resid_total"], window=int(sig_cfg["window"]), min_n=int(sig_cfg["min_residuals"]),
                               default=float(defaults["total"])) if not history.empty else float(defaults["total"]),
        "team_total": (current_sigma(team_resid["r"], window=2 * int(sig_cfg["window"]), min_n=2 * int(sig_cfg["min_residuals"]),
                                     default=float(defaults["team_total"])) if not history.empty else float(defaults["team_total"])),
    }
    # Served calibrator: every out-of-sample prediction made so far (identity without history).
    calibrator = ShrunkPlatt.from_params(params)
    if not history.empty and "p_raw" in history.columns:
        oos = history.dropna(subset=["p_raw", "home_win"]).sort_values(["local_date", "game_id"])
        calibrator.fit(oos["p_raw"].to_numpy(float), oos["home_win"].to_numpy(float))
    if not scheduled.empty and len(finals) >= int(lp.get("min_train_games", 100)):
        win = fit_win_model(design, finals.index, feature_cols, params, league)
        mm = fit_margin_model(design, finals.index, params)
        upcoming = pd.DataFrame({"game_id": scheduled["game_id"].values}, index=scheduled.index)
        upcoming["p_raw"] = predict_win(win, design, scheduled.index)
        upcoming["p_home"] = calibrator.transform(upcoming["p_raw"].to_numpy(float))
        upcoming["m_elorest"] = mm.predict(_xy(design, scheduled.index, list(params["margin_model"]["elo_rest_features"])))
        points = predict_points(design, rp, only_ids=set(upcoming["game_id"]))
        upcoming = combine_margin_total(design, upcoming, points, params)
        upcoming["train_cutoff"] = trained_through
        upcoming["n_train"] = len(finals)
    elif not scheduled.empty:
        # Too little history to fit (e.g. a brand-new league): Elo is the honest fallback.
        upcoming = pd.DataFrame({"game_id": scheduled["game_id"].values, "p_home": scheduled["p_elo"].values})
        upcoming["margin_pred"] = np.nan
        upcoming["total_pred"] = np.nan
        upcoming["train_cutoff"] = trained_through
        upcoming["n_train"] = len(finals)
    timings["serve"] = time.time() - t3
    return LeagueRun(league=league, design=design, feature_cols=feature_cols, bundle=bundle, history=history,
                     upcoming=upcoming, trained_through=trained_through, sigmas=sigmas, timings=timings,
                     calibration=calibrator.as_dict())


# ── metrics ───────────────────────────────────────────────────────────────────

def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def win_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """n / accuracy / Brier / log-loss. A probability within NO_PICK_BAND of 0.5 is NO
    pick: it is left out of accuracy (n_picks counts the rest) but kept in Brier and
    log-loss. Counting 0.5 as a home pick is what inflated the WNBA cold-start claim."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p) & ~np.isnan(y)
    y, p = y[ok], p[ok]
    if not len(y):
        return {"n": 0}
    pick = is_pick(p)
    accuracy = float(np.mean((p[pick] > 0.5) == (y[pick] == 1))) if pick.any() else float("nan")
    return {"n": int(len(y)), "n_picks": int(pick.sum()), "accuracy": accuracy,
            "brier": float(np.mean((p - y) ** 2)), "log_loss": log_loss(y, p)}


def _irls_logit(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by Newton/IRLS; returns (coefficients, standard errors)."""
    off = np.zeros(len(y)) if offset is None else offset
    beta = np.zeros(X.shape[1])
    with np.errstate(all="ignore"):
        for _ in range(100):
            mu = 1.0 / (1.0 + np.exp(-np.clip(off + X @ beta, -30, 30)))
            hess = X.T @ (X * (mu * (1 - mu))[:, None])
            step = np.linalg.solve(hess, X.T @ (y - mu))
            beta = beta + step
            if np.max(np.abs(step)) < 1e-10:
                break
        mu = 1.0 / (1.0 + np.exp(-np.clip(off + X @ beta, -30, 30)))
        cov = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    return beta, np.sqrt(np.diag(cov))


def calibration_metrics(y: np.ndarray, p: np.ndarray, *, bucket: float = 0.8) -> dict[str, Any]:
    """Calibration slope (logistic regression of y on logit p; 1 = calibrated) and
    calibration-in-the-large intercept (slope fixed at 1; 0 = calibrated), each with a
    95% Wald CI, plus the favourite-probability >= `bucket` bucket: mean stated
    probability vs how often that favourite won."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p) & ~np.isnan(y)
    y, p = y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6)
    if len(y) < 30:
        return {"n": int(len(y))}
    lg = np.log(p / (1 - p))
    b, se = _irls_logit(np.column_stack([np.ones(len(y)), lg]), y)
    b0, se0 = _irls_logit(np.ones((len(y), 1)), y, offset=lg)
    fav = np.maximum(p, 1 - p)
    won = np.where(p > 0.5, y, 1 - y)
    sel = fav >= bucket
    out: dict[str, Any] = {
        "n": int(len(y)),
        "slope": float(b[1]), "slope_ci": [float(b[1] - 1.96 * se[1]), float(b[1] + 1.96 * se[1])],
        "intercept": float(b0[0]), "intercept_ci": [float(b0[0] - 1.96 * se0[0]), float(b0[0] + 1.96 * se0[0])],
        f"bucket_{int(bucket * 100)}": {"n": int(sel.sum()),
                                        "mean_p": float(fav[sel].mean()) if sel.any() else None,
                                        "hit": float(won[sel].mean()) if sel.any() else None},
    }
    out["slope_ci_includes_1"] = bool(out["slope_ci"][0] <= 1.0 <= out["slope_ci"][1])
    out["intercept_ci_includes_0"] = bool(out["intercept_ci"][0] <= 0.0 <= out["intercept_ci"][1])
    return out


def paired_brier_diff(y: np.ndarray, p_model: np.ndarray, p_ref: np.ndarray) -> dict[str, float]:
    """Model minus reference Brier on the SAME games, with a 95% CI from the per-game
    differences (negative = model better). Small-n targets are judged on this, not on
    raw point thresholds (plan review, binding)."""
    y = np.asarray(y, dtype=float)
    a = np.asarray(p_model, dtype=float)
    b = np.asarray(p_ref, dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(a) & ~np.isnan(b)
    d = (a[ok] - y[ok]) ** 2 - (b[ok] - y[ok]) ** 2
    if len(d) < 2:
        return {"n": int(len(d))}
    se = float(np.std(d, ddof=1) / math.sqrt(len(d)))
    mean = float(np.mean(d))
    return {"n": int(len(d)), "diff": mean, "ci": [mean - 1.96 * se, mean + 1.96 * se]}


def interval_coverage(residual: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    z = np.abs(np.asarray(residual, dtype=float) / np.asarray(sigma, dtype=float))
    z = z[~np.isnan(z)]
    if not len(z):
        return {}
    return {"cov68": float(np.mean(z <= 1.0)), "cov80": float(np.mean(z <= 1.2816)), "cov95": float(np.mean(z <= 1.96))}


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
