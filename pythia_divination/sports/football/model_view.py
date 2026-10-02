"""NFL "model view": a compact, regularized, calibrated home-win model.

Why this exists. The old served models (a depth-8 HistGradientBoosting and a torch
MLP over every numeric column, ~440 features on ~1,300 games) were badly
overconfident: under a season walk-forward (2022-2025, n=1,084) the HGB recipe scored
Brier 0.2704 / log-loss 0.895, worse than a coin flip, while the de-vigged closing
moneyline scored 0.2105 / 0.608. The market is now the headline; this module is the
labelled "model view" shown next to it (and the headline only when no line is posted).

Design (every input is pregame-safe and available at serve time):
  * Walk-forward team ratings: exponentially decayed ridge fits of margin (SRS-style)
    and of offense/defense points, re-fit before every game week on games completed
    before that week only. They are computed from scores, so the daily export rebuilds
    them from the freshly ingested nflverse schedule.
  * An explicit FEATURE_ALLOWLIST (no "every numeric column"): ratings, rolling team and
    QB form differentials, rest and site context. No betting lines (the model view must
    be independent of the market it is compared with), no post-game columns (overtime,
    scores), no game-time weather and no game-week roster statuses (both are only
    known after the board is published, i.e. train/serve skew).
  * Ridge regression of the home margin; the prediction is the "model fair spread"
    (shown as modelLine only, never as a pick).
  * Platt scaling of that margin into P(home win), fit on PRIOR-season out-of-sample
    predictions only, so the probability is calibrated rather than overconfident.

Everything the exporter needs is a small JSON (model_params.json, committed next to
this file) applied with numpy alone, so serving does not depend on a pickled
scikit-learn object or on a gitignored artifact.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

PARAMS_PATH = Path(__file__).resolve().parent / "model_params.json"
METRICS_PATH = Path(__file__).resolve().parent / "metrics.json"

RATING_COLUMNS = ["rt_margin", "rt_home_pts", "rt_away_pts"]

# Explicit allowlist for the margin (and therefore home-win) model. Order matters: it
# is the coefficient order in model_params.json.
FEATURE_ALLOWLIST: list[str] = [
    *RATING_COLUMNS,
    "team_diff_point_diff_avg_last_3",
    "team_diff_point_diff_avg_last_10",
    "team_diff_offense_epa_total_avg_last_3",
    "team_diff_offense_epa_total_avg_last_10",
    "team_diff_passing_epa_avg_last_10",
    "team_diff_rushing_epa_avg_last_10",
    "team_diff_points_scored_avg_last_10",
    "team_diff_points_allowed_avg_last_10",
    "team_diff_won_avg_last_10",
    "team_diff_sacks_suffered_avg_last_10",
    "team_diff_def_sacks_avg_last_10",
    "team_diff_passing_interceptions_avg_last_10",
    "team_diff_def_interceptions_avg_last_10",
    "qb_diff_passing_epa_avg_last_3",
    "qb_diff_passing_epa_avg_last_10",
    "qb_diff_passing_cpoe_avg_last_10",
    "qb_diff_games_played_prior",
    "home_qb_games_played_prior",
    "away_qb_games_played_prior",
    "home_rest",
    "away_rest",
    "div_game",
    "is_neutral_site",
    "roof_is_closed",
]

# Allowlist for the (information-only) model fair total.
TOTAL_FEATURE_ALLOWLIST: list[str] = [
    "rt_home_pts",
    "rt_away_pts",
    "home_team_points_scored_avg_last_10",
    "away_team_points_scored_avg_last_10",
    "home_team_points_allowed_avg_last_10",
    "away_team_points_allowed_avg_last_10",
    "home_team_offense_epa_total_avg_last_10",
    "away_team_offense_epa_total_avg_last_10",
    "div_game",
    "is_neutral_site",
    "roof_is_closed",
]

DEFAULT_RATING_PARAMS = {"half_life_days": 200.0, "lam": 10.0}


# ── walk-forward team ratings ─────────────────────────────────────────────────

def _week_starts(frame: pd.DataFrame) -> np.ndarray:
    starts = frame.groupby(["season", "week"])["official_date"].transform("min")
    return pd.to_datetime(starts).values


def walkforward_ratings(games: pd.DataFrame, *, half_life_days: float = 200.0, lam: float = 10.0,
                        min_games: int = 60) -> pd.DataFrame:
    """Ratings for every row of `games` (completed or not), each fit ONLY on games
    completed strictly before the start of that row's (season, week).

    `games` needs: season, week, official_date, home_team, away_team, home_score,
    away_score, is_neutral_site. Returns rt_margin (home minus away incl. home field),
    rt_home_pts, rt_away_pts, rt_total, rt_n (training games used), index-aligned."""
    frame = games.reset_index(drop=True)
    out = pd.DataFrame(np.nan, index=frame.index, columns=["rt_margin", "rt_home_pts", "rt_away_pts", "rt_hfa", "rt_n"])
    if frame.empty:
        out["rt_total"] = np.nan
        return out.set_index(games.index)
    teams = sorted(set(frame["home_team"].astype(str)) | set(frame["away_team"].astype(str)))
    tix = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    official = pd.to_datetime(frame["official_date"]).values
    week_start = _week_starts(frame)
    home_score = pd.to_numeric(frame["home_score"], errors="coerce").to_numpy(float)
    away_score = pd.to_numeric(frame["away_score"], errors="coerce").to_numpy(float)
    done = ~(np.isnan(home_score) | np.isnan(away_score))
    hi_all = frame["home_team"].astype(str).map(tix).to_numpy()
    ai_all = frame["away_team"].astype(str).map(tix).to_numpy()
    if "is_neutral_site" in frame.columns:
        neutral_all = pd.to_numeric(frame["is_neutral_site"], errors="coerce").fillna(0).to_numpy(float)
    else:
        neutral_all = np.zeros(len(frame))

    d_idx = np.where(done)[0]
    d_dates = official[d_idx]
    # numpy 2 + macOS Accelerate emits spurious divide/overflow warnings from matmul on
    # these 0/+-1 design matrices; correctness is enforced by the finiteness check.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for ws in np.unique(week_start):
            prior = d_idx[d_dates < ws]
            if len(prior) < min_games:
                continue
            age = (np.datetime64(ws) - official[prior]).astype("timedelta64[D]").astype(float)
            w = 0.5 ** (age / float(half_life_days))
            h, a, nn = hi_all[prior], ai_all[prior], neutral_all[prior]
            n = len(prior)
            rows = np.arange(n)
            # margin: home - away = r_h - r_a + hfa * (1 - neutral)
            A = np.zeros((n, n_teams + 1))
            A[rows, h] = 1.0
            A[rows, a] = -1.0
            A[:, n_teams] = 1.0 - nn
            y = home_score[prior] - away_score[prior]
            R = lam * np.eye(n_teams + 1)
            R[n_teams, n_teams] = 1e-3
            beta = np.linalg.solve(A.T @ (A * w[:, None]) + R, A.T @ (w * y))
            # points: pts = mu + hfa_pts*(home & not neutral) + off_team - def_opp
            B = np.zeros((2 * n, 2 * n_teams + 2))
            B[rows, h] = 1.0
            B[rows, n_teams + a] = -1.0
            B[rows, 2 * n_teams] = 1.0
            B[rows, 2 * n_teams + 1] = 1.0 - nn
            B[n + rows, a] = 1.0
            B[n + rows, n_teams + h] = -1.0
            B[n + rows, 2 * n_teams] = 1.0
            yp = np.r_[home_score[prior], away_score[prior]]
            wp = np.r_[w, w]
            R2 = lam * np.eye(2 * n_teams + 2)
            R2[2 * n_teams, 2 * n_teams] = 1e-6
            R2[2 * n_teams + 1, 2 * n_teams + 1] = 1e-3
            gamma = np.linalg.solve(B.T @ (B * wp[:, None]) + R2, B.T @ (wp * yp))
            if not (np.isfinite(beta).all() and np.isfinite(gamma).all()):
                raise FloatingPointError(f"non-finite team ratings for week starting {ws}")
            sel = np.where(week_start == ws)[0]
            th, ta, neu = hi_all[sel], ai_all[sel], neutral_all[sel]
            out.loc[sel, "rt_margin"] = beta[th] - beta[ta] + beta[n_teams] * (1.0 - neu)
            out.loc[sel, "rt_home_pts"] = gamma[2 * n_teams] + gamma[2 * n_teams + 1] * (1.0 - neu) + gamma[th] - gamma[n_teams + ta]
            out.loc[sel, "rt_away_pts"] = gamma[2 * n_teams] + gamma[ta] - gamma[n_teams + th]
            out.loc[sel, "rt_hfa"] = beta[n_teams]
            out.loc[sel, "rt_n"] = float(n)
    out["rt_total"] = out["rt_home_pts"] + out["rt_away_pts"]
    out.index = games.index
    return out


def attach_ratings(games: pd.DataFrame, rating_params: Optional[dict] = None) -> pd.DataFrame:
    params = {**DEFAULT_RATING_PARAMS, **(rating_params or {})}
    ratings = walkforward_ratings(games, half_life_days=float(params["half_life_days"]), lam=float(params["lam"]))
    merged = games.drop(columns=[c for c in ratings.columns if c in games.columns])
    return pd.concat([merged, ratings], axis=1)


# ── linear model with numpy-only serving ──────────────────────────────────────

def feature_matrix(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Columns in allowlist order; absent columns become NaN (imputed at predict)."""
    cols = list(columns)
    return (frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan))


def fit_ridge(frame: pd.DataFrame, y: Iterable[float], columns: list[str],
              alphas: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Median-impute, standardize, RidgeCV. Returns plain-JSON params."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    X = feature_matrix(frame, columns)
    keep = [c for c in columns if X[c].notna().any()]
    X = X[keep]
    imputer = SimpleImputer(strategy="median").fit(X)
    Xi = imputer.transform(X)
    scaler = StandardScaler().fit(Xi)
    Xs = scaler.transform(Xi)
    model = RidgeCV(alphas=alphas if alphas is not None else np.logspace(-1, 4.5, 30)).fit(Xs, np.asarray(y, float))
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    return {
        "features": keep,
        "medians": [float(v) for v in imputer.statistics_],
        "means": [float(v) for v in scaler.mean_],
        "scales": [float(v) for v in scale],
        "coef": [float(v) for v in model.coef_],
        "intercept": float(model.intercept_),
        "alpha": float(model.alpha_),
    }


def predict_linear(params: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    X = feature_matrix(frame, params["features"]).to_numpy(float)
    medians = np.asarray(params["medians"], float)
    X = np.where(np.isnan(X), medians[None, :], X)
    Z = (X - np.asarray(params["means"], float)) / np.asarray(params["scales"], float)
    return Z @ np.asarray(params["coef"], float) + float(params["intercept"])


def _logistic_irls(x: np.ndarray, y: np.ndarray, *, offset: Optional[np.ndarray] = None,
                   with_slope: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Unpenalized logistic regression of y on [1, x] (or [1] with x as an offset) by
    Newton/IRLS. Returns (beta, covariance). Two parameters, so no solver tuning."""
    X = np.c_[np.ones_like(x), x] if with_slope else np.ones((len(x), 1))
    off = np.zeros_like(x) if offset is None else offset
    beta = np.zeros(X.shape[1])
    # errstate: see walkforward_ratings (spurious Accelerate matmul warnings); the
    # result is checked for finiteness instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(100):
            eta = np.clip(X @ beta + off, -30.0, 30.0)
            mu = 1.0 / (1.0 + np.exp(-eta))
            H = X.T @ (X * (mu * (1 - mu))[:, None])
            step = np.linalg.solve(H, X.T @ (y - mu))
            beta = beta + step
            if np.max(np.abs(step)) < 1e-10:
                break
        eta = np.clip(X @ beta + off, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        cov = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    if not (np.isfinite(beta).all() and np.isfinite(cov).all()):
        raise FloatingPointError("logistic fit did not converge to finite values")
    return beta, cov


def fit_platt(score: Iterable[float], y: Iterable[float]) -> dict[str, float]:
    """P(y=1) = sigmoid(a + b * score), fit on prior out-of-sample scores."""
    s = np.asarray(list(score), float)
    beta, _ = _logistic_irls(s, np.asarray(list(y), float))
    return {"a": float(beta[0]), "b": float(beta[1])}


def platt_predict(platt: dict[str, float], score: Iterable[float] | float) -> np.ndarray:
    s = np.asarray(score, float)
    z = np.clip(float(platt["a"]) + float(platt["b"]) * s, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


# ── training frame, walk-forward, final fit ───────────────────────────────────

SCORE_COLUMNS = ["home_score", "away_score"]
LINE_COLUMNS = ["home_moneyline", "away_moneyline", "spread_line", "home_spread_odds", "away_spread_odds",
                "total_line", "over_odds", "under_odds"]


def american_to_prob(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values) if not isinstance(values, (pd.Series, np.ndarray)) else values, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, -a / (-a + 100.0), 100.0 / (a + 100.0))


def market_home_probability(frame: pd.DataFrame) -> pd.Series:
    """De-vigged (proportional) closing/posted moneyline P(home win); NaN without a line."""
    ph = american_to_prob(pd.to_numeric(frame["home_moneyline"], errors="coerce"))
    pa = american_to_prob(pd.to_numeric(frame["away_moneyline"], errors="coerce"))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = ph / (ph + pa)
    return pd.Series(p, index=frame.index, dtype=float)


def prepare_training_frame(dataset: pd.DataFrame, games: pd.DataFrame,
                           rating_params: Optional[dict] = None) -> pd.DataFrame:
    """Dataset rows (pregame features) + scores and lines re-joined from the games table
    (the dataset drops them as leaky) + walk-forward ratings + labels."""
    frame = dataset.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    g = games.copy()
    g["game_id"] = g["game_id"].astype(str)
    join = [c for c in SCORE_COLUMNS + LINE_COLUMNS if c in g.columns]
    frame = frame.drop(columns=[c for c in join if c in frame.columns]).merge(g[["game_id", *join]], on="game_id", how="left")
    for c in join:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame["official_date"] = pd.to_datetime(frame["official_date"], errors="coerce", format="mixed")
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype(int)
    frame = frame.sort_values(["official_date", "game_id"]).reset_index(drop=True)
    frame["margin"] = frame["home_score"] - frame["away_score"]
    frame["pts_total"] = frame["home_score"] + frame["away_score"]
    # Ties are not a home win or a home loss: no binary label.
    frame["home_win"] = np.where(frame["margin"] > 0, 1.0, np.where(frame["margin"] < 0, 0.0, np.nan))
    frame["market_home_win_probability"] = market_home_probability(frame)
    return attach_ratings(frame, rating_params)


def tune_rating_params(frame: pd.DataFrame, validation_season: int) -> dict[str, float]:
    """Pick half-life / ridge strength by next-season RMSE on ONE validation season that
    precedes every test season (no test-season selection)."""
    sub = frame[frame["season"] <= validation_season].reset_index(drop=True)
    best: Optional[tuple[float, float, float]] = None
    for half_life in (100.0, 200.0, 400.0):
        for lam in (1.0, 3.0, 10.0):
            r = walkforward_ratings(sub, half_life_days=half_life, lam=lam)
            m = (sub["season"] == validation_season) & r["rt_margin"].notna() & sub["margin"].notna()
            if not m.any():
                continue
            err = float(np.sqrt(((sub["margin"] - r["rt_margin"])[m] ** 2).mean())
                        + np.sqrt(((sub["pts_total"] - r["rt_total"])[m] ** 2).mean()))
            if best is None or err < best[0]:
                best = (err, half_life, lam)
    if best is None:
        return dict(DEFAULT_RATING_PARAMS)
    return {"half_life_days": best[1], "lam": best[2]}


def walk_forward(frame: pd.DataFrame, *, test_seasons: Iterable[int], first_oos_season: int) -> pd.DataFrame:
    """Season walk-forward: for every season S >= first_oos_season the ridge models are
    fit on seasons < S only; the Platt calibration used for S is fit on the out-of-sample
    margins of seasons first_oos_season..S-1 only. Returns oos_margin, oos_total and
    model_home_win_probability (NaN outside test_seasons)."""
    tests = sorted({int(s) for s in test_seasons})
    out = pd.DataFrame(np.nan, index=frame.index, columns=["oos_margin", "oos_total", "model_home_win_probability"])
    if not tests:
        return out
    for season in range(int(first_oos_season), max(tests) + 1):
        train = frame[(frame["season"] < season) & frame["margin"].notna()]
        test = frame[frame["season"] == season]
        if train.empty or test.empty:
            continue
        margin_params = fit_ridge(train, train["margin"], FEATURE_ALLOWLIST)
        out.loc[test.index, "oos_margin"] = predict_linear(margin_params, test)
        total_params = fit_ridge(train, train["pts_total"], TOTAL_FEATURE_ALLOWLIST)
        out.loc[test.index, "oos_total"] = predict_linear(total_params, test)
    for season in tests:
        prior = frame[(frame["season"] >= first_oos_season) & (frame["season"] < season)
                      & frame["home_win"].notna() & out["oos_margin"].notna()]
        test = frame[frame["season"] == season]
        if prior.empty or test.empty:
            continue
        platt = fit_platt(out.loc[prior.index, "oos_margin"], prior["home_win"])
        out.loc[test.index, "model_home_win_probability"] = platt_predict(platt, out.loc[test.index, "oos_margin"])
    return out


def fit_final(frame: pd.DataFrame, *, train_through_season: int, first_oos_season: int,
              oos_margin: pd.Series) -> dict[str, Any]:
    """Serving params: ridge models on seasons <= train_through_season, Platt on the
    walk-forward out-of-sample margins of first_oos_season..train_through_season."""
    train = frame[(frame["season"] <= train_through_season) & frame["margin"].notna()]
    prior = frame[(frame["season"] >= first_oos_season) & (frame["season"] <= train_through_season)
                  & frame["home_win"].notna() & oos_margin.notna()]
    return {
        "margin_model": fit_ridge(train, train["margin"], FEATURE_ALLOWLIST),
        "total_model": fit_ridge(train, train["pts_total"], TOTAL_FEATURE_ALLOWLIST),
        "platt": fit_platt(oos_margin.loc[prior.index], prior["home_win"]),
        "train_rows": int(len(train)),
        "platt_rows": int(len(prior)),
    }


# ── params file ───────────────────────────────────────────────────────────────

def load_params(path: Path = PARAMS_PATH) -> Optional[dict[str, Any]]:
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def predict_model_view(params: dict[str, Any], games_with_features: pd.DataFrame) -> pd.DataFrame:
    """Model-view outputs for a frame that already carries the pregame features and
    ratings: model_margin (home), model_home_win_probability, model_total."""
    out = pd.DataFrame(index=games_with_features.index)
    margin = predict_linear(params["margin_model"], games_with_features)
    out["model_margin"] = margin
    out["model_home_win_probability"] = platt_predict(params["platt"], margin)
    if params.get("total_model"):
        out["model_total"] = predict_linear(params["total_model"], games_with_features)
    else:
        out["model_total"] = np.nan
    # Ratings need >= 60 completed prior games; without them the linear model falls
    # back to imputed medians, which is a weak but still calibrated estimate.
    return out


def logit(p: np.ndarray | float) -> np.ndarray:
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(q / (1.0 - q))


def calibration_fit(p: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    """Calibration slope/intercept of y on logit(p) with 95% Wald CIs (IRLS), plus
    calibration-in-the-large (intercept with slope fixed at 1). A calibrated forecast
    has slope 1 and intercept 0; unlike a per-decile rule this accounts for n."""
    x = logit(np.asarray(list(p), float))
    yy = np.asarray(list(y), float)
    beta, cov = _logistic_irls(x, yy)
    se = np.sqrt(np.diag(cov))
    # calibration-in-the-large: logit P(y) = a + 1 * logit(p)
    a_beta, a_cov = _logistic_irls(x, yy, offset=x, with_slope=False)
    a, se_a = float(a_beta[0]), float(math.sqrt(a_cov[0, 0]))
    return {
        "n": int(len(yy)),
        "intercept": round(float(beta[0]), 4),
        "intercept_ci95": [round(float(beta[0] - 1.96 * se[0]), 4), round(float(beta[0] + 1.96 * se[0]), 4)],
        "slope": round(float(beta[1]), 4),
        "slope_ci95": [round(float(beta[1] - 1.96 * se[1]), 4), round(float(beta[1] + 1.96 * se[1]), 4)],
        "calibration_in_the_large": round(float(a), 4),
        "calibration_in_the_large_ci95": [round(float(a - 1.96 * se_a), 4), round(float(a + 1.96 * se_a), 4)],
    }
