"""Walk-forward (strictly out-of-sample) Dixon-Coles predictions and their scoring.

Used twice, with the same code path:
  * the in-app Track Record (scripts/export_soccer_frontend_data.py): every league
    match since the record's first season, each predicted by a model fitted only on
    matches dated before that match's refit cut-off;
  * the acceptance backtest (scripts/eval_soccer_walkforward.py) that produced the
    numbers in sports/soccer/model_params.json.

Refit cadence: twice a week, cut-offs on Monday and Friday (``refitWeekdays``), so a
weekend round is predicted from a fit through Thursday and a midweek round from a
fit through Sunday -- close to what the daily export actually shows, and at least
weekly. A match dated on/after its cut-off is never in its own training set.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from .dixon_coles import DEFAULT_MAX_GOALS, DixonColesModel, score_matrices

MIN_TRAIN_MATCHES = 50


# ── fitting ──────────────────────────────────────────────────────────────────

def refit_cutoff(day: pd.Timestamp, weekdays: Iterable[int] = (0, 4)) -> pd.Timestamp:
    """The latest refit day (weekday in `weekdays`, Mon=0) on or before `day`."""
    day = pd.Timestamp(day).normalize()
    wanted = set(int(w) for w in weekdays)
    for back in range(7):
        cand = day - pd.Timedelta(days=back)
        if cand.weekday() in wanted:
            return cand
    return day


def decay_weights(dates: pd.Series, xi: float, reference: Optional[pd.Timestamp] = None) -> np.ndarray:
    """exp(-xi * days before the reference date (default: newest match))."""
    valid = pd.to_datetime(dates)
    ref = reference if reference is not None else valid.max()
    days = (ref - valid).dt.days.fillna(0).clip(lower=0).to_numpy(dtype=float)
    return np.exp(-float(xi) * days)


def training_window(history: pd.DataFrame, season: int, window_seasons: int,
                    before: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Matches of seasons [season - window + 1, season], dated strictly before `before`."""
    mask = (history["season_start"] >= season - (int(window_seasons) - 1)) & (history["season_start"] <= season)
    if before is not None:
        mask &= history["date"] < before
    return history[mask]


def fit_model(train: pd.DataFrame, params: dict[str, Any],
              warm_start: Optional[DixonColesModel] = None) -> Optional[DixonColesModel]:
    if len(train) < MIN_TRAIN_MATCHES:
        return None
    w = decay_weights(train["date"], params["timeDecayXiPerDay"])
    return DixonColesModel().fit(
        train["home"], train["away"],
        train["home_goals"].astype(int), train["away_goals"].astype(int),
        weights=w, l2=float(params["l2"]), warm_start=warm_start,
    )


def walk_forward(history: pd.DataFrame, params: dict[str, Any], start_season: int,
                 end_season: Optional[int] = None) -> pd.DataFrame:
    """Out-of-sample expected goals for every completed match of seasons
    [start_season, end_season] in ONE league's `history` (columns date, home, away,
    home_goals, away_goals, season_start; extra columns are carried through)."""
    hist = history.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date", kind="mergesort")
    mask = hist["season_start"] >= int(start_season)
    if end_season is not None:
        mask &= hist["season_start"] <= int(end_season)
    test = hist[mask].copy()
    if test.empty:
        return test.assign(lam=[], mu=[], rho=[], home_known=[], away_known=[], n_train=[], cutoff=[])
    weekdays = params.get("refitWeekdays", [0, 4])
    test["cutoff"] = test["date"].map(lambda d: refit_cutoff(d, weekdays))
    out: list[pd.DataFrame] = []
    prev: Optional[DixonColesModel] = None
    for (cutoff, season), grp in test.groupby(["cutoff", "season_start"], sort=True):
        train = training_window(hist, int(season), int(params["windowSeasons"]), before=cutoff)
        model = fit_model(train, params, warm_start=prev)
        if model is None:
            continue
        prev = model
        rates = [model.rates(h, a) for h, a in zip(grp["home"], grp["away"])]
        part = grp.copy()
        part["lam"] = [r[0] for r in rates]
        part["mu"] = [r[1] for r in rates]
        part["rho"] = model.rho
        part["home_known"] = [model.knows_team(h) for h in grp["home"]]
        part["away_known"] = [model.knows_team(a) for a in grp["away"]]
        part["n_train"] = len(train)
        out.append(part)
    if not out:
        return test.iloc[0:0].assign(lam=[], mu=[], rho=[], home_known=[], away_known=[], n_train=[])
    return pd.concat(out, ignore_index=True)


# ── derived probabilities ────────────────────────────────────────────────────

def derive(lam: np.ndarray, mu: np.ndarray, rho: np.ndarray, max_goals: int = DEFAULT_MAX_GOALS) -> dict[str, np.ndarray]:
    """Per-match market probabilities from the full score matrix."""
    m = score_matrices(lam, mu, rho, max_goals)
    i, j = np.indices(m.shape[1:])
    diff, tot = i - j, i + j
    out = {
        "pH": m[:, diff > 0].sum(1),
        "pD": m[:, diff == 0].sum(1),
        "pA": m[:, diff < 0].sum(1),
        "pBTTS": m[:, (i >= 1) & (j >= 1)].sum(1),
        "expGoals": (m * tot[None]).sum(axis=(1, 2)),
    }
    for line in (1.5, 2.5, 3.5):
        out[f"pOver{line}"] = m[:, tot > line].sum(1)
    flat = m.reshape(len(m), -1)
    out["csTop3"] = np.argsort(-flat, axis=1, kind="stable")[:, :3]   # flat index = i*(G+1)+j
    out["csTop3Prob"] = np.take_along_axis(flat, out["csTop3"], 1)
    return out


# ── scoring ──────────────────────────────────────────────────────────────────

def _outcome(hg: np.ndarray, ag: np.ndarray) -> np.ndarray:
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def metrics_1x2(P: np.ndarray, y: np.ndarray) -> dict[str, float]:
    P = np.clip(np.asarray(P, float), 1e-12, 1.0)
    P = P / P.sum(1, keepdims=True)
    Y = np.eye(3)[y]
    cp, cy = np.cumsum(P, 1)[:, :2], np.cumsum(Y, 1)[:, :2]
    return {
        "n": int(len(y)),
        "accuracy": float((P.argmax(1) == y).mean()) if len(y) else float("nan"),
        "brier": float(((P - Y) ** 2).sum(1).mean()) if len(y) else float("nan"),
        "rps": float((((cp - cy) ** 2).sum(1) / 2).mean()) if len(y) else float("nan"),
        "logLoss": float(-np.log(P[np.arange(len(y)), y]).mean()) if len(y) else float("nan"),
    }


def metrics_binary(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    if not len(y):
        return {"n": 0, "accuracy": float("nan"), "brier": float("nan"), "logLoss": float("nan")}
    return {
        "n": int(len(y)),
        "accuracy": float(((p > 0.5) == (y > 0.5)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "logLoss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
    }


def devig_rows(*odds: np.ndarray) -> np.ndarray:
    inv = np.stack([1.0 / np.asarray(o, float) for o in odds], axis=1)
    return inv / inv.sum(1, keepdims=True)


def score_predictions(preds: pd.DataFrame, max_goals: int = DEFAULT_MAX_GOALS) -> dict[str, Any]:
    """Model metrics for a block of walk-forward predictions, plus the de-vigged
    football-data closing-average market on the same matches when the odds columns
    are present (close_avg_home/draw/away, close_avg_over25/under25)."""
    if preds.empty:
        return {"n": 0}
    hg = preds["home_goals"].to_numpy(dtype=int)
    ag = preds["away_goals"].to_numpy(dtype=int)
    y = _outcome(hg, ag)
    d = derive(preds["lam"].to_numpy(float), preds["mu"].to_numpy(float), preds["rho"].to_numpy(float), max_goals)
    P = np.c_[d["pH"], d["pD"], d["pA"]]
    tot = hg + ag
    actual_cell = np.minimum(hg, max_goals) * (max_goals + 1) + np.minimum(ag, max_goals)
    out: dict[str, Any] = {"model": metrics_1x2(P, y)}
    out["overUnder25"] = metrics_binary(d["pOver2.5"], (tot > 2.5).astype(int))
    out["btts"] = metrics_binary(d["pBTTS"], ((hg > 0) & (ag > 0)).astype(int))
    out["correctScore"] = {
        "top1HitRate": float((d["csTop3"][:, 0] == actual_cell).mean()),
        "top3HitRate": float((d["csTop3"] == actual_cell[:, None]).any(1).mean()),
        "naiveTop3HitRate": float(np.isin(actual_cell, [1 * (max_goals + 1) + 1, 1 * (max_goals + 1), 2 * (max_goals + 1) + 1]).mean()),
    }
    out["goals"] = {"meanPredicted": float(d["expGoals"].mean()), "meanActual": float(tot.mean())}
    out["baselines"] = {
        "alwaysHomeAccuracy": float((y == 0).mean()),
        "over25BaseRate": float((tot > 2.5).mean()),
        "bttsBaseRate": float(((hg > 0) & (ag > 0)).mean()),
    }
    # constant base-rate log-losses (in-sample rates: a generous yardstick)
    for key, ybin in (("over25", (tot > 2.5).astype(float)), ("btts", ((hg > 0) & (ag > 0)).astype(float))):
        r = float(ybin.mean())
        if 0 < r < 1:
            out["baselines"][f"{key}BaseRateLogLoss"] = float(-(ybin * math.log(r) + (1 - ybin) * math.log(1 - r)).mean())
    cols = ["close_avg_home", "close_avg_draw", "close_avg_away"]
    if all(c in preds.columns for c in cols):
        ok = preds[cols].notna().all(1).to_numpy() & (preds[cols] > 1.0).all(1).to_numpy()
        if ok.any():
            Pm = devig_rows(*(preds.loc[ok, c].to_numpy(float) for c in cols))
            market = metrics_1x2(Pm, y[ok])
            market["favouriteWinRate"] = market.pop("accuracy")
            market["source"] = "football-data.co.uk closing average, de-vigged"
            out["market"] = market
            out["modelOnMarketRows"] = metrics_1x2(P[ok], y[ok])
    ou = ["close_avg_over25", "close_avg_under25"]
    if all(c in preds.columns for c in ou):
        ok = preds[ou].notna().all(1).to_numpy() & (preds[ou] > 1.0).all(1).to_numpy()
        if ok.any():
            pm = devig_rows(*(preds.loc[ok, c].to_numpy(float) for c in ou))[:, 0]
            out["marketOverUnder25"] = metrics_binary(pm, (tot[ok] > 2.5).astype(int))
    return out
