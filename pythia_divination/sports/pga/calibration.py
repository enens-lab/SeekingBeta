"""Walk-forward calibration of the golf board's Top-10 and Made-Cut probabilities.

The multitask model was trained with ``BCEWithLogitsLoss(pos_weight=...)``
(9.2 for top-10), which multiplies the odds it learns by roughly that weight:
its top-10 probabilities summed to 40-60 per event against about 11 actual
top-10 finishers, a Brier of 0.218 against 0.090 for a constant.

Calibration here keeps the model's ordering and fixes the scale, per event:

* top-10: ``logit(p) * slope + shift`` with the shift solved so the field's
  probabilities sum to the expected number of top-10 finishers (10 plus ties,
  estimated from earlier events, held to [9.5, 11]);
* made cut: same form, with the field's mean set to the cut rate of the same
  tournament's previous edition (no-cut events come out near 1), or, for a new
  event, of earlier events of similar field size;
* consistency: P(win) <= P(top 10) <= P(made cut) on every row.

``slope`` is fitted by maximum likelihood on events that finished before the
one being calibrated (1.0 until ``MIN_EVENTS_FOR_SLOPE`` exist). Nothing about
an event's own outcome is used to calibrate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-6
TOP10_TARGET_BOUNDS = (9.5, 11.0)
TOP10_DEFAULT_TARGET = 10.5
MIN_EVENTS_FOR_SLOPE = 5
SLOPE_GRID = np.round(np.arange(0.4, 2.01, 0.05), 2)
NO_CUT_RATE = 0.98


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def shift_to_sum(logits: np.ndarray, target: float, *, iterations: int = 60) -> np.ndarray:
    """Probabilities sigmoid(logits + c) with c chosen so they sum to ``target``."""
    logits = np.asarray(logits, dtype=float)
    n = len(logits)
    if n == 0:
        return logits
    target = float(min(max(target, EPS), n - EPS))
    lo, hi = -40.0, 40.0
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if _sigmoid(logits + mid).sum() > target:
            hi = mid
        else:
            lo = mid
    return _sigmoid(logits + (lo + hi) / 2.0)


def calibrate_event(raw: np.ndarray, *, slope: float, target_sum: float) -> np.ndarray:
    return shift_to_sum(_logit(raw) * slope, target_sum)


def _log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs, EPS, 1 - EPS)
    return float(np.sum(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def loglik_curve(raw: np.ndarray, labels: np.ndarray, target: float) -> np.ndarray:
    """Log-likelihood of one finished event at every slope in ``SLOPE_GRID``."""
    return np.array([_log_likelihood(calibrate_event(raw, slope=slope, target_sum=target), labels) for slope in SLOPE_GRID])


def fit_slope(curves: Iterable[np.ndarray]) -> float:
    """Slope maximising the summed likelihood of finished events (1.0 until enough exist)."""
    curves = list(curves)
    if len(curves) < MIN_EVENTS_FOR_SLOPE:
        return 1.0
    return float(SLOPE_GRID[int(np.argmax(np.sum(curves, axis=0)))])


@dataclass
class EventInfo:
    tournament_id: str
    tournament_name: str
    start: pd.Timestamp | None
    field_size: int


def _canonical(name: str) -> str:
    return " ".join(str(name or "").lower().replace("&", "and").split())


def previous_cut_rate(
    event: EventInfo,
    labelled_events: pd.DataFrame,
) -> float | None:
    """Made-cut share in the latest earlier edition of the same tournament, else in
    earlier events of similar field size. ``labelled_events`` has one row per
    finished event: tournament_id, tournament_name, start, field_size, cut_rate."""
    if event.start is None or labelled_events.empty:
        return None
    earlier = labelled_events[labelled_events["start"] < event.start]
    if earlier.empty:
        return None
    same = earlier[earlier["tournament_name"].map(_canonical) == _canonical(event.tournament_name)]
    if not same.empty:
        return float(same.sort_values("start")["cut_rate"].iloc[-1])
    similar = earlier[(earlier["field_size"] - event.field_size).abs() <= 15]
    pool = similar if len(similar) >= 3 else earlier
    return float(pool["cut_rate"].median())


def calibrate_board(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward calibrate ``top_10_probability`` and ``made_cut_probability``.

    ``predictions``: tournament_id, player_id, winner_probability, top_10_probability,
    made_cut_probability (raw model outputs) for the board's events.
    ``labels``: one row per (tournament_id, player_id) for every finished event in
    history (not just board events) with tournament_name, event_start_date,
    top_10, made_cut, won, event_field_size.

    Returns the predictions with calibrated columns (raw kept as ``*_raw``) and a
    per-event log of the targets and slopes used.
    """
    out = predictions.copy()
    out["top_10_probability_raw"] = out["top_10_probability"]
    out["made_cut_probability_raw"] = out["made_cut_probability"]

    lab = labels.drop_duplicates(subset=["tournament_id", "player_id"], keep="last").copy()
    lab["start"] = pd.to_datetime(lab["event_start_date"], errors="coerce")
    for column in ("top_10", "made_cut", "won"):
        lab[column] = pd.to_numeric(lab[column], errors="coerce")
    finished = lab.dropna(subset=["start", "made_cut"])
    event_table = (
        finished.groupby("tournament_id")
        .agg(
            tournament_name=("tournament_name", "first"),
            start=("start", "first"),
            field_size=("player_id", "size"),
            cut_rate=("made_cut", "mean"),
            winners=("won", "sum"),
        )
        .reset_index()
    )
    event_table = event_table[event_table["winners"] >= 1]

    keyed = out.merge(
        lab[["tournament_id", "player_id", "top_10", "made_cut", "start", "tournament_name"]].rename(
            columns={"tournament_name": "_label_name"}
        ),
        on=["tournament_id", "player_id"],
        how="left",
    )
    keyed["_start"] = keyed.groupby("tournament_id")["start"].transform("first")
    order = keyed.groupby("tournament_id")["_start"].first().sort_values(na_position="last")

    top10_history: list[np.ndarray] = []
    cut_history: list[np.ndarray] = []
    top10_counts: list[float] = []
    log: list[dict] = []
    calibrated_t10 = pd.Series(np.nan, index=keyed.index)
    calibrated_mc = pd.Series(np.nan, index=keyed.index)

    for tournament_id, start in order.items():
        rows = keyed.index[keyed["tournament_id"] == tournament_id]
        group = keyed.loc[rows]
        n = len(group)
        start_ts = None if pd.isna(start) else pd.Timestamp(start)
        name = str(group["tournament_name"].dropna().iloc[0]) if group["tournament_name"].notna().any() else ""

        prior_counts = top10_counts[:]
        t10_target = float(np.clip(np.mean(prior_counts), *TOP10_TARGET_BOUNDS)) if prior_counts else TOP10_DEFAULT_TARGET
        t10_target = min(t10_target, 0.9 * n)
        t10_slope = fit_slope(top10_history)
        p_t10 = calibrate_event(group["top_10_probability"].to_numpy(dtype=float), slope=t10_slope, target_sum=t10_target)

        info = EventInfo(str(tournament_id), name, start_ts, n)
        cut_rate = previous_cut_rate(info, event_table)
        if cut_rate is None:
            cut_rate = float(np.clip(group["made_cut_probability"].mean(), 0.3, NO_CUT_RATE))
        cut_rate = float(min(cut_rate, 1 - 1e-3))
        mc_slope = fit_slope(cut_history)
        p_mc = calibrate_event(group["made_cut_probability"].to_numpy(dtype=float), slope=mc_slope, target_sum=cut_rate * n)

        p_win = group["winner_probability"].to_numpy(dtype=float)
        p_t10 = np.maximum(p_t10, p_win)
        p_mc = np.maximum(p_mc, p_t10)
        calibrated_t10.loc[rows] = p_t10
        calibrated_mc.loc[rows] = p_mc
        log.append(
            {
                "tournament_id": str(tournament_id),
                "start": None if start_ts is None else start_ts.strftime("%Y-%m-%d"),
                "field": int(n),
                "top10_target": round(t10_target, 3),
                "top10_slope": t10_slope,
                "cut_rate_target": round(cut_rate, 4),
                "made_cut_slope": mc_slope,
            }
        )

        # Only now, after this event is calibrated, may its outcome inform later events.
        labelled = group.dropna(subset=["top_10", "made_cut"])
        if len(labelled) == n and n > 0:
            top10_history.append(
                loglik_curve(group["top_10_probability"].to_numpy(dtype=float), labelled["top_10"].to_numpy(dtype=float), t10_target)
            )
            cut_history.append(
                loglik_curve(group["made_cut_probability"].to_numpy(dtype=float), labelled["made_cut"].to_numpy(dtype=float), cut_rate * n)
            )
            top10_counts.append(float(labelled["top_10"].sum()))

    out["top_10_probability"] = calibrated_t10.to_numpy()
    out["made_cut_probability"] = calibrated_mc.to_numpy()
    return out, {"events": log}


def calibration_report(calibrated: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Brier / sums / ordering checks on events with complete labels."""
    lab = labels[["tournament_id", "player_id", "top_10", "made_cut", "won"]].drop_duplicates(
        subset=["tournament_id", "player_id"], keep="last"
    ).copy()
    for column in ("top_10", "made_cut", "won"):
        lab[column] = pd.to_numeric(lab[column], errors="coerce")
    frame = calibrated.merge(lab, on=["tournament_id", "player_id"], how="inner")
    complete = frame.groupby("tournament_id").filter(
        lambda g: g["top_10"].notna().all() and g["made_cut"].notna().all() and g["won"].sum() == 1
    )
    if complete.empty:
        return {"events": 0}
    sums = complete.groupby("tournament_id")["top_10_probability"].sum()
    t10_rate = complete["top_10"].mean()
    mc_rate = complete["made_cut"].mean()
    winners = complete.sort_values(["tournament_id", "winner_probability"], ascending=[True, False]).groupby("tournament_id").head(1)

    def brier(p, y):
        return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))

    return {
        "events": int(complete["tournament_id"].nunique()),
        "rows": int(len(complete)),
        "top10": {
            "brier": round(brier(complete["top_10_probability"], complete["top_10"]), 4),
            "brierRaw": round(brier(complete["top_10_probability_raw"], complete["top_10"]), 4),
            "brierConstant": round(brier(np.full(len(complete), t10_rate), complete["top_10"]), 4),
            "meanPredicted": round(float(complete["top_10_probability"].mean()), 4),
            "actualRate": round(float(t10_rate), 4),
            "sumPerEventMin": round(float(sums.min()), 3),
            "sumPerEventMax": round(float(sums.max()), 3),
            "sumPerEventMean": round(float(sums.mean()), 3),
        },
        "madeCut": {
            "brier": round(brier(complete["made_cut_probability"], complete["made_cut"]), 4),
            "brierRaw": round(brier(complete["made_cut_probability_raw"], complete["made_cut"]), 4),
            "brierConstant": round(brier(np.full(len(complete), mc_rate), complete["made_cut"]), 4),
            "meanPredicted": round(float(complete["made_cut_probability"].mean()), 4),
            "actualRate": round(float(mc_rate), 4),
        },
        "rowsMadeCutBelowTop10": int((complete["made_cut_probability"] < complete["top_10_probability"] - 1e-12).sum()),
        "topPick": {
            "events": int(len(winners)),
            "hits": int(winners["won"].sum()),
            "rate": round(float(winners["won"].mean()), 4),
        },
    }
