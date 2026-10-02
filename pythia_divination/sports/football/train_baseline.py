"""Train and honestly evaluate the NFL model view + market ladders.

    python -m sports.football.train_baseline [--rebuild-dataset] [-v]

What it does (all evaluation is a SEASON WALK-FORWARD: every number for season S comes
from models and calibrations fit on seasons < S only):
  1. Joins the pregame training dataset to scores and lines from the games table, adds
     walk-forward team ratings (half-life / ridge strength tuned on the first
     out-of-sample season only, never on a test season).
  2. Model view: ridge on the explicit FEATURE_ALLOWLIST (model_view.py) -> home
     margin, Platt-calibrated on prior-season out-of-sample margins -> P(home win).
     Ties have no binary label and are excluded from every win-probability metric.
  3. Served headline: the de-vigged nflverse moneyline where posted, else the model view.
  4. Market ladders: spread (key-number margin pmf), total and team totals centered on
     the posted line; Brier on the alt-line ladder and the push-rate gate the review
     requires before any push probability or whole-number alt line is published.
  5. Writes model_params.json (small, committed, read by the exporter) and metrics.json
     (per-season walk-forward metrics with the market beside every model number).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from sports.football import markets_nfl
from sports.football import model_view as mv
from sports.football.build_training_dataset import DEFAULT_FOOTBALL_DATA_ROOT, build_training_dataset
from sports.pga.storage import read_preferred_table

logger = logging.getLogger(__name__)

# Kept for importers (train_torch): columns that are never features.
_IDENTIFIER_COLUMNS = {
    "game_id", "season", "season_display", "week", "game_type", "gameday", "weekday", "gametime",
    "official_date", "away_team", "home_team", "away_team_key", "home_team_key", "location", "roof",
    "surface", "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name", "away_coach", "home_coach",
    "referee", "stadium_id", "stadium", "old_game_id", "gsis", "nfl_detail_id", "pfr", "pff", "espn", "ftn",
}
_LABEL_COLUMNS = {"home_win", "away_win"}

ANALYST_LADDER_OFFSETS = (-14, -10, -7, -6, -4, -3, -1, 1, 3, 4, 6, 7, 10, 14)
PUBLISHED_LADDER_OFFSETS = (-7, -3, 0, 3, 7)
PUSH_GATE_PP = 0.015


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate the NFL model view and market ladders.")
    parser.add_argument("--dataset", default=None, help="Explicit training dataset (parquet/csv).")
    parser.add_argument("--games", default=None, help="Explicit games table (parquet/csv).")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Re-ingest nflverse and rebuild the dataset first.")
    parser.add_argument("--season-start", type=int, default=2020)
    parser.add_argument("--season-end", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--first-oos-season", type=int, default=2021,
                        help="First season with out-of-sample predictions (ratings are tuned on it).")
    parser.add_argument("--eval-start", type=int, default=2022, help="First season of the reported walk-forward.")
    parser.add_argument("--params-out", default=str(mv.PARAMS_PATH))
    parser.add_argument("--metrics-out", default=str(mv.METRICS_PATH))
    parser.add_argument("--output-dir", default=None,
                        help="Optional extra dir for metrics.json + walk-forward predictions CSV (e.g. artifacts/...).")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


# ── metrics helpers ───────────────────────────────────────────────────────────

def _clip(p: Iterable[float]) -> np.ndarray:
    return np.clip(np.asarray(list(p) if not isinstance(p, (np.ndarray, pd.Series)) else p, float), 1e-6, 1 - 1e-6)


def binary_metrics(y: Iterable[float], p: Iterable[float]) -> dict[str, Any]:
    """Accuracy counts p == 0.5 as no pick (excluded from the accuracy denominator)."""
    from sklearn.metrics import roc_auc_score

    yy = np.asarray(list(y) if not isinstance(y, (np.ndarray, pd.Series)) else y, float)
    pp = _clip(p)
    picked = np.abs(pp - 0.5) > 1e-9
    out: dict[str, Any] = {
        "n": int(len(yy)),
        "accuracy": round(float(((pp[picked] > 0.5) == (yy[picked] == 1)).mean()), 4) if picked.any() else None,
        "brier": round(float(np.mean((pp - yy) ** 2)), 4),
        "log_loss": round(float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))), 4),
    }
    if len(np.unique(yy)) > 1:
        out["auc"] = round(float(roc_auc_score(yy, pp)), 4)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[Optional[float]]:
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 4), round(centre + half, 4)]


def paired_brier_diff(y: np.ndarray, p_model: np.ndarray, p_market: np.ndarray, *, reps: int = 2000,
                      seed: int = 7) -> dict[str, Any]:
    d = (_clip(p_model) - y) ** 2 - (_clip(p_market) - y) ** 2
    rng = np.random.default_rng(seed)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(reps)]
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                                        round(float(np.percentile(boots, 97.5)), 4)]}


def decile_table(y: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, 10):
        rows.append({"n": int(len(chunk)), "mean_p": round(float(p[chunk].mean()), 4),
                     "actual": round(float(y[chunk].mean()), 4),
                     "gap": round(float(y[chunk].mean() - p[chunk].mean()), 4)})
    return rows


# ── data ──────────────────────────────────────────────────────────────────────

def _load_table(path: Optional[str], stem: str) -> pd.DataFrame:
    if path:
        p = Path(path)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)
    base = DEFAULT_FOOTBALL_DATA_ROOT / "normalized"
    return read_preferred_table(base / f"{stem}.parquet", base / f"{stem}.csv")


def _regular_season_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["season"] = pd.to_numeric(g["season"], errors="coerce")
    g = g[g["game_type"].astype(str) == "REG"]
    for c in ("home_score", "away_score"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g


def season_status(games: pd.DataFrame) -> tuple[int, int]:
    """(current_season, train_through_season). The current season is the latest one in
    the schedule; if it still has unplayed games the model trains through the season
    before it, so every current-season game it scores is out of sample."""
    g = _regular_season_games(games)
    current = int(g["season"].max())
    unplayed = g[(g["season"] == current) & (g["home_score"].isna() | g["away_score"].isna())]
    return current, (current - 1 if len(unplayed) else current)


# ── ladder evaluation ─────────────────────────────────────────────────────────

def _ladder_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    need = ["margin", "spread_line", "pts_total", "total_line", "home_score", "away_score"]
    return frame.dropna(subset=need)


def fit_ladder_for(frame: pd.DataFrame) -> dict[str, Any]:
    d = _ladder_inputs(frame)
    return markets_nfl.fit_ladder_params(d["margin"], d["spread_line"], d["pts_total"], d["total_line"],
                                         d["home_score"], d["away_score"])


def evaluate_ladders(frame: pd.DataFrame, seasons: Iterable[int]) -> dict[str, Any]:
    """Walk-forward ladder evaluation: shape params for season S are fit on seasons < S."""
    spread_cells = {"analyst": [], "published": []}
    total_cells = {"analyst": [], "published": []}
    key_rows, team_rows, cover_rows, total_line_rows = [], [], [], []
    for season in sorted(seasons):
        prior = frame[frame["season"] < season]
        test = _ladder_inputs(frame[frame["season"] == season])
        if test.empty or _ladder_inputs(prior).empty:
            continue
        ladder = fit_ladder_for(prior)
        for row in test.itertuples(index=False):
            s, margin = float(row.spread_line), float(row.margin)
            values, probs, _ = markets_nfl.home_margin_distribution(s, row.home_spread_odds, row.away_spread_odds, ladder)
            pm = dict(zip(values.astype(int), probs))
            key_rows.append({"season": season, "spread": s, "margin": margin,
                             **{f"p{k:+d}": float(pm.get(k, 0.0)) for k in (-7, -3, 0, 3, 7)}})
            for tag, offsets in (("analyst", ANALYST_LADDER_OFFSETS), ("published", PUBLISHED_LADDER_OFFSETS)):
                for k in offsets:
                    thr = s + k
                    if margin == thr:
                        continue
                    p = markets_nfl.side_prices(values, probs, "home", -thr).win_ex_push
                    spread_cells[tag].append((p, float(margin > thr), k))
            home_cover = markets_nfl.side_prices(values, probs, "home", -s)
            cover_rows.append({"p": home_cover.win_ex_push, "push_p": home_cover.push, "push": float(margin == s),
                               "y": float(margin > s) if margin != s else np.nan, "whole": float(s == round(s)),
                               "abs_line": abs(s)})
            # totals
            tl, tot = float(row.total_line), float(row.pts_total)
            center, _ = markets_nfl.total_center(tl, row.over_odds, row.under_odds, ladder)
            sig_t = float(ladder["total_sigma"])
            for tag, offsets in (("analyst", ANALYST_LADDER_OFFSETS), ("published", PUBLISHED_LADDER_OFFSETS)):
                for k in offsets:
                    thr = tl + k
                    if tot == thr:
                        continue
                    p = markets_nfl.mk.total_prices_normal(center, sig_t, thr, "over").win_ex_push
                    total_cells[tag].append((p, float(tot > thr), k))
            posted = markets_nfl.mk.total_prices_normal(center, sig_t, tl, "over")
            total_line_rows.append({"push_p": posted.push, "push": float(tot == tl), "whole": float(tl == round(tl))})
            # team totals at x.5
            for side, pts, implied in zip(("home", "away"), (row.home_score, row.away_score),
                                          markets_nfl.implied_team_totals(s, tl)):
                line = math.floor(implied) + 0.5
                p = markets_nfl.mk.total_prices_normal(implied, float(ladder["team_total_sigma"]), line, "over").win_ex_push
                team_rows.append({"season": season, "side": side, "p": p, "y": float(pts > line)})

    def brier(cells: list[tuple[float, float, int]]) -> dict[str, Any]:
        if not cells:
            return {"cells": 0, "brier": None}
        p = np.array([c[0] for c in cells]); y = np.array([c[1] for c in cells])
        by_k = {}
        for k in sorted({c[2] for c in cells}):
            sel = np.array([c[2] == k for c in cells])
            by_k[f"{k:+d}"] = {"n": int(sel.sum()), "mean_p": round(float(p[sel].mean()), 4), "actual": round(float(y[sel].mean()), 4)}
        return {"cells": int(len(cells)), "brier": round(float(np.mean((p - y) ** 2)), 5), "by_offset": by_k}

    keys = pd.DataFrame(key_rows)
    covers = pd.DataFrame(cover_rows)
    tlines = pd.DataFrame(total_line_rows)
    teams = pd.DataFrame(team_rows)
    gate: dict[str, Any] = {"rule": "home-margin masses at +/-3 and +/-7 within 1.5 pp (n>=500); conditional push "
                                    "rates (n<500) must fall inside the observed Wilson 95% CI"}
    passed = True
    for k in (-7, -3, 3, 7):
        pred = float(keys[f"p{k:+d}"].mean()); actual = float((keys["margin"] == k).mean())
        ok = abs(pred - actual) <= PUSH_GATE_PP
        passed &= ok
        gate[f"margin_{k:+d}"] = {"n": int(len(keys)), "predicted": round(pred, 4), "actual": round(actual, 4),
                                  "gap_pp": round(100 * (pred - actual), 2), "pass": bool(ok)}
    for label, sel in (("whole_number_lines", covers["whole"] == 1),
                       ("line_abs_3", covers["abs_line"] == 3), ("line_abs_7", covers["abs_line"] == 7)):
        sub = covers[sel]
        k, n = int(sub["push"].sum()), int(len(sub))
        ci = wilson(k, n)
        pred = float(sub["push_p"].mean()) if n else float("nan")
        ok = bool(n and ci[0] is not None and ci[0] <= pred <= ci[1])
        if label == "whole_number_lines" and n >= 500:
            ok = ok and abs(pred - k / n) <= PUSH_GATE_PP
        passed &= ok
        gate[f"push_{label}"] = {"n": n, "predicted": round(pred, 4), "actual": round(k / n, 4) if n else None,
                                 "actual_ci95": ci, "gap_pp": round(100 * (pred - k / n), 2) if n else None, "pass": ok}
    gate["passed"] = bool(passed)
    wt = tlines[tlines["whole"] == 1]
    k, n = int(wt["push"].sum()), int(len(wt))
    tci = wilson(k, n)
    tpred = float(wt["push_p"].mean()) if n else float("nan")
    total_gate = {"n": n, "predicted": round(tpred, 4), "actual": round(k / n, 4) if n else None, "actual_ci95": tci,
                  "gap_pp": round(100 * (tpred - k / n), 2) if n else None}
    total_gate["passed"] = bool(n and tci[0] <= tpred <= tci[1] and abs(tpred - k / n) <= PUSH_GATE_PP)
    cover_ok = covers.dropna(subset=["y"])
    team_summary = {}
    for side in ("home", "away"):
        t = teams[teams["side"] == side]
        team_summary[side] = {"n": int(len(t)), "brier": round(float(np.mean((t["p"] - t["y"]) ** 2)), 5),
                              "brier_constant_in_hindsight": round(float(np.var(t["y"])), 5),
                              "mean_p": round(float(t["p"].mean()), 4), "over_rate": round(float(t["y"].mean()), 4)}
    return {
        "spread_ladder": {"analyst_offsets": brier(spread_cells["analyst"]), "published_offsets": brier(spread_cells["published"])},
        "total_ladder": {"analyst_offsets": brier(total_cells["analyst"]), "published_offsets": brier(total_cells["published"])},
        "posted_spread_home_cover": {"n": int(len(cover_ok)), "mean_p": round(float(cover_ok["p"].mean()), 4),
                                     "actual": round(float(cover_ok["y"].mean()), 4)},
        "team_totals_x5": team_summary,
        "push_gate_spread": gate,
        "push_gate_total": total_gate,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def train_baseline(args: argparse.Namespace) -> dict[str, Any]:
    if args.rebuild_dataset:
        build_training_dataset(SimpleNamespace(season_start=args.season_start, season_end=args.season_end,
                                               refresh_source_data=True, output_root=None, snapshot_tag=None,
                                               verbose=args.verbose))
    dataset = _load_table(args.dataset, "football_training_dataset_latest")
    games = _load_table(args.games, "football_games_latest")
    current_season, train_through = season_status(games)

    base = mv.prepare_training_frame(dataset, games)
    rating_params = mv.tune_rating_params(base, validation_season=args.first_oos_season)
    frame = mv.attach_ratings(base, rating_params)
    test_seasons = list(range(args.first_oos_season + 1, current_season + 1))
    wf = mv.walk_forward(frame, test_seasons=test_seasons, first_oos_season=args.first_oos_season)
    frame = pd.concat([frame, wf], axis=1)
    frame["served_home_win_probability"] = frame["market_home_win_probability"].where(
        frame["market_home_win_probability"].notna(), frame["model_home_win_probability"])
    frame["served_basis"] = np.where(frame["market_home_win_probability"].notna(), "market", "model")

    eval_seasons = list(range(args.eval_start, train_through + 1))
    labelled = frame[frame["home_win"].notna()]

    def block(sel: pd.Series) -> dict[str, Any]:
        d = labelled[sel.reindex(labelled.index, fill_value=False)]
        d = d[d["model_home_win_probability"].notna()]
        if d.empty:
            return {"n": 0}
        y = d["home_win"].to_numpy(float)
        out = {
            "n": int(len(d)),
            "served": binary_metrics(y, d["served_home_win_probability"]),
            "served_basis_counts": d["served_basis"].value_counts().to_dict(),
            "model_view": binary_metrics(y, d["model_home_win_probability"]),
            "always_home_accuracy": round(float(y.mean()), 4),
        }
        mkt = d[d["market_home_win_probability"].notna()]
        if not mkt.empty:
            ym = mkt["home_win"].to_numpy(float)
            pm = mkt["market_home_win_probability"].to_numpy(float)
            out["market"] = binary_metrics(ym, pm)
            fav = np.abs(pm - 0.5) > 1e-9
            out["market_favourite_win_rate"] = round(float(((pm[fav] > 0.5) == (ym[fav] == 1)).mean()), 4)
            out["model_minus_market_brier"] = paired_brier_diff(ym, mkt["model_home_win_probability"].to_numpy(float), pm)
        return out

    per_season = {str(s): block(frame["season"] == s) for s in eval_seasons}
    pooled_sel = frame["season"].isin(eval_seasons)
    pooled = block(pooled_sel)
    pooled_rows = labelled[pooled_sel.reindex(labelled.index, fill_value=False)].dropna(subset=["model_home_win_probability"])
    y_pool = pooled_rows["home_win"].to_numpy(float)
    pooled["served_calibration"] = mv.calibration_fit(pooled_rows["served_home_win_probability"], y_pool)
    pooled["model_view_calibration"] = mv.calibration_fit(pooled_rows["model_home_win_probability"], y_pool)
    pooled["served_deciles_informational"] = decile_table(y_pool, pooled_rows["served_home_win_probability"].to_numpy(float))
    in_progress = {str(s): block(frame["season"] == s) for s in range(train_through + 1, current_season + 1)}

    # margin / total accuracy of the information-only model lines vs the posted lines
    lines = frame[pooled_sel & frame["margin"].notna() & frame["spread_line"].notna() & frame["oos_margin"].notna()]
    fair_lines = {
        "n": int(len(lines)),
        "model_margin_rmse": round(float(np.sqrt(np.mean((lines["oos_margin"] - lines["margin"]) ** 2))), 3),
        "market_spread_rmse": round(float(np.sqrt(np.mean((lines["spread_line"] - lines["margin"]) ** 2))), 3),
        "model_total_rmse": round(float(np.sqrt(np.mean((lines["oos_total"] - lines["pts_total"]) ** 2))), 3),
        "market_total_rmse": round(float(np.sqrt(np.mean((lines["total_line"] - lines["pts_total"]) ** 2))), 3),
    }

    ladders = evaluate_ladders(frame, eval_seasons)
    final_ladder = fit_ladder_for(frame[frame["season"] <= train_through])
    final_ladder["push_validated"] = bool(ladders["push_gate_spread"]["passed"])
    final_ladder["total_push_validated"] = bool(ladders["push_gate_total"]["passed"])
    final_ladder["fit_seasons"] = [int(frame["season"].min()), int(train_through)]

    final = mv.fit_final(frame, train_through_season=train_through, first_oos_season=args.first_oos_season,
                         oos_margin=frame["oos_margin"])
    model_version = f"nfl-mv1-ridge-platt-thru{train_through}"
    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    params = {
        "model_version": model_version,
        "trained_at": trained_at,
        "train_through_season": int(train_through),
        "first_oos_season": int(args.first_oos_season),
        "rating_params": rating_params,
        "feature_allowlist": mv.FEATURE_ALLOWLIST,
        "total_feature_allowlist": mv.TOTAL_FEATURE_ALLOWLIST,
        **final,
        "ladder": final_ladder,
    }
    metrics = {
        "league": "nfl",
        "model_version": model_version,
        "generated_at": trained_at,
        "method": ("Season walk-forward: models for season S fit on seasons < S; Platt calibration for S fit on "
                   "out-of-sample margins of seasons {first}..S-1; ties excluded from win metrics; accuracy counts "
                   "p == 0.5 as no pick. Served = de-vigged nflverse moneyline where posted, else model view. "
                   "Market lines are nflverse closing lines, not the earlier lines a board is published with."
                   ).format(first=args.first_oos_season),
        "rating_params": rating_params,
        "feature_count": len(mv.FEATURE_ALLOWLIST),
        "feature_allowlist": mv.FEATURE_ALLOWLIST,
        "eval_seasons": eval_seasons,
        "pooled": pooled,
        "per_season": per_season,
        "in_progress_season": in_progress,
        "fair_lines_information_only": fair_lines,
        "ladders": ladders,
    }
    _write_json(Path(args.params_out), params)
    _write_json(Path(args.metrics_out), metrics)
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "metrics.json", metrics)
        cols = ["game_id", "season", "week", "official_date", "away_team", "home_team", "home_win", "margin",
                "market_home_win_probability", "model_home_win_probability", "served_home_win_probability",
                "served_basis", "oos_margin", "oos_total"]
        frame[frame["season"] >= args.eval_start][cols].to_csv(out / "walkforward_predictions.csv", index=False)
    logger.info("NFL walk-forward pooled %s: %s", eval_seasons, json.dumps(pooled, default=str)[:1500])
    logger.info("Ladders: %s", json.dumps({k: v for k, v in ladders.items() if k != "team_totals_x5"}, default=str)[:2500])
    return metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    tmp.replace(path)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    train_baseline(args)


if __name__ == "__main__":
    main()
