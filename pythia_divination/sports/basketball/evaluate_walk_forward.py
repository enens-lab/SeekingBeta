"""Walk-forward evaluation of the served NBA/WNBA models (acceptance evidence for P1-4 / P1-12).

Runs exactly the code the exporter runs (sports/basketball/honest_model.run_league)
and scores the out-of-sample predictions:

  * winner: accuracy / Brier / log-loss of the served model next to Elo alone,
    always-home and better-record baselines, and -- when a closing-line file is
    given -- the de-vigged closing moneyline on the same games;
  * the first 14-day block of every season (cold-start check);
  * margin / total MAE of the P1-12 model and the 68/80/95% interval coverage of
    its normal predictive distribution with the rolling out-of-sample sigma;
  * the training-cutoff invariant: no graded game dated on/before its model's cutoff.

The closing-line file is internal QA only (ESPN lines are not licensed for display;
see markets_plan.md section 9). It is a pickle/CSV with columns league, date (UTC),
home_name, away_name, p_home_mkt, spread_home, total_line.

    python -m sports.basketball.evaluate_walk_forward --market-file /path/market.pkl \
        --out sports/basketball/walk_forward_metrics.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.basketball import honest_model as hm  # noqa: E402

logger = logging.getLogger(__name__)
NORMALIZED_DIR = ROOT / "data" / "sports" / "basketball" / "normalized"


def _r(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
            return None
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def match_market(history: pd.DataFrame, design: pd.DataFrame, market: pd.DataFrame, league: str) -> pd.DataFrame:
    """Attach closing lines by team nickname + tip time within 14 hours."""
    mk = market.loc[market["league"] == league].copy()
    if mk.empty:
        return history.assign(p_home_mkt=np.nan, spread_home=np.nan, total_line=np.nan)
    a = design[["game_id", "tip_utc", "home_name", "away_name"]].copy()
    a["k"] = a["home_name"].str.lower().str.strip() + "|" + a["away_name"].str.lower().str.strip()
    mk["k"] = mk["home_name"].str.lower().str.strip() + "|" + mk["away_name"].str.lower().str.strip()
    mk["date"] = pd.to_datetime(mk["date"])
    j = a.merge(mk[["k", "date", "p_home_mkt", "spread_home", "total_line"]], on="k", how="inner")
    j = j.loc[(j["tip_utc"] - j["date"]).abs() < pd.Timedelta(hours=14)]
    j = j.sort_values("game_id").drop_duplicates("game_id")
    return history.merge(j[["game_id", "p_home_mkt", "spread_home", "total_line"]], on="game_id", how="left")


def _calib(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    def rr(v: Any) -> Any:
        if isinstance(v, list):
            return [_r(x) for x in v]
        if isinstance(v, dict):
            return {k: rr(x) for k, x in v.items()}
        return _r(v)
    return rr(hm.calibration_metrics(y, p))


def _winner_block(g: pd.DataFrame, *, with_market: bool) -> dict[str, Any]:
    y = g["home_win"].to_numpy(float)
    out: dict[str, Any] = {"n": int(len(g))}
    out["model"] = {k: _r(v) for k, v in hm.win_metrics(y, g["p_home"].to_numpy(float)).items()}
    if "p_raw" in g.columns:
        out["model_uncalibrated"] = {k: _r(v) for k, v in hm.win_metrics(y, g["p_raw"].to_numpy(float)).items()}
    out["model_calibration"] = _calib(y, g["p_home"].to_numpy(float))
    out["elo"] = {k: _r(v) for k, v in hm.win_metrics(y, g["p_elo"].to_numpy(float)).items()}
    out["model_minus_elo_brier"] = {k: (_r(v) if not isinstance(v, list) else [_r(x) for x in v])
                                    for k, v in hm.paired_brier_diff(y, g["p_home"].to_numpy(float),
                                                                     g["p_elo"].to_numpy(float)).items()}
    out["always_home_accuracy"] = _r(float(np.mean(y == 1))) if len(y) else None
    rec = g.dropna(subset=["home_games_before", "away_games_before"])
    rec = rec.loc[(rec["home_games_before"] > 0) & (rec["away_games_before"] > 0)]
    if len(rec):
        hp = rec["home_wins_before"] / rec["home_games_before"]
        ap = rec["away_wins_before"] / rec["away_games_before"]
        out["better_record_accuracy"] = _r(float(np.mean((hp >= ap).astype(int) == rec["home_win"])))
        out["better_record_n"] = int(len(rec))
    if with_market:
        m = g.dropna(subset=["p_home_mkt"])
        if len(m):
            ym = m["home_win"].to_numpy(float)
            diff = hm.paired_brier_diff(ym, m["p_home"].to_numpy(float), m["p_home_mkt"].to_numpy(float))
            out["market_matched"] = {
                "n": int(len(m)),
                "market": {k: _r(v) for k, v in hm.win_metrics(ym, m["p_home_mkt"].to_numpy(float)).items()},
                "model": {k: _r(v) for k, v in hm.win_metrics(ym, m["p_home"].to_numpy(float)).items()},
                "elo": {k: _r(v) for k, v in hm.win_metrics(ym, m["p_elo"].to_numpy(float)).items()},
                "model_minus_market_brier": {k: (_r(v) if not isinstance(v, list) else [_r(x) for x in v])
                                             for k, v in diff.items()},
                "market_calibration": _calib(ym, m["p_home_mkt"].to_numpy(float)),
            }
    return out


def _regression_block(g: pd.DataFrame, *, with_market: bool) -> dict[str, Any]:
    g = g.dropna(subset=["margin_pred", "total_pred"])
    out: dict[str, Any] = {"n": int(len(g))}
    if not len(g):
        return out
    out["margin_mae"] = _r((g["margin"] - g["margin_pred"]).abs().mean(), 3)
    out["margin_mae_ratings_only"] = _r((g["margin"] - g["r_margin"]).abs().mean(), 3)
    out["margin_mae_elo_rest_only"] = _r((g["margin"] - g["m_elorest"]).abs().mean(), 3)
    out["total_mae"] = _r((g["total"] - g["total_pred"]).abs().mean(), 3)
    out["margin_bias"] = _r((g["margin"] - g["margin_pred"]).mean(), 3)
    out["total_bias"] = _r((g["total"] - g["total_pred"]).mean(), 3)
    out["sigma_margin_mean"] = _r(g["sigma_margin"].mean(), 2)
    out["sigma_total_mean"] = _r(g["sigma_total"].mean(), 2)
    out["margin_coverage"] = {k: _r(v) for k, v in hm.interval_coverage(g["resid_margin"], g["sigma_margin"]).items()}
    out["total_coverage"] = {k: _r(v) for k, v in hm.interval_coverage(g["resid_total"], g["sigma_total"]).items()}
    if with_market:
        m = g.dropna(subset=["spread_home", "total_line"])
        if len(m):
            out["market_matched"] = {
                "n": int(len(m)),
                "market_margin_mae": _r((m["margin"] + m["spread_home"]).abs().mean(), 3),
                "model_margin_mae": _r((m["margin"] - m["margin_pred"]).abs().mean(), 3),
                "market_total_mae": _r((m["total"] - m["total_line"]).abs().mean(), 3),
                "model_total_mae": _r((m["total"] - m["total_pred"]).abs().mean(), 3),
            }
    return out


def evaluate_league(league: str, params: dict[str, Any], *, today: pd.Timestamp, market: pd.DataFrame | None,
                    normalized_dir: Path = NORMALIZED_DIR) -> tuple[dict[str, Any], pd.DataFrame]:
    t0 = time.time()
    tables = hm.load_league_tables(normalized_dir, league)
    run = hm.run_league(tables, params, today=today, upcoming_ids=set(), with_history=True)
    if run is None:
        return {"league": league, "error": "no data"}, pd.DataFrame()
    design = run.design
    H = run.history.merge(design[["game_id", "kind", "home_wins_before", "home_games_before", "away_wins_before",
                                  "away_games_before", "home_name", "away_name", "tip_utc"]], on="game_id", how="left")
    with_market = market is not None
    if with_market:
        H = match_market(H, design, market, league)
    warmup = H.loc[~H["published"]]
    H = H.loc[H["published"]].copy()
    win = H.dropna(subset=["p_home"])
    report: dict[str, Any] = {
        "league": league,
        "seconds": round(time.time() - t0, 1),
        "timings": {k: round(v, 1) for k, v in run.timings.items()},
        "n_features": len(run.feature_cols),
        "trained_through": str(run.trained_through.date()) if run.trained_through is not None else None,
        "current_sigmas": {k: _r(v, 2) for k, v in run.sigmas.items()},
        "served_calibration": run.calibration,
        "warmup_seasons": sorted(warmup["season"].astype(str).unique().tolist()),
        "cutoff_violations": int((H["local_date"] <= H["train_cutoff"]).sum()),
        "winner": {}, "first_block": {}, "regression": {},
    }
    for season, g in list(win.groupby("season")) + [("pooled", win)]:
        report["winner"][str(season)] = _winner_block(g, with_market=with_market)
        report["winner"][str(season) + "_regular_season"] = _winner_block(g.loc[g["kind"] == "regular"], with_market=with_market)
    for season, g in win.groupby("season"):
        first = g.loc[g["block_start"] == g["block_start"].min()]
        report["first_block"][str(season)] = {
            "n": int(len(first)), "start": str(first["block_start"].min().date()),
            "model_log_loss": _r(hm.log_loss(first["home_win"].to_numpy(float), first["p_home"].to_numpy(float))),
            "elo_log_loss": _r(hm.log_loss(first["home_win"].to_numpy(float), first["p_elo"].to_numpy(float))),
            "home_win_rate": _r(float(first["home_win"].mean())),
        }
        if with_market and first["p_home_mkt"].notna().all():
            report["first_block"][str(season)]["market_log_loss"] = _r(
                hm.log_loss(first["home_win"].to_numpy(float), first["p_home_mkt"].to_numpy(float)))
    reg = H.loc[H["season"].isin(win["season"].unique())]
    for season, g in list(reg.groupby("season")) + [("pooled", reg)]:
        report["regression"][str(season)] = _regression_block(g, with_market=with_market)
        report["regression"][str(season) + "_regular_season"] = _regression_block(g.loc[g["kind"] == "regular"], with_market=with_market)
    return report, H


def _rl(d: dict[str, Any]) -> dict[str, Any]:
    return {k: ([_r(x) for x in v] if isinstance(v, list) else _r(v)) for k, v in d.items()}


def _calibration_check(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    c = hm.calibration_metrics(y, p)
    b = c.get("bucket_80") or {}
    gap = (abs(b["mean_p"] - b["hit"]) if b.get("n") else None)
    return {
        "target": "slope CI includes 1, intercept CI includes 0, p>=0.8 bucket |mean p - hit| <= 0.04",
        "n": c.get("n"),
        "slope": _r(c.get("slope"), 3), "slope_ci": [_r(x, 3) for x in c.get("slope_ci", [])],
        "intercept": _r(c.get("intercept"), 3), "intercept_ci": [_r(x, 3) for x in c.get("intercept_ci", [])],
        "bucket_80": {"n": b.get("n"), "mean_p": _r(b.get("mean_p")), "hit": _r(b.get("hit")), "gap": _r(gap)},
        "passed": bool(c.get("slope_ci_includes_1") and c.get("intercept_ci_includes_0") and gap is not None and gap <= 0.04),
    }


def _winner_check(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["home_win"].to_numpy(float)
    out: dict[str, Any] = {"n": int(len(frame)),
                           "model": _rl(hm.win_metrics(y, frame["p_home"].to_numpy(float))),
                           "elo": _rl(hm.win_metrics(y, frame["p_elo"].to_numpy(float))),
                           "model_minus_elo_brier": _rl(hm.paired_brier_diff(y, frame["p_home"].to_numpy(float),
                                                                               frame["p_elo"].to_numpy(float)))}
    mk = frame.dropna(subset=["p_home_mkt"]) if "p_home_mkt" in frame.columns else frame.iloc[:0]
    if len(mk):
        ym = mk["home_win"].to_numpy(float)
        out["market_n"] = int(len(mk))
        out["market"] = _rl(hm.win_metrics(ym, mk["p_home_mkt"].to_numpy(float)))
        out["model_on_market_games"] = _rl(hm.win_metrics(ym, mk["p_home"].to_numpy(float)))
        out["model_minus_market_brier"] = _rl(hm.paired_brier_diff(ym, mk["p_home"].to_numpy(float),
                                                                    mk["p_home_mkt"].to_numpy(float)))
    return out


def _regression_check(reg: pd.DataFrame) -> dict[str, Any]:
    reg = reg.dropna(subset=["margin_pred", "total_pred"])
    out = {"n": int(len(reg)),
           "margin_mae": _r((reg["margin"] - reg["margin_pred"]).abs().mean(), 3),
           "total_mae": _r((reg["total"] - reg["total_pred"]).abs().mean(), 3),
           "margin_cov80": _r(hm.interval_coverage(reg["resid_margin"], reg["sigma_margin"]).get("cov80")),
           "total_cov80": _r(hm.interval_coverage(reg["resid_total"], reg["sigma_total"]).get("cov80"))}
    if "spread_home" in reg.columns:
        m = reg.dropna(subset=["spread_home", "total_line"])
        if len(m):
            out["market_n"] = int(len(m))
            out["market_margin_mae"] = _r((m["margin"] + m["spread_home"]).abs().mean(), 3)
            out["market_total_mae"] = _r((m["total"] - m["total_line"]).abs().mean(), 3)
    return out


def acceptance(reports: dict[str, dict[str, Any]], frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """The plan's P1-4 / P1-12 checks on the plan's own subsets (regular season),
    plus the review's binding additions: calibration slope/intercept CIs, the p>=0.8
    bucket, and paired CIs against the market wherever n < 500."""
    out: dict[str, Any] = {}
    nba = frames.get("nba")
    if nba is not None and not nba.empty:
        sub = nba.dropna(subset=["p_home"])
        sub = sub.loc[sub["season"].isin(["2024-25", "2025-26"]) & (sub["kind"] == "regular")]
        w = _winner_check(sub)
        out["nba_pooled_brier"] = {"target": "<= 0.2090, market shown", **w,
                                   "passed": bool(w["model"].get("brier") is not None and w["model"]["brier"] <= 0.2090)}
        out["nba_calibration"] = _calibration_check(sub["home_win"].to_numpy(float), sub["p_home"].to_numpy(float))
        reg = _regression_check(nba.loc[nba["season"].isin(["2024-25", "2025-26"]) & (nba["kind"] == "regular")])
        out["nba_margin_total"] = {
            "target": "margin MAE <= 11.15, total MAE <= 14.9, 80% coverage 0.78-0.82", **reg,
            "passed": bool(reg["margin_mae"] <= 11.15 and reg["total_mae"] <= 14.9
                           and 0.78 <= reg["margin_cov80"] <= 0.82 and 0.78 <= reg["total_cov80"] <= 0.82)}
    wnba = frames.get("wnba")
    if wnba is not None and not wnba.empty:
        wr = wnba.dropna(subset=["p_home"])
        wr = wr.loc[wr["kind"] == "regular"]
        w25 = wr.loc[wr["season"] == "2025"].sort_values(["tip_utc", "game_id"]).iloc[60:]
        w26 = wr.loc[wr["season"] == "2026"]
        pooled = pd.concat([w25, w26])
        w = _winner_check(pooled)
        out["wnba_pooled_brier"] = {"target": "<= 0.2085 (n < 500 per season: judge on the paired CI vs market)",
                                    "subset": "2025 regular season after its 60th game + 2026 regular season", **w,
                                    "passed": bool(w["model"].get("brier") is not None and w["model"]["brier"] <= 0.2085)}
        w2 = _winner_check(w26)
        out["wnba_2026"] = {"target": "accuracy >= 0.70 and log_loss <= 0.56 (n=330 < 500: judge on the paired CI vs market)",
                            **w2,
                            "passed": bool(w2["model"].get("accuracy", 0) >= 0.70 and w2["model"].get("log_loss", 9) <= 0.56),
                            "paired_ci_vs_market_includes_0": bool(
                                "model_minus_market_brier" in w2
                                and w2["model_minus_market_brier"]["ci"][0] <= 0.0 <= w2["model_minus_market_brier"]["ci"][1])}
        out["wnba_calibration"] = _calibration_check(wr.loc[wr["season"].isin(["2025", "2026"]), "home_win"].to_numpy(float),
                                                     wr.loc[wr["season"].isin(["2025", "2026"]), "p_home"].to_numpy(float))
        out["wnba_margin_total"] = _regression_check(wnba.loc[wnba["season"].isin(["2025", "2026"]) & (wnba["kind"] == "regular")])
    out["first_block_log_loss"] = {
        lg: {s: {"model": v["model_log_loss"], "elo": v["elo_log_loss"], "market": v.get("market_log_loss"),
                 "n": v["n"], "home_win_rate": v["home_win_rate"], "passed": bool(v["model_log_loss"] < 0.693)}
             for s, v in rep.get("first_block", {}).items()}
        for lg, rep in reports.items()}
    out["cutoff_violations"] = {lg: rep.get("cutoff_violations") for lg, rep in reports.items()}
    return out


def _apply_overrides(params: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    params = copy.deepcopy(params)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = params
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leagues", nargs="+", default=["nba", "wnba"])
    parser.add_argument("--market-file", default=None, help="Closing lines (pickle/CSV), internal QA only.")
    parser.add_argument("--today", default=None, help="Evaluate as of this local date (default: today).")
    parser.add_argument("--out", default=None, help="Write the metrics JSON here.")
    parser.add_argument("--predictions-out", default=None, help="Optional pickle of the per-game OOS predictions.")
    parser.add_argument("--set", action="append", default=[], help="Param override, e.g. win_model.C=0.02")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    params = _apply_overrides(hm.load_params(), args.set)
    today = pd.Timestamp(args.today) if args.today else pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    market = None
    if args.market_file:
        path = Path(args.market_file)
        market = pd.read_pickle(path) if path.suffix in (".pkl", ".pickle") else pd.read_csv(path)
    reports, frames = {}, {}
    for league in args.leagues:
        reports[league], frames[league] = evaluate_league(league, params, today=today, market=market)
    result = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "as_of": str(today.date()),
        "model_version": hm.model_version(params),
        "params_overrides": args.set,
        "market_source": ("ESPN closing lines matched by team and tip time (internal QA only, not displayed)"
                          if market is not None else None),
        "method": ("14-day expanding-window walk-forward: every graded game is predicted by models fit only on games "
                   "from earlier local dates; Elo/rest/ratings are pregame by construction."),
        "acceptance": acceptance(reports, frames),
        "leagues": reports,
    }
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(json.dumps(result["acceptance"], indent=2, default=str))
    if args.predictions_out:
        pd.to_pickle(frames, args.predictions_out)


if __name__ == "__main__":
    main()
