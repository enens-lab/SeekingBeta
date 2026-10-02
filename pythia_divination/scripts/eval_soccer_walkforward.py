"""Acceptance backtest for the soccer Dixon-Coles model (plan items P1-2 / P1-7).

Runs the SAME walk-forward the in-app Track Record uses (sports/soccer/walkforward.py:
refit Mon + Fri on matches strictly before the cut-off, 5-season window, L2 toward
the league mean) over a block of seasons, and scores it next to the de-vigged
football-data closing-average market on the same matches.

Input: a directory of football-data.co.uk CSVs named <FD_CODE>_<YYZZ>.csv (E0, SP1,
I1, D1, F1), i.e. the files sports/soccer/client.py caches. Nothing is downloaded.

    python scripts/eval_soccer_walkforward.py --raw-dir <dir> --test-start 2023 --test-end 2026 \
        [--out results.json] [--check]

--check exits non-zero if any acceptance threshold in model_params.json fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports.soccer.client import normalize_football_data  # noqa: E402
from sports.soccer.constants import LEAGUE_CONFIGS, football_data_season_code, load_model_params, season_label  # noqa: E402
from sports.soccer.walkforward import score_predictions, walk_forward  # noqa: E402

FD_CODES = [str(v["fd_code"]) for v in LEAGUE_CONFIGS.values() if v.get("fd_code")]

# Acceptance thresholds (plan P1-2 / P1-7, TEST = 2023-24..2026-27, n~5,506).
ACCEPTANCE = {
    "x12LogLossMax": 0.986,
    "rpsMax": 0.1995,
    "overUnder25LogLossMax": 0.677,
    "bttsLogLossMax": 0.686,
    "correctScoreTop3Min": 0.32,
    "meanGoalsAbsErrorMax": 0.05,
    "lastSeasonLogLossMax": 0.985,
}


def load_league(raw_dir: Path, fd_code: str, first_season: int, last_season: int) -> pd.DataFrame:
    frames = []
    for start in range(first_season, last_season + 1):
        path = raw_dir / f"{fd_code}_{football_data_season_code(start)}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, encoding="latin-1")
        frames.append(normalize_football_data(raw, season_start=start, keep_odds=True))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    df["season_start"] = df["season_start"].astype(int)
    df["league"] = fd_code
    return df.sort_values(["date", "home"], kind="mergesort").reset_index(drop=True)


def _run(args):
    raw_dir, fd_code, test_start, test_end, params = args
    hist = load_league(Path(raw_dir), fd_code, test_start - int(params["windowSeasons"]) + 1, test_end)
    if hist.empty:
        return pd.DataFrame()
    return walk_forward(hist, params, start_season=test_start, end_season=test_end)


def _flat(block: dict) -> dict:
    m = block.get("model", {})
    row = {
        "n": m.get("n"), "acc": m.get("accuracy"), "rps": m.get("rps"), "logloss": m.get("logLoss"),
        "brier": m.get("brier"),
        "ou25_ll": block.get("overUnder25", {}).get("logLoss"),
        "btts_ll": block.get("btts", {}).get("logLoss"),
        "cs_top3": block.get("correctScore", {}).get("top3HitRate"),
        "goals_pred": block.get("goals", {}).get("meanPredicted"),
        "goals_act": block.get("goals", {}).get("meanActual"),
    }
    mk = block.get("market")
    if mk:
        row.update({"mkt_n": mk.get("n"), "mkt_fav": mk.get("favouriteWinRate"), "mkt_rps": mk.get("rps"),
                    "mkt_ll": mk.get("logLoss")})
    mou = block.get("marketOverUnder25")
    if mou:
        row["mkt_ou25_ll"] = mou.get("logLoss")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--test-start", type=int, default=2023)
    ap.add_argument("--test-end", type=int, default=2026)
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    params = load_model_params()

    t0 = time.time()
    with Pool(min(len(FD_CODES), os.cpu_count() or 2)) as pool:
        parts = pool.map(_run, [(args.raw_dir, fd, args.test_start, args.test_end, params) for fd in FD_CODES])
    preds = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    elapsed = time.time() - t0

    result = {"params": {k: params[k] for k in ("modelVersion", "l2", "windowSeasons", "timeDecayXiPerDay", "refitWeekdays")},
              "split": f"{season_label(args.test_start)}..{season_label(args.test_end)}",
              "seconds": round(elapsed, 1),
              "overall": score_predictions(preds), "bySeason": {}, "byLeague": {}}
    for s, g in preds.groupby("season_start"):
        result["bySeason"][season_label(int(s))] = score_predictions(g)
    for lg, g in preds.groupby("league"):
        result["byLeague"][lg] = score_predictions(g)

    o = result["overall"]
    last = result["bySeason"].get(season_label(args.test_end), {})
    checks = {
        "x12LogLoss": (o["model"]["logLoss"], o["model"]["logLoss"] <= ACCEPTANCE["x12LogLossMax"]),
        "rps": (o["model"]["rps"], o["model"]["rps"] <= ACCEPTANCE["rpsMax"]),
        "overUnder25LogLoss": (o["overUnder25"]["logLoss"], o["overUnder25"]["logLoss"] <= ACCEPTANCE["overUnder25LogLossMax"]),
        "bttsLogLoss": (o["btts"]["logLoss"], o["btts"]["logLoss"] <= ACCEPTANCE["bttsLogLossMax"]),
        "correctScoreTop3": (o["correctScore"]["top3HitRate"], o["correctScore"]["top3HitRate"] >= ACCEPTANCE["correctScoreTop3Min"]),
        "meanGoalsError": (o["goals"]["meanPredicted"] - o["goals"]["meanActual"],
                           abs(o["goals"]["meanPredicted"] - o["goals"]["meanActual"]) <= ACCEPTANCE["meanGoalsAbsErrorMax"]),
        "lastSeasonLogLoss": (last.get("model", {}).get("logLoss"),
                              (last.get("model", {}).get("logLoss") or 9) <= ACCEPTANCE["lastSeasonLogLossMax"]),
    }
    result["acceptance"] = {k: {"value": round(v, 4) if v is not None else None, "pass": bool(ok), "threshold": t}
                            for (k, (v, ok)), t in zip(checks.items(), ACCEPTANCE.values())}

    pd.set_option("display.width", 220)
    table = pd.DataFrame({**{"ALL": _flat(o)}, **{k: _flat(v) for k, v in result["bySeason"].items()},
                          **{k: _flat(v) for k, v in result["byLeague"].items()}}).T
    print(table.round(4).to_string())
    print(f"\nwalk-forward {len(preds)} matches in {elapsed:.0f}s")
    for k, v in result["acceptance"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k:20s} {v['value']}  (threshold {v['threshold']})")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=float))
        preds.to_parquet(Path(args.out).with_suffix(".parquet"))
    if args.check and not all(v["pass"] for v in result["acceptance"].values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
