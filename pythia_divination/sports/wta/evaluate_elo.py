"""Measure the walk-forward tennis Elo (match level) against the legacy Elo.

    python -m sports.wta.evaluate_elo                 # 2023-2025 report
    python -m sports.wta.evaluate_elo --json out.json # also write the numbers

Legacy = the pre-2026-10 ``process_match_history`` Elo: keyed on raw ids
(so every ATP player resets on the 2025 id-scheme change), K=32 overall,
K=16 surface, and the overall rating alone as the prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sports.wta.elo import (  # noqa: E402
    evaluation_mask,
    load_params,
    match_metrics,
    normalize_identifier,
    prepare_matches,
    run_elo,
)
from sports.wta.ingest import load_combined_matches  # noqa: E402


def legacy_elo_probabilities(prepared: pd.DataFrame) -> np.ndarray:
    """P(winner) from the old fixed-K, raw-id Elo, in ``prepared`` order."""
    ratings: dict[str, float] = {}
    out = np.empty(len(prepared))
    for index, (tour, wid, lid) in enumerate(
        zip(prepared["tour"].values, prepared["winner_id"].values, prepared["loser_id"].values)
    ):
        w = f"{tour}:{normalize_identifier(wid)}"
        lo = f"{tour}:{normalize_identifier(lid)}"
        rw, rl = ratings.get(w, 1500.0), ratings.get(lo, 1500.0)
        expected = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))
        out[index] = expected
        ratings[w] = rw + 32.0 * (1.0 - expected)
        ratings[lo] = rl - 32.0 * (1.0 - expected)
    return out


def ranking_favourite_accuracy(frame: pd.DataFrame) -> dict[str, Any]:
    wr = pd.to_numeric(frame.get("winner_rank"), errors="coerce")
    lr = pd.to_numeric(frame.get("loser_rank"), errors="coerce")
    ok = wr.notna() & lr.notna() & (wr != lr)
    if not ok.any():
        return {"n": 0, "accuracy": None}
    return {"n": int(ok.sum()), "accuracy": float((wr[ok] < lr[ok]).mean())}


def evaluate(
    matches: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    start: int = 20230101,
    end: int = 20260101,
) -> dict[str, Any]:
    params = params or load_params()
    prepared = prepare_matches(matches, params)
    values, _, _ = run_elo(prepared, params)
    prepared = pd.concat([prepared, values], axis=1)
    prepared["p_legacy"] = legacy_elo_probabilities(prepared)
    mask = evaluation_mask(prepared, start, end)
    window = prepared[mask].copy()
    window["year"] = window["tourney_date"] // 10000

    def block(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "elo": match_metrics(frame["p_winner"].values),
            "legacy_elo": match_metrics(frame["p_legacy"].values),
            "ranking_favourite": ranking_favourite_accuracy(frame),
        }

    report: dict[str, Any] = {
        "params_version": params.get("version"),
        "window": [start, end],
        "all": block(window),
        "by_tour": {tour: block(group) for tour, group in window.groupby("tour")},
        "by_tour_year": {
            f"{tour}-{int(year)}": block(group) for (tour, year), group in window.groupby(["tour", "year"])
        },
    }
    return report


def _fmt(metrics: dict[str, Any]) -> str:
    if not metrics or not metrics.get("n"):
        return "n=0"
    return "n=%d acc=%.4f brier=%.4f ll=%.4f" % (
        metrics["n"],
        metrics["accuracy"],
        metrics["brier"],
        metrics["log_loss"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=20230101)
    parser.add_argument("--end", type=int, default=20260101)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    matches = load_combined_matches()
    report = evaluate(matches, start=args.start, end=args.end)
    print(f"params {report['params_version']} window {args.start}..{args.end}")
    for label, block in [("ALL", report["all"])] + list(report["by_tour"].items()) + list(report["by_tour_year"].items()):
        rank = block["ranking_favourite"]
        print(
            f"{label:10s} elo[{_fmt(block['elo'])}]  legacy[{_fmt(block['legacy_elo'])}]  "
            f"rank-fav acc={rank['accuracy'] if rank['accuracy'] is None else round(rank['accuracy'], 4)} (n={rank['n']})"
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
