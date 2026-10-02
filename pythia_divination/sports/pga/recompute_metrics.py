"""Correct the stored golf multitask metrics.json from its validation predictions.

The artifact's ``per_target.won.top_pick_hit_rate`` was 0.0087, which is the
share of winners among all validation rows, not the share of events whose top
pick won (13.7%, 7 of 51 events with a known winner). This rewrites that value,
records the original under ``corrections``, and adds the walk-forward
calibrated top-10 / made-cut scores the boards now show.

    python -m sports.pga.recompute_metrics            # rewrite metrics.json
    python -m sports.pga.recompute_metrics --dry-run  # print only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sports.pga.calibration import calibrate_board, calibration_report  # noqa: E402

ARTIFACT_DIR = PACKAGE_ROOT / "artifacts" / "pga_neural_multitask_torch"
DATASET_PATH = PACKAGE_ROOT / "data" / "sports" / "pga" / "normalized" / "golf_training_dataset_latest.csv"
LABEL_COLUMNS = {"tournament_id", "tournament_name", "player_id", "event_start_date", "won", "top_10", "made_cut", "tour", "event_field_size"}


def _binary(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
    ).astype(float)


def load_labels(path: Path = DATASET_PATH) -> pd.DataFrame:
    labels = pd.read_csv(path, usecols=lambda column: column in LABEL_COLUMNS, low_memory=False)
    for column in ("won", "top_10", "made_cut"):
        labels[column] = _binary(labels[column])
    labels["player_id"] = pd.to_numeric(labels["player_id"], errors="coerce")
    labels = labels.dropna(subset=["tournament_id", "player_id"])
    return labels[labels["tour"].astype(str).str.upper() == "PGA"]


def top_pick_summary(predictions: pd.DataFrame, labels: pd.DataFrame, probability_column: str) -> dict:
    frame = predictions.merge(
        labels[["tournament_id", "player_id", "won"]].drop_duplicates(subset=["tournament_id", "player_id"]),
        on=["tournament_id", "player_id"],
        how="left",
    )
    winners = frame.groupby("tournament_id")["won"].transform("sum")
    gradable = frame[winners == 1]
    picks = gradable.sort_values(["tournament_id", probability_column], ascending=[True, False]).groupby("tournament_id").head(1)
    return {
        "top_pick_hit_rate": round(float(picks["won"].mean()), 4) if len(picks) else None,
        "top_pick_hits": int(picks["won"].sum()),
        "top_pick_events": int(len(picks)),
        "validation_events": int(frame["tournament_id"].nunique()),
    }


def recompute(artifact_dir: Path = ARTIFACT_DIR, dataset_path: Path = DATASET_PATH) -> dict:
    metrics = json.loads((artifact_dir / "metrics.json").read_text())
    predictions = pd.read_csv(artifact_dir / "validation_predictions.csv")
    predictions["player_id"] = pd.to_numeric(predictions["player_id"], errors="coerce")
    labels = load_labels(dataset_path)
    probability_column = "won_normalized_probability" if "won_normalized_probability" in predictions.columns else "won_probability"

    summary = top_pick_summary(predictions, labels, probability_column)
    inputs = predictions[["tournament_id", "tournament_name", "player_id"]].copy()
    inputs["winner_probability"] = predictions[probability_column].astype(float)
    inputs["top_10_probability"] = predictions["top_10_probability"].astype(float)
    inputs["made_cut_probability"] = predictions["made_cut_probability"].astype(float)
    calibrated, _ = calibrate_board(inputs, labels)
    report = calibration_report(calibrated, labels)

    won = metrics.setdefault("per_target", {}).setdefault("won", {})
    previous = won.get("top_pick_hit_rate")
    corrections = metrics.setdefault("corrections", [])
    if not any(item.get("field") == "per_target.won.top_pick_hit_rate" for item in corrections):
        corrections.append(
            {
                "field": "per_target.won.top_pick_hit_rate",
                "previous_value": previous,
                "reason": "was the share of winners among validation rows; now the share of events (with the winner in the held-out field) whose top pick won",
                "corrected_at": datetime.now(timezone.utc).date().isoformat(),
            }
        )
    won.update(summary)
    metrics["calibrated_heads"] = {
        "method": "walk-forward per-event logit shift + slope (sports/pga/calibration.py), as served on the boards",
        "events": report.get("events"),
        "top_10": report.get("top10"),
        "made_cut": report.get("madeCut"),
        "rows_made_cut_below_top_10": report.get("rowsMadeCutBelowTop10"),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    metrics = recompute(args.artifact_dir, args.dataset)
    print(json.dumps({"won": metrics["per_target"]["won"], "calibrated_heads": metrics["calibrated_heads"]}, indent=2))
    if not args.dry_run:
        (args.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Wrote {args.artifact_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
