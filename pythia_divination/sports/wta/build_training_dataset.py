"""Build the ATP/WTA training dataset for the tournament ranker."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sports.wta.feature_engineering import add_rolling_features, build_player_event_features, process_match_history

DEFAULT_WTA_DATA_ROOT = PACKAGE_ROOT / "data" / "sports" / "wta"

logger = logging.getLogger(__name__)


def build_wta_dataset() -> None:
    """Execute the full pipeline to build the ATP/WTA training dataset."""
    normalized_dir = DEFAULT_WTA_DATA_ROOT / "normalized"
    parquet_path = normalized_dir / "tennis_matches_combined.parquet"
    csv_path = normalized_dir / "tennis_matches_combined.csv"

    logger.info("Loading match data...")
    if parquet_path.exists():
        try:
            matches = pd.read_parquet(parquet_path)
        except ImportError:
            logger.warning("Parquet engine unavailable; falling back to %s", csv_path)
            if not csv_path.exists():
                logger.error("Combined matches not found. Run ingest.py first.")
                return
            matches = pd.read_csv(csv_path, low_memory=False)
    elif csv_path.exists():
        matches = pd.read_csv(csv_path, low_memory=False)
    else:
        logger.error("Combined matches not found. Run ingest.py first.")
        return

    logger.info("Processing match history (Elo and technical stats)...")
    history_with_elo = process_match_history(matches)

    logger.info("Building player-event features...")
    player_events = build_player_event_features(history_with_elo)

    logger.info("Adding rolling form features...")
    dataset = add_rolling_features(player_events)

    output_path = DEFAULT_WTA_DATA_ROOT / "normalized" / "wta_training_dataset_latest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.info("Dataset built successfully: %s", output_path)
    logger.info("Rows: %s, Columns: %s", len(dataset), len(dataset.columns))
    logger.info("Tournaments: %s", dataset["tournament_id"].nunique())

    duplicate_events = (
        dataset[["tournament_id", "tour"]]
        .drop_duplicates()
        .groupby("tournament_id")
        .size()
        .gt(1)
        .sum()
    )

    counts_by_tour_year = {}
    if "date" in dataset.columns:
        years = pd.to_datetime(dataset["date"].astype(str), format="%Y%m%d", errors="coerce").dt.year
        summary = (
            dataset.assign(year=years)
            .groupby(["tour", "year"], dropna=False)
            .size()
            .reset_index(name="rows")
        )
        for row in summary.to_dict(orient="records"):
            tour = str(row["tour"])
            year = "unknown" if pd.isna(row["year"]) else str(int(row["year"]))
            counts_by_tour_year.setdefault(tour, {})[year] = int(row["rows"])

    manifest = {
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "tournaments": int(dataset["tournament_id"].nunique()),
        "duplicate_canonical_tournament_ids": int(duplicate_events),
        "counts_by_tour_year": counts_by_tour_year,
        "columns_present": dataset.columns.tolist(),
    }
    manifest_path = DEFAULT_WTA_DATA_ROOT / "normalized" / "wta_training_dataset_manifest_latest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Dataset manifest written to %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    build_wta_dataset()
