"""Build the WTA training dataset for the tournament ranker."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.wta.feature_engineering import (
    process_match_history,
    build_player_event_features,
    add_rolling_features
)

DEFAULT_WTA_DATA_ROOT = ROOT / "data" / "sports" / "wta"

logger = logging.getLogger(__name__)

def build_wta_dataset():
    """Execute the full pipeline to build the WTA training dataset."""
    matches_path = DEFAULT_WTA_DATA_ROOT / "normalized" / "wta_matches_combined.parquet"
    if not matches_path.exists():
        logger.error("Combined matches not found. Run ingest.py first.")
        return
        
    logger.info("Loading match data...")
    matches = pd.read_parquet(matches_path)
    
    logger.info("Processing match history (Elo & technical stats)...")
    history_with_elo = process_match_history(matches)
    
    logger.info("Building player-event features...")
    player_events = build_player_event_features(history_with_elo)
    
    logger.info("Adding rolling form features...")
    dataset = add_rolling_features(player_events)
    
    # Save the training dataset
    output_path = DEFAULT_WTA_DATA_ROOT / "normalized" / "wta_training_dataset_latest.csv"
    dataset.to_csv(output_path, index=False)
    logger.info("Dataset built successfully: %s", output_path)
    logger.info("Rows: %s, Columns: %s", len(dataset), len(dataset.columns))
    
    # Tournament stats
    tourneys = dataset["tournament_id"].nunique()
    logger.info("Tournaments: %s", tourneys)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    build_wta_dataset()
