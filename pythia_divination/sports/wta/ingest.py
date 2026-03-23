"""Ingest historical WTA match data from Jeff Sackmann's tennis_wta repository."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.pga.storage import DEFAULT_PGA_DATA_ROOT # Reuse storage logic if applicable

# Specific WTA data root
DEFAULT_WTA_DATA_ROOT = ROOT / "data" / "sports" / "wta"

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"

def _download_file(url: str, dest: Path) -> bool:
    """Download a file from a URL to a destination path."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        return False

def ingest_wta_matches(years: list[int], force: bool = False) -> pd.DataFrame:
    """Download and merge WTA match data for specified years."""
    all_matches = []
    
    for year in years:
        filename = f"wta_matches_{year}.csv"
        url = f"{BASE_URL}/{filename}"
        dest = DEFAULT_WTA_DATA_ROOT / "raw" / filename
        
        if not dest.exists() or force:
            logger.info("Downloading match data for year %s...", year)
            if not _download_file(url, dest):
                continue
        else:
            logger.info("Using cached match data for year %s.", year)
            
        try:
            df = pd.read_csv(dest, low_memory=False)
            all_matches.append(df)
        except Exception as e:
            logger.error("Failed to read %s: %s", dest, e)
            
    if not all_matches:
        return pd.DataFrame()
        
    combined = pd.concat(all_matches, ignore_index=True)
    logger.info("Combined %s match rows across several years.", len(combined))
    
    # Fix types for Parquet
    for col in combined.columns:
        if combined[col].dtype == "object":
            combined[col] = combined[col].astype(str).replace("nan", "")
            
    # Save combined parquet
    output_path = DEFAULT_WTA_DATA_ROOT / "normalized" / "wta_matches_combined.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    logger.info("Saved combined matches to %s", output_path)
    
    return combined

def ingest_wta_players(force: bool = False) -> pd.DataFrame:
    """Download WTA player profile data."""
    filename = "wta_players.csv"
    url = f"{BASE_URL}/{filename}"
    dest = DEFAULT_WTA_DATA_ROOT / "raw" / filename
    
    if not dest.exists() or force:
        logger.info("Downloading player profiles...")
        _download_file(url, dest)
        
    df = pd.read_csv(dest, low_memory=False)
    output_path = DEFAULT_WTA_DATA_ROOT / "normalized" / "wta_players.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest WTA data from Jeff Sackmann's repo.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--force", action="store_true", help="Force re-download of data.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    years = list(range(args.start_year, args.end_year + 1))
    ingest_wta_matches(years, force=args.force)
    ingest_wta_players(force=args.force)

if __name__ == "__main__":
    main()
