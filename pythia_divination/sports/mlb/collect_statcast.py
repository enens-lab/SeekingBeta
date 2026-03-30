"""Collect Baseball Savant Statcast exports and build daily aggregate tables."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
from pybaseball import statcast

from sports.mlb.statcast_enrichment import build_daily_group_features, normalize_statcast
from sports.pga.storage import write_csv, write_json, write_parquet

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATCAST_ROOT = ROOT / "data" / "sports" / "mlb" / "statcast"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Statcast pitch-level exports in date chunks.")
    parser.add_argument("--start-date", required=True, help="Inclusive ISO start date.")
    parser.add_argument("--end-date", required=True, help="Inclusive ISO end date.")
    parser.add_argument("--chunk-days", type=int, default=7, help="Days per Statcast fetch chunk.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument("--snapshot-tag", default=None, help="Optional explicit run tag.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _iter_date_chunks(start_date: date, end_date: date, chunk_days: int) -> Iterator[tuple[date, date]]:
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def collect_statcast(args: argparse.Namespace) -> dict[str, object]:
    base_dir = Path(args.output_root) if args.output_root else DEFAULT_STATCAST_ROOT
    snapshot_tag = args.snapshot_tag or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    raw_dir = base_dir / "raw" / snapshot_tag
    normalized_dir = base_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date")
    dataset_label = f"{start_date.isoformat()}_to_{end_date.isoformat()}"

    chunk_frames: list[pd.DataFrame] = []
    chunk_manifest: list[dict[str, object]] = []

    for chunk_start, chunk_end in _iter_date_chunks(start_date, end_date, args.chunk_days):
        logger.info("Fetching Statcast chunk %s -> %s", chunk_start, chunk_end)
        frame = statcast(chunk_start.isoformat(), chunk_end.isoformat(), verbose=False, parallel=False)
        chunk_slug = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}"
        raw_csv = raw_dir / f"statcast_{chunk_slug}.csv"
        write_csv(raw_csv, frame)
        write_parquet(raw_dir / f"statcast_{chunk_slug}.parquet", frame)
        chunk_frames.append(frame)
        chunk_manifest.append(
            {
                "chunk_start": chunk_start.isoformat(),
                "chunk_end": chunk_end.isoformat(),
                "row_count": int(len(frame)),
                "csv_path": str(raw_csv),
            }
        )

    combined = pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame()
    normalized = normalize_statcast(combined) if not combined.empty else pd.DataFrame()

    pitcher_daily = (
        build_daily_group_features(normalized.dropna(subset=["pitcher_id"]), group_col="pitcher_id", prefix="pitcher")
        if not normalized.empty and "pitcher_id" in normalized.columns
        else pd.DataFrame()
    )
    team_daily = (
        build_daily_group_features(normalized.dropna(subset=["batting_team"]), group_col="batting_team", prefix="batting_team")
        if not normalized.empty and "batting_team" in normalized.columns
        else pd.DataFrame()
    )

    write_csv(normalized_dir / "statcast_combined_latest.csv", combined)
    write_parquet(normalized_dir / "statcast_combined_latest.parquet", combined)
    write_csv(normalized_dir / f"statcast_combined_{dataset_label}_latest.csv", combined)
    write_parquet(normalized_dir / f"statcast_combined_{dataset_label}_latest.parquet", combined)
    write_csv(normalized_dir / "statcast_normalized_latest.csv", normalized)
    write_parquet(normalized_dir / "statcast_normalized_latest.parquet", normalized)
    write_csv(normalized_dir / f"statcast_normalized_{dataset_label}_latest.csv", normalized)
    write_parquet(normalized_dir / f"statcast_normalized_{dataset_label}_latest.parquet", normalized)
    write_csv(normalized_dir / "statcast_pitcher_daily_latest.csv", pitcher_daily)
    write_parquet(normalized_dir / "statcast_pitcher_daily_latest.parquet", pitcher_daily)
    write_csv(normalized_dir / f"statcast_pitcher_daily_{dataset_label}_latest.csv", pitcher_daily)
    write_parquet(normalized_dir / f"statcast_pitcher_daily_{dataset_label}_latest.parquet", pitcher_daily)
    write_csv(normalized_dir / "statcast_team_daily_latest.csv", team_daily)
    write_parquet(normalized_dir / "statcast_team_daily_latest.parquet", team_daily)
    write_csv(normalized_dir / f"statcast_team_daily_{dataset_label}_latest.csv", team_daily)
    write_parquet(normalized_dir / f"statcast_team_daily_{dataset_label}_latest.parquet", team_daily)

    manifest = {
        "snapshot_tag": snapshot_tag,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "dataset_label": dataset_label,
        "chunk_days": args.chunk_days,
        "chunk_count": len(chunk_manifest),
        "combined_rows": int(len(combined)),
        "normalized_rows": int(len(normalized)),
        "pitcher_daily_rows": int(len(pitcher_daily)),
        "team_daily_rows": int(len(team_daily)),
        "raw_dir": str(raw_dir),
        "normalized_dir": str(normalized_dir),
        "chunks": chunk_manifest,
    }
    write_json(raw_dir / "statcast_manifest.json", manifest)
    write_json(normalized_dir / "statcast_manifest_latest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    manifest = collect_statcast(args)
    print("=" * 72)
    print("MLB STATCAST COLLECTION COMPLETE")
    print("=" * 72)
    print(json.dumps(manifest, indent=2))
    print("=" * 72)


if __name__ == "__main__":
    main()
