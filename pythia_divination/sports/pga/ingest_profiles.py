"""Collect stable public PGA player profile data for model features."""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.pga.client import PGATourStatsClient  # noqa: E402
from sports.pga.storage import build_ingestion_paths, write_csv, write_json, write_parquet  # noqa: E402

logger = logging.getLogger(__name__)


PROFILE_COLUMNS = [
    "player_id",
    "player_name",
    "profile_source_path",
    "profile_first_name",
    "profile_last_name",
    "profile_country",
    "profile_country_code",
    "profile_born_date",
    "profile_born_year",
    "profile_age_current",
    "profile_turned_pro_year",
    "profile_birthplace",
    "profile_college",
    "profile_height_text",
    "profile_height_inches",
    "profile_equipment_sponsor_primary",
    "profile_equipment_sponsor_count",
    "profile_has_equipment_sponsor",
    "profile_social_count",
    "profile_has_instagram",
    "profile_has_x",
    "profile_has_twitter",
    "profile_has_facebook",
    "profile_has_tiktok",
    "profile_has_youtube",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PGA player profiles from the public player pages.")
    parser.add_argument("--player-id", action="append", default=[], help="Explicit player ID(s) to fetch.")
    parser.add_argument("--player-name", action="append", default=[], help="Optional player name(s), matched by order to --player-id.")
    parser.add_argument("--input-csv", default=None, help="Optional CSV with player_id and optional player_name columns.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit after de-duplication.")
    parser.add_argument("--output-root", default=None, help="Override output root. Defaults to pythia_divination/data/sports/pga.")
    parser.add_argument("--snapshot-tag", default=None, help="Optional explicit run tag for raw snapshot storage.")
    parser.add_argument("--refresh", action="store_true", help="Force refetch even if normalized profile data exists.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent fetch workers.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _normalize_player_index(player_index: pd.DataFrame) -> pd.DataFrame:
    frame = player_index.copy()
    required = {"player_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"player_index missing required columns: {sorted(missing)}")
    frame["player_id"] = frame["player_id"].astype(str).str.strip()
    if "player_name" not in frame.columns:
        frame["player_name"] = None
    frame["player_name"] = frame["player_name"].astype(object)
    if "country" not in frame.columns:
        frame["country"] = None
    frame = frame[frame["player_id"].astype(str).str.len() > 0]
    frame = frame.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    return frame


def _player_frame_from_args(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv:
        frame = pd.read_csv(args.input_csv, low_memory=False)
        return _normalize_player_index(frame)

    names = list(args.player_name)
    rows: list[dict[str, Any]] = []
    for index, player_id in enumerate(args.player_id):
        rows.append(
            {
                "player_id": str(player_id),
                "player_name": names[index] if index < len(names) else None,
            }
        )
    if not rows:
        raise ValueError("Provide --input-csv or at least one --player-id")
    return _normalize_player_index(pd.DataFrame(rows))


def _load_existing_profiles(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    frame = pd.read_csv(path, low_memory=False)
    for column in PROFILE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[PROFILE_COLUMNS]


def _fetch_profile_record(player_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    client = PGATourStatsClient()
    payload = client.get_player_profile(
        player_id=str(player_row["player_id"]),
        player_name=player_row.get("player_name"),
    )
    return payload["flattened"], payload


def collect_player_profiles(
    player_index: pd.DataFrame,
    *,
    output_root: Path | str | None = None,
    snapshot_tag: str | None = None,
    refresh: bool = False,
    limit: int | None = None,
    workers: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = build_ingestion_paths(base_dir=Path(output_root) if output_root else None, snapshot_tag=snapshot_tag)
    normalized_path = paths.normalized_dir / "player_profiles_latest.csv"
    existing = _load_existing_profiles(normalized_path)
    player_frame = _normalize_player_index(player_index)
    if limit is not None:
        player_frame = player_frame.head(max(0, limit)).reset_index(drop=True)

    existing_ids = set(existing["player_id"].astype(str)) if not existing.empty and not refresh else set()
    pending = player_frame[~player_frame["player_id"].astype(str).isin(existing_ids)].reset_index(drop=True)

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not pending.empty:
        max_workers = max(1, min(int(workers), len(pending)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_profile_record, row): row
                for row in pending.to_dict(orient="records")
            }
            for future in as_completed(futures):
                row = futures[future]
                player_id = str(row["player_id"])
                try:
                    flattened, payload = future.result()
                    records.append(flattened)
                    write_json(
                        paths.raw_dir / f"player_profile_{player_id}.json",
                        {
                            "player_id": player_id,
                            "player_name": row.get("player_name"),
                            "payload": payload,
                        },
                    )
                except Exception as exc:  # pragma: no cover - network variability
                    logger.warning("Failed fetching player profile %s (%s): %s", row.get("player_name"), player_id, exc)
                    errors.append(
                        {
                            "player_id": player_id,
                            "player_name": row.get("player_name"),
                            "error": str(exc),
                        }
                    )

    new_profiles = pd.DataFrame(records, columns=PROFILE_COLUMNS) if records else pd.DataFrame(columns=PROFILE_COLUMNS)
    combined = pd.concat([existing, new_profiles], ignore_index=True, sort=False)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["player_id"], keep="last")
        combined = combined.sort_values(["player_name", "player_id"], na_position="last").reset_index(drop=True)
    else:
        combined = pd.DataFrame(columns=PROFILE_COLUMNS)

    write_csv(normalized_path, combined)
    write_parquet(paths.normalized_dir / "player_profiles_latest.parquet", combined)
    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "requested_players": int(len(player_frame)),
        "fetched_players": int(len(new_profiles)),
        "cached_players": int(len(existing_ids & set(player_frame["player_id"].astype(str)))),
        "successful_profiles": int(len(combined)),
        "failed_profiles": int(len(errors)),
        "normalized_path": str(normalized_path),
        "raw_dir": str(paths.raw_dir),
        "errors": errors,
    }
    write_json(paths.raw_dir / "player_profiles_manifest.json", manifest)
    write_json(paths.normalized_dir / "player_profiles_manifest_latest.json", manifest)
    return combined, manifest


def run_profile_ingestion(args: argparse.Namespace) -> dict[str, Any]:
    player_frame = _player_frame_from_args(args)
    _, manifest = collect_player_profiles(
        player_frame,
        output_root=args.output_root,
        snapshot_tag=args.snapshot_tag,
        refresh=args.refresh,
        limit=args.limit,
        workers=args.workers,
    )
    return manifest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    manifest = run_profile_ingestion(args)
    print("=" * 72)
    print("PGA PLAYER PROFILE INGESTION COMPLETE")
    print("=" * 72)
    print(f"Requested players:   {manifest['requested_players']}")
    print(f"Successful profiles: {manifest['successful_profiles']}")
    print(f"Failed profiles:     {manifest['failed_profiles']}")
    print(f"Normalized output:   {manifest['normalized_path']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
