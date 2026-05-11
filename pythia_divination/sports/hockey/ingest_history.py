"""Backfill Hockey history and boxscores from the official NHL API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.hockey.client import (  # noqa: E402
    HockeyStatsClient,
    flatten_club_schedule_payload,
    flatten_club_stats,
    flatten_game_bundle,
    flatten_player_boxscores,
    flatten_roster,
)
from sports.hockey.constants import (  # noqa: E402
    CURRENT_SEASON_START,
    DEFAULT_SEASON_END,
    DEFAULT_SEASON_START,
    REGULAR_SEASON_GAME_TYPE,
    SUPPORTED_GAME_STATES_COMPLETED,
    TEAM_CODES,
    season_id,
    season_label,
    season_start_years_for_range,
)
from sports.pga.storage import build_ingestion_paths, write_csv, write_json, write_parquet  # noqa: E402

logger = logging.getLogger(__name__)
DEFAULT_HOCKEY_DATA_ROOT = ROOT / "data" / "sports" / "hockey"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Hockey history from the official NHL API.")
    parser.add_argument("--season-start", type=int, default=DEFAULT_SEASON_START, help="First season start year to include.")
    parser.add_argument("--season-end", type=int, default=DEFAULT_SEASON_END, help="Last season start year to include.")
    parser.add_argument("--include-game-details", action="store_true", help="Fetch detailed game boxscores for completed regular-season games.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on completed-game detail fetches.")
    parser.add_argument("--max-workers", type=int, default=12, help="Parallel workers for boxscore fetches.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument("--snapshot-tag", default=None, help="Optional explicit run tag.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()



def _fetch_game_detail(game_id: str) -> dict[str, Any] | None:
    client = HockeyStatsClient()
    bundle = client.get_game_bundle_optional(game_id)
    if bundle is None:
        return None
    return {
        "game_id": game_id,
        "detail_frame": flatten_game_bundle(bundle),
        "player_frame": flatten_player_boxscores(bundle),
        "raw_payload": bundle.boxscore,
    }



def run_ingestion(args: argparse.Namespace) -> dict[str, object]:
    base_dir = Path(args.output_root) if args.output_root else DEFAULT_HOCKEY_DATA_ROOT
    paths = build_ingestion_paths(base_dir=base_dir, snapshot_tag=args.snapshot_tag)
    season_years = season_start_years_for_range(args.season_start, args.season_end)
    client = HockeyStatsClient()

    schedule_frames: list[pd.DataFrame] = []
    club_stat_frames: list[pd.DataFrame] = []
    roster_frames: list[pd.DataFrame] = []

    for start_year in season_years:
        season_value = season_id(start_year)
        logger.info("Downloading Hockey club schedules, rosters, and club stats for %s...", season_label(start_year))
        for team_code in TEAM_CODES:
            schedule_payload = client.get_club_schedule_season(team_code, season_value)
            schedule_frames.append(flatten_club_schedule_payload(schedule_payload, team_code=team_code))
            write_json(paths.raw_dir / f"schedule_{team_code}_{season_value}.json", schedule_payload)

            roster_payload = client.get_roster(team_code, season_value)
            roster_frames.append(flatten_roster(roster_payload, team_code=team_code, season_id=season_value))
            write_json(paths.raw_dir / f"roster_{team_code}_{season_value}.json", roster_payload)

            club_stats_payload = client.get_club_stats(team_code, season_value, REGULAR_SEASON_GAME_TYPE)
            club_stat_frames.append(flatten_club_stats(club_stats_payload, team_code=team_code, season_id=season_value))
            write_json(paths.raw_dir / f"club_stats_{team_code}_{season_value}.json", club_stats_payload)

    schedule_df = pd.concat(schedule_frames, ignore_index=True).drop_duplicates(subset=["game_id"]).sort_values(["official_date", "game_id"]).reset_index(drop=True) if schedule_frames else pd.DataFrame()
    roster_df = pd.concat(roster_frames, ignore_index=True).drop_duplicates(subset=["season_id", "team_key", "player_id"]).reset_index(drop=True) if roster_frames else pd.DataFrame()
    club_stats_df = pd.concat(club_stat_frames, ignore_index=True).drop_duplicates(subset=["season_id", "team_key", "player_id", "player_type"]).reset_index(drop=True) if club_stat_frames else pd.DataFrame()

    completed = schedule_df.loc[
        (pd.to_numeric(schedule_df.get("game_type"), errors="coerce") == REGULAR_SEASON_GAME_TYPE)
        & schedule_df["game_state"].isin(SUPPORTED_GAME_STATES_COMPLETED)
    ].copy()
    if args.limit is not None:
        completed = completed.head(max(0, int(args.limit)))

    details_df = pd.DataFrame()
    player_df = pd.DataFrame()
    if args.include_game_details and not completed.empty:
        detail_rows: list[pd.DataFrame] = []
        player_rows: list[pd.DataFrame] = []
        game_ids = completed["game_id"].astype(str).tolist()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
            future_map = {executor.submit(_fetch_game_detail, game_id): game_id for game_id in game_ids}
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                if result is None:
                    continue
                logger.info("Fetched Hockey boxscore for %s", result["game_id"])
                detail_rows.append(result["detail_frame"])
                player_rows.append(result["player_frame"])
                write_json(paths.raw_dir / f"hockey_game_{result['game_id']}.json", result["raw_payload"])
        if detail_rows:
            details_df = pd.concat(detail_rows, ignore_index=True).drop_duplicates(subset=["league", "game_id"]).sort_values(["official_date", "game_id"]).reset_index(drop=True)
        if player_rows:
            player_df = pd.concat(player_rows, ignore_index=True).drop_duplicates(subset=["league", "game_id", "team_id", "player_id", "player_type"]).sort_values(["official_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)

    write_csv(paths.raw_dir / "hockey_schedule_raw_latest.csv", schedule_df)
    write_parquet(paths.raw_dir / "hockey_schedule_raw_latest.parquet", schedule_df)
    write_csv(paths.normalized_dir / "hockey_schedule_latest.csv", schedule_df)
    write_parquet(paths.normalized_dir / "hockey_schedule_latest.parquet", schedule_df)
    write_csv(paths.normalized_dir / "hockey_rosters_latest.csv", roster_df)
    write_parquet(paths.normalized_dir / "hockey_rosters_latest.parquet", roster_df)
    write_csv(paths.normalized_dir / "hockey_club_stats_latest.csv", club_stats_df)
    write_parquet(paths.normalized_dir / "hockey_club_stats_latest.parquet", club_stats_df)
    if not details_df.empty:
        write_csv(paths.normalized_dir / "hockey_game_details_latest.csv", details_df)
        write_parquet(paths.normalized_dir / "hockey_game_details_latest.parquet", details_df)
    if not player_df.empty:
        write_csv(paths.normalized_dir / "hockey_player_boxscores_latest.csv", player_df)
        write_parquet(paths.normalized_dir / "hockey_player_boxscores_latest.parquet", player_df)

    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "season_start": int(args.season_start),
        "season_end": int(args.season_end),
        "season_display_values": [season_label(year) for year in season_years],
        "current_season": season_label(CURRENT_SEASON_START),
        "schedule_rows": int(len(schedule_df)),
        "regular_season_games": int((pd.to_numeric(schedule_df.get("game_type"), errors="coerce") == REGULAR_SEASON_GAME_TYPE).sum()) if not schedule_df.empty else 0,
        "completed_regular_season_games": int(len(completed)),
        "detail_rows": int(len(details_df)),
        "player_boxscore_rows": int(len(player_df)),
        "club_stats_rows": int(len(club_stats_df)),
        "roster_rows": int(len(roster_df)),
        "normalized_dir": str(paths.normalized_dir),
    }
    write_json(paths.normalized_dir / "hockey_history_manifest_latest.json", manifest)
    logger.info("Hockey history manifest: %s", json.dumps(manifest, indent=2))
    return manifest



def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_ingestion(args)


if __name__ == "__main__":
    main()
