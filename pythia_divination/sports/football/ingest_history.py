"""Download nflverse Football history into durable raw and normalized tables."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import warnings

import pandas as pd

from sports.football.client import FootballStatsClient
from sports.football.constants import (
    CURRENT_SEASON,
    DEFAULT_SEASON_END,
    DEFAULT_SEASON_START,
    season_years_for_range,
)
from sports.pga.storage import build_ingestion_paths, write_csv, write_json, write_parquet

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOOTBALL_DATA_ROOT = ROOT / "data" / "sports" / "football"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Football history from nflverse release assets.")
    parser.add_argument("--season-start", type=int, default=DEFAULT_SEASON_START, help="First season to include.")
    parser.add_argument("--season-end", type=int, default=DEFAULT_SEASON_END, help="Last season to include.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument("--snapshot-tag", default=None, help="Optional explicit run tag.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _sanitize_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sanitized = frame.copy()
    for column in sanitized.columns:
        series = sanitized[column]
        if series.dtype == "object":
            sanitized[column] = series.where(series.notna(), "").astype(str)
    return sanitized


def _write_with_parquet_fallback(csv_path: Path, parquet_path: Path, frame: pd.DataFrame) -> None:
    write_csv(csv_path, frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        write_parquet(parquet_path, _sanitize_for_parquet(frame))


def run_ingestion(args: argparse.Namespace) -> dict[str, object]:
    base_dir = Path(args.output_root) if args.output_root else DEFAULT_FOOTBALL_DATA_ROOT
    paths = build_ingestion_paths(base_dir=base_dir, snapshot_tag=args.snapshot_tag)
    seasons = season_years_for_range(args.season_start, args.season_end)
    client = FootballStatsClient()

    logger.info("Downloading Football schedule history...")
    games = client.get_games()
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games = games.loc[games["season"].isin(seasons)].copy()
    games = games.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)

    team_frames: list[pd.DataFrame] = []
    player_frames: list[pd.DataFrame] = []
    roster_frames: list[pd.DataFrame] = []
    for season in seasons:
        logger.info("Downloading Football team/player/roster week tables for %s...", season)
        team_df = client.get_team_week_stats(season)
        player_df = client.get_player_week_stats(season)
        roster_df = client.get_weekly_rosters(season)
        team_frames.append(team_df)
        player_frames.append(player_df)
        roster_frames.append(roster_df)
        write_csv(paths.raw_dir / f"stats_team_week_{season}.csv", team_df)
        write_csv(paths.raw_dir / f"stats_player_week_{season}.csv", player_df)
        write_csv(paths.raw_dir / f"roster_weekly_{season}.csv", roster_df)

    logger.info("Downloading Football player directory...")
    players = client.get_players()

    team_week = pd.concat(team_frames, ignore_index=True).drop_duplicates() if team_frames else pd.DataFrame()
    player_week = pd.concat(player_frames, ignore_index=True).drop_duplicates() if player_frames else pd.DataFrame()
    weekly_rosters = pd.concat(roster_frames, ignore_index=True).drop_duplicates() if roster_frames else pd.DataFrame()

    write_csv(paths.raw_dir / "games.csv", games)
    write_csv(paths.raw_dir / "players.csv", players)

    _write_with_parquet_fallback(paths.normalized_dir / "football_games_latest.csv", paths.normalized_dir / "football_games_latest.parquet", games)
    _write_with_parquet_fallback(paths.normalized_dir / "football_team_week_stats_latest.csv", paths.normalized_dir / "football_team_week_stats_latest.parquet", team_week)
    _write_with_parquet_fallback(paths.normalized_dir / "football_player_week_stats_latest.csv", paths.normalized_dir / "football_player_week_stats_latest.parquet", player_week)
    _write_with_parquet_fallback(paths.normalized_dir / "football_weekly_rosters_latest.csv", paths.normalized_dir / "football_weekly_rosters_latest.parquet", weekly_rosters)
    _write_with_parquet_fallback(paths.normalized_dir / "football_players_latest.csv", paths.normalized_dir / "football_players_latest.parquet", players)

    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "season_start": int(args.season_start),
        "season_end": int(args.season_end),
        "seasons": seasons,
        "current_season": CURRENT_SEASON,
        "games_rows": int(len(games)),
        "team_week_rows": int(len(team_week)),
        "player_week_rows": int(len(player_week)),
        "weekly_roster_rows": int(len(weekly_rosters)),
        "players_rows": int(len(players)),
        "normalized_dir": str(paths.normalized_dir),
    }
    write_json(paths.normalized_dir / "football_history_manifest_latest.json", manifest)
    logger.info("Football history manifest: %s", json.dumps(manifest, indent=2))
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
