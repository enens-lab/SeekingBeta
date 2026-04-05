"""Build a Football matchup training dataset from nflverse schedule and weekly stat tables."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from sports.football.constants import DEFAULT_SEASON_END, DEFAULT_SEASON_START
from sports.football.feature_engineering import (
    add_matchup_differentials,
    attach_pregame_qb_features,
    attach_pregame_roster_features,
    attach_pregame_team_features,
    build_qb_week_logs,
    build_roster_week_summaries,
    build_team_game_logs,
    prepare_games,
)
from sports.football.ingest_history import DEFAULT_FOOTBALL_DATA_ROOT, run_ingestion
from sports.pga.storage import build_ingestion_paths, read_preferred_table, write_csv, write_json, write_parquet

logger = logging.getLogger(__name__)

_LEAKY_COLUMNS = {
    "away_score",
    "home_score",
    "result",
    "total",
    "point_margin_home",
    "point_total",
    "away_win",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Football home-win training dataset.")
    parser.add_argument("--season-start", type=int, default=DEFAULT_SEASON_START, help="First season to include.")
    parser.add_argument("--season-end", type=int, default=DEFAULT_SEASON_END, help="Last season to include.")
    parser.add_argument("--refresh-source-data", action="store_true", help="Refresh source data before building.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument("--snapshot-tag", default=None, help="Optional explicit run tag.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _load_table(base_dir: Path, stem: str) -> pd.DataFrame:
    frame = read_preferred_table(
        base_dir / "normalized" / f"{stem}.parquet",
        base_dir / "normalized" / f"{stem}.csv",
    )
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str)
    return frame


def build_training_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    base_dir = Path(args.output_root) if args.output_root else DEFAULT_FOOTBALL_DATA_ROOT
    paths = build_ingestion_paths(base_dir=base_dir, snapshot_tag=args.snapshot_tag)

    if args.refresh_source_data:
        run_ingestion(
            SimpleNamespace(
                season_start=args.season_start,
                season_end=args.season_end,
                output_root=str(base_dir),
                snapshot_tag=args.snapshot_tag,
                verbose=args.verbose,
            )
        )

    games_df = _load_table(base_dir, "football_games_latest")
    team_week_df = _load_table(base_dir, "football_team_week_stats_latest")
    player_week_df = _load_table(base_dir, "football_player_week_stats_latest")
    weekly_rosters_df = _load_table(base_dir, "football_weekly_rosters_latest")

    games = prepare_games(
        games_df,
        season_start=args.season_start,
        season_end=args.season_end,
        require_completed=True,
    )
    if games.empty:
        raise ValueError("No completed regular-season Football games were available for dataset building.")

    team_logs = build_team_game_logs(games, team_week_df)
    qb_logs = build_qb_week_logs(player_week_df)
    roster_summaries = build_roster_week_summaries(weekly_rosters_df)

    dataset = attach_pregame_team_features(games, team_logs)
    dataset = attach_pregame_qb_features(dataset, qb_logs)
    dataset = attach_pregame_roster_features(dataset, roster_summaries)
    dataset = add_matchup_differentials(dataset)
    dataset = dataset.drop(columns=[column for column in _LEAKY_COLUMNS if column in dataset.columns])
    dataset = dataset.sort_values(["official_date", "season", "week", "game_id"]).reset_index(drop=True)

    write_csv(paths.normalized_dir / "football_training_dataset_latest.csv", dataset)
    write_parquet(paths.normalized_dir / "football_training_dataset_latest.parquet", dataset)
    write_csv(paths.normalized_dir / "football_team_game_logs_latest.csv", team_logs)
    write_parquet(paths.normalized_dir / "football_team_game_logs_latest.parquet", team_logs)
    write_csv(paths.normalized_dir / "football_qb_week_logs_latest.csv", qb_logs)
    write_parquet(paths.normalized_dir / "football_qb_week_logs_latest.parquet", qb_logs)
    write_csv(paths.normalized_dir / "football_roster_week_summaries_latest.csv", roster_summaries)
    write_parquet(paths.normalized_dir / "football_roster_week_summaries_latest.parquet", roster_summaries)

    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "season_start": int(args.season_start),
        "season_end": int(args.season_end),
        "games_rows": int(len(games)),
        "team_week_rows": int(len(team_week_df)),
        "player_week_rows": int(len(player_week_df)),
        "weekly_roster_rows": int(len(weekly_rosters_df)),
        "team_game_log_rows": int(len(team_logs)),
        "qb_week_log_rows": int(len(qb_logs)),
        "roster_summary_rows": int(len(roster_summaries)),
        "dataset_rows": int(len(dataset)),
        "dataset_columns": int(len(dataset.columns)),
        "season_values": sorted({int(value) for value in dataset["season"].dropna().unique()}),
        "output_dataset": str(paths.normalized_dir / "football_training_dataset_latest.csv"),
    }
    write_json(paths.normalized_dir / "football_training_dataset_manifest_latest.json", manifest)
    logger.info("Football dataset built: %s", json.dumps(manifest, indent=2))
    return dataset, manifest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    build_training_dataset(args)


if __name__ == "__main__":
    main()
