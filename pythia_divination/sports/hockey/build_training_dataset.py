"""Build a Hockey matchup training dataset from official NHL schedule and boxscore history."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from sports.hockey.feature_engineering import (
    add_matchup_differentials,
    attach_pregame_goalie_features,
    attach_pregame_team_features,
    build_goalie_game_logs,
    build_team_game_logs,
    prepare_games,
)
from sports.hockey.ingest_history import DEFAULT_HOCKEY_DATA_ROOT, run_ingestion
from sports.pga.storage import build_ingestion_paths, read_preferred_table, write_csv, write_json, write_parquet

logger = logging.getLogger(__name__)

_LEAKY_COLUMNS = {
    "away_score",
    "home_score",
    "away_win",
    "home_win",
    "goal_diff_home",
    "total_goals",
    "game_state_detail",
    "game_schedule_state_detail",
    "away_goals_skaters",
    "home_goals_skaters",
    "away_assists",
    "home_assists",
    "away_points",
    "home_points",
    "away_hits",
    "home_hits",
    "away_blocked_shots",
    "home_blocked_shots",
    "away_pim",
    "home_pim",
    "away_power_play_goals",
    "home_power_play_goals",
    "away_skaters_shots",
    "home_skaters_shots",
    "away_giveaways",
    "home_giveaways",
    "away_takeaways",
    "home_takeaways",
    "away_faceoff_win_pct",
    "home_faceoff_win_pct",
    "away_shooting_pct",
    "home_shooting_pct",
    "away_starter_goalie_saves",
    "home_starter_goalie_saves",
    "away_starter_goalie_goals_against",
    "home_starter_goalie_goals_against",
    "away_starter_goalie_shots_against",
    "home_starter_goalie_shots_against",
    "away_starter_goalie_save_pct",
    "home_starter_goalie_save_pct",
    "away_starter_goalie_toi_minutes",
    "home_starter_goalie_toi_minutes",
}



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Hockey home-win training dataset.")
    parser.add_argument("--season-start", type=int, default=2024, help="First season start year to include.")
    parser.add_argument("--season-end", type=int, default=2025, help="Last season start year to include.")
    parser.add_argument("--refresh-source-data", action="store_true", help="Refresh source data before building.")
    parser.add_argument("--history-limit", type=int, default=None, help="Optional cap on completed game detail refreshes.")
    parser.add_argument("--max-workers", type=int, default=12, help="Parallel workers for boxscore refresh.")
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
    if "official_date" in frame.columns:
        frame["official_date"] = pd.to_datetime(frame["official_date"], errors="coerce", format="mixed")
    return frame



def build_training_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    base_dir = Path(args.output_root) if args.output_root else DEFAULT_HOCKEY_DATA_ROOT
    paths = build_ingestion_paths(base_dir=base_dir, snapshot_tag=args.snapshot_tag)

    if args.refresh_source_data:
        run_ingestion(
            SimpleNamespace(
                season_start=args.season_start,
                season_end=args.season_end,
                include_game_details=True,
                limit=args.history_limit,
                max_workers=args.max_workers,
                output_root=str(base_dir),
                snapshot_tag=args.snapshot_tag,
                verbose=args.verbose,
            )
        )

    schedule_df = _load_table(base_dir, "hockey_schedule_latest")
    details_df = _load_table(base_dir, "hockey_game_details_latest")
    player_boxscores_df = _load_table(base_dir, "hockey_player_boxscores_latest")

    if details_df.empty:
        raise ValueError("No Hockey game details are available yet. Run with --refresh-source-data after regular-season games exist.")
    if player_boxscores_df.empty:
        raise ValueError("No Hockey player boxscores are available yet. Run with --refresh-source-data after regular-season games exist.")

    games = prepare_games(schedule_df, details_df, season_start=args.season_start, season_end=args.season_end, require_completed=True)
    if games.empty:
        raise ValueError("No completed regular-season Hockey games were available for dataset building.")

    team_logs = build_team_game_logs(games)
    goalie_logs = build_goalie_game_logs(player_boxscores_df)
    dataset = attach_pregame_team_features(games, team_logs)
    dataset = attach_pregame_goalie_features(dataset, goalie_logs)
    dataset = add_matchup_differentials(dataset)
    dataset = dataset.drop(columns=[column for column in _LEAKY_COLUMNS if column in dataset.columns])
    dataset = dataset.sort_values(["official_date", "game_id"]).reset_index(drop=True)

    write_csv(paths.normalized_dir / "hockey_training_dataset_latest.csv", dataset)
    write_parquet(paths.normalized_dir / "hockey_training_dataset_latest.parquet", dataset)
    write_csv(paths.normalized_dir / "hockey_team_game_logs_latest.csv", team_logs)
    write_parquet(paths.normalized_dir / "hockey_team_game_logs_latest.parquet", team_logs)
    write_csv(paths.normalized_dir / "hockey_goalie_game_logs_latest.csv", goalie_logs)
    write_parquet(paths.normalized_dir / "hockey_goalie_game_logs_latest.parquet", goalie_logs)

    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "season_start": int(args.season_start),
        "season_end": int(args.season_end),
        "schedule_rows": int(len(schedule_df)),
        "detail_rows": int(len(details_df)),
        "player_boxscore_rows": int(len(player_boxscores_df)),
        "games_rows": int(len(games)),
        "team_game_log_rows": int(len(team_logs)),
        "goalie_game_log_rows": int(len(goalie_logs)),
        "dataset_rows": int(len(dataset)),
        "dataset_columns": int(len(dataset.columns)),
        "season_display_values": sorted({str(value) for value in dataset["season_display"].dropna().unique()}),
        "output_dataset": str(paths.normalized_dir / "hockey_training_dataset_latest.csv"),
    }
    write_json(paths.normalized_dir / "hockey_training_dataset_manifest_latest.json", manifest)
    logger.info("Hockey dataset built: %s", json.dumps(manifest, indent=2))
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
