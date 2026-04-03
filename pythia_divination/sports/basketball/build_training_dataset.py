"""Build the first basketball training dataset from current-season NBA/WNBA schedules and boxscores."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from sports.basketball.feature_engineering import (
    add_matchup_differentials,
    attach_pregame_team_features,
    build_team_game_logs,
    prepare_games,
)
from sports.basketball.ingest_history import run_ingestion
from sports.basketball.constants import LEAGUE_CONFIGS
from sports.basketball.rotation_features import (
    attach_expected_rotation_features,
    attach_pregame_rotation_features,
    build_expected_rotation_game_logs,
    build_player_game_logs,
    build_rotation_game_logs,
)
from sports.pga.storage import build_ingestion_paths, read_preferred_table, write_csv, write_json, write_parquet

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASKETBALL_DATA_ROOT = ROOT / "data" / "sports" / "basketball"

_LEAKY_COLUMNS = {
    "away_team_wins",
    "away_team_losses",
    "home_team_wins",
    "home_team_losses",
    "away_score",
    "home_score",
    "away_is_winner",
    "home_is_winner",
    "winner_team_id",
    "status_code_detail",
    "status_text_detail",
    "attendance",
    "away_points",
    "away_assists",
    "away_rebounds_total",
    "away_rebounds_offensive",
    "away_rebounds_defensive",
    "away_turnovers",
    "away_steals",
    "away_blocks",
    "away_fouls_personal",
    "away_field_goals_made",
    "away_field_goals_attempted",
    "away_field_goal_pct",
    "away_three_pointers_made",
    "away_three_pointers_attempted",
    "away_three_point_pct",
    "away_free_throws_made",
    "away_free_throws_attempted",
    "away_free_throw_pct",
    "away_bench_points",
    "away_fast_break_points",
    "away_paint_points",
    "away_second_chance_points",
    "away_true_shooting_pct",
    "away_effective_fg_pct",
    "away_time_leading",
    "away_largest_lead",
    "away_points_off_turnovers",
    "home_points",
    "home_assists",
    "home_rebounds_total",
    "home_rebounds_offensive",
    "home_rebounds_defensive",
    "home_turnovers",
    "home_steals",
    "home_blocks",
    "home_fouls_personal",
    "home_field_goals_made",
    "home_field_goals_attempted",
    "home_field_goal_pct",
    "home_three_pointers_made",
    "home_three_pointers_attempted",
    "home_three_point_pct",
    "home_free_throws_made",
    "home_free_throws_attempted",
    "home_free_throw_pct",
    "home_bench_points",
    "home_fast_break_points",
    "home_paint_points",
    "home_second_chance_points",
    "home_true_shooting_pct",
    "home_effective_fg_pct",
    "home_time_leading",
    "home_largest_lead",
    "home_points_off_turnovers",
}


def _to_datetime_mixed(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a basketball matchup training dataset.")
    parser.add_argument(
        "--league",
        default="all",
        choices=["all", "nba", "wnba"],
        help="League to build. 'all' concatenates NBA and WNBA current-season data.",
    )
    parser.add_argument("--refresh-source-data", action="store_true", help="Refresh basketball source data from the web.")
    parser.add_argument("--history-limit", type=int, default=None, help="Optional completed-game detail limit per league.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers for boxscore refresh.")
    parser.add_argument("--output-root", default=None, help="Override output root. Defaults to pythia_divination/data/sports/basketball.")
    parser.add_argument("--snapshot-tag", default=None, help="Optional explicit run tag for raw snapshot storage.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _load_table_for_label(base_dir: Path, stem: str, label: str) -> pd.DataFrame:
    frame = read_preferred_table(
        base_dir / "normalized" / f"{stem}_{label}_latest.parquet",
        base_dir / "normalized" / f"{stem}_{label}_latest.csv",
    )
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str)
    if "official_date" in frame.columns:
        frame["official_date"] = _to_datetime_mixed(frame["official_date"])
    if "season_display" in frame.columns:
        frame["season_display"] = frame["season_display"].astype(str)
    return frame


def _selected_leagues(arg: str) -> list[str]:
    return ["nba", "wnba"] if arg == "all" else [arg]


def _refresh_history(leagues: list[str], args: argparse.Namespace, base_dir: Path) -> None:
    run_ingestion(
        SimpleNamespace(
            league="all" if len(leagues) > 1 else leagues[0],
            season=None,
            limit=args.history_limit,
            include_game_details=True,
            max_workers=args.max_workers,
            output_root=str(base_dir),
            snapshot_tag=args.snapshot_tag,
            verbose=args.verbose,
        )
    )


def _load_history_tables(leagues: list[str], base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    for league in leagues:
        schedule_frames.append(_load_table_for_label(base_dir, "schedule", league))
        try:
            detail_frames.append(_load_table_for_label(base_dir, "game_details", league))
        except FileNotFoundError:
            logger.warning("No game detail table found yet for %s", league)
    schedule_df = pd.concat(schedule_frames, ignore_index=True).drop_duplicates(subset=["league", "game_id"])
    detail_df = pd.concat(detail_frames, ignore_index=True).drop_duplicates(subset=["league", "game_id"]) if detail_frames else pd.DataFrame()
    return schedule_df, detail_df


def _load_player_boxscores(leagues: list[str], base_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for league in leagues:
        try:
            frames.append(_load_table_for_label(base_dir, "player_boxscores", league))
        except FileNotFoundError:
            logger.warning("No player boxscore table found yet for %s", league)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["league", "game_id", "team_id", "player_id"])


def build_training_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    base_dir = Path(args.output_root) if args.output_root else DEFAULT_BASKETBALL_DATA_ROOT
    paths = build_ingestion_paths(base_dir=base_dir, snapshot_tag=args.snapshot_tag)
    leagues = _selected_leagues(args.league)

    if args.refresh_source_data:
        _refresh_history(leagues, args, base_dir)

    schedule_df, details_df = _load_history_tables(leagues, base_dir)
    player_boxscores_df = _load_player_boxscores(leagues, base_dir)
    if details_df.empty:
        raise ValueError("No basketball game details are available yet. Run with --refresh-source-data once regular-season games have completed.")
    if player_boxscores_df.empty:
        raise ValueError("No basketball player boxscores are available yet. Run with --refresh-source-data once regular-season games have completed.")

    games = prepare_games(schedule_df, details_df, require_completed=True)
    if games.empty:
        raise ValueError("No completed regular-season basketball games were available for dataset building.")

    team_logs = build_team_game_logs(games)
    player_logs = build_player_game_logs(player_boxscores_df)
    rotation_logs = build_rotation_game_logs(player_boxscores_df)
    expected_rotation_logs = build_expected_rotation_game_logs(games, player_logs)
    dataset = attach_pregame_team_features(games, team_logs)
    dataset = attach_pregame_rotation_features(dataset, rotation_logs)
    dataset = attach_expected_rotation_features(dataset, expected_rotation_logs)
    dataset = add_matchup_differentials(dataset)
    dataset = dataset.drop(columns=[column for column in _LEAKY_COLUMNS if column in dataset.columns])
    dataset = dataset.sort_values(["official_date", "game_id"]).reset_index(drop=True)

    label = "basketball" if args.league == "all" else f"basketball_{args.league}"
    write_csv(paths.normalized_dir / f"{label}_training_dataset_latest.csv", dataset)
    write_parquet(paths.normalized_dir / f"{label}_training_dataset_latest.parquet", dataset)
    write_csv(paths.normalized_dir / f"team_game_logs_{args.league}_latest.csv", team_logs)
    write_parquet(paths.normalized_dir / f"team_game_logs_{args.league}_latest.parquet", team_logs)
    write_csv(paths.normalized_dir / f"player_game_logs_{args.league}_latest.csv", player_logs)
    write_parquet(paths.normalized_dir / f"player_game_logs_{args.league}_latest.parquet", player_logs)
    write_csv(paths.normalized_dir / f"rotation_game_logs_{args.league}_latest.csv", rotation_logs)
    write_parquet(paths.normalized_dir / f"rotation_game_logs_{args.league}_latest.parquet", rotation_logs)
    write_csv(paths.normalized_dir / f"expected_rotation_game_logs_{args.league}_latest.csv", expected_rotation_logs)
    write_parquet(paths.normalized_dir / f"expected_rotation_game_logs_{args.league}_latest.parquet", expected_rotation_logs)

    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "league": args.league,
        "leagues_included": leagues,
        "schedule_rows": int(len(schedule_df)),
        "detail_rows": int(len(details_df)),
        "player_boxscore_rows": int(len(player_boxscores_df)),
        "games_rows": int(len(games)),
        "team_game_log_rows": int(len(team_logs)),
        "player_game_log_rows": int(len(player_logs)),
        "rotation_game_log_rows": int(len(rotation_logs)),
        "expected_rotation_game_log_rows": int(len(expected_rotation_logs)),
        "dataset_rows": int(len(dataset)),
        "dataset_columns": int(len(dataset.columns)),
        "season_display_values": sorted({str(value) for value in dataset["season_display"].dropna().unique()}),
        "output_dataset": str(paths.normalized_dir / f"{label}_training_dataset_latest.csv"),
        "current_season_only": len({str(value) for value in dataset["season_display"].dropna().unique()}) <= 1
        and all(str(value) in {LEAGUE_CONFIGS[league].current_season for league in leagues} for value in dataset["season_display"].dropna().unique()),
    }
    write_json(paths.normalized_dir / f"{label}_training_dataset_manifest_latest.json", manifest)
    return dataset, manifest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _, manifest = build_training_dataset(args)
    logger.info("Basketball dataset built: %s", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
