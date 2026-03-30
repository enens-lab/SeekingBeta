"""Build the first MLB training dataset from historical schedules and game details."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from sports.mlb.client import MLBStatsClient, flatten_person, flatten_teams
from sports.mlb.feature_engineering import (
    add_matchup_differentials,
    build_starter_game_logs,
    build_team_game_logs,
    merge_starter_features,
    merge_team_features,
    prepare_games,
)
from sports.mlb.ingest_history import run_ingestion
from sports.mlb.roster_features import (
    build_batter_game_logs,
    build_bullpen_feature_frame,
    build_lineup_feature_frame,
    build_reliever_game_logs,
    extract_roster_tables,
    merge_bullpen_features,
    merge_lineup_features,
)
from sports.mlb.statcast_enrichment import enrich_dataset_with_statcast
from sports.pga.storage import build_ingestion_paths, read_preferred_table, write_csv, write_json, write_parquet

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MLB_DATA_ROOT = ROOT / "data" / "sports" / "mlb"
DEFAULT_STATCAST_ROOT = DEFAULT_MLB_DATA_ROOT / "statcast" / "normalized"

_LEAKY_GAME_COLUMNS = {
    "away_wins",
    "away_losses",
    "home_wins",
    "home_losses",
    "away_score",
    "home_score",
    "away_is_winner",
    "home_is_winner",
    "winner_team_id",
    "winner_team_name",
    "winning_pitcher_id",
    "losing_pitcher_id",
    "save_pitcher_id",
    "away_starter_summary",
    "away_starter_games_started",
    "away_starter_innings_pitched",
    "away_starter_outs_recorded",
    "away_starter_hits_allowed",
    "away_starter_runs_allowed",
    "away_starter_earned_runs",
    "away_starter_walks",
    "away_starter_strikeouts",
    "away_starter_home_runs_allowed",
    "away_starter_pitches_thrown",
    "away_starter_strikes",
    "away_starter_balls",
    "away_starter_batters_faced",
    "home_starter_summary",
    "home_starter_games_started",
    "home_starter_innings_pitched",
    "home_starter_outs_recorded",
    "home_starter_hits_allowed",
    "home_starter_runs_allowed",
    "home_starter_earned_runs",
    "home_starter_walks",
    "home_starter_strikeouts",
    "home_starter_home_runs_allowed",
    "home_starter_pitches_thrown",
    "home_starter_strikes",
    "home_starter_balls",
    "home_starter_batters_faced",
    "away_batting_hits",
    "away_batting_runs",
    "away_batting_home_runs",
    "away_batting_walks",
    "away_batting_strikeouts",
    "away_batting_total_bases",
    "away_batting_plate_appearances",
    "away_batting_left_on_base",
    "away_batting_stolen_bases",
    "away_batting_obp",
    "away_batting_slg",
    "away_batting_ops",
    "away_pitching_hits_allowed",
    "away_pitching_runs_allowed",
    "away_pitching_earned_runs",
    "away_pitching_walks",
    "away_pitching_strikeouts",
    "away_pitching_home_runs_allowed",
    "away_pitching_outs",
    "away_pitching_innings_pitched",
    "away_pitching_whip",
    "away_pitching_pitches_thrown",
    "away_pitching_strikes",
    "away_pitching_balls",
    "away_pitching_batters_faced",
    "home_batting_hits",
    "home_batting_runs",
    "home_batting_home_runs",
    "home_batting_walks",
    "home_batting_strikeouts",
    "home_batting_total_bases",
    "home_batting_plate_appearances",
    "home_batting_left_on_base",
    "home_batting_stolen_bases",
    "home_batting_obp",
    "home_batting_slg",
    "home_batting_ops",
    "home_pitching_hits_allowed",
    "home_pitching_runs_allowed",
    "home_pitching_earned_runs",
    "home_pitching_walks",
    "home_pitching_strikeouts",
    "home_pitching_home_runs_allowed",
    "home_pitching_outs",
    "home_pitching_innings_pitched",
    "home_pitching_whip",
    "home_pitching_pitches_thrown",
    "home_pitching_strikes",
    "home_pitching_balls",
    "home_pitching_batters_faced",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an MLB matchup training dataset.")
    parser.add_argument("--season", type=int, default=None, help="Single season to build.")
    parser.add_argument("--season-start", type=int, default=None, help="First season to include.")
    parser.add_argument("--season-end", type=int, default=None, help="Last season to include.")
    parser.add_argument("--refresh-source-data", action="store_true", help="Refresh MLB source data from the web.")
    parser.add_argument("--history-limit", type=int, default=None, help="Optional game detail limit per season.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers for MLB game-detail refresh.")
    parser.add_argument("--profile-workers", type=int, default=12, help="Parallel workers for pitcher profile refresh.")
    parser.add_argument("--refresh-pitcher-profiles", action="store_true", help="Refresh pitcher profiles from MLB Stats API.")
    parser.add_argument("--statcast-path", action="append", default=[], help="Optional local Statcast CSV/parquet path. Repeat to supply multiple files.")
    parser.add_argument(
        "--disable-statcast-cache",
        action="store_true",
        help="Do not auto-load cached Statcast daily features from the default MLB Statcast directory.",
    )
    parser.add_argument("--output-root", default=None, help="Override output root. Defaults to pythia_divination/data/sports/mlb.")
    parser.add_argument("--snapshot-tag", default=None, help="Optional explicit run tag for raw snapshot storage.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _resolve_seasons(args: argparse.Namespace) -> list[int]:
    if args.season:
        return [args.season]
    if args.season_start and args.season_end:
        return list(range(args.season_start, args.season_end + 1))
    current_year = pd.Timestamp.utcnow().year
    return [current_year - 1, current_year]


def _load_table_for_label(base_dir: Path, stem: str, label: str) -> pd.DataFrame:
    return read_preferred_table(
        base_dir / "normalized" / f"{stem}_{label}_latest.parquet",
        base_dir / "normalized" / f"{stem}_{label}_latest.csv",
    )


def _resolve_statcast_paths(args: argparse.Namespace) -> list[str]:
    explicit_paths = [str(Path(path)) for path in args.statcast_path if path]
    if args.disable_statcast_cache:
        return explicit_paths

    cached_paths: list[str] = []
    labeled_patterns = (
        "statcast_pitcher_daily_*_latest.parquet",
        "statcast_team_daily_*_latest.parquet",
        "statcast_pitcher_daily_*_latest.csv",
        "statcast_team_daily_*_latest.csv",
    )
    for pattern in labeled_patterns:
        for candidate in sorted(DEFAULT_STATCAST_ROOT.glob(pattern)):
            if candidate.name in {
                "statcast_pitcher_daily_latest.parquet",
                "statcast_team_daily_latest.parquet",
                "statcast_pitcher_daily_latest.csv",
                "statcast_team_daily_latest.csv",
            }:
                continue
            cached_paths.append(str(candidate))

    if not cached_paths:
        for candidate in (
            DEFAULT_STATCAST_ROOT / "statcast_pitcher_daily_latest.parquet",
            DEFAULT_STATCAST_ROOT / "statcast_team_daily_latest.parquet",
            DEFAULT_STATCAST_ROOT / "statcast_pitcher_daily_latest.csv",
            DEFAULT_STATCAST_ROOT / "statcast_team_daily_latest.csv",
        ):
            if candidate.exists():
                cached_paths.append(str(candidate))

    merged_paths = explicit_paths + [path for path in cached_paths if path not in explicit_paths]
    return merged_paths


def _refresh_history(seasons: list[int], args: argparse.Namespace, base_dir: Path) -> None:
    for season in seasons:
        logger.info("Refreshing MLB season history for %s", season)
        run_ingestion(
            SimpleNamespace(
                season=season,
                date=None,
                start_date=None,
                end_date=None,
                team_id=None,
                game_type="R",
                limit=args.history_limit,
                include_game_details=True,
                include_win_probability=False,
                max_workers=args.max_workers,
                output_root=str(base_dir),
                snapshot_tag=None,
                verbose=args.verbose,
            )
        )


def _load_history_tables(seasons: list[int], base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    for season in seasons:
        label = str(season)
        schedule_frames.append(_load_table_for_label(base_dir, "schedule", label))
        detail_frames.append(_load_table_for_label(base_dir, "game_details", label))
    schedule_df = pd.concat(schedule_frames, ignore_index=True).drop_duplicates(subset=["game_pk"])
    detail_df = pd.concat(detail_frames, ignore_index=True).drop_duplicates(subset=["game_pk"])
    return schedule_df, detail_df


def _load_team_meta(seasons: list[int], base_dir: Path) -> pd.DataFrame:
    client = MLBStatsClient()
    season = max(seasons)
    payload = client.get_teams(season=season)
    frame = flatten_teams(payload)
    write_csv(base_dir / "normalized" / "teams_latest.csv", frame)
    write_parquet(base_dir / "normalized" / "teams_latest.parquet", frame)
    return frame


def _load_pitcher_profiles(
    games: pd.DataFrame,
    *,
    base_dir: Path,
    refresh: bool,
    profile_workers: int,
) -> pd.DataFrame:
    target_path_csv = base_dir / "normalized" / "pitcher_profiles_latest.csv"
    target_path_parquet = base_dir / "normalized" / "pitcher_profiles_latest.parquet"
    existing = None
    if not refresh and (target_path_parquet.exists() or target_path_csv.exists()):
        existing = read_preferred_table(target_path_parquet, target_path_csv)

    pitcher_ids = pd.Series(
        pd.concat(
            [
                games["away_probable_pitcher_id"].dropna(),
                games["home_probable_pitcher_id"].dropna(),
            ],
            ignore_index=True,
        ).unique()
    ).dropna()
    pitcher_ids = [int(value) for value in pitcher_ids if pd.notna(value)]
    if not pitcher_ids:
        return pd.DataFrame()

    if existing is not None and set(existing.get("player_id", [])) >= set(pitcher_ids):
        if not target_path_parquet.exists():
            cached = existing.copy()
            for column in ("position_code", "position_name", "bat_side_code", "bat_side_description", "pitch_hand_code", "pitch_hand_description"):
                if column in cached.columns:
                    cached[column] = cached[column].astype("string")
            write_parquet(target_path_parquet, cached)
        return existing

    rows: list[pd.DataFrame] = []
    existing_ids = set(existing["player_id"].tolist()) if existing is not None and not existing.empty else set()
    missing_ids = [pitcher_id for pitcher_id in pitcher_ids if pitcher_id not in existing_ids]

    def _fetch_person_frame(pitcher_id: int) -> pd.DataFrame:
        payload = MLBStatsClient().get_person(pitcher_id)
        return flatten_person(payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(profile_workers))) as executor:
        future_map = {executor.submit(_fetch_person_frame, pitcher_id): pitcher_id for pitcher_id in missing_ids}
        for future in concurrent.futures.as_completed(future_map):
            pitcher_id = future_map[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # pragma: no cover - network variability
                logger.warning("Skipping pitcher profile %s: %s", pitcher_id, exc)
    new_profiles = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined = pd.concat([existing, new_profiles], ignore_index=True) if existing is not None else new_profiles
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["player_id"], keep="last").reset_index(drop=True)
        for column in ("position_code", "position_name", "bat_side_code", "bat_side_description", "pitch_hand_code", "pitch_hand_description"):
            if column in combined.columns:
                combined[column] = combined[column].astype("string")
        write_csv(target_path_csv, combined)
        write_parquet(target_path_parquet, combined)
    return combined


def _load_history_raw_dirs(seasons: list[int], base_dir: Path) -> list[Path]:
    raw_dirs: list[Path] = []
    for season in seasons:
        manifest_path = base_dir / "normalized" / f"history_manifest_{season}_latest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        raw_dir = Path(manifest.get("raw_dir", ""))
        if raw_dir.exists():
            raw_dirs.append(raw_dir)
    return raw_dirs


def _drop_leaky_current_game_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    removable = [column for column in dataset.columns if column in _LEAKY_GAME_COLUMNS]
    return dataset.drop(columns=removable, errors="ignore")


def build_training_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    seasons = _resolve_seasons(args)
    base_dir = Path(args.output_root) if args.output_root else DEFAULT_MLB_DATA_ROOT
    paths = build_ingestion_paths(base_dir=base_dir, snapshot_tag=args.snapshot_tag)

    if args.refresh_source_data:
        _refresh_history(seasons, args, base_dir)

    schedule_df, details_df = _load_history_tables(seasons, base_dir)
    team_meta_df = _load_team_meta(seasons, base_dir)

    merged_games = prepare_games(schedule_df, details_df, team_meta_df=team_meta_df)
    pitcher_profiles_df = _load_pitcher_profiles(
        merged_games,
        base_dir=base_dir,
        refresh=args.refresh_pitcher_profiles,
        profile_workers=args.profile_workers,
    )
    games = prepare_games(
        schedule_df,
        details_df,
        team_meta_df=team_meta_df,
        pitcher_profiles_df=pitcher_profiles_df,
    )

    team_logs = build_team_game_logs(games)
    starter_logs = build_starter_game_logs(games)
    raw_dirs = _load_history_raw_dirs(seasons, base_dir)
    lineup_roster, batter_game_rows, bullpen_roster, reliever_game_rows = extract_roster_tables(raw_dirs)
    batter_logs = build_batter_game_logs(batter_game_rows)
    reliever_logs = build_reliever_game_logs(reliever_game_rows)
    lineup_features = build_lineup_feature_frame(lineup_roster, batter_logs)
    bullpen_features = build_bullpen_feature_frame(bullpen_roster, reliever_logs)
    statcast_paths = _resolve_statcast_paths(args)

    dataset = merge_team_features(games, team_logs)
    dataset = merge_starter_features(dataset, starter_logs)
    dataset = merge_lineup_features(dataset, lineup_features)
    dataset = merge_bullpen_features(dataset, bullpen_features)
    dataset = add_matchup_differentials(dataset)
    dataset = enrich_dataset_with_statcast(dataset, statcast_paths)
    dataset = _drop_leaky_current_game_columns(dataset)
    dataset = dataset.sort_values(["official_date", "game_pk"]).reset_index(drop=True)

    write_csv(paths.normalized_dir / "team_game_logs_latest.csv", team_logs)
    write_parquet(paths.normalized_dir / "team_game_logs_latest.parquet", team_logs)
    write_csv(paths.normalized_dir / "starter_game_logs_latest.csv", starter_logs)
    write_parquet(paths.normalized_dir / "starter_game_logs_latest.parquet", starter_logs)
    write_csv(paths.normalized_dir / "lineup_roster_latest.csv", lineup_roster)
    write_parquet(paths.normalized_dir / "lineup_roster_latest.parquet", lineup_roster)
    write_csv(paths.normalized_dir / "batter_game_logs_latest.csv", batter_logs)
    write_parquet(paths.normalized_dir / "batter_game_logs_latest.parquet", batter_logs)
    write_csv(paths.normalized_dir / "bullpen_roster_latest.csv", bullpen_roster)
    write_parquet(paths.normalized_dir / "bullpen_roster_latest.parquet", bullpen_roster)
    write_csv(paths.normalized_dir / "reliever_game_logs_latest.csv", reliever_logs)
    write_parquet(paths.normalized_dir / "reliever_game_logs_latest.parquet", reliever_logs)
    write_csv(paths.normalized_dir / "lineup_features_latest.csv", lineup_features)
    write_parquet(paths.normalized_dir / "lineup_features_latest.parquet", lineup_features)
    write_csv(paths.normalized_dir / "bullpen_features_latest.csv", bullpen_features)
    write_parquet(paths.normalized_dir / "bullpen_features_latest.parquet", bullpen_features)
    write_csv(paths.normalized_dir / "mlb_training_dataset_latest.csv", dataset)
    write_parquet(paths.normalized_dir / "mlb_training_dataset_latest.parquet", dataset)

    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "seasons": seasons,
        "schedule_rows": int(len(schedule_df)),
        "detail_rows": int(len(details_df)),
        "games_rows": int(len(games)),
        "team_game_log_rows": int(len(team_logs)),
        "starter_game_log_rows": int(len(starter_logs)),
        "lineup_roster_rows": int(len(lineup_roster)),
        "batter_game_log_rows": int(len(batter_logs)),
        "bullpen_roster_rows": int(len(bullpen_roster)),
        "reliever_game_log_rows": int(len(reliever_logs)),
        "lineup_feature_rows": int(len(lineup_features)),
        "bullpen_feature_rows": int(len(bullpen_features)),
        "dataset_rows": int(len(dataset)),
        "dataset_columns": int(dataset.shape[1]),
        "statcast_paths": list(statcast_paths),
        "history_raw_dirs": [str(path) for path in raw_dirs],
        "raw_dir": str(paths.raw_dir),
        "normalized_dir": str(paths.normalized_dir),
    }
    write_json(paths.raw_dir / "training_dataset_manifest.json", manifest)
    write_json(paths.normalized_dir / "mlb_training_dataset_manifest_latest.json", manifest)
    return dataset, manifest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    dataset, manifest = build_training_dataset(args)
    print("=" * 72)
    print("MLB TRAINING DATASET COMPLETE")
    print("=" * 72)
    print(json.dumps(manifest, indent=2))
    print(f"Columns sample: {list(dataset.columns[:20])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
