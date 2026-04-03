"""CLI entrypoint for collecting basketball season history and archived boxscores."""

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

from sports.basketball.client import (  # noqa: E402
    BasketballGameBundle,
    BasketballStatsClient,
    flatten_game_bundle,
    flatten_player_boxscores,
    flatten_schedule,
    flatten_schedule_from_game_bundle,
)
from sports.basketball.constants import (  # noqa: E402
    DEFAULT_LEAGUES,
    LEAGUE_CONFIGS,
    archive_game_id_prefix,
    normalize_season_label,
    season_labels_for_year_range,
)
from sports.pga.storage import build_ingestion_paths, read_preferred_table, write_csv, write_json, write_parquet  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect basketball season history and boxscores.")
    parser.add_argument(
        "--league",
        default="all",
        choices=["all", *DEFAULT_LEAGUES],
        help="League to ingest. 'all' collects both Basketball leagues.",
    )
    parser.add_argument(
        "--season",
        default=None,
        help="Optional explicit season label. NBA accepts values like 2024-25 or 2024. WNBA accepts values like 2024.",
    )
    parser.add_argument("--season-start-year", type=int, default=None, help="Optional first season start year for archive backfill.")
    parser.add_argument("--season-end-year", type=int, default=None, help="Optional last season start year for archive backfill.")
    parser.add_argument("--limit", type=int, default=None, help="Limit completed games when fetching details.")
    parser.add_argument(
        "--include-game-details",
        action="store_true",
        help="Fetch boxscore-derived details for completed games. Archive backfills always fetch boxscores.",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers for completed-game detail fetches.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output root. Defaults to pythia_divination/data/sports/basketball.",
    )
    parser.add_argument("--snapshot-tag", default=None, help="Optional explicit run tag for raw snapshot storage.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _default_output_root() -> Path:
    return ROOT / "data" / "sports" / "basketball"


def _selected_leagues(arg: str) -> list[str]:
    return list(DEFAULT_LEAGUES) if arg == "all" else [arg]


def _season_labels_for_league(league: str, args: argparse.Namespace) -> list[str]:
    if args.season_start_year is not None or args.season_end_year is not None:
        start_year = args.season_start_year if args.season_start_year is not None else args.season_end_year
        end_year = args.season_end_year if args.season_end_year is not None else args.season_start_year
        if start_year is None or end_year is None:
            raise ValueError("Both season_start_year and season_end_year must resolve to integers.")
        return season_labels_for_year_range(league, start_year, end_year)
    if args.season:
        return [normalize_season_label(league, args.season)]
    return [LEAGUE_CONFIGS[league].current_season]


def _fetch_game_detail(league: str, game_id: str, *, season_display: str, optional: bool = False) -> dict[str, Any] | None:
    client = BasketballStatsClient()
    bundle = client.get_game_bundle_optional(league, game_id) if optional else client.get_game_bundle(league, game_id)
    if bundle is None:
        return None
    return {
        "league": league,
        "season_display": season_display,
        "game_id": game_id,
        "schedule_frame": flatten_schedule_from_game_bundle(bundle, season_display=season_display),
        "detail_frame": flatten_game_bundle(bundle),
        "player_frame": flatten_player_boxscores(bundle),
        "raw_payload": {"boxscore": bundle.boxscore},
    }


def _archive_game_ids(league: str, season_display: str, *, limit: int | None = None) -> list[str]:
    config = LEAGUE_CONFIGS[league]
    prefix = archive_game_id_prefix(league, season_display)
    upper_bound = limit if limit is not None else config.archive_max_game_number
    return [f"{prefix}{index:05d}" for index in range(1, int(upper_bound) + 1)]


def _load_existing_current_season_tables(base_dir: Path, league: str, season_display: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    try:
        schedule_df = read_preferred_table(
            base_dir / "normalized" / f"schedule_{league}_latest.parquet",
            base_dir / "normalized" / f"schedule_{league}_latest.csv",
        )
    except FileNotFoundError:
        return None

    def _optional_table(stem: str) -> pd.DataFrame:
        try:
            return read_preferred_table(
                base_dir / "normalized" / f"{stem}_{league}_latest.parquet",
                base_dir / "normalized" / f"{stem}_{league}_latest.csv",
            )
        except FileNotFoundError:
            return pd.DataFrame()

    details_df = _optional_table("game_details")
    players_df = _optional_table("player_boxscores")

    if "season_display" in schedule_df.columns:
        schedule_df = schedule_df.loc[schedule_df["season_display"].astype(str) == str(season_display)].copy()
    if "season_display" in details_df.columns:
        details_df = details_df.loc[details_df["season_display"].astype(str) == str(season_display)].copy()
    if "season_display" in players_df.columns:
        players_df = players_df.loc[players_df["season_display"].astype(str) == str(season_display)].copy()

    if schedule_df.empty:
        return None
    return schedule_df, details_df, players_df


def _load_archived_season_from_raw_cache(base_dir: Path, league: str, season_display: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    raw_root = base_dir / "raw"
    if not raw_root.exists():
        return None

    snapshot_files: dict[Path, list[Path]] = {}
    pattern = f"{league}_{season_display}_game_*.json"
    for candidate in raw_root.glob(f"*/{pattern}"):
        snapshot_files.setdefault(candidate.parent, []).append(candidate)

    if not snapshot_files:
        return None

    snapshot_dir, files = max(snapshot_files.items(), key=lambda item: len(item[1]))
    if not files:
        return None

    schedule_rows: list[pd.DataFrame] = []
    detail_rows: list[pd.DataFrame] = []
    player_rows: list[pd.DataFrame] = []
    for path in sorted(files):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        boxscore = payload.get("boxscore")
        if not isinstance(boxscore, dict):
            continue
        game_id = path.stem.split("_game_")[-1]
        bundle = BasketballGameBundle(league=league, game_id=game_id, boxscore=boxscore)
        schedule_rows.append(flatten_schedule_from_game_bundle(bundle, season_display=season_display))
        detail_rows.append(flatten_game_bundle(bundle))
        player_rows.append(flatten_player_boxscores(bundle))

    if not schedule_rows:
        return None

    schedule_df = pd.concat(schedule_rows, ignore_index=True).drop_duplicates(subset=["league", "game_id"]).sort_values(["official_date", "game_id"])
    details_df = pd.concat(detail_rows, ignore_index=True).drop_duplicates(subset=["league", "game_id"]).sort_values(["game_date_time_utc", "game_id"])
    players_df = (
        pd.concat(player_rows, ignore_index=True)
        .drop_duplicates(subset=["league", "game_id", "team_id", "player_id"])
        .sort_values(["game_date_time_utc", "game_id", "team_id", "player_id"])
    )

    manifest = {
        "season_display": season_display,
        "source": "raw_snapshot_cache",
        "snapshot_tag": snapshot_dir.name,
        "schedule_rows": int(len(schedule_df)),
        "regular_season_games": int(len(schedule_df)),
        "completed_regular_season_games": int(len(schedule_df)),
        "detail_row_count": int(len(details_df)),
        "player_boxscore_row_count": int(len(players_df)),
        "discovered_games": int(len(schedule_df)),
    }
    return schedule_df, details_df, players_df, manifest


def _ingest_live_current_season(
    client: BasketballStatsClient,
    league: str,
    season_display: str,
    args: argparse.Namespace,
    paths: Any,
    *,
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    try:
        payload = client.get_schedule(league, season=season_display)
        schedule_df = flatten_schedule(payload, league=league)
        details_df = pd.DataFrame()
        players_df = pd.DataFrame()

        write_json(paths.raw_dir / f"schedule_{league}_{season_display}.json", payload)
        completed = schedule_df.loc[(schedule_df["status_code"] == 3) & (schedule_df["is_regular_season"] == True)].copy()  # noqa: E712
        if args.limit is not None:
            completed = completed.head(max(0, args.limit))

        if args.include_game_details and not completed.empty:
            completed_games = completed["game_id"].astype(str).tolist()
            detail_rows: list[pd.DataFrame] = []
            player_rows: list[pd.DataFrame] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
                future_map = {
                    executor.submit(_fetch_game_detail, league, game_id, season_display=season_display, optional=False): game_id
                    for game_id in completed_games
                }
                for future in concurrent.futures.as_completed(future_map):
                    game_id = future_map[future]
                    result = future.result()
                    if result is None:
                        continue
                    logger.info("Fetched %s live boxscore for %s", league.upper(), result["game_id"])
                    detail_rows.append(result["detail_frame"])
                    player_rows.append(result["player_frame"])
                    write_json(paths.raw_dir / f"{league}_{season_display}_game_{result['game_id']}.json", result["raw_payload"])
            if detail_rows:
                details_df = pd.concat(detail_rows, ignore_index=True).drop_duplicates(subset=["league", "game_id"])
            if player_rows:
                players_df = (
                    pd.concat(player_rows, ignore_index=True)
                    .drop_duplicates(subset=["league", "game_id", "team_id", "player_id"])
                )

        manifest = {
            "season_display": season_display,
            "source": "live_schedule_feed",
            "schedule_rows": int(len(schedule_df)),
            "regular_season_games": int(schedule_df["is_regular_season"].sum()),
            "completed_regular_season_games": int(len(completed)),
            "detail_row_count": int(len(details_df)),
            "player_boxscore_row_count": int(len(players_df)),
        }
        return schedule_df, details_df, players_df, manifest
    except Exception as exc:
        cached = _load_existing_current_season_tables(base_dir, league, season_display)
        if cached is None:
            raise
        logger.warning(
            "Falling back to cached current-season %s tables for %s after live schedule fetch failed: %s",
            league.upper(),
            season_display,
            exc,
        )
        schedule_df, details_df, players_df = cached
        completed = schedule_df.loc[(schedule_df["status_code"] == 3) & (schedule_df["is_regular_season"] == True)].copy() if "status_code" in schedule_df.columns else schedule_df.copy()
        manifest = {
            "season_display": season_display,
            "source": "existing_local_cache",
            "schedule_rows": int(len(schedule_df)),
            "regular_season_games": int(schedule_df["is_regular_season"].sum()) if "is_regular_season" in schedule_df.columns else int(len(schedule_df)),
            "completed_regular_season_games": int(len(completed)),
            "detail_row_count": int(len(details_df)),
            "player_boxscore_row_count": int(len(players_df)),
        }
        return schedule_df, details_df, players_df, manifest


def _ingest_archived_season(
    league: str,
    season_display: str,
    args: argparse.Namespace,
    paths: Any,
    *,
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidate_game_ids = _archive_game_ids(league, season_display, limit=args.limit)
    schedule_rows: list[pd.DataFrame] = []
    detail_rows: list[pd.DataFrame] = []
    player_rows: list[pd.DataFrame] = []
    discovered_games = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        future_map = {
            executor.submit(_fetch_game_detail, league, game_id, season_display=season_display, optional=True): game_id
            for game_id in candidate_game_ids
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            if result is None:
                continue
            discovered_games += 1
            logger.info("Fetched %s archived boxscore for %s (%s)", league.upper(), result["game_id"], season_display)
            schedule_rows.append(result["schedule_frame"])
            detail_rows.append(result["detail_frame"])
            player_rows.append(result["player_frame"])
            write_json(paths.raw_dir / f"{league}_{season_display}_game_{result['game_id']}.json", result["raw_payload"])

    schedule_df = (
        pd.concat(schedule_rows, ignore_index=True).drop_duplicates(subset=["league", "game_id"]).sort_values(["official_date", "game_id"])
        if schedule_rows
        else pd.DataFrame()
    )
    details_df = (
        pd.concat(detail_rows, ignore_index=True).drop_duplicates(subset=["league", "game_id"]).sort_values(["game_date_time_utc", "game_id"])
        if detail_rows
        else pd.DataFrame()
    )
    players_df = (
        pd.concat(player_rows, ignore_index=True)
        .drop_duplicates(subset=["league", "game_id", "team_id", "player_id"])
        .sort_values(["game_date_time_utc", "game_id", "team_id", "player_id"])
        if player_rows
        else pd.DataFrame()
    )

    manifest = {
        "season_display": season_display,
        "source": "archived_boxscore_enumeration",
        "candidate_game_ids": int(len(candidate_game_ids)),
        "schedule_rows": int(len(schedule_df)),
        "regular_season_games": int(len(schedule_df)),
        "completed_regular_season_games": int(len(schedule_df)),
        "detail_row_count": int(len(details_df)),
        "player_boxscore_row_count": int(len(players_df)),
        "discovered_games": int(discovered_games),
    }
    cached = _load_archived_season_from_raw_cache(base_dir, league, season_display)
    if cached is not None and len(cached[0]) > len(schedule_df):
        logger.info(
            "Using cached raw snapshot for %s %s because it contains more games (%s > %s).",
            league.upper(),
            season_display,
            len(cached[0]),
            len(schedule_df),
        )
        return cached
    return schedule_df, details_df, players_df, manifest


def run_ingestion(args: argparse.Namespace) -> dict[str, object]:
    client = BasketballStatsClient()
    base_dir = Path(args.output_root) if args.output_root else _default_output_root()
    paths = build_ingestion_paths(base_dir=base_dir, snapshot_tag=args.snapshot_tag)

    league_manifests: dict[str, dict[str, Any]] = {}
    all_schedule_frames: list[pd.DataFrame] = []
    all_detail_frames: list[pd.DataFrame] = []
    all_player_frames: list[pd.DataFrame] = []

    for league in _selected_leagues(args.league):
        season_labels = _season_labels_for_league(league, args)
        league_schedule_frames: list[pd.DataFrame] = []
        league_detail_frames: list[pd.DataFrame] = []
        league_player_frames: list[pd.DataFrame] = []
        season_manifests: dict[str, dict[str, Any]] = {}

        for season_display in season_labels:
            is_current_live_season = season_display == LEAGUE_CONFIGS[league].current_season
            if is_current_live_season:
                schedule_df, details_df, players_df, season_manifest = _ingest_live_current_season(
                    client,
                    league,
                    season_display,
                    args,
                    paths,
                    base_dir=base_dir,
                )
            else:
                schedule_df, details_df, players_df, season_manifest = _ingest_archived_season(
                    league,
                    season_display,
                    args,
                    paths,
                    base_dir=base_dir,
                )

            if schedule_df.empty:
                logger.warning("No %s games found for season %s", league, season_display)
                continue

            league_schedule_frames.append(schedule_df)
            if not details_df.empty:
                league_detail_frames.append(details_df)
            if not players_df.empty:
                league_player_frames.append(players_df)
            season_manifests[season_display] = season_manifest

        if not league_schedule_frames:
            continue

        combined_schedule = (
            pd.concat(league_schedule_frames, ignore_index=True)
            .drop_duplicates(subset=["league", "game_id"])
            .sort_values(["official_date", "game_id"])
            .reset_index(drop=True)
        )
        combined_details = (
            pd.concat(league_detail_frames, ignore_index=True)
            .drop_duplicates(subset=["league", "game_id"])
            .sort_values(["game_date_time_utc", "game_id"])
            .reset_index(drop=True)
            if league_detail_frames
            else pd.DataFrame()
        )
        combined_players = (
            pd.concat(league_player_frames, ignore_index=True)
            .drop_duplicates(subset=["league", "game_id", "team_id", "player_id"])
            .sort_values(["game_date_time_utc", "game_id", "team_id", "player_id"])
            .reset_index(drop=True)
            if league_player_frames
            else pd.DataFrame()
        )

        write_csv(paths.normalized_dir / f"schedule_{league}_latest.csv", combined_schedule)
        write_parquet(paths.normalized_dir / f"schedule_{league}_latest.parquet", combined_schedule)
        if not combined_details.empty:
            write_csv(paths.normalized_dir / f"game_details_{league}_latest.csv", combined_details)
            write_parquet(paths.normalized_dir / f"game_details_{league}_latest.parquet", combined_details)
        if not combined_players.empty:
            write_csv(paths.normalized_dir / f"player_boxscores_{league}_latest.csv", combined_players)
            write_parquet(paths.normalized_dir / f"player_boxscores_{league}_latest.parquet", combined_players)

        manifest = {
            "league": league,
            "league_display_name": LEAGUE_CONFIGS[league].display_name,
            "seasons_included": list(season_manifests.keys()),
            "schedule_rows": int(len(combined_schedule)),
            "regular_season_games": int(combined_schedule["is_regular_season"].sum()),
            "completed_regular_season_games": int(len(combined_schedule.loc[combined_schedule["status_code"] == 3])),
            "detail_row_count": int(len(combined_details)),
            "player_boxscore_row_count": int(len(combined_players)),
            "raw_dir": str(paths.raw_dir),
            "normalized_dir": str(paths.normalized_dir),
            "season_manifests": season_manifests,
        }
        write_json(paths.normalized_dir / f"history_manifest_{league}_latest.json", manifest)
        league_manifests[league] = manifest

        all_schedule_frames.append(combined_schedule)
        if not combined_details.empty:
            all_detail_frames.append(combined_details)
        if not combined_players.empty:
            all_player_frames.append(combined_players)

    combined_schedule = (
        pd.concat(all_schedule_frames, ignore_index=True).drop_duplicates(subset=["league", "game_id"]).sort_values(["official_date", "game_id"])
        if all_schedule_frames
        else pd.DataFrame()
    )
    combined_details = (
        pd.concat(all_detail_frames, ignore_index=True).drop_duplicates(subset=["league", "game_id"]).sort_values(["game_date_time_utc", "game_id"])
        if all_detail_frames
        else pd.DataFrame()
    )
    combined_players = (
        pd.concat(all_player_frames, ignore_index=True)
        .drop_duplicates(subset=["league", "game_id", "team_id", "player_id"])
        .sort_values(["game_date_time_utc", "game_id", "team_id", "player_id"])
        if all_player_frames
        else pd.DataFrame()
    )

    if not combined_schedule.empty:
        write_csv(paths.normalized_dir / "schedule_latest.csv", combined_schedule)
        write_parquet(paths.normalized_dir / "schedule_latest.parquet", combined_schedule)
    if not combined_details.empty:
        write_csv(paths.normalized_dir / "game_details_latest.csv", combined_details)
        write_parquet(paths.normalized_dir / "game_details_latest.parquet", combined_details)
    if not combined_players.empty:
        write_csv(paths.normalized_dir / "player_boxscores_latest.csv", combined_players)
        write_parquet(paths.normalized_dir / "player_boxscores_latest.parquet", combined_players)

    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "leagues": league_manifests,
        "schedule_rows": int(len(combined_schedule)),
        "detail_row_count": int(len(combined_details)),
        "player_boxscore_row_count": int(len(combined_players)),
        "raw_dir": str(paths.raw_dir),
        "normalized_dir": str(paths.normalized_dir),
    }
    write_json(paths.raw_dir / "history_manifest.json", manifest)
    write_json(paths.normalized_dir / "history_manifest_latest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    manifest = run_ingestion(args)
    logger.info("Completed basketball ingestion for leagues: %s", ", ".join(manifest.get("leagues", {}).keys()))


if __name__ == "__main__":
    main()
