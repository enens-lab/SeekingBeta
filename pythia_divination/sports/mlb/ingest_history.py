"""CLI entrypoint for collecting MLB schedule history and game-level summaries."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.mlb.client import (  # noqa: E402
    MLBStatsClient,
    flatten_context_metrics,
    flatten_game_bundle,
    flatten_schedule,
    flatten_win_probability,
)
from sports.mlb.constants import DEFAULT_REGULAR_SEASON_GAME_TYPES  # noqa: E402
from sports.pga.storage import build_ingestion_paths, write_csv, write_json, write_parquet  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect MLB schedule history and game summaries.")
    parser.add_argument("--season", type=int, default=None, help="Season year to ingest.")
    parser.add_argument("--date", default=None, help="Single ISO date to ingest.")
    parser.add_argument("--start-date", default=None, help="ISO start date for schedule ingestion.")
    parser.add_argument("--end-date", default=None, help="ISO end date for schedule ingestion.")
    parser.add_argument("--team-id", type=int, default=None, help="Filter schedule by one team.")
    parser.add_argument(
        "--game-type",
        default=",".join(DEFAULT_REGULAR_SEASON_GAME_TYPES),
        help="Comma-separated MLB game type codes. Defaults to regular season only ('R').",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit completed games when fetching details.")
    parser.add_argument(
        "--include-game-details",
        action="store_true",
        help="Fetch live-feed derived details for completed games.",
    )
    parser.add_argument(
        "--include-win-probability",
        action="store_true",
        help="Fetch per-event win probability history for completed games.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel workers for completed-game detail fetches.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output root. Defaults to pythia_divination/data/sports/mlb.",
    )
    parser.add_argument("--snapshot-tag", default=None, help="Optional explicit run tag for raw snapshot storage.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _default_output_root() -> Path:
    return ROOT / "data" / "sports" / "mlb"


def _resolve_schedule_window(args: argparse.Namespace) -> dict[str, object]:
    if args.date:
        return {"game_date": args.date}
    if args.start_date or args.end_date:
        return {"start_date": args.start_date, "end_date": args.end_date}
    if args.season:
        return {"start_date": f"{args.season}-03-01", "end_date": f"{args.season}-11-30", "season": args.season}
    today = date.today().isoformat()
    return {"game_date": today}


def _dataset_label(args: argparse.Namespace, schedule_kwargs: dict[str, object]) -> str:
    if args.season:
        return str(args.season)
    if args.date:
        return str(args.date)
    start_date = schedule_kwargs.get("start_date")
    end_date = schedule_kwargs.get("end_date")
    if start_date and end_date:
        return f"{start_date}_to_{end_date}"
    if start_date:
        return str(start_date)
    return "latest"


def _fetch_game_detail(
    game: dict[str, Any],
    *,
    include_win_probability: bool,
) -> dict[str, Any]:
    game_pk = int(game["game_pk"])
    client = MLBStatsClient()
    bundle = client.get_game_bundle(
        game_pk,
        include_boxscore=True,
        include_play_by_play=False,
        include_context_metrics=True,
        include_win_probability=include_win_probability,
    )
    return {
        "game_pk": game_pk,
        "game": game,
        "bundle": bundle,
        "detail_frame": flatten_game_bundle(bundle),
        "context_frame": flatten_context_metrics(game_pk, bundle.context_metrics) if bundle.context_metrics else pd.DataFrame(),
        "win_prob_frame": flatten_win_probability(game_pk, bundle.win_probability) if bundle.win_probability else pd.DataFrame(),
        "raw_payload": {
            "live_feed": bundle.live_feed,
            "boxscore": bundle.boxscore,
            "context_metrics": bundle.context_metrics,
            "win_probability": bundle.win_probability,
        },
    }


def run_ingestion(args: argparse.Namespace) -> dict[str, object]:
    client = MLBStatsClient()
    base_dir = Path(args.output_root) if args.output_root else _default_output_root()
    paths = build_ingestion_paths(base_dir=base_dir, snapshot_tag=args.snapshot_tag)

    schedule_kwargs = _resolve_schedule_window(args)
    dataset_label = _dataset_label(args, schedule_kwargs)
    schedule_payload = client.get_schedule(
        team_id=args.team_id,
        game_type=args.game_type,
        hydrate="team,linescore",
        **schedule_kwargs,
    )
    schedule_df = flatten_schedule(schedule_payload)
    if schedule_df.empty:
        raise ValueError("No MLB games matched the provided schedule filters.")

    write_json(paths.raw_dir / "schedule.json", schedule_payload)
    write_csv(paths.normalized_dir / "schedule_latest.csv", schedule_df)
    write_parquet(paths.normalized_dir / "schedule_latest.parquet", schedule_df)
    write_csv(paths.normalized_dir / f"schedule_{dataset_label}_latest.csv", schedule_df)
    write_parquet(paths.normalized_dir / f"schedule_{dataset_label}_latest.parquet", schedule_df)

    completed = schedule_df.loc[schedule_df["status_code"] == "F"].copy()
    if args.limit is not None:
        completed = completed.head(max(0, args.limit))

    detail_rows: list[pd.DataFrame] = []
    context_rows: list[pd.DataFrame] = []
    win_prob_rows: list[pd.DataFrame] = []
    detail_manifest: list[dict[str, object]] = []

    if args.include_game_details:
        completed_games = completed.to_dict(orient="records")
        max_workers = max(1, int(args.max_workers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _fetch_game_detail,
                    game,
                    include_win_probability=args.include_win_probability,
                ): game
                for game in completed_games
            }
            for future in concurrent.futures.as_completed(future_map):
                game = future_map[future]
                game_pk = int(game["game_pk"])
                logger.info("Fetched MLB game details for %s (%s)", game.get("official_date"), game_pk)
                result = future.result()
                detail_rows.append(result["detail_frame"])
                if not result["context_frame"].empty:
                    context_rows.append(result["context_frame"])
                if not result["win_prob_frame"].empty:
                    win_prob_rows.append(result["win_prob_frame"])
                write_json(paths.raw_dir / f"game_{game_pk}.json", result["raw_payload"])
                detail_manifest.append(
                    {
                        "game_pk": game_pk,
                        "official_date": game.get("official_date"),
                        "away_team_name": game.get("away_team_name"),
                        "home_team_name": game.get("home_team_name"),
                        "included_win_probability": bool(args.include_win_probability),
                    }
                )

    if detail_rows:
        details_df = pd.concat(detail_rows, ignore_index=True)
        write_csv(paths.normalized_dir / "game_details_latest.csv", details_df)
        write_parquet(paths.normalized_dir / "game_details_latest.parquet", details_df)
        write_csv(paths.normalized_dir / f"game_details_{dataset_label}_latest.csv", details_df)
        write_parquet(paths.normalized_dir / f"game_details_{dataset_label}_latest.parquet", details_df)
    else:
        details_df = pd.DataFrame()

    if context_rows:
        context_df = pd.concat(context_rows, ignore_index=True)
        write_csv(paths.normalized_dir / "game_context_metrics_latest.csv", context_df)
        write_parquet(paths.normalized_dir / "game_context_metrics_latest.parquet", context_df)
        write_csv(paths.normalized_dir / f"game_context_metrics_{dataset_label}_latest.csv", context_df)
        write_parquet(paths.normalized_dir / f"game_context_metrics_{dataset_label}_latest.parquet", context_df)
    else:
        context_df = pd.DataFrame()

    if win_prob_rows:
        win_prob_df = pd.concat(win_prob_rows, ignore_index=True)
        write_csv(paths.normalized_dir / "game_win_probability_latest.csv", win_prob_df)
        write_parquet(paths.normalized_dir / "game_win_probability_latest.parquet", win_prob_df)
        write_csv(paths.normalized_dir / f"game_win_probability_{dataset_label}_latest.csv", win_prob_df)
        write_parquet(paths.normalized_dir / f"game_win_probability_{dataset_label}_latest.parquet", win_prob_df)
    else:
        win_prob_df = pd.DataFrame()

    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "dataset_label": dataset_label,
        "schedule_rows": int(len(schedule_df)),
        "completed_games": int(len(completed)),
        "detailed_games": int(len(detail_manifest)),
        "detail_row_count": int(len(details_df)),
        "context_metric_rows": int(len(context_df)),
        "win_probability_rows": int(len(win_prob_df)),
        "schedule_filters": schedule_kwargs,
        "team_id": args.team_id,
        "game_type": args.game_type,
        "raw_dir": str(paths.raw_dir),
        "normalized_dir": str(paths.normalized_dir),
    }
    write_json(paths.raw_dir / "history_manifest.json", manifest)
    write_json(paths.normalized_dir / "history_manifest_latest.json", manifest)
    write_json(paths.normalized_dir / f"history_manifest_{dataset_label}_latest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    manifest = run_ingestion(args)
    print("=" * 72)
    print("MLB HISTORY INGESTION COMPLETE")
    print("=" * 72)
    print(json.dumps(manifest, indent=2))
    print("=" * 72)


if __name__ == "__main__":
    main()
