"""CLI entrypoint for collecting PGA Tour season schedule and tournament results."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.pga.client import (  # noqa: E402
    PGATourStatsClient,
    flatten_schedule,
    flatten_tournament_results,
)
from sports.pga.storage import build_ingestion_paths, write_csv, write_json, write_parquet  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect PGA Tour season schedule and tournament result labels.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year to ingest. Defaults to the currently rendered schedule season.",
    )
    parser.add_argument(
        "--tournament-id",
        action="append",
        default=[],
        help="Collect only the provided tournament ID(s). Repeat the flag for multiple tournaments.",
    )
    parser.add_argument(
        "--include-status",
        action="append",
        default=[],
        help="Include schedule rows matching the given status. Defaults to COMPLETED only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tournaments fetched after filtering.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output root. Defaults to pythia_divination/data/sports/pga.",
    )
    parser.add_argument(
        "--snapshot-tag",
        default=None,
        help="Optional explicit run tag for raw snapshot storage.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _select_tournaments(schedule_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = schedule_df.copy()
    if args.tournament_id:
        ids = {str(value) for value in args.tournament_id}
        selected = selected[selected["tournament_id"].astype(str).isin(ids)]
    else:
        statuses = {status.upper() for status in args.include_status} if args.include_status else {"COMPLETED"}
        selected = selected[selected["status"].astype(str).str.upper().isin(statuses)]

    if args.limit is not None:
        selected = selected.head(max(0, args.limit))
    return selected.reset_index(drop=True)


def _partition_result_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a combined result frame into all, individual, and team-event subsets."""
    if frame.empty:
        return frame.copy(), frame.copy(), frame.copy()

    partitioned = frame.copy()
    if "team_event" not in partitioned.columns:
        partitioned["team_event"] = False
    partitioned["team_event"] = partitioned["team_event"].fillna(False).astype(bool)

    individual = partitioned.loc[~partitioned["team_event"]].reset_index(drop=True)
    team = partitioned.loc[partitioned["team_event"]].reset_index(drop=True)
    return partitioned.reset_index(drop=True), individual, team


def run_history_ingestion(args: argparse.Namespace) -> dict[str, object]:
    client = PGATourStatsClient()
    paths = build_ingestion_paths(
        base_dir=Path(args.output_root) if args.output_root else None,
        snapshot_tag=args.snapshot_tag,
    )

    schedule_payload = client.get_schedule(args.season)
    season = schedule_payload.get("season")
    schedule_df = flatten_schedule(schedule_payload)
    selected_tournaments = _select_tournaments(schedule_df, args)
    if selected_tournaments.empty:
        raise ValueError("No tournaments matched the provided season/filter arguments")

    write_json(paths.raw_dir / f"schedule_{season}.json", schedule_payload)
    write_csv(paths.normalized_dir / f"schedule_{season}_latest.csv", schedule_df)
    write_parquet(paths.normalized_dir / f"schedule_{season}_latest.parquet", schedule_df)

    tournament_frames: list[pd.DataFrame] = []
    tournament_manifest: list[dict[str, object]] = []
    excluded_tournaments: list[dict[str, object]] = []
    tournament_errors: list[dict[str, object]] = []
    for tournament in selected_tournaments.to_dict(orient="records"):
        logger.info(
            "Fetching tournament leaderboard for %s (%s)",
            tournament["tournament_name"],
            tournament["tournament_id"],
        )
        try:
            payload = client.get_tournament_leaderboard(
                season=tournament["season"],
                tournament_id=tournament["tournament_id"],
                tournament_name=tournament["tournament_name"],
            )
            leaderboard = payload["leaderboard"]
            tournament_meta = payload["tournament"]
            frame = flatten_tournament_results(leaderboard, tournament_meta)
            tournament_frames.append(frame)

            winner = None
            if not frame.empty and frame["won"].any():
                winner = frame.loc[frame["won"], "player_name"].iloc[0]

            tournament_manifest.append(
                {
                    "tournament_id": tournament["tournament_id"],
                    "tournament_name": tournament["tournament_name"],
                    "status": tournament["status"],
                    "row_count": int(len(frame)),
                    "leaderboard_path": payload["path"],
                    "leaderboard_query_name": payload.get("leaderboard_query_name"),
                    "winner": winner,
                }
            )

            slug = str(tournament["tournament_id"]).lower()
            write_json(
                paths.raw_dir / f"leaderboard_{slug}.json",
                {"path": payload["path"], "tournament": tournament_meta, "leaderboard": leaderboard},
            )
            write_csv(paths.normalized_dir / f"leaderboard_{slug}_latest.csv", frame)
            write_parquet(paths.normalized_dir / f"leaderboard_{slug}_latest.parquet", frame)
        except Exception as exc:
            if "Unsupported cup/match-play leaderboard format" in str(exc):
                logger.info(
                    "Excluding tournament %s (%s): %s",
                    tournament["tournament_name"],
                    tournament["tournament_id"],
                    exc,
                )
                excluded_tournaments.append(
                    {
                        "tournament_id": tournament["tournament_id"],
                        "tournament_name": tournament["tournament_name"],
                        "status": tournament["status"],
                        "reason": str(exc),
                    }
                )
                continue
            logger.warning(
                "Skipping tournament %s (%s): %s",
                tournament["tournament_name"],
                tournament["tournament_id"],
                exc,
            )
            tournament_errors.append(
                {
                    "tournament_id": tournament["tournament_id"],
                    "tournament_name": tournament["tournament_name"],
                    "status": tournament["status"],
                    "error": str(exc),
                }
            )

    non_empty_frames = [frame for frame in tournament_frames if frame is not None and not frame.empty]
    combined = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame()
    combined_all, combined_individual, combined_team = _partition_result_frames(combined)

    write_csv(paths.normalized_dir / f"season_tournament_results_{season}_latest.csv", combined_all)
    write_csv(paths.normalized_dir / f"season_tournament_labels_{season}_latest.csv", combined_all)
    write_parquet(paths.normalized_dir / f"season_tournament_results_{season}_latest.parquet", combined_all)
    write_parquet(paths.normalized_dir / f"season_tournament_labels_{season}_latest.parquet", combined_all)

    write_csv(paths.normalized_dir / f"season_tournament_results_individual_{season}_latest.csv", combined_individual)
    write_csv(paths.normalized_dir / f"season_tournament_labels_individual_{season}_latest.csv", combined_individual)
    write_parquet(paths.normalized_dir / f"season_tournament_results_individual_{season}_latest.parquet", combined_individual)
    write_parquet(paths.normalized_dir / f"season_tournament_labels_individual_{season}_latest.parquet", combined_individual)

    write_csv(paths.normalized_dir / f"season_tournament_results_team_{season}_latest.csv", combined_team)
    write_csv(paths.normalized_dir / f"season_tournament_labels_team_{season}_latest.csv", combined_team)
    write_parquet(paths.normalized_dir / f"season_tournament_results_team_{season}_latest.parquet", combined_team)
    write_parquet(paths.normalized_dir / f"season_tournament_labels_team_{season}_latest.parquet", combined_team)

    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "season": season,
        "selected_tournaments": int(len(selected_tournaments)),
        "successful_tournaments": int(len(tournament_manifest)),
        "excluded_tournaments_count": int(len(excluded_tournaments)),
        "combined_row_count": int(len(combined_all)),
        "individual_row_count": int(len(combined_individual)),
        "team_row_count": int(len(combined_team)),
        "individual_tournament_count": int(combined_individual["tournament_id"].nunique()) if not combined_individual.empty else 0,
        "team_tournament_count": int(combined_team["tournament_id"].nunique()) if not combined_team.empty else 0,
        "tournament_manifest": tournament_manifest,
        "excluded_tournaments": excluded_tournaments,
        "tournament_errors": tournament_errors,
        "raw_dir": str(paths.raw_dir),
        "normalized_dir": str(paths.normalized_dir),
    }
    write_json(paths.raw_dir / f"history_manifest_{season}.json", manifest)
    write_json(paths.normalized_dir / f"history_manifest_{season}_latest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    manifest = run_history_ingestion(args)
    print("=" * 72)
    print("PGA HISTORY INGESTION COMPLETE")
    print("=" * 72)
    print(f"Season:               {manifest['season']}")
    print(f"Tournaments fetched:  {manifest['selected_tournaments']}")
    print(f"Excluded tournaments: {manifest['excluded_tournaments_count']}")
    print(f"Combined result rows: {manifest['combined_row_count']}")
    print(f"Raw snapshots:        {manifest['raw_dir']}")
    print(f"Normalized output:    {manifest['normalized_dir']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
