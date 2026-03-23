"""CLI entrypoint for collecting PGA Tour stat data for model feature snapshots."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.pga.client import (  # noqa: E402
    PGATourStatsClient,
    build_player_feature_snapshot,
    flatten_current_leaders,
    flatten_stat_catalog,
    flatten_stat_detail,
)
from sports.pga.constants import DEFAULT_TRACKED_STATS, DEFAULT_TOUR_CODE, TrackedPGAStat  # noqa: E402
from sports.pga.storage import build_ingestion_paths, write_csv, write_json, write_parquet  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect PGA Tour stats and store normalized feature snapshots.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Season year to request from PGA Tour when supported.",
    )
    parser.add_argument(
        "--tour-code",
        default=DEFAULT_TOUR_CODE,
        help="PGA Tour code to request. Default is 'R' for PGA Tour.",
    )
    parser.add_argument(
        "--stat-id",
        action="append",
        default=[],
        help="Collect only the provided stat ID(s). Repeat the flag for multiple stats.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tracked stats collected (useful for smoke tests).",
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


def _select_tracked_stats(stat_ids: list[str], limit: int | None) -> tuple[TrackedPGAStat, ...]:
    if stat_ids:
        requested = []
        known = {stat.stat_id: stat for stat in DEFAULT_TRACKED_STATS}
        for stat_id in stat_ids:
            stat_id = str(stat_id)
            requested.append(
                known.get(stat_id, TrackedPGAStat(stat_id, f"stat_{stat_id}", f"PGA stat {stat_id}"))
            )
        tracked = tuple(requested)
    else:
        tracked = DEFAULT_TRACKED_STATS

    if limit is not None:
        tracked = tracked[: max(0, limit)]
    return tracked


def run_ingestion(args: argparse.Namespace) -> dict[str, object]:
    tracked_stats = _select_tracked_stats(args.stat_id, args.limit)
    if not tracked_stats:
        raise ValueError("No PGA stats selected for ingestion")

    client = PGATourStatsClient()
    paths = build_ingestion_paths(
        base_dir=Path(args.output_root) if args.output_root else None,
        snapshot_tag=args.snapshot_tag,
    )

    bundle = client.get_stats_index()
    stat_catalog = flatten_stat_catalog(bundle.get("statOverview") or {})
    current_leaders = flatten_current_leaders(bundle.get("currentLeaders") or {})

    write_json(paths.raw_dir / "stats_index.json", bundle)
    write_csv(paths.normalized_dir / "stat_catalog_latest.csv", stat_catalog)
    write_parquet(paths.normalized_dir / "stat_catalog_latest.parquet", stat_catalog)
    if not current_leaders.empty:
        write_csv(paths.normalized_dir / "current_leaders_latest.csv", current_leaders)
        write_parquet(paths.normalized_dir / "current_leaders_latest.parquet", current_leaders)

    detail_frames = []
    detail_manifest = []
    detail_errors = []
    for tracked_stat in tracked_stats:
        try:
            payload = client.get_stat_detail(
                tracked_stat.stat_id,
                year=args.year,
                tour_code=args.tour_code,
            )
            frame = flatten_stat_detail(payload, tracked_stat=tracked_stat)
            detail_frames.append(frame)
            detail_manifest.append(
                {
                    "stat_id": tracked_stat.stat_id,
                    "slug": tracked_stat.slug,
                    "title": tracked_stat.title,
                    "row_count": int(len(frame)),
                    "output_csv": str(paths.normalized_dir / f"stat_detail_{tracked_stat.slug}_latest.csv"),
                }
            )
            write_json(paths.raw_dir / f"stat_{tracked_stat.slug}_{tracked_stat.stat_id}.json", payload)
            write_csv(paths.normalized_dir / f"stat_detail_{tracked_stat.slug}_latest.csv", frame)
            write_parquet(paths.normalized_dir / f"stat_detail_{tracked_stat.slug}_latest.parquet", frame)
        except Exception as exc:
            logger.warning(
                "Skipping PGA stat %s (%s): %s",
                tracked_stat.stat_id,
                tracked_stat.title,
                exc,
            )
            detail_errors.append(
                {
                    "stat_id": tracked_stat.stat_id,
                    "slug": tracked_stat.slug,
                    "title": tracked_stat.title,
                    "error": str(exc),
                }
            )

    feature_snapshot = build_player_feature_snapshot(detail_frames)
    if feature_snapshot.empty:
        raise RuntimeError("No PGA stat detail frames were successfully collected.")
    write_csv(paths.normalized_dir / "player_feature_snapshot_latest.csv", feature_snapshot)
    write_parquet(paths.normalized_dir / "player_feature_snapshot_latest.parquet", feature_snapshot)
    if args.year is not None:
        write_csv(paths.normalized_dir / f"player_feature_snapshot_{args.year}_latest.csv", feature_snapshot)
        write_parquet(paths.normalized_dir / f"player_feature_snapshot_{args.year}_latest.parquet", feature_snapshot)

    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "tour_code": args.tour_code,
        "year": args.year,
        "tracked_stats": detail_manifest,
        "tracked_stat_errors": detail_errors,
        "current_tournament_name": (bundle.get("tournament") or {}).get("tournamentName"),
        "current_tournament_id": (bundle.get("tournament") or {}).get("id"),
        "feature_row_count": int(len(feature_snapshot)),
        "feature_column_count": int(len(feature_snapshot.columns)),
        "raw_dir": str(paths.raw_dir),
        "normalized_dir": str(paths.normalized_dir),
    }
    write_json(paths.raw_dir / "manifest.json", manifest)
    write_json(paths.normalized_dir / "manifest_latest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    manifest = run_ingestion(args)
    print("=" * 72)
    print("PGA INGESTION COMPLETE")
    print("=" * 72)
    print(f"Snapshot tag:       {manifest['snapshot_tag']}")
    print(f"Tour code:          {manifest['tour_code']}")
    print(f"Year:               {manifest['year']}")
    print(f"Tournament:         {manifest['current_tournament_name']} ({manifest['current_tournament_id']})")
    print(f"Tracked stat count: {len(manifest['tracked_stats'])}")
    print(f"Feature rows:       {manifest['feature_row_count']}")
    print(f"Feature columns:    {manifest['feature_column_count']}")
    print(f"Raw snapshots:      {manifest['raw_dir']}")
    print(f"Normalized output:  {manifest['normalized_dir']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
