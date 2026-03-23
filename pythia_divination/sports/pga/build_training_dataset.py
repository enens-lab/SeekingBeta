"""Build the first PGA training dataset by joining stat snapshots with tournament labels."""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.pga.constants import DEFAULT_TOUR_CODE  # noqa: E402
from sports.pga.feature_engineering import (  # noqa: E402
    add_schedule_identifiers,
    engineer_training_features,
)
from sports.pga.ingest import run_ingestion  # noqa: E402
from sports.pga.ingest_history import run_history_ingestion  # noqa: E402
from sports.pga.ingest_profiles import collect_player_profiles  # noqa: E402
from sports.pga.storage import (  # noqa: E402
    build_ingestion_paths,
    read_preferred_table,
    write_csv,
    write_json,
    write_parquet,
    write_partitioned_parquet,
)

logger = logging.getLogger(__name__)

_MONTH_TO_NUMBER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
_LABEL_COLUMNS = {"won", "top_5", "top_10", "made_cut", "withdrawn"}
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a first PGA training dataset from stat snapshots and tournament labels.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Target season to build. Defaults to the current schedule season rendered by PGA Tour.",
    )
    parser.add_argument(
        "--season-start",
        type=int,
        default=None,
        help="Optional first season year to include for multi-season history joins.",
    )
    parser.add_argument(
        "--season-end",
        type=int,
        default=None,
        help="Optional last season year to include for multi-season history joins.",
    )
    parser.add_argument(
        "--tour-code",
        default=DEFAULT_TOUR_CODE,
        help="PGA Tour code to request. Default is 'R'.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=None,
        help="Optional limit on completed tournaments to include.",
    )
    parser.add_argument(
        "--stat-limit",
        type=int,
        default=None,
        help="Optional limit on tracked stats to fetch.",
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
        "--snapshot-mode",
        default="prior_season",
        choices=["prior_season", "same_season", "none"],
        help="How to join public PGA stat snapshots. Defaults to prior_season to reduce leakage.",
    )
    parser.add_argument(
        "--refresh-source-data",
        action="store_true",
        help="Refresh schedules, results, and stat snapshots from the web instead of reusing local normalized/parquet tables.",
    )
    parser.add_argument(
        "--profile-limit",
        type=int,
        default=None,
        help="Optional limit on player profiles fetched during dataset assembly.",
    )
    parser.add_argument(
        "--refresh-profiles",
        action="store_true",
        help="Refresh cached PGA player profiles instead of reusing the normalized snapshot.",
    )
    parser.add_argument(
        "--profile-workers",
        type=int,
        default=8,
        help="Concurrent player profile fetch workers.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _coerce_currency(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace("$", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _coerce_first_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    filtered = "".join(character for character in text if character.isdigit() or character == ".")
    if not filtered:
        return None
    try:
        return float(filtered)
    except ValueError:
        return None


def _safe_divide(numerator: Any, denominator: Any) -> float | None:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den == 0 or math.isnan(num) or math.isnan(den):
        return None
    return num / den


def _slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _NON_ALNUM_PATTERN.sub("_", text).strip("_")
    return text or "unknown"


def _prepare_schedule_features(schedule_df: pd.DataFrame) -> pd.DataFrame:
    frame = schedule_df.copy()
    frame["month_number"] = frame["month"].map(_MONTH_TO_NUMBER)
    frame["purse_value"] = frame["purse"].map(_coerce_currency)
    frame["champion_earnings_value"] = frame["champion_earnings"].map(_coerce_currency)
    frame["fedex_points_value"] = frame["standings_value"].map(_coerce_first_number)
    frame["is_us_event"] = frame["course_country"].fillna("").eq("United States of America").astype(int)
    frame["is_signature_like_event"] = (
        pd.to_numeric(frame["purse_value"], errors="coerce").fillna(0).ge(18_000_000)
        | pd.to_numeric(frame["fedex_points_value"], errors="coerce").fillna(0).ge(700)
    ).astype(int)
    if "season" in frame.columns:
        frame = frame.reset_index(drop=True)
        frame["season_event_index"] = frame.groupby("season").cumcount() + 1
        frame["season_event_total"] = frame.groupby("season")["tournament_id"].transform("count")
        frame["season_progress_pct"] = frame["season_event_index"] / frame["season_event_total"].replace(0, pd.NA)
        if "purse_value" in frame.columns:
            frame["purse_rank_pct_in_season"] = frame.groupby("season")["purse_value"].rank(
                pct=True,
                ascending=False,
                method="average",
            )
        if "fedex_points_value" in frame.columns:
            frame["fedex_points_rank_pct_in_season"] = frame.groupby("season")["fedex_points_value"].rank(
                pct=True,
                ascending=False,
                method="average",
            )
    else:
        frame["season_event_index"] = range(1, len(frame) + 1)
    return add_schedule_identifiers(frame)


def _compute_tournament_context(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    frame["field_size"] = frame.groupby("tournament_id")["player_id"].transform("nunique")

    aggregations = {}
    for column in ("sg_total__avg", "owgr__rank", "scoring_average__avg"):
        if column in frame.columns:
            aggregations[column] = ["mean", "median"]
    if aggregations:
        grouped = frame.groupby("tournament_id").agg(aggregations)
        grouped.columns = [
            f"field_{column}_{agg}"
            for column, agg in grouped.columns.to_flat_index()
        ]
        grouped = grouped.reset_index()
        frame = frame.merge(grouped, on="tournament_id", how="left")

    if "owgr__rank" in frame.columns:
        frame["owgr_rank_pct_in_field"] = frame.groupby("tournament_id")["owgr__rank"].rank(pct=True, ascending=True)
    if "sg_total__avg" in frame.columns:
        frame["sg_total_pct_in_field"] = frame.groupby("tournament_id")["sg_total__avg"].rank(pct=True, ascending=False)
    if "scoring_average__avg" in frame.columns:
        frame["scoring_avg_pct_in_field"] = frame.groupby("tournament_id")["scoring_average__avg"].rank(pct=True, ascending=True)

    frame["implied_uniform_win_probability"] = frame["field_size"].map(lambda size: _safe_divide(1.0, size))
    return frame


def _collect_feature_snapshots(
    *,
    snapshot_years: list[int],
    args: argparse.Namespace,
    paths: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[int]]:
    feature_frames: list[pd.DataFrame] = []
    feature_manifests: list[dict[str, Any]] = []
    loaded_seasons: list[int] = []

    for season in snapshot_years:
        manifest: dict[str, Any] | None = None
        snapshot: pd.DataFrame | None = None
        if not getattr(args, "refresh_source_data", False):
            try:
                snapshot = _load_local_table(paths.normalized_dir, f"player_feature_snapshot_{season}_latest")
                manifest = {
                    "year": int(season),
                    "source": "local_store",
                    "normalized_dir": str(paths.normalized_dir),
                    "feature_row_count": int(len(snapshot)),
                    "feature_column_count": int(len(snapshot.columns)),
                }
            except FileNotFoundError:
                snapshot = None
        if snapshot is None:
            feature_args = SimpleNamespace(
                year=season,
                tour_code=args.tour_code,
                stat_id=[],
                limit=args.stat_limit,
                output_root=str(paths.root),
                snapshot_tag=paths.snapshot_tag,
                verbose=args.verbose,
            )
            manifest = run_ingestion(feature_args)
            normalized_dir = Path(manifest["normalized_dir"])
            snapshot = _load_local_table(normalized_dir, f"player_feature_snapshot_{season}_latest")
        snapshot["feature_snapshot_source_year"] = season
        write_csv(paths.normalized_dir / f"player_feature_snapshot_{season}_latest.csv", snapshot)
        write_parquet(paths.normalized_dir / f"player_feature_snapshot_{season}_latest.parquet", snapshot)
        feature_frames.append(snapshot)
        feature_manifests.append(manifest)
        if season is not None:
            loaded_seasons.append(int(season))

    if not feature_frames:
        return (
            pd.DataFrame(columns=["player_id", "player_name", "country", "feature_snapshot_source_year"]),
            [],
            [],
        )

    feature_snapshot = pd.concat(feature_frames, ignore_index=True, sort=False)
    return feature_snapshot, feature_manifests, loaded_seasons


def _resolve_seasons(args: argparse.Namespace) -> list[int | None]:
    if args.season_start is not None or args.season_end is not None:
        if args.season_start is None or args.season_end is None:
            raise ValueError("Both --season-start and --season-end are required for multi-season builds")
        if args.season_end < args.season_start:
            raise ValueError("--season-end must be greater than or equal to --season-start")
        return list(range(args.season_start, args.season_end + 1))
    return [args.season]


def _prepare_profile_features(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    categorical_specs = (
        ("profile_equipment_sponsor_primary", "profile_equipment_brand", None, 2),
        ("profile_country_code", "profile_country_code", 12, 5),
        ("profile_college", "profile_college", 12, 4),
    )
    for source_column, prefix, top_n, min_count in categorical_specs:
        if source_column not in frame.columns:
            continue
        valid = frame[["player_id", source_column]].drop_duplicates(subset=["player_id"])
        valid[source_column] = valid[source_column].fillna("").astype(str).str.strip()
        valid = valid[valid[source_column].str.len() > 0]
        if valid.empty:
            continue
        counts = valid[source_column].value_counts()
        if top_n is not None:
            counts = counts.head(top_n)
        categories = [category for category, count in counts.items() if count >= min_count]
        if not categories:
            continue
        source_values = frame[source_column].fillna("").astype(str).str.strip()
        for category in categories:
            column = f"{prefix}_{_slugify(category)}"
            frame[column] = source_values.eq(category).astype(int)
    return frame


def _load_local_table(normalized_dir: Path, stem: str) -> pd.DataFrame:
    return read_preferred_table(
        normalized_dir / f"{stem}.parquet",
        normalized_dir / f"{stem}.csv",
    )


def build_training_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root) if args.output_root else None
    paths = build_ingestion_paths(base_dir=output_root, snapshot_tag=args.snapshot_tag)
    seasons = _resolve_seasons(args)
    history_manifests: list[dict[str, Any]] = []
    schedule_frames: list[pd.DataFrame] = []
    label_frames: list[pd.DataFrame] = []
    for season in seasons:
        manifest: dict[str, Any] | None = None
        schedule_frame: pd.DataFrame | None = None
        label_frame: pd.DataFrame | None = None
        if season is not None and not getattr(args, "refresh_source_data", False):
            try:
                schedule_frame = _load_local_table(paths.normalized_dir, f"schedule_{season}_latest")
                label_frame = _load_local_table(paths.normalized_dir, f"season_tournament_labels_{season}_latest")
                manifest = {
                    "season": int(season),
                    "source": "local_store",
                    "selected_tournaments": int(schedule_frame["tournament_id"].nunique()),
                    "combined_row_count": int(len(label_frame)),
                    "normalized_dir": str(paths.normalized_dir),
                }
            except FileNotFoundError:
                schedule_frame = None
                label_frame = None

        if schedule_frame is None or label_frame is None:
            history_args = SimpleNamespace(
                season=season,
                tournament_id=[],
                include_status=[],
                limit=args.history_limit,
                output_root=str(paths.root),
                snapshot_tag=paths.snapshot_tag,
                verbose=args.verbose,
            )
            manifest = run_history_ingestion(history_args)
            season_year = int(manifest["season"])
            schedule_frame = _load_local_table(paths.normalized_dir, f"schedule_{season_year}_latest")
            label_frame = _load_local_table(paths.normalized_dir, f"season_tournament_labels_{season_year}_latest")

        history_manifests.append(manifest)
        schedule_frames.append(schedule_frame)
        label_frames.append(label_frame)

    actual_seasons = sorted(
        {
            int(value)
            for frame in schedule_frames
            for value in pd.to_numeric(frame.get("season"), errors="coerce").dropna().astype(int).tolist()
        }
    )
    if args.snapshot_mode == "prior_season":
        snapshot_years = sorted({season - 1 for season in actual_seasons})
    elif args.snapshot_mode == "same_season":
        snapshot_years = sorted(set(actual_seasons))
    else:
        snapshot_years = []

    feature_snapshot, feature_manifests, feature_snapshot_seasons = _collect_feature_snapshots(
        snapshot_years=snapshot_years,
        args=args,
        paths=paths,
    )

    normalized_dir = paths.normalized_dir
    labels = pd.concat(label_frames, ignore_index=True)
    schedule = pd.concat(schedule_frames, ignore_index=True)
    schedule = _prepare_schedule_features(schedule)
    player_index = (
        labels[["player_id", "player_name", "country"]]
        .drop_duplicates(subset=["player_id"])
        .reset_index(drop=True)
    )
    profiles, profile_manifest = collect_player_profiles(
        player_index,
        output_root=str(paths.root),
        snapshot_tag=paths.snapshot_tag,
        refresh=getattr(args, "refresh_profiles", False),
        limit=getattr(args, "profile_limit", None),
        workers=getattr(args, "profile_workers", 8),
    )

    feature_snapshot = feature_snapshot.rename(
        columns={
            "player_name": "feature_player_name",
            "country": "feature_country_name",
        }
    )
    if args.snapshot_mode == "prior_season":
        labels["feature_snapshot_lookup_year"] = pd.to_numeric(labels["season_year"], errors="coerce") - 1
    elif args.snapshot_mode == "same_season":
        labels["feature_snapshot_lookup_year"] = pd.to_numeric(labels["season_year"], errors="coerce")
    else:
        labels["feature_snapshot_lookup_year"] = pd.NA

    if not feature_snapshot.empty and args.snapshot_mode != "none":
        dataset = labels.merge(
            feature_snapshot,
            how="left",
            left_on=["feature_snapshot_lookup_year", "player_id"],
            right_on=["feature_snapshot_source_year", "player_id"],
            indicator=True,
        )
        dataset["feature_snapshot_found"] = dataset["_merge"].eq("both")
        dataset["feature_snapshot_season_year"] = dataset["feature_snapshot_source_year"]
        dataset["feature_snapshot_display_season"] = dataset["feature_snapshot_source_year"]
        dataset = dataset.drop(columns=["_merge", "feature_snapshot_source_year"], errors="ignore")
    else:
        dataset = labels.copy()
        dataset["feature_snapshot_found"] = False
        dataset["feature_snapshot_season_year"] = pd.NA
        dataset["feature_snapshot_display_season"] = pd.NA

    if "feature_player_name" in dataset.columns:
        dataset["player_name"] = dataset["player_name"].fillna(dataset["feature_player_name"])
    if not profiles.empty:
        profiles = profiles.rename(columns={"player_name": "profile_player_name"})
        dataset = dataset.merge(profiles, how="left", on="player_id")
        if "profile_player_name" in dataset.columns:
            dataset["player_name"] = dataset["player_name"].fillna(dataset["profile_player_name"])
        dataset["profile_snapshot_found"] = dataset["profile_source_path"].notna()
    else:
        dataset["profile_snapshot_found"] = False

    schedule_columns = [
        "tournament_id",
        "display_date",
        "month",
        "month_number",
        "season_event_index",
        "event_sort_key",
        "champion_name",
        "champion_player_id",
        "champion_earnings_value",
        "purse_value",
        "fedex_points_value",
        "course_name",
        "course_city",
        "course_state_code",
        "course_country",
        "course_key",
        "is_us_event",
        "is_signature_like_event",
        "season_event_total",
        "season_progress_pct",
        "purse_rank_pct_in_season",
        "fedex_points_rank_pct_in_season",
        "standings_heading",
        "standings_value",
    ]
    dataset = dataset.merge(
        schedule[schedule_columns],
        how="left",
        on="tournament_id",
    )
    dataset = _prepare_profile_features(dataset)
    dataset = _compute_tournament_context(dataset)
    dataset = engineer_training_features(dataset)

    dataset["feature_snapshot_mode"] = (
        "prior_season_public_stats_snapshot"
        if args.snapshot_mode == "prior_season"
        else "season_public_stats_snapshot"
        if args.snapshot_mode == "same_season"
        else "none"
    )
    dataset["feature_leakage_warning"] = bool(args.snapshot_mode == "same_season")

    ordered_front = [
        "season_year",
        "tournament_id",
        "tournament_name",
        "season_event_index",
        "display_date",
        "month",
        "month_number",
        "player_id",
        "player_name",
        "country",
        "feature_snapshot_found",
        "feature_snapshot_lookup_year",
        "feature_snapshot_season_year",
        "feature_snapshot_display_season",
        "feature_snapshot_mode",
        "feature_leakage_warning",
        "profile_snapshot_found",
        "won",
        "top_5",
        "top_10",
        "made_cut",
        "withdrawn",
    ]
    ordered_front = [column for column in ordered_front if column in dataset.columns]
    remaining = [column for column in dataset.columns if column not in ordered_front]
    dataset = dataset[ordered_front + remaining]

    dataset_path = normalized_dir / "pga_training_dataset_latest.csv"
    write_csv(dataset_path, dataset)
    dataset_parquet_path = normalized_dir / "pga_training_dataset_latest.parquet"
    write_parquet(dataset_parquet_path, dataset)
    partitioned_dataset_paths = write_partitioned_parquet(
        normalized_dir / "warehouse" / "pga_training_dataset",
        dataset,
        partition_column="season_year",
    )

    label_counts = {
        label: int(dataset[label].fillna(False).astype(bool).sum())
        for label in _LABEL_COLUMNS
        if label in dataset.columns
    }
    manifest = {
        "snapshot_tag": paths.snapshot_tag,
        "season": history_manifests[-1]["season"],
        "seasons": [manifest["season"] for manifest in history_manifests],
        "row_count": int(len(dataset)),
        "column_count": int(len(dataset.columns)),
        "unique_tournaments": int(dataset["tournament_id"].nunique()),
        "feature_snapshot_found_rate": float(dataset["feature_snapshot_found"].mean()) if not dataset.empty else 0.0,
        "profile_snapshot_found_rate": float(dataset["profile_snapshot_found"].mean()) if not dataset.empty else 0.0,
        "feature_snapshot_seasons": feature_snapshot_seasons,
        "feature_snapshot_mode": dataset["feature_snapshot_mode"].iloc[0] if not dataset.empty else args.snapshot_mode,
        "feature_leakage_warning": bool(args.snapshot_mode == "same_season"),
        "label_counts": label_counts,
        "feature_manifests": feature_manifests,
        "history_manifests": history_manifests,
        "profile_manifest": profile_manifest,
        "dataset_path": str(dataset_path),
        "dataset_parquet_path": str(dataset_parquet_path),
        "partitioned_dataset_paths": partitioned_dataset_paths,
        "refresh_source_data": bool(getattr(args, "refresh_source_data", False)),
    }
    write_json(normalized_dir / "pga_training_dataset_manifest_latest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    manifest = build_training_dataset(args)
    print("=" * 72)
    print("PGA TRAINING DATASET COMPLETE")
    print("=" * 72)
    print(f"Seasons:                   {manifest['seasons']}")
    print(f"Rows:                      {manifest['row_count']}")
    print(f"Columns:                   {manifest['column_count']}")
    print(f"Tournaments:               {manifest['unique_tournaments']}")
    print(f"Feature coverage:          {manifest['feature_snapshot_found_rate']:.2%}")
    print(f"Feature snapshot seasons:  {manifest['feature_snapshot_seasons']}")
    print(f"Dataset path:              {manifest['dataset_path']}")
    if manifest["feature_leakage_warning"]:
        print("Leakage note:              season public stats snapshots still include end-of-season leakage")
    else:
        print("Leakage note:              reduced; public stat priors come from prior seasons only")
    print("=" * 72)


if __name__ == "__main__":
    main()
