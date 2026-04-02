"""Optional Statcast enrichment for MLB matchup datasets.

Supports either raw pitch-level Statcast exports or pre-aggregated daily
feature tables written by ``sports.mlb.collect_statcast``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path, low_memory=False)
            raise
    return pd.read_csv(path, low_memory=False)


def _normalize_daily_table(
    frame: pd.DataFrame,
    *,
    id_col: str,
    date_col: str = "game_date",
    coerce_numeric: bool = False,
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[date_col] = pd.to_datetime(normalized[date_col], errors="coerce").dt.normalize()
    if coerce_numeric and id_col in normalized.columns:
        normalized[id_col] = pd.to_numeric(normalized[id_col], errors="coerce")
    allowed_columns = [
        column
        for column in normalized.columns
        if column in {id_col, date_col} or column.endswith("_avg_last_5") or column.endswith("_avg_last_15")
    ]
    normalized = normalized[allowed_columns]
    return normalized


def normalize_statcast(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "game_date" not in normalized.columns:
        raise ValueError("Statcast file must include a game_date column.")
    normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce").dt.normalize()

    if "pitcher" in normalized.columns:
        normalized["pitcher_id"] = pd.to_numeric(normalized["pitcher"], errors="coerce")
    elif "pitcher_id" in normalized.columns:
        normalized["pitcher_id"] = pd.to_numeric(normalized["pitcher_id"], errors="coerce")

    if "home_team" in normalized.columns and "away_team" in normalized.columns and "inning_topbot" in normalized.columns:
        top_mask = normalized["inning_topbot"].astype(str).str.lower().eq("top")
        normalized["batting_team"] = normalized["home_team"]
        normalized.loc[top_mask, "batting_team"] = normalized.loc[top_mask, "away_team"]
        normalized["fielding_team"] = normalized["away_team"]
        normalized.loc[top_mask, "fielding_team"] = normalized.loc[top_mask, "home_team"]

    for column in ("launch_speed", "launch_angle", "release_speed", "estimated_woba_using_speedangle", "woba_value"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "launch_speed" in normalized.columns:
        normalized["hard_hit_flag"] = normalized["launch_speed"].ge(95).astype(float)
    if "events" in normalized.columns:
        events = normalized["events"].astype(str).str.lower()
        normalized["barrel_like_flag"] = events.isin({"home_run", "triple", "double"}).astype(float)
    return normalized


def build_daily_group_features(frame: pd.DataFrame, *, group_col: str, prefix: str) -> pd.DataFrame:
    if frame.empty or group_col not in frame.columns:
        return pd.DataFrame()

    metrics = {
        "statcast_plate_appearances": frame.groupby([group_col, "game_date"]).size(),
    }
    if "launch_speed" in frame.columns:
        metrics["statcast_avg_launch_speed"] = frame.groupby([group_col, "game_date"])["launch_speed"].mean()
    if "launch_angle" in frame.columns:
        metrics["statcast_avg_launch_angle"] = frame.groupby([group_col, "game_date"])["launch_angle"].mean()
    if "estimated_woba_using_speedangle" in frame.columns:
        metrics["statcast_avg_xwoba"] = frame.groupby([group_col, "game_date"])["estimated_woba_using_speedangle"].mean()
    if "release_speed" in frame.columns:
        metrics["statcast_avg_release_speed"] = frame.groupby([group_col, "game_date"])["release_speed"].mean()
    if "hard_hit_flag" in frame.columns:
        metrics["statcast_hard_hit_rate"] = frame.groupby([group_col, "game_date"])["hard_hit_flag"].mean()
    if "barrel_like_flag" in frame.columns:
        metrics["statcast_barrel_like_rate"] = frame.groupby([group_col, "game_date"])["barrel_like_flag"].mean()

    grouped = pd.concat(metrics, axis=1).reset_index().sort_values([group_col, "game_date"])
    metric_columns = [column for column in grouped.columns if column not in {group_col, "game_date"}]
    for column in metric_columns:
        series = grouped.groupby(group_col)[column]
        grouped[f"{column}_avg_last_5"] = series.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        grouped[f"{column}_avg_last_15"] = series.transform(lambda s: s.shift(1).rolling(15, min_periods=1).mean())

    rename_map = {column: f"{prefix}_{column}" for column in grouped.columns if column not in {group_col, "game_date"}}
    grouped = grouped.rename(columns=rename_map)
    return grouped


def _merge_asof_features(
    base: pd.DataFrame,
    history: pd.DataFrame,
    *,
    group_col: str,
    date_col: str,
    feature_prefix: str,
    rename_group_to: str,
) -> pd.DataFrame:
    if base.empty or history.empty:
        return base

    history_frame = history.copy()
    history_frame[date_col] = pd.to_datetime(history_frame[date_col], errors="coerce").dt.normalize()
    history_frame = history_frame.dropna(subset=[group_col, date_col]).sort_values([group_col, date_col]).reset_index(drop=True)

    feature_columns = [column for column in history_frame.columns if column not in {group_col, date_col}]
    output_parts: list[pd.DataFrame] = []

    for key, base_group in base.groupby(rename_group_to, sort=False):
        history_group = history_frame.loc[history_frame[group_col] == key, [date_col, *feature_columns]].copy()
        base_sorted = base_group.sort_values(["official_date", "game_pk"]).reset_index(drop=True)
        if history_group.empty:
            for column in feature_columns:
                base_sorted[f"{feature_prefix}{column}"] = pd.NA
            output_parts.append(base_sorted)
            continue
        merged = pd.merge_asof(
            base_sorted,
            history_group.sort_values(date_col),
            left_on="official_date",
            right_on=date_col,
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.drop(columns=[date_col], errors="ignore")
        merged = merged.rename(columns={column: f"{feature_prefix}{column}" for column in feature_columns})
        output_parts.append(merged)

    return pd.concat(output_parts, ignore_index=True)


def enrich_dataset_with_statcast(dataset: pd.DataFrame, statcast_paths: Iterable[str] | None) -> pd.DataFrame:
    """Optionally merge rolling Statcast team/pitcher features onto the matchup dataset.

    ``statcast_paths`` can point at:
    - raw pitch-level Statcast exports, or
    - pre-aggregated daily feature tables such as
      ``statcast_pitcher_daily_latest.parquet`` and
      ``statcast_team_daily_latest.parquet``.
    """
    paths = [Path(path) for path in (statcast_paths or []) if path]
    if not paths:
        return dataset

    raw_frames: list[pd.DataFrame] = []
    pitcher_frames: list[pd.DataFrame] = []
    team_frames: list[pd.DataFrame] = []

    for path in paths:
        if not path.exists():
            continue
        frame = _read_table(path)
        columns = set(frame.columns)
        if {"pitcher_id", "game_date"}.issubset(columns) and any(column.startswith("pitcher_statcast_") for column in columns):
            pitcher_frames.append(_normalize_daily_table(frame, id_col="pitcher_id", coerce_numeric=True))
            continue
        if {"batting_team", "game_date"}.issubset(columns) and any(column.startswith("batting_team_statcast_") for column in columns):
            team_frames.append(_normalize_daily_table(frame, id_col="batting_team"))
            continue
        raw_frames.append(frame)

    pitcher_features = pd.DataFrame()
    team_features = pd.DataFrame()

    if raw_frames:
        statcast = normalize_statcast(pd.concat(raw_frames, ignore_index=True))
        if {"pitcher_id", "game_date"}.issubset(statcast.columns):
            pitcher_features = build_daily_group_features(
                statcast.dropna(subset=["pitcher_id"]),
                group_col="pitcher_id",
                prefix="pitcher",
            )
        if {"batting_team", "game_date"}.issubset(statcast.columns):
            team_features = build_daily_group_features(
                statcast.dropna(subset=["batting_team"]),
                group_col="batting_team",
                prefix="batting_team",
            )

    if pitcher_frames:
        pitcher_features = pd.concat([pitcher_features, *pitcher_frames], ignore_index=True) if not pitcher_features.empty else pd.concat(pitcher_frames, ignore_index=True)
        pitcher_features = pitcher_features.drop_duplicates(subset=["pitcher_id", "game_date"], keep="last").reset_index(drop=True)
    if team_frames:
        team_features = pd.concat([team_features, *team_frames], ignore_index=True) if not team_features.empty else pd.concat(team_frames, ignore_index=True)
        team_features = team_features.drop_duplicates(subset=["batting_team", "game_date"], keep="last").reset_index(drop=True)

    enriched = dataset.copy()
    enriched["official_date"] = pd.to_datetime(enriched["official_date"], errors="coerce").dt.normalize()

    if not pitcher_features.empty:
        away_base = enriched[["game_pk", "official_date", "away_probable_pitcher_id"]].copy()
        away_base["away_probable_pitcher_id"] = pd.to_numeric(away_base["away_probable_pitcher_id"], errors="coerce")
        away_pitcher = _merge_asof_features(
            away_base,
            pitcher_features,
            group_col="pitcher_id",
            date_col="game_date",
            feature_prefix="away_",
            rename_group_to="away_probable_pitcher_id",
        )
        enriched = enriched.merge(away_pitcher, on=["game_pk", "official_date", "away_probable_pitcher_id"], how="left")

        home_base = enriched[["game_pk", "official_date", "home_probable_pitcher_id"]].copy()
        home_base["home_probable_pitcher_id"] = pd.to_numeric(home_base["home_probable_pitcher_id"], errors="coerce")
        home_pitcher = _merge_asof_features(
            home_base,
            pitcher_features,
            group_col="pitcher_id",
            date_col="game_date",
            feature_prefix="home_",
            rename_group_to="home_probable_pitcher_id",
        )
        enriched = enriched.merge(home_pitcher, on=["game_pk", "official_date", "home_probable_pitcher_id"], how="left")

    if not team_features.empty and "away_team_abbreviation" in enriched.columns and "home_team_abbreviation" in enriched.columns:
        away_base = enriched[["game_pk", "official_date", "away_team_abbreviation"]].copy()
        away_team = _merge_asof_features(
            away_base,
            team_features,
            group_col="batting_team",
            date_col="game_date",
            feature_prefix="away_",
            rename_group_to="away_team_abbreviation",
        )
        enriched = enriched.merge(away_team, on=["game_pk", "official_date", "away_team_abbreviation"], how="left")

        home_base = enriched[["game_pk", "official_date", "home_team_abbreviation"]].copy()
        home_team = _merge_asof_features(
            home_base,
            team_features,
            group_col="batting_team",
            date_col="game_date",
            feature_prefix="home_",
            rename_group_to="home_team_abbreviation",
        )
        enriched = enriched.merge(home_team, on=["game_pk", "official_date", "home_team_abbreviation"], how="left")

    return enriched
