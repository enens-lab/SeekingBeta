from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports.mlb.availability_features import build_transaction_feature_frame, merge_transaction_features
from sports.mlb.build_training_dataset import DEFAULT_MLB_DATA_ROOT, _resolve_statcast_paths
from sports.mlb.branding import get_team_branding
from sports.mlb.client import (
    MLBStatsClient,
    flatten_game_bundle,
    flatten_game_players,
    flatten_roster,
    flatten_schedule,
    flatten_teams,
    flatten_transactions,
)
from sports.mlb.feature_engineering import (
    add_matchup_differentials,
    attach_pregame_starter_features,
    attach_pregame_team_features,
    prepare_games,
)
from sports.mlb.roster_features import (
    build_bullpen_feature_frame,
    build_lineup_feature_frame,
    build_projected_bullpen_roster,
    build_projected_lineup_roster,
    merge_bullpen_features,
    merge_lineup_features,
)
from sports.mlb.statcast_enrichment import enrich_dataset_with_statcast
from sports.pga.storage import read_preferred_table

PROJ_ROOT = Path(__file__).resolve().parents[2]
MLB_DATA_ROOT = DEFAULT_MLB_DATA_ROOT
NORMALIZED_DIR = MLB_DATA_ROOT / "normalized"
FRONTEND_DATA_DIR = PROJ_ROOT / "pythia_prophecy" / "frontend" / "src" / "data"
DATASET_PATH = NORMALIZED_DIR / "mlb_training_dataset_latest.csv"
HISTORICAL_PREDICTIONS_PATH = DIV_ROOT / "artifacts" / "mlb_baseline" / "hist_gradient_boosting" / "home_win" / "validation_predictions.csv"
MODEL_PATH = DIV_ROOT / "artifacts" / "mlb_baseline" / "hist_gradient_boosting" / "home_win" / "model.joblib"
FEATURE_COLUMNS_PATH = DIV_ROOT / "artifacts" / "mlb_baseline" / "hist_gradient_boosting" / "home_win" / "feature_columns.csv"
UPCOMING_LOOKAHEAD_DAYS = 7
MAX_AVAILABLE_DATES = 7


def _optional_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _load_table(stem: str) -> pd.DataFrame:
    parquet_path = NORMALIZED_DIR / f"{stem}.parquet"
    csv_path = NORMALIZED_DIR / f"{stem}.csv"
    try:
        return read_preferred_table(parquet_path, csv_path)
    except ImportError:
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise


def _load_table_optional(stem: str) -> pd.DataFrame:
    try:
        return _load_table(stem)
    except FileNotFoundError:
        return pd.DataFrame()


def _load_live_teams_table(client: MLBStatsClient, *, season: int | None = None) -> pd.DataFrame:
    payload = client.get_teams(season=season)
    frame = flatten_teams(payload)
    if frame.empty:
        return frame
    frame = frame.sort_values(["season", "team_id"], ascending=[False, True]).reset_index(drop=True)
    return frame


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = values.clip(-6.0, 6.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _coerce_date_key(value: object) -> str | None:
    if value in (None, ""):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.normalize().strftime("%Y-%m-%d")


def _date_key_to_label(value: str) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return value
    return timestamp.strftime("%a, %b %d").replace(" 0", " ")


def _safe_value(row: Any, field: str) -> float | None:
    value = getattr(row, field, np.nan)
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_score(value: float | None, low: float, high: float, *, inverse: bool = False) -> float | None:
    if value is None or high <= low:
        return None
    clipped = min(max(value, low), high)
    ratio = (clipped - low) / (high - low)
    if inverse:
        ratio = 1.0 - ratio
    return round(ratio * 100.0, 1)


def _normalize_metric(value: float | None, low: float, high: float, *, inverse: bool = False) -> float | None:
    if value is None or high <= low:
        return None
    clipped = min(max(value, low), high)
    ratio = (clipped - low) / (high - low)
    if inverse:
        ratio = 1.0 - ratio
    return round(ratio * 100.0, 1)


def _combine_metric(*weighted_values: tuple[float, float | None]) -> float | None:
    total_weight = 0.0
    total_value = 0.0
    for weight, value in weighted_values:
        if value is None:
            continue
        total_weight += weight
        total_value += weight * value
    if total_weight == 0:
        return None
    return round(total_value / total_weight, 1)


def _format_record_prior(games_played: float | None, win_pct: float | None) -> str | None:
    if games_played is None or win_pct is None or games_played <= 0:
        return None
    wins = int(round(games_played * win_pct))
    wins = max(0, min(wins, int(round(games_played))))
    losses = max(int(round(games_played)) - wins, 0)
    return f"{wins}-{losses}"


def _format_recent_form(row: Any, side: str) -> str | None:
    wins_rate = _safe_value(row, f"{side}_team_won_avg_last_5")
    run_diff = _safe_value(row, f"{side}_team_run_diff_avg_last_10")
    runs = _safe_value(row, f"{side}_team_runs_scored_avg_last_5")
    parts: list[str] = []
    if wins_rate is not None:
        wins = max(0, min(5, int(round(wins_rate * 5))))
        parts.append(f"{wins}-{5 - wins} last 5")
    if run_diff is not None:
        parts.append(f"{run_diff:+.1f} run diff")
    if runs is not None:
        parts.append(f"{runs:.1f} runs/game")
    return " | ".join(parts) or None


def _format_bullpen_summary(row: Any, side: str) -> str | None:
    leverage = _safe_value(row, f"{side}_bullpen_high_leverage_arms_count")
    short_rest = _safe_value(row, f"{side}_bullpen_high_leverage_short_rest_count")
    era_like = _safe_value(row, f"{side}_bullpen_avg_era_like_avg_last_5")
    parts: list[str] = []
    if leverage is not None:
        parts.append(f"{int(round(leverage))} leverage arms")
    if short_rest is not None:
        parts.append(f"{int(round(short_rest))} short rest")
    if era_like is not None:
        parts.append(f"{era_like:.2f} ERA-like")
    return " | ".join(parts) or None


def _format_availability_summary(row: Any, side: str) -> str | None:
    il_adds = int(round(_safe_value(row, f"{side}_availability_il_additions_last_14") or 0))
    il_activations = int(round(_safe_value(row, f"{side}_availability_il_activations_last_14") or 0))
    roster_moves = int(round(_safe_value(row, f"{side}_availability_transactions_last_14") or 0))
    if il_adds == 0 and il_activations == 0 and roster_moves == 0:
        return "Quiet roster over last 14 days"
    return f"{il_adds} IL adds | {il_activations} activations | {roster_moves} moves"


def _format_lineup_continuity(row: Any, side: str) -> str | None:
    overlap = _safe_value(row, f"{side}_lineup_prev_game_overlap")
    coverage = _safe_value(row, f"{side}_lineup_history_coverage")
    if overlap is not None:
        return f"{overlap * 100:.0f}% overlap vs prior lineup"
    if coverage is not None:
        return f"{coverage * 100:.0f}% lineup history coverage"
    return None


def _format_weather_summary(row: Any) -> str | None:
    condition = _optional_text(getattr(row, "weather_condition", None))
    temp = _safe_value(row, "weather_temp_f")
    wind = _optional_text(getattr(row, "weather_wind", None))
    parts = [part for part in [condition, f"{temp:.0f} deg F" if temp is not None else None, wind] if part]
    return " | ".join(parts) or None


def _mlb_headshot_url(player_id: int | None) -> str | None:
    if not player_id:
        return None
    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"w_360,q_auto:best/v1/people/{int(player_id)}/headshot/67/current"
    )


def _safe_int_value(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bat_side_label(code: str | None) -> str | None:
    normalized = (code or "").strip().upper()
    return {
        "L": "L bat",
        "R": "R bat",
        "S": "Switch bat",
    }.get(normalized)


def _build_mlb_lineup_entry(record: dict[str, Any], *, historical: bool) -> dict[str, Any]:
    player_name = _optional_text(record.get("batter_name")) or "Unknown player"
    player_id = _safe_int_value(record.get("batter_id"))
    lineup_slot = _safe_int_value(record.get("lineup_slot"))
    position = _optional_text(record.get("batter_position"))
    bat_side = _optional_text(record.get("batter_bat_side") or record.get("bat_side_code"))
    subtitle_parts = [f"#{lineup_slot}" if lineup_slot is not None else None, position, _bat_side_label(bat_side)]

    stats: list[dict[str, str]] = []
    performance_summary: str | None = None
    radar_metrics: list[dict[str, float]] | None = None

    if historical:
        at_bats = _safe_int_value(record.get("at_bats")) or 0
        hits = _safe_int_value(record.get("hits")) or 0
        runs = _safe_int_value(record.get("runs")) or 0
        rbi = _safe_int_value(record.get("rbi")) or 0
        home_runs = _safe_int_value(record.get("home_runs")) or 0
        walks = _safe_int_value(record.get("walks")) or 0
        strikeouts = _safe_int_value(record.get("strikeouts")) or 0

        stats = [
            {"label": "Game", "value": f"{hits}-{at_bats}" if at_bats > 0 else "0-0"},
            {"label": "R", "value": str(runs)},
            {"label": "RBI", "value": str(rbi)},
            {"label": "HR", "value": str(home_runs)},
        ]
        performance_summary = f"BB {walks} | K {strikeouts}"
        on_base_rate = ((hits + walks) / max(at_bats + walks, 1)) if (at_bats + walks) > 0 else 0.0
        discipline = _combine_metric(
            (0.45, _clamp_score(float(walks), 0.0, 3.0)),
            (0.55, _clamp_score(float(strikeouts), 0.0, 4.0, inverse=True)),
        )
        game_impact = _combine_metric(
            (0.35, _clamp_score(float(hits), 0.0, 4.0)),
            (0.35, _clamp_score(float(rbi), 0.0, 4.0)),
            (0.30, _clamp_score(float(home_runs), 0.0, 2.0)),
        )
        radar_metrics = [
            {"label": "Contact", "value": _clamp_score(float(hits), 0.0, 4.0) or 0.0},
            {"label": "Power", "value": _clamp_score(float(home_runs), 0.0, 2.0) or 0.0},
            {"label": "On-Base", "value": _clamp_score(on_base_rate, 0.20, 0.70) or 0.0},
            {"label": "Discipline", "value": discipline or 0.0},
            {"label": "Run Production", "value": _clamp_score(float(rbi + runs), 0.0, 6.0) or 0.0},
            {"label": "Game Impact", "value": game_impact or 0.0},
        ]
    else:
        ops_last_10 = _safe_value(type("Row", (), record), "ops_like_avg_last_10")
        hits_last_5 = _safe_value(type("Row", (), record), "hits_avg_last_5")
        home_runs_last_10 = _safe_value(type("Row", (), record), "home_runs_avg_last_10")
        rbi_last_5 = _safe_value(type("Row", (), record), "rbi_avg_last_5")
        obp_last_10 = _safe_value(type("Row", (), record), "obp_like_avg_last_10")
        walks_last_5 = _safe_value(type("Row", (), record), "walks_avg_last_5")

        if ops_last_10 is not None:
            stats.append({"label": "OPS L10", "value": f"{ops_last_10:.3f}"})
        if hits_last_5 is not None:
            stats.append({"label": "Hits L5", "value": f"{hits_last_5:.1f}"})
        if home_runs_last_10 is not None:
            stats.append({"label": "HR L10", "value": f"{home_runs_last_10:.1f}"})
        if rbi_last_5 is not None:
            stats.append({"label": "RBI L5", "value": f"{rbi_last_5:.1f}"})
        if len(stats) < 4 and obp_last_10 is not None:
            stats.append({"label": "OBP L10", "value": f"{obp_last_10:.3f}"})
        if len(stats) < 4 and walks_last_5 is not None:
            stats.append({"label": "BB L5", "value": f"{walks_last_5:.1f}"})

        discipline = _combine_metric(
            (0.45, _clamp_score(walks_last_5, 0.0, 2.0)),
            (0.55, _clamp_score(_safe_value(type("Row", (), record), "strikeouts_avg_last_10"), 0.0, 2.5, inverse=True)),
        )
        run_production = _combine_metric(
            (0.55, _clamp_score(rbi_last_5, 0.0, 2.5)),
            (0.45, _clamp_score(hits_last_5, 0.0, 2.2)),
        )
        radar_metrics = [
            {"label": "Contact", "value": _clamp_score(hits_last_5, 0.0, 2.2) or 0.0},
            {"label": "Power", "value": _clamp_score(home_runs_last_10, 0.0, 0.8) or 0.0},
            {"label": "On-Base", "value": _clamp_score(obp_last_10, 0.24, 0.45) or 0.0},
            {"label": "Discipline", "value": discipline or 0.0},
            {"label": "Run Production", "value": run_production or 0.0},
            {"label": "Recent Form", "value": _clamp_score(ops_last_10, 0.50, 1.10) or 0.0},
        ]

    return {
        "playerId": player_id,
        "playerName": player_name,
        "lineupSlot": lineup_slot,
        "position": position,
        "batSide": bat_side,
        "performanceSummary": performance_summary,
        "radarMetrics": radar_metrics or [],
        "profile": {
            "imageUrl": _mlb_headshot_url(player_id),
            "subtitle": " | ".join(part for part in subtitle_parts if part) or "Projected lineup",
            "country": None,
            "stats": stats[:4],
        },
    }


def _merge_projected_lineup_history(projected_lineups: pd.DataFrame, batter_logs: pd.DataFrame) -> pd.DataFrame:
    if projected_lineups.empty or batter_logs.empty:
        return projected_lineups.copy()

    lineup = projected_lineups.copy()
    lineup["official_date"] = pd.to_datetime(lineup["official_date"], errors="coerce").dt.normalize()
    history = batter_logs.copy()
    history["official_date"] = pd.to_datetime(history["official_date"], errors="coerce").dt.normalize()

    history_columns = [
        "official_date",
        "batter_id",
        "plate_appearances_avg_last_10",
        "hits_avg_last_5",
        "home_runs_avg_last_10",
        "rbi_avg_last_5",
        "ops_like_avg_last_10",
        "obp_like_avg_last_10",
        "walks_avg_last_5",
    ]
    history = history.loc[:, [column for column in history_columns if column in history.columns]]
    history = history.dropna(subset=["official_date", "batter_id"]).sort_values(["official_date", "batter_id"])
    lineup = lineup.dropna(subset=["official_date", "batter_id"]).sort_values(["official_date", "batter_id"])
    if history.empty or lineup.empty:
        return projected_lineups.copy()

    merged = pd.merge_asof(
        lineup,
        history,
        on="official_date",
        by="batter_id",
        direction="backward",
        suffixes=("", "_history"),
    )
    return merged.sort_values(["game_pk", "team_side", "lineup_slot"]).reset_index(drop=True)


def _build_projected_lineup_map(projected_lineups: pd.DataFrame, batter_logs: pd.DataFrame) -> dict[int, dict[str, list[dict[str, Any]]]]:
    if projected_lineups.empty:
        return {}

    enriched = _merge_projected_lineup_history(projected_lineups, batter_logs)
    lineup_map: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for (game_pk, team_side), group in enriched.groupby(["game_pk", "team_side"], sort=False):
        entries = [
            _build_mlb_lineup_entry(record, historical=False)
            for record in group.sort_values("lineup_slot").to_dict(orient="records")
        ]
        lineup_map.setdefault(int(game_pk), {})[str(team_side)] = entries
    return lineup_map


def _load_historical_lineup_map(game_pks: set[int]) -> dict[int, dict[str, list[dict[str, Any]]]]:
    if not game_pks:
        return {}

    lineup_roster = _load_table_optional("lineup_roster_latest")
    batter_logs = _load_table_optional("batter_game_logs_latest")
    if lineup_roster.empty:
        return {}

    lineup = lineup_roster.loc[lineup_roster["game_pk"].isin(game_pks)].copy()
    if lineup.empty:
        return {}

    if not batter_logs.empty:
        performance_columns = [
            "game_pk",
            "team_side",
            "team_id",
            "batter_id",
            "at_bats",
            "hits",
            "runs",
            "rbi",
            "home_runs",
            "walks",
            "strikeouts",
        ]
        performance = batter_logs.loc[
            batter_logs["game_pk"].isin(game_pks),
            [column for column in performance_columns if column in batter_logs.columns],
        ].copy()
        lineup = lineup.merge(
            performance,
            on=["game_pk", "team_side", "team_id", "batter_id"],
            how="left",
        )

    lineup_map: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for (game_pk, team_side), group in lineup.groupby(["game_pk", "team_side"], sort=False):
        entries = [
            _build_mlb_lineup_entry(record, historical=True)
            for record in group.sort_values("lineup_slot").to_dict(orient="records")
        ]
        lineup_map.setdefault(int(game_pk), {})[str(team_side)] = entries
    return lineup_map


def _build_team_details(row: Any, side: str) -> dict[str, Any]:
    team_id = getattr(row, f"{side}_team_id", None)
    abbreviation = _optional_text(getattr(row, f"{side}_team_abbreviation", None))
    branding = get_team_branding(int(team_id) if team_id is not None and not pd.isna(team_id) else None, abbreviation)
    return {
        **branding,
        "recordPrior": _format_record_prior(
            _safe_value(row, f"{side}_team_games_played_prior"),
            _safe_value(row, f"{side}_team_win_pct_prior"),
        ),
        "recentForm": _format_recent_form(row, side),
        "bullpenSummary": _format_bullpen_summary(row, side),
        "availabilitySummary": _format_availability_summary(row, side),
        "lineupContinuity": _format_lineup_continuity(row, side),
        "venue": _optional_text(getattr(row, "venue_name", None)) or "Ballpark",
        "weather": _format_weather_summary(row),
    }


def _build_starter_profile(row: Any, side: str) -> dict[str, Any] | None:
    starter_name = _optional_text(getattr(row, f"{side}_probable_pitcher_name", None))
    if not starter_name:
        return None

    pitcher_id_raw = getattr(row, f"{side}_probable_pitcher_id", None)
    pitcher_id = int(pitcher_id_raw) if pitcher_id_raw is not None and not pd.isna(pitcher_id_raw) else None
    age = _safe_value(row, f"{side}_starter_profile_current_age")
    pitch_hand = _optional_text(getattr(row, f"{side}_starter_profile_pitch_hand", None))
    height = _optional_text(getattr(row, f"{side}_starter_profile_height", None))
    birth_country = _optional_text(getattr(row, f"{side}_starter_profile_birth_country", None))
    era_like = _safe_value(row, f"{side}_starter_era_like_avg_last_5")
    whip = _safe_value(row, f"{side}_starter_whip_avg_last_5")
    strikeouts = _safe_value(row, f"{side}_starter_strikeouts_avg_last_5")
    innings = _safe_value(row, f"{side}_starter_innings_pitched_avg_last_5")
    days_rest = _safe_value(row, f"{side}_starter_days_rest")

    stats: list[dict[str, str]] = []
    if era_like is not None:
        stats.append({"label": "ERA-like", "value": f"{era_like:.2f}"})
    if whip is not None:
        stats.append({"label": "WHIP-like", "value": f"{whip:.2f}"})
    if strikeouts is not None:
        stats.append({"label": "K Avg", "value": f"{strikeouts:.1f}"})
    if innings is not None:
        stats.append({"label": "IP Avg", "value": f"{innings:.1f}"})
    if len(stats) < 4 and days_rest is not None:
        stats.append({"label": "Rest", "value": f"{days_rest:.0f} days"})

    subtitle_parts = []
    if pitch_hand:
        subtitle_parts.append(f"{pitch_hand}-handed")
    if age is not None:
        subtitle_parts.append(f"Age {age:.0f}")
    if height:
        subtitle_parts.append(height)

    return {
        "imageUrl": _mlb_headshot_url(pitcher_id),
        "subtitle": " | ".join(subtitle_parts) if subtitle_parts else "Probable starter",
        "country": birth_country,
        "stats": stats[:4],
    }


def _build_starter_radar(row: Any, side: str) -> list[dict[str, float]] | None:
    prefix = f"{side}_starter_"
    run_prevention = _combine_metric(
        (0.55, _normalize_metric(_safe_value(row, f"{prefix}era_like_avg_last_5"), 1.75, 6.75, inverse=True)),
        (0.30, _normalize_metric(_safe_value(row, f"{prefix}whip_avg_last_5"), 0.9, 1.7, inverse=True)),
        (0.15, _normalize_metric(_safe_value(row, f"{prefix}home_runs_allowed_avg_last_5"), 0.2, 1.8, inverse=True)),
    )
    strikeout_ability = _combine_metric(
        (0.7, _normalize_metric(_safe_value(row, f"{prefix}strikeouts_avg_last_5"), 3.0, 10.0)),
        (0.3, _normalize_metric(_safe_value(row, f"{prefix}strike_pct_avg_last_5"), 0.58, 0.72)),
    )
    command = _combine_metric(
        (0.55, _normalize_metric(_safe_value(row, f"{prefix}walks_avg_last_5"), 0.8, 4.5, inverse=True)),
        (0.45, _normalize_metric(_safe_value(row, f"{prefix}strike_pct_avg_last_5"), 0.58, 0.72)),
    )
    contact_suppression = _combine_metric(
        (0.4, _normalize_metric(_safe_value(row, f"{prefix}hits_allowed_avg_last_5"), 4.0, 9.5, inverse=True)),
        (0.35, _normalize_metric(_safe_value(row, f"{prefix}whip_avg_last_5"), 0.9, 1.7, inverse=True)),
        (0.25, _normalize_metric(_safe_value(row, f"{prefix}home_runs_allowed_avg_last_5"), 0.2, 1.8, inverse=True)),
    )
    durability = _combine_metric(
        (0.35, _normalize_metric(_safe_value(row, f"{prefix}innings_pitched_avg_last_5"), 4.0, 7.5)),
        (0.25, _normalize_metric(_safe_value(row, f"{prefix}batters_faced_avg_last_5"), 18.0, 29.0)),
        (0.2, _normalize_metric(_safe_value(row, f"{prefix}pitches_thrown_avg_last_5"), 68.0, 108.0)),
        (0.2, _normalize_metric(_safe_value(row, f"{prefix}starts_prior"), 3.0, 30.0)),
    )
    recent_form = _combine_metric(
        (0.35, _normalize_metric(_safe_value(row, f"{prefix}won_avg_last_3"), 0.0, 1.0)),
        (0.35, _normalize_metric(_safe_value(row, f"{prefix}earned_runs_avg_last_3"), 0.5, 5.5, inverse=True)),
        (0.3, _normalize_metric(_safe_value(row, f"{prefix}era_like_avg_last_3"), 1.75, 6.75, inverse=True)),
    )

    values = [
        ("Run Prevention", run_prevention),
        ("Strikeout Ability", strikeout_ability),
        ("Command", command),
        ("Contact Suppression", contact_suppression),
        ("Durability", durability),
        ("Recent Form", recent_form),
    ]
    if all(value is None for _, value in values):
        return None
    return [{"label": label, "value": value or 0.0} for label, value in values]


def _build_available_dates(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    if schedule.empty:
        return []
    dates = (
        schedule.assign(official_date=pd.to_datetime(schedule["official_date"], errors="coerce").dt.normalize())
        .dropna(subset=["official_date"])
        .groupby("official_date", as_index=False)
        .agg(gameCount=("game_pk", "count"))
        .sort_values("official_date")
        .head(MAX_AVAILABLE_DATES)
    )
    return [
        {
            "dateKey": row.official_date.strftime("%Y-%m-%d"),
            "label": _date_key_to_label(row.official_date.strftime("%Y-%m-%d")),
            "gameCount": int(row.gameCount),
        }
        for row in dates.itertuples(index=False)
    ]


def _resolve_selected_date(available_dates: list[dict[str, Any]], requested_date: str | None) -> str | None:
    requested = _coerce_date_key(requested_date)
    available_keys = {item["dateKey"] for item in available_dates}
    if requested and requested in available_keys:
        return requested
    if available_dates:
        return str(available_dates[0]["dateKey"])
    return requested


def _fallback_predict_upcoming(frame: pd.DataFrame) -> pd.DataFrame:
    inference = frame.copy()
    score = pd.Series(0.14, index=inference.index, dtype=float)
    score += 1.35 * _numeric_series(inference, "delta_win_pct_prior")
    score += 0.06 * _numeric_series(inference, "delta_run_diff_avg_last_10")
    score += 0.10 * _numeric_series(inference, "delta_runs_scored_avg_last_5")
    score -= 0.10 * _numeric_series(inference, "delta_runs_allowed_avg_last_5")
    score -= 0.30 * _numeric_series(inference, "delta_era_like_avg_last_5")
    score -= 0.22 * _numeric_series(inference, "delta_whip_avg_last_5")
    score += 0.02 * _numeric_series(inference, "delta_strikeouts_avg_last_5")
    score -= 0.02 * _numeric_series(inference, "delta_walks_avg_last_5")
    score += 0.03 * _numeric_series(inference, "delta_days_rest")
    score += 0.08 * (
        _numeric_series(inference, "home_availability_il_activations_last_14")
        - _numeric_series(inference, "away_availability_il_activations_last_14")
    )
    score -= 0.08 * (
        _numeric_series(inference, "home_availability_il_additions_last_14")
        - _numeric_series(inference, "away_availability_il_additions_last_14")
    )
    probabilities = _sigmoid(score)
    inference["home_win_probability"] = probabilities
    inference["away_win_probability"] = 1.0 - probabilities
    inference["prediction_source"] = "heuristic_fallback"
    return inference


def _load_historical_source() -> pd.DataFrame:
    predictions = pd.read_csv(HISTORICAL_PREDICTIONS_PATH)
    dataset = pd.read_csv(DATASET_PATH, low_memory=False)
    merged = predictions.merge(
        dataset.drop_duplicates(subset=["game_pk"]),
        on="game_pk",
        how="left",
        suffixes=("", "_dataset"),
    )
    official_date_column = "official_date_dataset" if "official_date_dataset" in merged.columns else "official_date"
    merged["official_date"] = pd.to_datetime(merged[official_date_column], errors="coerce")
    merged = merged.dropna(subset=["game_pk", "official_date", "away_team_name", "home_team_name"])
    return merged.sort_values(["official_date", "game_pk"], ascending=[False, False]).reset_index(drop=True)


def _build_backtests(frame: pd.DataFrame) -> list[dict[str, Any]]:
    historical_lineups = _load_historical_lineup_map(set(pd.to_numeric(frame["game_pk"], errors="coerce").dropna().astype(int).tolist()))
    backtests: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        home_prob = float(row.home_win_probability)
        away_prob = max(0.0, 1.0 - home_prob)
        actual_winner = row.home_team_name if int(row.home_win) == 1 else row.away_team_name
        ranked = sorted(
            [
                {
                    "rank": 1,
                    "playerName": row.home_team_name,
                    "winProbability": home_prob * 100.0,
                    "actualWinner": actual_winner == row.home_team_name,
                    "side": "home",
                },
                {
                    "rank": 1,
                    "playerName": row.away_team_name,
                    "winProbability": away_prob * 100.0,
                    "actualWinner": actual_winner == row.away_team_name,
                    "side": "away",
                },
            ],
            key=lambda item: item["winProbability"],
            reverse=True,
        )
        for idx, item in enumerate(ranked, start=1):
            item["rank"] = idx

        date_key = int(row.official_date.strftime("%Y%m%d"))
        backtests.append(
            {
                "year": int(row.season) if pd.notna(row.season) else int(row.official_date.year),
                "tournament": f"{row.away_team_name} at {row.home_team_name}",
                "tour": "Baseball",
                "venue": row.venue_name or "Ballpark",
                "awayTeam": row.away_team_name,
                "homeTeam": row.home_team_name,
                "predictedWinner": ranked[0]["playerName"],
                "predictedTop3": [entry["playerName"] for entry in ranked],
                "predictedTop5": [entry["playerName"] for entry in ranked],
                "actualWinner": actual_winner,
                "hitStatus": "Top Pick" if ranked[0]["playerName"] == actual_winner else "Miss",
                "prob": ranked[0]["winProbability"] / 100.0,
                "fullField": ranked,
                "latestDate": date_key,
                "tournamentId": f"mlb-{row.game_pk}",
                "scheduledDate": date_key,
                "course": _optional_text(row.venue_name) or "Ballpark",
                "predictions": ranked,
                "homeStarter": _optional_text(row.home_probable_pitcher_name),
                "awayStarter": _optional_text(row.away_probable_pitcher_name),
                "awayStarterProfile": _build_starter_profile(row, "away"),
                "homeStarterProfile": _build_starter_profile(row, "home"),
                "awayStarterRadar": _build_starter_radar(row, "away"),
                "homeStarterRadar": _build_starter_radar(row, "home"),
                "awayTeamDetails": _build_team_details(row, "away"),
                "homeTeamDetails": _build_team_details(row, "home"),
                "awayLineup": historical_lineups.get(int(row.game_pk), {}).get("away", []),
                "homeLineup": historical_lineups.get(int(row.game_pk), {}).get("home", []),
            }
        )
    return backtests


def _load_upcoming_schedule(client: MLBStatsClient) -> pd.DataFrame:
    today = pd.Timestamp.utcnow().normalize()
    end_date = today + pd.Timedelta(days=UPCOMING_LOOKAHEAD_DAYS - 1)
    payload = client.get_schedule(
        start_date=today.date().isoformat(),
        end_date=end_date.date().isoformat(),
        game_type="R",
        hydrate="probablePitcher,team,linescore",
    )
    schedule = flatten_schedule(payload)
    if schedule.empty:
        return schedule
    schedule["game_date"] = pd.to_datetime(schedule["game_date"], utc=True, errors="coerce")
    schedule["official_date"] = pd.to_datetime(schedule["official_date"], errors="coerce")
    preview_mask = ~schedule["status_abstract"].isin(["Final", "Completed Early", "Cancelled", "Postponed"])
    probable_mask = schedule["away_team_name"].notna() & schedule["home_team_name"].notna()
    upcoming = schedule.loc[preview_mask & probable_mask].copy()
    return upcoming.sort_values(["official_date", "game_date", "game_pk"]).reset_index(drop=True)


def _fetch_preview_bundles(client: MLBStatsClient, schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, Any], pd.DataFrame]:
    detail_frames: list[pd.DataFrame] = []
    bundle_map: dict[int, Any] = {}
    player_frames: list[pd.DataFrame] = []

    for row in schedule.itertuples(index=False):
        bundle = client.get_game_bundle(
            int(row.game_pk),
            include_boxscore=False,
            include_context_metrics=False,
            include_play_by_play=False,
            include_win_probability=False,
        )
        bundle_map[int(row.game_pk)] = bundle
        detail_frames.append(flatten_game_bundle(bundle))
        players = flatten_game_players(bundle.live_feed)
        if not players.empty:
            players["game_pk"] = int(row.game_pk)
            player_frames.append(players)

    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    player_profiles = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()
    return details, bundle_map, player_profiles


def _augment_pitcher_profiles(
    pitcher_profiles: pd.DataFrame,
    preview_player_profiles: pd.DataFrame,
    upcoming_games: pd.DataFrame,
) -> pd.DataFrame:
    profiles = pitcher_profiles.copy()
    if preview_player_profiles.empty:
        return profiles

    needed_ids = pd.Series(
        pd.concat(
            [
                upcoming_games["away_probable_pitcher_id"].dropna(),
                upcoming_games["home_probable_pitcher_id"].dropna(),
            ],
            ignore_index=True,
        )
    ).dropna()
    if needed_ids.empty:
        return profiles

    needed_ids = {int(value) for value in needed_ids.tolist()}
    preview_subset = preview_player_profiles.loc[
        preview_player_profiles["player_id"].isin(needed_ids)
    ].drop_duplicates(subset=["player_id"])
    if preview_subset.empty:
        return profiles

    rename_map = {"position_abbreviation": "position_abbreviation"}
    preview_subset = preview_subset.rename(columns=rename_map)
    combined = pd.concat([profiles, preview_subset], ignore_index=True, sort=False)
    return combined.drop_duplicates(subset=["player_id"], keep="last").reset_index(drop=True)


def _build_active_roster_frame(
    client: MLBStatsClient,
    upcoming_games: pd.DataFrame,
    preview_player_profiles: pd.DataFrame,
) -> pd.DataFrame:
    roster_frames: list[pd.DataFrame] = []
    preview_profiles = preview_player_profiles.drop_duplicates(subset=["player_id"]) if not preview_player_profiles.empty else pd.DataFrame()
    roster_cache: dict[tuple[int, str], pd.DataFrame] = {}

    for row in upcoming_games.itertuples(index=False):
        official_date = pd.to_datetime(row.official_date, errors="coerce")
        for side, team_id, team_name in (
            ("away", int(row.away_team_id), row.away_team_name),
            ("home", int(row.home_team_id), row.home_team_name),
        ):
            cache_key = (team_id, official_date.date().isoformat())
            if cache_key not in roster_cache:
                payload = client.get_team_roster(team_id, season=int(row.season), roster_type="active", date=official_date.date().isoformat())
                roster_cache[cache_key] = flatten_roster(payload, team_id=team_id, roster_type="active")
            roster = roster_cache[cache_key].copy()
            if roster.empty:
                continue
            roster["game_pk"] = int(row.game_pk)
            roster["official_date"] = official_date
            roster["season"] = int(row.season)
            roster["team_side"] = side
            roster["team_name"] = team_name
            roster_frames.append(roster)

    if not roster_frames:
        return pd.DataFrame()

    active_roster = pd.concat(roster_frames, ignore_index=True)
    if preview_profiles.empty:
        active_roster["current_age"] = np.nan
        active_roster["weight"] = np.nan
        active_roster["bat_side"] = pd.NA
        active_roster["pitch_hand"] = pd.NA
        active_roster["mlb_debut_date"] = pd.NA
        active_roster["years_since_debut"] = np.nan
        return active_roster

    active_roster = active_roster.merge(
        preview_profiles[
            [
                "player_id",
                "current_age",
                "weight",
                "bat_side",
                "pitch_hand",
                "mlb_debut_date",
                "position_abbreviation",
            ]
        ].drop_duplicates(subset=["player_id"]),
        on="player_id",
        how="left",
        suffixes=("", "_profile"),
    )
    active_roster["position_abbreviation"] = active_roster["position_abbreviation"].fillna(active_roster["position_abbreviation_profile"])
    active_roster = active_roster.drop(columns=["position_abbreviation_profile"], errors="ignore")
    active_roster["years_since_debut"] = (
        pd.to_datetime(active_roster["official_date"], errors="coerce")
        - pd.to_datetime(active_roster["mlb_debut_date"], errors="coerce")
    ).dt.days / 365.25
    return active_roster


def _build_transaction_frame(client: MLBStatsClient, upcoming_games: pd.DataFrame) -> pd.DataFrame:
    if upcoming_games.empty:
        return pd.DataFrame()

    team_rows = pd.concat(
        [
            upcoming_games[["away_team_id", "away_team_name", "official_date"]].rename(columns={"away_team_id": "team_id", "away_team_name": "team_name"}),
            upcoming_games[["home_team_id", "home_team_name", "official_date"]].rename(columns={"home_team_id": "team_id", "home_team_name": "team_name"}),
        ],
        ignore_index=True,
    )
    team_rows["official_date"] = pd.to_datetime(team_rows["official_date"], errors="coerce")
    window_start = (team_rows["official_date"].min() - timedelta(days=30)).date().isoformat()
    window_end = team_rows["official_date"].max().date().isoformat()

    transaction_frames: list[pd.DataFrame] = []
    for team_id, team_name in team_rows[["team_id", "team_name"]].drop_duplicates().itertuples(index=False):
        payload = client.get_transactions(start_date=window_start, end_date=window_end, team_id=int(team_id))
        frame = flatten_transactions(payload)
        if frame.empty:
            continue
        frame["team_id"] = int(team_id)
        frame["team_name"] = team_name
        transaction_frames.append(frame)
    return pd.concat(transaction_frames, ignore_index=True) if transaction_frames else pd.DataFrame()


def _build_upcoming_dataset(
    selected_date: str | None = None,
) -> tuple[pd.DataFrame, str | None, list[dict[str, Any]], dict[int, dict[str, list[dict[str, Any]]]]]:
    client = MLBStatsClient()
    full_schedule = _load_upcoming_schedule(client)
    available_dates = _build_available_dates(full_schedule)
    resolved_selected_date = _resolve_selected_date(available_dates, selected_date)
    if full_schedule.empty or not resolved_selected_date:
        return pd.DataFrame(), resolved_selected_date, available_dates, {}

    schedule = full_schedule.loc[
        pd.to_datetime(full_schedule["official_date"], errors="coerce").dt.strftime("%Y-%m-%d") == resolved_selected_date
    ].copy()
    if schedule.empty:
        return pd.DataFrame(), resolved_selected_date, available_dates, {}

    details, _, preview_player_profiles = _fetch_preview_bundles(client, schedule)
    season = int(pd.to_numeric(schedule["season"], errors="coerce").dropna().iloc[0]) if schedule["season"].notna().any() else None
    team_meta = _load_table_optional("teams_latest")
    if team_meta.empty:
        team_meta = _load_live_teams_table(client, season=season)
    pitcher_profiles = _load_table_optional("pitcher_profiles_latest")
    pitcher_profiles = _augment_pitcher_profiles(pitcher_profiles, preview_player_profiles, schedule)

    games = prepare_games(
        schedule,
        details,
        team_meta_df=team_meta,
        pitcher_profiles_df=pitcher_profiles,
        require_completed=False,
    )

    team_logs = _load_table_optional("team_game_logs_latest")
    starter_logs = _load_table_optional("starter_game_logs_latest")
    batter_logs = _load_table_optional("batter_game_logs_latest")
    reliever_logs = _load_table_optional("reliever_game_logs_latest")
    historical_lineups = _load_table_optional("lineup_roster_latest")

    games = attach_pregame_team_features(games, team_logs)
    games = attach_pregame_starter_features(games, starter_logs)

    active_roster = _build_active_roster_frame(client, games, preview_player_profiles)
    lineup_map: dict[int, dict[str, list[dict[str, Any]]]] = {}
    if not batter_logs.empty:
        projected_lineups = build_projected_lineup_roster(active_roster, batter_logs, historical_lineups=historical_lineups)
        lineup_features = build_lineup_feature_frame(projected_lineups, batter_logs)
        games = merge_lineup_features(games, lineup_features)
        lineup_map = _build_projected_lineup_map(projected_lineups, batter_logs)

    if not reliever_logs.empty:
        projected_bullpen = build_projected_bullpen_roster(
            active_roster,
            probable_pitchers=pd.concat(
                [
                    games[["game_pk", "away_team_id", "away_probable_pitcher_id"]]
                    .rename(columns={"away_team_id": "team_id", "away_probable_pitcher_id": "probable_pitcher_id"})
                    .assign(team_side="away"),
                    games[["game_pk", "home_team_id", "home_probable_pitcher_id"]]
                    .rename(columns={"home_team_id": "team_id", "home_probable_pitcher_id": "probable_pitcher_id"})
                    .assign(team_side="home"),
                ],
                ignore_index=True,
            ),
        )
        bullpen_features = build_bullpen_feature_frame(projected_bullpen, reliever_logs)
        games = merge_bullpen_features(games, bullpen_features)

    transactions = _build_transaction_frame(client, games)
    transaction_features = build_transaction_feature_frame(games, transactions)
    games = merge_transaction_features(games, transaction_features)

    games = add_matchup_differentials(games)
    statcast_paths = _resolve_statcast_paths(
        type("Args", (), {"statcast_path": [], "disable_statcast_cache": False})()
    )
    games = enrich_dataset_with_statcast(games, statcast_paths)
    games = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    return games, resolved_selected_date, available_dates, lineup_map


def _predict_upcoming(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        return _fallback_predict_upcoming(frame)

    try:
        estimator = joblib.load(MODEL_PATH)
        feature_columns = pd.read_csv(FEATURE_COLUMNS_PATH)["feature"].tolist()
    except Exception:
        return _fallback_predict_upcoming(frame)

    inference = frame.copy()
    for column in feature_columns:
        if column not in inference.columns:
            inference[column] = np.nan
    x = inference[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    try:
        probabilities = estimator.predict_proba(x)[:, 1]
    except Exception:
        return _fallback_predict_upcoming(frame)
    inference["home_win_probability"] = probabilities
    inference["away_win_probability"] = 1.0 - probabilities
    inference["prediction_source"] = "mlb_baseline_model"
    return inference


def _build_upcoming_boards(
    frame: pd.DataFrame,
    lineup_map: dict[int, dict[str, list[dict[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    boards: list[dict[str, Any]] = []
    lineup_map = lineup_map or {}
    for row in frame.itertuples(index=False):
        home_prob = float(row.home_win_probability)
        away_prob = float(row.away_win_probability)
        ranked = sorted(
            [
                {
                    "rank": 1,
                    "playerName": row.home_team_name,
                    "winProbability": home_prob * 100.0,
                    "side": "home",
                },
                {
                    "rank": 1,
                    "playerName": row.away_team_name,
                    "winProbability": away_prob * 100.0,
                    "side": "away",
                },
            ],
            key=lambda item: item["winProbability"],
            reverse=True,
        )
        for idx, item in enumerate(ranked, start=1):
            item["rank"] = idx

        scheduled = pd.to_datetime(row.official_date, errors="coerce")
        date_key = int(scheduled.strftime("%Y%m%d")) if pd.notna(scheduled) else None
        boards.append(
            {
                "id": f"mlb-{row.game_pk}",
                "name": f"{row.away_team_name} at {row.home_team_name}",
                "tour": "Baseball",
                "course": _optional_text(row.venue_name) or "Ballpark",
                "venue": _optional_text(row.venue_name) or "Ballpark",
                "scheduledDate": date_key,
                "latestDate": date_key,
                "predictedWinner": ranked[0]["playerName"],
                "awayTeam": row.away_team_name,
                "homeTeam": row.home_team_name,
                "awayStarter": _optional_text(row.away_probable_pitcher_name),
                "homeStarter": _optional_text(row.home_probable_pitcher_name),
                "awayStarterProfile": _build_starter_profile(row, "away"),
                "homeStarterProfile": _build_starter_profile(row, "home"),
                "awayTeamDetails": _build_team_details(row, "away"),
                "homeTeamDetails": _build_team_details(row, "home"),
                "awayStarterRadar": _build_starter_radar(row, "away"),
                "homeStarterRadar": _build_starter_radar(row, "home"),
                "awayAvailability": {
                    "ilAdds14": int(getattr(row, "away_availability_il_additions_last_14", 0) or 0),
                    "ilActivations14": int(getattr(row, "away_availability_il_activations_last_14", 0) or 0),
                    "rosterMoves14": int(getattr(row, "away_availability_transactions_last_14", 0) or 0),
                },
                "homeAvailability": {
                    "ilAdds14": int(getattr(row, "home_availability_il_additions_last_14", 0) or 0),
                    "ilActivations14": int(getattr(row, "home_availability_il_activations_last_14", 0) or 0),
                    "rosterMoves14": int(getattr(row, "home_availability_transactions_last_14", 0) or 0),
                },
                "projectedLineupContext": {
                    "awayCoverage": float(getattr(row, "away_lineup_history_coverage", np.nan))
                    if pd.notna(getattr(row, "away_lineup_history_coverage", np.nan))
                    else None,
                    "homeCoverage": float(getattr(row, "home_lineup_history_coverage", np.nan))
                    if pd.notna(getattr(row, "home_lineup_history_coverage", np.nan))
                    else None,
                    "awayContinuity": float(getattr(row, "away_lineup_prev_game_overlap", np.nan))
                    if pd.notna(getattr(row, "away_lineup_prev_game_overlap", np.nan))
                    else None,
                    "homeContinuity": float(getattr(row, "home_lineup_prev_game_overlap", np.nan))
                    if pd.notna(getattr(row, "home_lineup_prev_game_overlap", np.nan))
                    else None,
                },
                "predictionSource": _optional_text(getattr(row, "prediction_source", None)),
                "awayLineup": lineup_map.get(int(row.game_pk), {}).get("away", []),
                "homeLineup": lineup_map.get(int(row.game_pk), {}).get("home", []),
                "predictions": ranked,
            }
        )
    return boards


def build_live_upcoming_payload(selected_date: str | None = None) -> dict[str, Any]:
    dataset, resolved_selected_date, available_dates, lineup_map = _build_upcoming_dataset(selected_date=selected_date)
    dataset = _predict_upcoming(dataset)
    upcoming = _build_upcoming_boards(dataset, lineup_map)
    return {
        "selectedDate": resolved_selected_date,
        "availableDates": available_dates,
        "upcoming": upcoming,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "divination_live_mlb_feed",
    }


def export_mlb_frontend_data() -> None:
    historical_frame = _load_historical_source()
    backtests = _build_backtests(historical_frame)
    upcoming_payload = build_live_upcoming_payload()
    upcoming = upcoming_payload["upcoming"]

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DATA_DIR / "mlb_historical_backtests.json").write_text(json.dumps(backtests, indent=2))
    (FRONTEND_DATA_DIR / "mlb_upcoming_tournaments.json").write_text(json.dumps(upcoming, indent=2))

    print(f"Exported {len(backtests)} MLB historical boards")
    print(f"Exported {len(upcoming)} MLB upcoming boards")


if __name__ == "__main__":
    export_mlb_frontend_data()
