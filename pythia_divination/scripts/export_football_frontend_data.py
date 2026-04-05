from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports.football.branding import get_team_branding
from sports.football.build_training_dataset import DEFAULT_FOOTBALL_DATA_ROOT
from sports.football.feature_engineering import (
    add_matchup_differentials,
    attach_pregame_qb_features,
    attach_pregame_roster_features,
    attach_pregame_team_features,
    prepare_games,
)
from sports.pga.storage import read_preferred_table

if torch is not None:
    from sports.football.torch_model import FootballTorchModel

PROJ_ROOT = Path(__file__).resolve().parents[2]
FOOTBALL_DATA_ROOT = DEFAULT_FOOTBALL_DATA_ROOT
NORMALIZED_DIR = FOOTBALL_DATA_ROOT / "normalized"
FRONTEND_DATA_DIR = PROJ_ROOT / "pythia_prophecy" / "frontend" / "src" / "data"
BASELINE_ARTIFACTS_DIR = DIV_ROOT / "artifacts" / "football_baseline" / "nfl" / "hist_gradient_boosting" / "home_win"
TORCH_ARTIFACTS_DIR = DIV_ROOT / "artifacts" / "football_torch" / "nfl" / "home_win"
MAX_AVAILABLE_DATES = 7


def _to_datetime_mixed(values: object) -> pd.Series | pd.Timestamp:
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _load_table_optional(stem: str) -> pd.DataFrame:
    parquet_path = NORMALIZED_DIR / f"{stem}.parquet"
    csv_path = NORMALIZED_DIR / f"{stem}.csv"
    existing = [path for path in (parquet_path, csv_path) if path.exists()]
    if not existing:
        return pd.DataFrame()
    try:
        frame = read_preferred_table(parquet_path, csv_path)
    except (FileNotFoundError, ImportError):
        if not csv_path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(csv_path, low_memory=False)
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str)
    if "official_date" in frame.columns:
        frame["official_date"] = _to_datetime_mixed(frame["official_date"])
    return frame


def _date_key(value: object) -> str | None:
    timestamp = _to_datetime_mixed(value)
    if pd.isna(timestamp):
        return None
    return timestamp.normalize().strftime("%Y-%m-%d")


def _date_key_int(value: object) -> int | None:
    timestamp = _to_datetime_mixed(value)
    if pd.isna(timestamp):
        return None
    return int(timestamp.normalize().strftime("%Y%m%d"))


def _date_label(value: str) -> str:
    timestamp = _to_datetime_mixed(value)
    if pd.isna(timestamp):
        return value
    return timestamp.strftime("%a, %b %d").replace(" 0", " ")


def _build_available_dates(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    if schedule.empty:
        return []
    counts = (
        schedule.assign(date_key=schedule["official_date"].map(_date_key))
        .dropna(subset=["date_key"])
        .groupby("date_key")
        .size()
        .sort_index()
    )
    return [
        {"dateKey": str(date_key), "label": _date_label(str(date_key)), "gameCount": int(count)}
        for date_key, count in counts.head(MAX_AVAILABLE_DATES).items()
    ]


def _resolve_selected_date(available_dates: list[dict[str, Any]], selected_date: str | None) -> str | None:
    if not available_dates:
        return None
    valid = {row["dateKey"] for row in available_dates}
    if selected_date and selected_date in valid:
        return selected_date
    return available_dates[0]["dateKey"]


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = values.clip(-6.0, 6.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _load_baseline_predictor() -> dict[str, Any] | None:
    model_path = BASELINE_ARTIFACTS_DIR / "model.joblib"
    feature_columns_path = BASELINE_ARTIFACTS_DIR / "feature_columns.csv"
    if not model_path.exists() or not feature_columns_path.exists():
        return None
    return {
        "model": joblib.load(model_path),
        "feature_columns": pd.read_csv(feature_columns_path)["feature"].tolist(),
        "source": "nfl_baseline_model",
    }


def _load_torch_predictor() -> dict[str, Any] | None:
    if torch is None:
        return None
    model_path = TORCH_ARTIFACTS_DIR / "model.pt"
    imputer_path = TORCH_ARTIFACTS_DIR / "imputer.joblib"
    scaler_path = TORCH_ARTIFACTS_DIR / "scaler.joblib"
    if not all(path.exists() for path in (model_path, imputer_path, scaler_path)):
        return None
    checkpoint = torch.load(model_path, map_location="cpu")
    model = FootballTorchModel(
        input_dim=int(checkpoint["input_dim"]),
        hidden_width=int(checkpoint["hidden_width"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "feature_columns": list(checkpoint["feature_columns"]),
        "imputer": joblib.load(imputer_path),
        "scaler": joblib.load(scaler_path),
        "source": "nfl_torch_model",
    }


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _best_predictor() -> dict[str, Any] | None:
    baseline = _load_baseline_predictor()
    torch_predictor = _load_torch_predictor()
    if baseline is None:
        return torch_predictor
    if torch_predictor is None:
        return baseline
    baseline_auc = float(_load_metrics(BASELINE_ARTIFACTS_DIR / "metrics.json").get("roc_auc") or 0.0)
    torch_auc = float(_load_metrics(TORCH_ARTIFACTS_DIR / "metrics.json").get("roc_auc") or 0.0)
    return torch_predictor if torch_auc >= baseline_auc else baseline


def _heuristic_probabilities(frame: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=frame.index, dtype=float)
    score += pd.to_numeric(frame.get("team_diff_win_pct_prior", 0.0), errors="coerce").fillna(0.0) * 1.6
    score += pd.to_numeric(frame.get("team_diff_point_diff_avg_last_5", 0.0), errors="coerce").fillna(0.0) * 0.12
    score += pd.to_numeric(frame.get("team_diff_passing_epa_avg_last_5", 0.0), errors="coerce").fillna(0.0) * 1.8
    score += pd.to_numeric(frame.get("qb_diff_passing_epa_avg_last_5", 0.0), errors="coerce").fillna(0.0) * 1.5
    score += pd.to_numeric(frame.get("qb_diff_passing_yards_avg_last_5", 0.0), errors="coerce").fillna(0.0) * 0.004
    score += pd.to_numeric(frame.get("roster_diff_availability_score", 0.0), errors="coerce").fillna(0.0) * 0.9
    score -= pd.to_numeric(frame.get("roster_diff_unavailable_skill_count", 0.0), errors="coerce").fillna(0.0) * 0.18
    score -= pd.to_numeric(frame.get("roster_diff_unavailable_offensive_line_count", 0.0), errors="coerce").fillna(0.0) * 0.14
    score += pd.to_numeric(frame.get("home_is_favorite", 0.0), errors="coerce").fillna(0.0) * 0.15
    return _sigmoid(score)


def _predict_games(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    predictor = _best_predictor()
    if predictor is None:
        output["home_win_probability"] = _heuristic_probabilities(output)
        output["prediction_source"] = "nfl_heuristic_fallback"
        return output

    feature_columns = [column for column in predictor["feature_columns"] if column in output.columns]
    features = output[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if predictor["source"] == "nfl_torch_model":
        imputer = predictor["imputer"]
        scaler = predictor["scaler"]
        transformed = scaler.transform(imputer.transform(features)).astype(np.float32)
        with torch.no_grad():
            tensor = torch.from_numpy(transformed)
            probabilities = torch.sigmoid(predictor["model"](tensor)).detach().cpu().numpy()
    else:
        probabilities = predictor["model"].predict_proba(features)[:, 1]
    output["home_win_probability"] = probabilities
    output["prediction_source"] = predictor["source"]
    return output


def _build_player_trends(player_week_stats: pd.DataFrame) -> pd.DataFrame:
    frame = player_week_stats.copy()
    if frame.empty:
        return frame
    frame = frame.loc[frame["season_type"] == "REG"].copy()
    frame = frame.loc[frame["position"].fillna("").isin(["QB", "RB", "WR", "TE", "FB"])].copy()
    if frame.empty:
        return frame

    numeric_columns = [
        "season", "week", "attempts", "passing_yards", "passing_tds", "passing_interceptions", "passing_epa", "passing_cpoe",
        "carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_epa",
        "fantasy_points",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["player_id"] = frame["player_id"].astype(str)
    frame["sort_key"] = frame["season"].fillna(0).astype(int) * 100 + frame["week"].fillna(0).astype(int)
    frame["scrimmage_yards"] = frame.get("rushing_yards", 0).fillna(0) + frame.get("receiving_yards", 0).fillna(0)
    frame["touch_volume"] = frame.get("carries", 0).fillna(0) + frame.get("targets", 0).fillna(0)
    frame = frame.sort_values(["player_id", "sort_key", "game_id"]).reset_index(drop=True)

    rolling_metrics = [
        "attempts", "passing_yards", "passing_tds", "passing_interceptions", "passing_epa", "passing_cpoe",
        "carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_epa",
        "fantasy_points", "scrimmage_yards", "touch_volume",
    ]
    windows = (3, 5)
    for metric in rolling_metrics:
        grouped = frame.groupby("player_id")[metric]
        for window in windows:
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )
    return frame


def _build_roster_history_map(weekly_rosters: pd.DataFrame, player_trends: pd.DataFrame) -> dict[tuple[int, int, str], list[dict[str, Any]]]:
    if weekly_rosters.empty:
        return {}

    roster = weekly_rosters.copy()
    roster = roster.loc[roster["game_type"] == "REG"].copy()
    roster = roster.loc[roster["position"].fillna("").isin(["QB", "RB", "WR", "TE", "FB"])].copy()
    if roster.empty:
        return {}

    roster["season"] = pd.to_numeric(roster["season"], errors="coerce")
    roster["week"] = pd.to_numeric(roster["week"], errors="coerce")
    roster["years_exp"] = pd.to_numeric(roster.get("years_exp"), errors="coerce")
    roster["player_id"] = roster["gsis_id"].astype(str)
    roster["player_name"] = roster["full_name"].fillna("")
    roster["sort_key"] = roster["season"].fillna(0).astype(int) * 100 + roster["week"].fillna(0).astype(int)
    roster = roster.sort_values(["player_id", "sort_key", "team"]).reset_index(drop=True)

    if not player_trends.empty:
        history = player_trends.sort_values(["player_id", "sort_key", "game_id"]).copy()
        keep_columns = [
            "player_id", "sort_key", "player_name", "headshot_url", "position", "team",
            "passing_yards_avg_last_3", "passing_tds_avg_last_3", "passing_epa_avg_last_3", "passing_cpoe_avg_last_3",
            "passing_interceptions_avg_last_3", "rushing_yards_avg_last_3", "rushing_yards_avg_last_5",
            "rushing_tds_avg_last_5", "targets_avg_last_5", "receiving_yards_avg_last_5", "receiving_tds_avg_last_5",
            "scrimmage_yards_avg_last_5", "touch_volume_avg_last_5", "fantasy_points_avg_last_5",
        ]
        history = history[[column for column in keep_columns if column in history.columns]]
        roster = pd.merge_asof(
            roster,
            history,
            on="sort_key",
            by="player_id",
            direction="backward",
            allow_exact_matches=False,
            suffixes=("", "_history"),
        )

    priority_map = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FB": 4}
    lineup_map: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for (season, week, team), group in roster.groupby(["season", "week", "team"], sort=False):
        active = group.loc[group["status"].fillna("").eq("ACT")].copy()
        if active.empty:
            lineup_map[(int(season), int(week), str(team))] = []
            continue
        active["priority"] = active["position"].map(lambda value: priority_map.get(str(value), 99))
        active["recent_value"] = pd.to_numeric(active.get("fantasy_points_avg_last_5"), errors="coerce").fillna(0.0)
        active = active.sort_values(["priority", "recent_value", "years_exp"], ascending=[True, False, False])

        selections: list[pd.Series] = []
        for target_position in ["QB", "RB", "WR", "WR", "TE"]:
            available = active.loc[active["position"] == target_position]
            for candidate in available.itertuples(index=False):
                if all(str(getattr(candidate, 'player_id')) != str(getattr(existing, 'player_id')) for existing in selections):
                    selections.append(candidate)  # type: ignore[arg-type]
                    break
        if not selections:
            selections = list(active.head(5).itertuples(index=False))
        entries = [_build_live_lineup_entry(index + 1, player) for index, player in enumerate(selections[:5])]
        lineup_map[(int(season), int(week), str(team))] = entries
    return lineup_map


def _value(row: Any, field: str) -> float | None:
    return _safe_float(getattr(row, field, np.nan))


def _normalize_metric(value: float | None, low: float, high: float, *, inverse: bool = False) -> float:
    if value is None or high <= low:
        return 0.0
    clipped = min(max(value, low), high)
    ratio = (clipped - low) / (high - low)
    if inverse:
        ratio = 1.0 - ratio
    return round(ratio * 100.0, 1)


def _build_live_qb_radar(row: Any, side: str) -> list[dict[str, float]]:
    return [
        {"label": "Passing", "value": _normalize_metric(_value(row, f"{side}_qb_passing_yards_avg_last_3"), 120.0, 340.0)},
        {"label": "Efficiency", "value": _normalize_metric(_value(row, f"{side}_qb_passing_epa_avg_last_3"), -6.0, 15.0)},
        {"label": "Accuracy", "value": _normalize_metric(_value(row, f"{side}_qb_passing_cpoe_avg_last_3"), -10.0, 12.0)},
        {"label": "Ball Security", "value": _normalize_metric(_value(row, f"{side}_qb_passing_interceptions_avg_last_3"), 0.0, 2.0, inverse=True)},
        {"label": "Rush", "value": _normalize_metric(_value(row, f"{side}_qb_rushing_yards_avg_last_3"), 0.0, 50.0)},
        {"label": "Form", "value": _normalize_metric(_value(row, f"{side}_qb_fantasy_points_avg_last_5"), 8.0, 30.0)},
    ]


def _build_historical_qb_radar(player_row: pd.Series) -> list[dict[str, float]]:
    return [
        {"label": "Passing", "value": _normalize_metric(_safe_float(player_row.get("passing_yards")), 120.0, 400.0)},
        {"label": "Efficiency", "value": _normalize_metric(_safe_float(player_row.get("passing_epa")), -8.0, 20.0)},
        {"label": "Accuracy", "value": _normalize_metric(_safe_float(player_row.get("passing_cpoe")), -10.0, 15.0)},
        {"label": "Ball Security", "value": _normalize_metric(_safe_float(player_row.get("passing_interceptions")), 0.0, 3.0, inverse=True)},
        {"label": "Rush", "value": _normalize_metric(_safe_float(player_row.get("rushing_yards")), 0.0, 80.0)},
        {"label": "Impact", "value": _normalize_metric(_safe_float(player_row.get("fantasy_points")), 8.0, 35.0)},
    ]


def _build_live_qb_profile(row: Any, side: str) -> dict[str, Any]:
    player_id = _optional_text(getattr(row, f"{side}_qb_id", None))
    qb_name = _optional_text(getattr(row, f"{side}_qb_name", None))
    return {
        "imageUrl": None,
        "subtitle": "Starting quarterback" if qb_name else "Quarterback pending",
        "country": None,
        "stats": [
            {"label": "Pass Yds L3", "value": f"{(_value(row, f'{side}_qb_passing_yards_avg_last_3') or 0):.0f}"},
            {"label": "TD L3", "value": f"{(_value(row, f'{side}_qb_passing_tds_avg_last_3') or 0):.1f}"},
            {"label": "EPA L3", "value": f"{(_value(row, f'{side}_qb_passing_epa_avg_last_3') or 0):+.1f}"},
            {"label": "Rush Yds L3", "value": f"{(_value(row, f'{side}_qb_rushing_yards_avg_last_3') or 0):.0f}"},
        ],
    }


def _build_live_lineup_entry(slot: int, player: Any) -> dict[str, Any]:
    position = _optional_text(getattr(player, "position", None))
    player_name = _optional_text(getattr(player, "player_name", None)) or _optional_text(getattr(player, "full_name", None)) or "Player"
    fantasy = _safe_float(getattr(player, "fantasy_points_avg_last_5", None))
    pass_yards = _safe_float(getattr(player, "passing_yards_avg_last_3", None))
    pass_tds = _safe_float(getattr(player, "passing_tds_avg_last_3", None))
    rush_yards = _safe_float(getattr(player, "rushing_yards_avg_last_5", None))
    rec_yards = _safe_float(getattr(player, "receiving_yards_avg_last_5", None))
    targets = _safe_float(getattr(player, "targets_avg_last_5", None))
    touches = _safe_float(getattr(player, "touch_volume_avg_last_5", None))
    scrimmage = _safe_float(getattr(player, "scrimmage_yards_avg_last_5", None))

    if position == "QB":
        stats = [
            {"label": "Pass Yds L3", "value": f"{(pass_yards or 0):.0f}"},
            {"label": "TD L3", "value": f"{(pass_tds or 0):.1f}"},
            {"label": "Rush Yds L3", "value": f"{(_safe_float(getattr(player, 'rushing_yards_avg_last_3', None)) or 0):.0f}"},
            {"label": "FPTS L5", "value": f"{(fantasy or 0):.1f}"},
        ]
        summary = f"{(pass_yards or 0):.0f} pass yds | {(pass_tds or 0):.1f} TD"
        radar = [
            {"label": "Passing", "value": _normalize_metric(pass_yards, 120.0, 340.0)},
            {"label": "Efficiency", "value": _normalize_metric(_safe_float(getattr(player, 'passing_epa_avg_last_3', None)), -6.0, 15.0)},
            {"label": "Accuracy", "value": _normalize_metric(_safe_float(getattr(player, 'passing_cpoe_avg_last_3', None)), -10.0, 12.0)},
            {"label": "Ball Security", "value": _normalize_metric(_safe_float(getattr(player, 'passing_interceptions_avg_last_3', None)), 0.0, 2.0, inverse=True)},
            {"label": "Rush", "value": _normalize_metric(_safe_float(getattr(player, 'rushing_yards_avg_last_3', None)), 0.0, 50.0)},
            {"label": "Form", "value": _normalize_metric(fantasy, 8.0, 30.0)},
        ]
    else:
        stats = [
            {"label": "Scrim Yds L5", "value": f"{(scrimmage or 0):.0f}"},
            {"label": "Touches L5", "value": f"{(touches or 0):.1f}"},
            {"label": "Targets L5", "value": f"{(targets or 0):.1f}"},
            {"label": "FPTS L5", "value": f"{(fantasy or 0):.1f}"},
        ]
        summary = f"{(scrimmage or (rush_yards or 0) + (rec_yards or 0)):.0f} yds | {(fantasy or 0):.1f} FPTS"
        radar = [
            {"label": "Usage", "value": _normalize_metric(touches, 1.0, 22.0)},
            {"label": "Yards", "value": _normalize_metric(scrimmage, 10.0, 140.0)},
            {"label": "Receiving", "value": _normalize_metric(rec_yards, 0.0, 100.0)},
            {"label": "Rushing", "value": _normalize_metric(rush_yards, 0.0, 100.0)},
            {"label": "Scoring", "value": _normalize_metric(_safe_float(getattr(player, 'receiving_tds_avg_last_5', None)) or _safe_float(getattr(player, 'rushing_tds_avg_last_5', None)), 0.0, 1.2)},
            {"label": "Form", "value": _normalize_metric(fantasy, 3.0, 25.0)},
        ]

    return {
        "playerId": None,
        "playerName": player_name,
        "lineupSlot": slot,
        "position": position,
        "performanceSummary": summary,
        "profile": {
            "imageUrl": _optional_text(getattr(player, "headshot_url", None)),
            "subtitle": f"{position} | {int((_safe_float(getattr(player, 'years_exp', None)) or 0))} yrs exp" if position else "Key player",
            "country": None,
            "stats": stats,
        },
        "radarMetrics": radar,
    }


def _build_historical_lineup_entry(slot: int, player_row: pd.Series) -> dict[str, Any]:
    position = _optional_text(player_row.get("position"))
    player_name = _optional_text(player_row.get("player_name")) or "Player"
    passing_yards = _safe_float(player_row.get("passing_yards"))
    passing_tds = _safe_float(player_row.get("passing_tds"))
    rushing_yards = _safe_float(player_row.get("rushing_yards"))
    receiving_yards = _safe_float(player_row.get("receiving_yards"))
    touchdowns = (_safe_float(player_row.get("rushing_tds")) or 0) + (_safe_float(player_row.get("receiving_tds")) or 0)
    fantasy = _safe_float(player_row.get("fantasy_points"))

    if position == "QB":
        stats = [
            {"label": "Pass Yds", "value": f"{(passing_yards or 0):.0f}"},
            {"label": "TD", "value": f"{(passing_tds or 0):.0f}"},
            {"label": "Rush Yds", "value": f"{(rushing_yards or 0):.0f}"},
            {"label": "FPTS", "value": f"{(fantasy or 0):.1f}"},
        ]
        summary = f"{(passing_yards or 0):.0f} pass yds | {(passing_tds or 0):.0f} TD"
        radar = _build_historical_qb_radar(player_row)
    else:
        stats = [
            {"label": "Scrim Yds", "value": f"{((rushing_yards or 0) + (receiving_yards or 0)):.0f}"},
            {"label": "TD", "value": f"{touchdowns:.0f}"},
            {"label": "Targets", "value": f"{(_safe_float(player_row.get('targets')) or 0):.0f}"},
            {"label": "FPTS", "value": f"{(fantasy or 0):.1f}"},
        ]
        summary = f"{((rushing_yards or 0) + (receiving_yards or 0)):.0f} yds | {touchdowns:.0f} TD"
        radar = [
            {"label": "Usage", "value": _normalize_metric((_safe_float(player_row.get('carries')) or 0) + (_safe_float(player_row.get('targets')) or 0), 1.0, 25.0)},
            {"label": "Yards", "value": _normalize_metric((rushing_yards or 0) + (receiving_yards or 0), 10.0, 180.0)},
            {"label": "Receiving", "value": _normalize_metric(receiving_yards, 0.0, 120.0)},
            {"label": "Rushing", "value": _normalize_metric(rushing_yards, 0.0, 120.0)},
            {"label": "Scoring", "value": _normalize_metric(touchdowns, 0.0, 3.0)},
            {"label": "Impact", "value": _normalize_metric(fantasy, 3.0, 35.0)},
        ]

    return {
        "playerId": None,
        "playerName": player_name,
        "lineupSlot": slot,
        "position": position,
        "performanceSummary": summary,
        "profile": {
            "imageUrl": _optional_text(player_row.get("headshot_url")),
            "subtitle": "Recorded game line",
            "country": None,
            "stats": stats,
        },
        "radarMetrics": radar,
    }


def _build_live_team_details(row: Any, side: str) -> dict[str, Any]:
    branding = get_team_branding(getattr(row, f"{side}_team", None))
    games_played = _value(row, f"{side}_team_games_played_prior")
    win_pct = _value(row, f"{side}_team_win_pct_prior")
    if games_played is not None and win_pct is not None and games_played > 0:
        wins = int(round(games_played * win_pct))
        losses = max(int(round(games_played)) - wins, 0)
        record = f"{wins}-{losses}"
    else:
        record = None

    recent_form_parts: list[str] = []
    win_last_5 = _value(row, f"{side}_team_won_avg_last_5")
    if win_last_5 is not None:
        wins_last_5 = max(0, min(5, int(round(win_last_5 * 5))))
        recent_form_parts.append(f"{wins_last_5}-{5 - wins_last_5} last 5")
    passing_epa = _value(row, f"{side}_team_passing_epa_avg_last_5")
    if passing_epa is not None:
        recent_form_parts.append(f"pass EPA {passing_epa:+.1f}")
    point_diff = _value(row, f"{side}_team_point_diff_avg_last_5")
    if point_diff is not None:
        recent_form_parts.append(f"{point_diff:+.1f} diff")

    inactive_count = _safe_int(getattr(row, f"{side}_roster_inactive_count", None)) or 0
    reserve_count = _safe_int(getattr(row, f"{side}_roster_reserve_count", None)) or 0
    skill_absences = _safe_int(getattr(row, f"{side}_roster_unavailable_skill_count", None)) or 0
    offense_summary = (
        f"QB EPA {(_value(row, f'{side}_qb_passing_epa_avg_last_5') or 0):+.1f}"
        f" | Rush {(_value(row, f'{side}_team_rushing_yards_avg_last_5') or 0):.0f}/g"
    )
    depth_summary = (
        f"skill ready {(_safe_int(getattr(row, f'{side}_roster_available_skill_count', None)) or 0)}"
        f" | OL ready {(_safe_int(getattr(row, f'{side}_roster_available_offensive_line_count', None)) or 0)}"
    )
    weather = []
    temp = _value(row, "weather_temp")
    wind = _value(row, "weather_wind")
    if temp is not None:
        weather.append(f"{temp:.0f} F")
    if wind is not None:
        weather.append(f"{wind:.0f} mph wind")
    roof = _optional_text(getattr(row, 'roof', None))
    if roof:
        weather.append(roof.title())

    return {
        "abbreviation": branding.get("abbreviation"),
        "logoUrl": branding.get("logoUrl"),
        "primaryColor": branding.get("primaryColor"),
        "secondaryColor": branding.get("secondaryColor"),
        "recordPrior": record,
        "recentForm": " | ".join(recent_form_parts) or None,
        "bullpenSummary": offense_summary,
        "availabilitySummary": f"{inactive_count} inactive | {reserve_count} reserve | {skill_absences} skill absences",
        "lineupContinuity": depth_summary,
        "venue": _optional_text(getattr(row, "stadium", None)),
        "weather": " | ".join(weather) or None,
    }


def _historical_featured_players(player_week: pd.DataFrame, game_id: str, team: str, qb_id: str | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, list[dict[str, float]] | None]:
    frame = player_week.loc[(player_week["game_id"].astype(str) == str(game_id)) & (player_week["team"] == team)].copy()
    frame = frame.loc[frame["position"].fillna("").isin(["QB", "RB", "WR", "TE", "FB"])].copy()
    if frame.empty:
        return [], None, None, None
    frame["priority"] = frame["position"].map({"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FB": 4}).fillna(9)
    frame["impact"] = pd.to_numeric(frame.get("fantasy_points"), errors="coerce").fillna(0.0)
    frame = frame.sort_values(["priority", "impact", "targets", "carries"], ascending=[True, False, False, False])
    lineups = [_build_historical_lineup_entry(index + 1, row) for index, (_, row) in enumerate(frame.head(5).iterrows())]
    qb_row = frame.loc[frame["position"] == "QB"]
    if qb_id:
        qb_row = frame.loc[frame["player_id"].astype(str) == str(qb_id)] if (frame["player_id"].astype(str) == str(qb_id)).any() else qb_row
    if qb_row.empty:
        qb_series = None
    else:
        qb_series = qb_row.iloc[0]
    if qb_series is None:
        return lineups, lineups[0] if lineups else None, None, None
    starter_profile = {
        "imageUrl": _optional_text(qb_series.get("headshot_url")),
        "subtitle": "Starting quarterback",
        "country": None,
        "stats": [
            {"label": "Pass Yds", "value": f"{(_safe_float(qb_series.get('passing_yards')) or 0):.0f}"},
            {"label": "TD", "value": f"{(_safe_float(qb_series.get('passing_tds')) or 0):.0f}"},
            {"label": "INT", "value": f"{(_safe_float(qb_series.get('passing_interceptions')) or 0):.0f}"},
            {"label": "Rush Yds", "value": f"{(_safe_float(qb_series.get('rushing_yards')) or 0):.0f}"},
        ],
    }
    starter_radar = _build_historical_qb_radar(qb_series)
    featured = next((entry for entry in lineups if entry.get("position") == "QB"), lineups[0] if lineups else None)
    return lineups, featured, starter_profile, starter_radar


def _build_live_predictions(row: Any) -> list[dict[str, Any]]:
    home_prob = (_value(row, "home_win_probability") or 0.5) * 100.0
    away_prob = 100.0 - home_prob
    away_profile = {
        "subtitle": _build_live_team_details(row, "away").get("recentForm"),
        "stats": [
            {"label": "Win %", "value": f"{((_value(row, 'away_team_win_pct_prior') or 0) * 100):.0f}%"},
            {"label": "Pass EPA L5", "value": f"{(_value(row, 'away_team_passing_epa_avg_last_5') or 0):+.1f}"},
            {"label": "QB EPA L5", "value": f"{(_value(row, 'away_qb_passing_epa_avg_last_5') or 0):+.1f}"},
        ],
    }
    home_profile = {
        "subtitle": _build_live_team_details(row, "home").get("recentForm"),
        "stats": [
            {"label": "Win %", "value": f"{((_value(row, 'home_team_win_pct_prior') or 0) * 100):.0f}%"},
            {"label": "Pass EPA L5", "value": f"{(_value(row, 'home_team_passing_epa_avg_last_5') or 0):+.1f}"},
            {"label": "QB EPA L5", "value": f"{(_value(row, 'home_qb_passing_epa_avg_last_5') or 0):+.1f}"},
        ],
    }
    return [
        {"rank": 1 if away_prob >= home_prob else 2, "playerName": getattr(row, "away_team"), "winProbability": away_prob, "side": "away", "profile": away_profile},
        {"rank": 1 if home_prob > away_prob else 2, "playerName": getattr(row, "home_team"), "winProbability": home_prob, "side": "home", "profile": home_profile},
    ]


def build_live_upcoming_payload(selected_date: str | None = None) -> dict[str, Any]:
    games_df = _load_table_optional("football_games_latest")
    team_logs = _load_table_optional("football_team_game_logs_latest")
    qb_logs = _load_table_optional("football_qb_week_logs_latest")
    roster_summaries = _load_table_optional("football_roster_week_summaries_latest")
    player_week = _load_table_optional("football_player_week_stats_latest")
    weekly_rosters = _load_table_optional("football_weekly_rosters_latest")

    full_schedule = prepare_games(games_df, require_completed=False)
    if not full_schedule.empty:
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        full_schedule = full_schedule.loc[full_schedule["official_date"] >= today].copy()
        full_schedule = full_schedule.sort_values(["official_date", "season", "week", "game_id"]).reset_index(drop=True)

    available_dates = _build_available_dates(full_schedule)
    resolved_selected_date = _resolve_selected_date(available_dates, selected_date)
    if full_schedule.empty or not resolved_selected_date:
        return {
            "selectedDate": resolved_selected_date,
            "availableDates": available_dates,
            "upcoming": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "divination_live_football_feed",
        }

    schedule = full_schedule.loc[full_schedule["official_date"].map(_date_key) == resolved_selected_date].copy()
    games = attach_pregame_team_features(schedule, team_logs)
    games = attach_pregame_qb_features(games, qb_logs)
    games = attach_pregame_roster_features(games, roster_summaries)
    games = add_matchup_differentials(games)
    scored_games = _predict_games(games)

    player_trends = _build_player_trends(player_week)
    roster_history_map = _build_roster_history_map(weekly_rosters, player_trends)

    boards: list[dict[str, Any]] = []
    for row in scored_games.itertuples(index=False):
        scheduled_date = _date_key_int(getattr(row, "official_date", None))
        away_lineup = roster_history_map.get((int(getattr(row, "season")), int(getattr(row, "week")), str(getattr(row, "away_team"))), [])
        home_lineup = roster_history_map.get((int(getattr(row, "season")), int(getattr(row, "week")), str(getattr(row, "home_team"))), [])
        away_featured = next((entry for entry in away_lineup if entry.get("position") == "QB"), away_lineup[0] if away_lineup else None)
        home_featured = next((entry for entry in home_lineup if entry.get("position") == "QB"), home_lineup[0] if home_lineup else None)
        home_prob = (_value(row, "home_win_probability") or 0.5) * 100.0
        away_prob = 100.0 - home_prob
        predicted_winner = getattr(row, "home_team") if home_prob >= away_prob else getattr(row, "away_team")
        away_availability = {
            "ilAdds14": _safe_int(getattr(row, "away_roster_inactive_count", None)),
            "ilActivations14": _safe_int(getattr(row, "away_roster_available_skill_count", None)),
            "rosterMoves14": _safe_int(getattr(row, "away_roster_reserve_count", None)),
        }
        home_availability = {
            "ilAdds14": _safe_int(getattr(row, "home_roster_inactive_count", None)),
            "ilActivations14": _safe_int(getattr(row, "home_roster_available_skill_count", None)),
            "rosterMoves14": _safe_int(getattr(row, "home_roster_reserve_count", None)),
        }
        boards.append(
            {
                "id": f"football-{getattr(row, 'game_id')}",
                "name": f"{getattr(row, 'away_team')} at {getattr(row, 'home_team')}",
                "tour": "Football",
                "course": _optional_text(getattr(row, "stadium", None)) or "Stadium",
                "venue": _optional_text(getattr(row, "stadium", None)) or "Stadium",
                "scheduledDate": scheduled_date,
                "latestDate": scheduled_date,
                "predictedWinner": predicted_winner,
                "awayTeam": getattr(row, "away_team"),
                "homeTeam": getattr(row, "home_team"),
                "awayStarter": _optional_text(getattr(row, "away_qb_name", None)),
                "homeStarter": _optional_text(getattr(row, "home_qb_name", None)),
                "awayStarterProfile": _build_live_qb_profile(row, "away"),
                "homeStarterProfile": _build_live_qb_profile(row, "home"),
                "awayStarterRadar": _build_live_qb_radar(row, "away"),
                "homeStarterRadar": _build_live_qb_radar(row, "home"),
                "awayTeamDetails": _build_live_team_details(row, "away"),
                "homeTeamDetails": _build_live_team_details(row, "home"),
                "awayAvailability": away_availability,
                "homeAvailability": home_availability,
                "predictionSource": _optional_text(getattr(row, "prediction_source", None)),
                "awayLineup": away_lineup,
                "homeLineup": home_lineup,
                "awayFeaturedPlayer": away_featured,
                "homeFeaturedPlayer": home_featured,
                "predictions": _build_live_predictions(row),
            }
        )

    return {
        "selectedDate": resolved_selected_date,
        "availableDates": available_dates,
        "upcoming": boards,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "divination_live_football_feed",
    }


def _historical_boards() -> list[dict[str, Any]]:
    predictions_paths = [
        TORCH_ARTIFACTS_DIR / "validation_predictions.csv",
        BASELINE_ARTIFACTS_DIR / "validation_predictions.csv",
    ]
    predictions_path = next((path for path in predictions_paths if path.exists()), None)
    if predictions_path is None:
        return []

    predictions = pd.read_csv(predictions_path)
    predictions["game_id"] = predictions["game_id"].astype(str)
    predictions["official_date"] = _to_datetime_mixed(predictions["official_date"])
    dataset = _load_table_optional("football_training_dataset_latest")
    if dataset.empty:
        return []
    dataset["official_date"] = _to_datetime_mixed(dataset["official_date"])
    dataset = dataset.drop_duplicates(subset=["game_id"])
    merged = predictions.merge(
        dataset,
        on=["game_id", "official_date", "season", "week", "away_team", "home_team"],
        how="left",
    )
    player_week = _load_table_optional("football_player_week_stats_latest")
    if not player_week.empty:
        player_week["game_id"] = player_week["game_id"].astype(str)

    boards: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        home_prob = (_safe_float(getattr(row, "home_win_probability", None)) or 0.5)
        away_prob = 1.0 - home_prob
        predicted_winner = getattr(row, "home_team") if home_prob >= away_prob else getattr(row, "away_team")
        actual_winner = getattr(row, "home_team") if int(getattr(row, "home_win", 0) or 0) == 1 else getattr(row, "away_team")
        away_lineup, away_featured, away_starter_profile, away_starter_radar = _historical_featured_players(player_week, str(getattr(row, "game_id")), str(getattr(row, "away_team")), _optional_text(getattr(row, "away_qb_id", None)))
        home_lineup, home_featured, home_starter_profile, home_starter_radar = _historical_featured_players(player_week, str(getattr(row, "game_id")), str(getattr(row, "home_team")), _optional_text(getattr(row, "home_qb_id", None)))
        boards.append(
            {
                "year": int(_to_datetime_mixed(getattr(row, "official_date")).year),
                "tournament": f"{getattr(row, 'away_team')} at {getattr(row, 'home_team')}",
                "tour": "Football",
                "hitStatus": "Top Pick" if predicted_winner == actual_winner else "Miss",
                "predictedWinner": predicted_winner,
                "actualWinner": actual_winner,
                "prob": max(home_prob, away_prob),
                "venue": _optional_text(getattr(row, "stadium", None)) or "Stadium",
                "course": _optional_text(getattr(row, "stadium", None)) or "Stadium",
                "latestDate": _date_key_int(getattr(row, "official_date", None)),
                "scheduledDate": _date_key_int(getattr(row, "official_date", None)),
                "awayTeam": getattr(row, "away_team"),
                "homeTeam": getattr(row, "home_team"),
                "awayStarter": _optional_text(getattr(row, "away_qb_name", None)),
                "homeStarter": _optional_text(getattr(row, "home_qb_name", None)),
                "awayStarterProfile": away_starter_profile,
                "homeStarterProfile": home_starter_profile,
                "awayStarterRadar": away_starter_radar,
                "homeStarterRadar": home_starter_radar,
                "awayTeamDetails": _build_live_team_details(row, "away"),
                "homeTeamDetails": _build_live_team_details(row, "home"),
                "awayLineup": away_lineup,
                "homeLineup": home_lineup,
                "awayFeaturedPlayer": away_featured,
                "homeFeaturedPlayer": home_featured,
            }
        )

    return sorted(boards, key=lambda item: (item.get("latestDate") or 0, item.get("tournament") or ""), reverse=True)


def export_football_frontend_data() -> None:
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    historical = _historical_boards()
    upcoming_payload = build_live_upcoming_payload()
    (FRONTEND_DATA_DIR / "football_historical_backtests.json").write_text(json.dumps(historical, indent=2))
    (FRONTEND_DATA_DIR / "football_upcoming_tournaments.json").write_text(json.dumps(upcoming_payload["upcoming"], indent=2))
    print(f"Exported {len(historical)} Football historical boards")
    print(f"Exported {len(upcoming_payload['upcoming'])} Football upcoming boards")


if __name__ == "__main__":
    export_football_frontend_data()
