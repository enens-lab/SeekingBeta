from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - torch is optional at runtime
    torch = None  # type: ignore[assignment]

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports.basketball.build_training_dataset import DEFAULT_BASKETBALL_DATA_ROOT
from sports.basketball.client import BasketballStatsClient, flatten_schedule
from sports.basketball.constants import LEAGUE_CONFIGS
from sports.basketball.feature_engineering import (
    add_matchup_differentials,
    attach_pregame_team_features,
    prepare_games,
)
from sports.basketball.rotation_features import (
    attach_expected_rotation_features,
    build_projected_rotation_map,
)
if torch is not None:
    from sports.basketball.torch_model import BasketballTorchModel
from sports.pga.storage import read_preferred_table

PROJ_ROOT = Path(__file__).resolve().parents[2]
BASKETBALL_DATA_ROOT = DEFAULT_BASKETBALL_DATA_ROOT
NORMALIZED_DIR = BASKETBALL_DATA_ROOT / "normalized"
FRONTEND_DATA_DIR = PROJ_ROOT / "pythia_prophecy" / "frontend" / "src" / "data"
ARTIFACTS_ROOT = DIV_ROOT / "artifacts" / "basketball_baseline"
TORCH_ARTIFACTS_ROOT = DIV_ROOT / "artifacts" / "basketball_torch"
UPCOMING_LOOKAHEAD_DAYS = 7
MAX_AVAILABLE_DATES = 7
logger = logging.getLogger(__name__)
_TABLE_CACHE: dict[tuple[str, tuple[str, ...]], tuple[tuple[tuple[str, float], ...], pd.DataFrame]] = {}
_PREDICTOR_CACHE: dict[tuple[str, str], tuple[tuple[tuple[str, float], ...], dict[str, Any]]] = {}


def _to_datetime_mixed(values: object) -> pd.Series | pd.Timestamp:
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _display_tour(league: str) -> str:
    return "Women's Basketball" if league == "wnba" else "Basketball"


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _paths_signature(paths: list[Path]) -> tuple[tuple[str, float], ...]:
    return tuple((str(path), path.stat().st_mtime if path.exists() else -1.0) for path in paths)


def _load_table_optional(*candidates: Path) -> pd.DataFrame:
    existing = [candidate for candidate in candidates if candidate.exists()]
    if not existing:
        return pd.DataFrame()
    try:
        frame = read_preferred_table(*candidates)
    except (FileNotFoundError, ImportError):
        csv_candidates = [candidate for candidate in candidates if candidate.suffix == ".csv" and candidate.exists()]
        if not csv_candidates:
            return pd.DataFrame()
        frame = pd.read_csv(csv_candidates[0], low_memory=False)
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str)
    if "official_date" in frame.columns:
        frame["official_date"] = _to_datetime_mixed(frame["official_date"])
    if "season_display" in frame.columns:
        frame["season_display"] = frame["season_display"].astype(str)
    return frame


def _load_league_table(stem: str, leagues: list[str]) -> pd.DataFrame:
    candidates = [
        candidate
        for league in leagues
        for candidate in (
            NORMALIZED_DIR / f"{stem}_{league}_latest.parquet",
            NORMALIZED_DIR / f"{stem}_{league}_latest.csv",
        )
    ]
    cache_key = (stem, tuple(leagues))
    signature = _paths_signature(candidates)
    cached = _TABLE_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1].copy(deep=False)

    frames: list[pd.DataFrame] = []
    for league in leagues:
        frame = _load_table_optional(
            NORMALIZED_DIR / f"{stem}_{league}_latest.parquet",
            NORMALIZED_DIR / f"{stem}_{league}_latest.csv",
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "game_id" in combined.columns:
        combined["game_id"] = combined["game_id"].astype(str)
    if "official_date" in combined.columns:
        combined["official_date"] = _to_datetime_mixed(combined["official_date"])
    _TABLE_CACHE[cache_key] = (signature, combined)
    return combined.copy(deep=False)


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
        return int(value)
    except (TypeError, ValueError):
        return None


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = values.clip(-6.0, 6.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _load_torch_predictor(league: str) -> dict[str, Any] | None:
    if torch is None:
        return None

    model_dir = TORCH_ARTIFACTS_ROOT / league / "home_win"
    model_path = model_dir / "model.pt"
    imputer_path = model_dir / "imputer.joblib"
    scaler_path = model_dir / "scaler.joblib"
    required = [model_path, imputer_path, scaler_path]
    if not all(path.exists() for path in required):
        return None

    cache_key = ("torch", league)
    signature = _paths_signature(required)
    cached = _PREDICTOR_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1]

    checkpoint = torch.load(model_path, map_location="cpu")
    model = BasketballTorchModel(
        input_dim=int(checkpoint["input_dim"]),
        hidden_width=int(checkpoint["hidden_width"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictor = {
        "model": model,
        "feature_columns": list(checkpoint["feature_columns"]),
        "imputer": joblib.load(imputer_path),
        "scaler": joblib.load(scaler_path),
    }
    _PREDICTOR_CACHE[cache_key] = (signature, predictor)
    return predictor


def _load_baseline_predictor(league: str) -> dict[str, Any] | None:
    model_dir = ARTIFACTS_ROOT / league / "hist_gradient_boosting" / "home_win"
    model_path = model_dir / "model.joblib"
    feature_columns_path = model_dir / "feature_columns.csv"
    required = [model_path, feature_columns_path]
    if not all(path.exists() for path in required):
        return None

    cache_key = ("baseline", league)
    signature = _paths_signature(required)
    cached = _PREDICTOR_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1]

    predictor = {
        "model": joblib.load(model_path),
        "feature_columns": pd.read_csv(feature_columns_path)["feature"].tolist(),
    }
    _PREDICTOR_CACHE[cache_key] = (signature, predictor)
    return predictor


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
    rows: list[dict[str, Any]] = []
    date_counts = (
        schedule.assign(date_key=schedule["official_date"].map(_date_key))
        .dropna(subset=["date_key"])
        .groupby("date_key")
        .size()
        .sort_index()
    )
    for date_key, count in date_counts.head(MAX_AVAILABLE_DATES).items():
        rows.append({"dateKey": str(date_key), "label": _date_label(str(date_key)), "gameCount": int(count)})
    return rows


def _resolve_selected_date(available_dates: list[dict[str, Any]], selected_date: str | None) -> str | None:
    if not available_dates:
        return None
    valid = {row["dateKey"] for row in available_dates}
    if selected_date and selected_date in valid:
        return selected_date
    return available_dates[0]["dateKey"]


def _team_logo_url(league: str, team_id: int | None) -> str | None:
    if not team_id:
        return None
    if league == "wnba":
        return f"https://cdn.wnba.com/logos/wnba/{int(team_id)}/global/L/logo.svg"
    return f"https://cdn.nba.com/logos/nba/{int(team_id)}/global/L/logo.svg"


def _player_headshot_url(league: str, player_id: int | None) -> str | None:
    if not player_id:
        return None
    if league == "wnba":
        return f"https://cdn.wnba.com/headshots/wnba/latest/1040x760/{int(player_id)}.png"
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png"


def _format_record_prior(games_played: float | None, win_pct: float | None) -> str | None:
    if games_played is None or win_pct is None or games_played <= 0:
        return None
    wins = int(round(games_played * win_pct))
    wins = max(0, min(wins, int(round(games_played))))
    losses = max(int(round(games_played)) - wins, 0)
    return f"{wins}-{losses}"


def _format_recent_form(row: Any, side: str) -> str | None:
    wins_rate = _safe_float(getattr(row, f"{side}_team_won_avg_last_5", np.nan))
    net_rating = _safe_float(getattr(row, f"{side}_team_net_rating_est_avg_last_10", np.nan))
    points = _safe_float(getattr(row, f"{side}_team_points_scored_avg_last_5", np.nan))
    parts: list[str] = []
    if wins_rate is not None:
        wins = max(0, min(5, int(round(wins_rate * 5))))
        parts.append(f"{wins}-{5 - wins} last 5")
    if net_rating is not None:
        parts.append(f"{net_rating:+.1f} net")
    if points is not None:
        parts.append(f"{points:.1f} pts")
    return " | ".join(parts) or None


def _format_availability_summary(row: Any, side: str) -> str | None:
    inactive_core = _safe_int(getattr(row, f"{side}_availability_likely_inactive_core_players", np.nan)) or 0
    absent_rotation = _safe_int(getattr(row, f"{side}_availability_likely_absent_rotation_players", np.nan)) or 0
    core_rating = _safe_float(getattr(row, f"{side}_availability_core_availability_rating", np.nan))
    parts: list[str] = []
    if inactive_core:
        parts.append(f"{inactive_core} core out")
    if absent_rotation:
        parts.append(f"{absent_rotation} rotation out")
    if core_rating is not None:
        parts.append(f"{core_rating * 100:.0f}% core ready")
    return " | ".join(parts) or "Rotation mostly intact"


def _format_lineup_continuity(row: Any, side: str) -> str | None:
    continuity = _safe_float(getattr(row, f"{side}_availability_expected_lineup_continuity_last_game", np.nan))
    starters = _safe_float(getattr(row, f"{side}_availability_expected_starter_continuity_last_game", np.nan))
    if continuity is not None:
        return f"{continuity * 100:.0f}% rotation overlap"
    if starters is not None:
        return f"{starters * 100:.0f}% starter overlap"
    return None


def _build_team_details(row: Any, side: str) -> dict[str, Any]:
    league = str(getattr(row, "league"))
    team_id = _safe_int(getattr(row, f"{side}_team_id"))
    return {
        "teamId": team_id,
        "abbreviation": _optional_text(getattr(row, f"{side}_team_tricode", None)),
        "logoUrl": _team_logo_url(league, team_id),
        "recordPrior": _format_record_prior(
            _safe_float(getattr(row, f"{side}_team_games_played_prior", np.nan)),
            _safe_float(getattr(row, f"{side}_team_win_pct_prior", np.nan)),
        ),
        "recentForm": _format_recent_form(row, side),
        "availabilitySummary": _format_availability_summary(row, side),
        "lineupContinuity": _format_lineup_continuity(row, side),
        "venue": _optional_text(getattr(row, "arena_name", None)),
    }


def _build_prediction_team_profile(row: Any, side: str) -> dict[str, Any]:
    return {
        "subtitle": _format_recent_form(row, side),
        "stats": [
            {"label": "Win %", "value": f"{(_safe_float(getattr(row, f'{side}_team_win_pct_prior', np.nan)) or 0) * 100:.0f}%"},
            {"label": "Net L10", "value": f"{(_safe_float(getattr(row, f'{side}_team_net_rating_est_avg_last_10', np.nan)) or 0):+.1f}"},
            {"label": "Core Ready", "value": f"{((_safe_float(getattr(row, f'{side}_availability_core_availability_rating', np.nan)) or 0) * 100):.0f}%"},
        ],
    }


def _featured_player(lineup: list[dict[str, Any]]) -> dict[str, Any] | None:
    return lineup[0] if lineup else None


def _historical_player_radar(row: Any) -> list[dict[str, float]]:
    points = _safe_float(getattr(row, "points", None))
    assists = _safe_float(getattr(row, "assists", None))
    rebounds = _safe_float(getattr(row, "rebounds_total", None))
    steals = _safe_float(getattr(row, "steals", None))
    blocks = _safe_float(getattr(row, "blocks", None))
    minutes = _safe_float(getattr(row, "minutes", None))
    fg_pct = _safe_float(getattr(row, "field_goal_pct", None))
    defense = (steals or 0.0) + 1.35 * (blocks or 0.0)
    return [
        {"label": "Scoring", "value": float(max(0.0, min(100.0, ((points or 0.0) / 30.0) * 100.0)))},
        {"label": "Playmaking", "value": float(max(0.0, min(100.0, ((assists or 0.0) / 10.0) * 100.0)))},
        {"label": "Rebounding", "value": float(max(0.0, min(100.0, ((rebounds or 0.0) / 14.0) * 100.0)))},
        {"label": "Defense", "value": float(max(0.0, min(100.0, (defense / 4.0) * 100.0)))},
        {"label": "Shooting", "value": float(max(0.0, min(100.0, ((fg_pct or 0.0) * 100.0))))},
        {"label": "Minutes", "value": float(max(0.0, min(100.0, ((minutes or 0.0) / 38.0) * 100.0)))},
    ]


def _load_live_schedule(
    client: BasketballStatsClient,
    *,
    include_completed: bool = False,
    calendar_year: int | None = None,
) -> pd.DataFrame:
    def _cached_schedule_payload(league: str) -> dict[str, Any] | None:
        pattern = f"schedule_{league}_{LEAGUE_CONFIGS[league].current_season}.json"
        raw_root = BASKETBALL_DATA_ROOT / "raw"
        candidates = sorted(raw_root.glob(f"*/{pattern}"), reverse=True)
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text())
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    frames: list[pd.DataFrame] = []
    today = datetime.now(timezone.utc).date()
    max_date = today + timedelta(days=UPCOMING_LOOKAHEAD_DAYS)
    target_year = calendar_year or today.year
    for league in ("nba", "wnba"):
        try:
            payload = client.get_schedule(league)
        except Exception as exc:
            payload = _cached_schedule_payload(league)
            if payload is not None:
                logger.warning("Falling back to cached %s schedule snapshot for live Basketball export: %s", league.upper(), exc)
                schedule = flatten_schedule(payload, league=league)
            else:
                schedule = _load_league_table("schedule", [league])
                if schedule.empty:
                    logger.warning("Unable to load live %s schedule and no cached snapshot was found: %s", league.upper(), exc)
                    continue
                logger.warning("Falling back to normalized %s schedule table for live Basketball export: %s", league.upper(), exc)
        else:
            schedule = flatten_schedule(payload, league=league)
        if schedule.empty:
            continue
        schedule["official_date"] = _to_datetime_mixed(schedule["official_date"])
        schedule = schedule.loc[(schedule["is_regular_season"] == True)].copy()  # noqa: E712
        if include_completed:
            schedule = schedule.loc[
                (schedule["status_code"] == 3)
                & schedule["official_date"].notna()
                & (schedule["official_date"].dt.year == target_year)
                & (schedule["official_date"].dt.date <= today)
            ].copy()
        else:
            schedule = schedule.loc[
                (schedule["status_code"] != 3)
                & (schedule["official_date"].dt.date >= today)
                & (schedule["official_date"].dt.date <= max_date)
            ].copy()
        if not schedule.empty:
            frames.append(schedule)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["official_date", "league", "game_id"]).reset_index(drop=True)


def _predict_games(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    scored_frames: list[pd.DataFrame] = []
    for league, group in frame.groupby("league", sort=False):
        inference = group.copy()
        torch_predictor = _load_torch_predictor(league)
        if torch_predictor is not None:
            try:
                x = (
                    inference.reindex(columns=torch_predictor["feature_columns"])
                    .apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                )
                imputed = torch_predictor["imputer"].transform(x)
                scaled = torch_predictor["scaler"].transform(imputed).astype(np.float32, copy=False)
                tensor = torch.from_numpy(scaled)
                with torch.no_grad():
                    probabilities = torch.sigmoid(torch_predictor["model"](tensor)).cpu().numpy()
                inference["home_win_probability"] = probabilities
                inference["away_win_probability"] = 1.0 - probabilities
                inference["prediction_source"] = f"{league}_torch_model"
                scored_frames.append(inference)
                continue
            except Exception:
                logger.exception("Basketball torch inference failed for %s; falling back to baseline model.", league.upper())

        baseline_predictor = _load_baseline_predictor(league)
        if baseline_predictor is not None:
            try:
                x = (
                    inference.reindex(columns=baseline_predictor["feature_columns"])
                    .apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                )
                probabilities = baseline_predictor["model"].predict_proba(x)[:, 1]
                inference["home_win_probability"] = probabilities
                inference["away_win_probability"] = 1.0 - probabilities
                inference["prediction_source"] = f"{league}_baseline_model"
                scored_frames.append(inference)
                continue
            except Exception:
                logger.exception("Basketball baseline inference failed for %s; falling back to heuristic scorer.", league.upper())

        score = pd.Series(0.0, index=inference.index, dtype=float)
        weighted_columns = [
            ("matchup_diff_win_pct_prior", 1.2),
            ("matchup_diff_net_rating_est_avg_last_10", 0.35),
            ("availability_diff_core_availability_rating", 1.0),
            ("availability_diff_expected_rotation_points_total", 0.015),
            ("home_rest_advantage", 0.18),
        ]
        for column, weight in weighted_columns:
            if column in inference.columns:
                score += pd.to_numeric(inference[column], errors="coerce").fillna(0.0) * weight
        probabilities = _sigmoid(score)
        inference["home_win_probability"] = probabilities
        inference["away_win_probability"] = 1.0 - probabilities
        inference["prediction_source"] = f"{league}_heuristic_fallback"
        scored_frames.append(inference)

    return pd.concat(scored_frames, ignore_index=True).sort_values(["official_date", "league", "game_id"]).reset_index(drop=True)


def _build_upcoming_boards(frame: pd.DataFrame, rotation_map: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    boards: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        home_prob = float(getattr(row, "home_win_probability", 0.5))
        away_prob = float(getattr(row, "away_win_probability", 0.5))
        scheduled_date = _date_key_int(getattr(row, "official_date", None))
        game_id = str(getattr(row, "game_id"))
        predicted_winner = getattr(row, "home_team_name") if home_prob >= away_prob else getattr(row, "away_team_name")
        predictions = [
            {
                "rank": 1 if away_prob > home_prob else 2,
                "playerName": getattr(row, "away_team_name"),
                "winProbability": away_prob * 100.0,
                "side": "away",
                "profile": _build_prediction_team_profile(row, "away"),
            },
            {
                "rank": 1 if home_prob >= away_prob else 2,
                "playerName": getattr(row, "home_team_name"),
                "winProbability": home_prob * 100.0,
                "side": "home",
                "profile": _build_prediction_team_profile(row, "home"),
            },
        ]
        predictions = sorted(predictions, key=lambda item: item["winProbability"], reverse=True)
        for index, item in enumerate(predictions, start=1):
            item["rank"] = index

        away_availability = {
            "ilAdds14": _safe_int(getattr(row, "away_availability_likely_inactive_core_players", np.nan)) or 0,
            "ilActivations14": _safe_int(getattr(row, "away_availability_expected_starters", np.nan)) or 0,
            "rosterMoves14": _safe_int(getattr(row, "away_availability_expected_rotation_players", np.nan)) or 0,
        }
        home_availability = {
            "ilAdds14": _safe_int(getattr(row, "home_availability_likely_inactive_core_players", np.nan)) or 0,
            "ilActivations14": _safe_int(getattr(row, "home_availability_expected_starters", np.nan)) or 0,
            "rosterMoves14": _safe_int(getattr(row, "home_availability_expected_rotation_players", np.nan)) or 0,
        }

        away_lineup = rotation_map.get(game_id, {}).get("away", [])
        home_lineup = rotation_map.get(game_id, {}).get("home", [])
        away_featured_player = _featured_player(away_lineup)
        home_featured_player = _featured_player(home_lineup)

        boards.append(
            {
                "id": f"basketball-{getattr(row, 'league')}-{game_id}",
                "name": f"{getattr(row, 'away_team_name')} at {getattr(row, 'home_team_name')}",
                "tour": _display_tour(str(getattr(row, "league"))),
                "course": _optional_text(getattr(row, "arena_name", None)) or "Arena",
                "venue": _optional_text(getattr(row, "arena_name", None)) or "Arena",
                "scheduledDate": scheduled_date,
                "latestDate": scheduled_date,
                "predictedWinner": predicted_winner,
                "awayTeam": getattr(row, "away_team_name"),
                "homeTeam": getattr(row, "home_team_name"),
                "awayTeamDetails": _build_team_details(row, "away"),
                "homeTeamDetails": _build_team_details(row, "home"),
                "awayAvailability": away_availability,
                "homeAvailability": home_availability,
                "predictionSource": _optional_text(getattr(row, "prediction_source", None)),
                "awayLineup": away_lineup,
                "homeLineup": home_lineup,
                "awayFeaturedPlayer": away_featured_player,
                "homeFeaturedPlayer": home_featured_player,
                "predictions": predictions,
            }
        )
    return boards


def build_live_upcoming_payload(selected_date: str | None = None) -> dict[str, Any]:
    client = BasketballStatsClient()
    full_schedule = _load_live_schedule(client)
    available_dates = _build_available_dates(full_schedule)
    resolved_selected_date = _resolve_selected_date(available_dates, selected_date)
    if full_schedule.empty or not resolved_selected_date:
        return {
            "selectedDate": resolved_selected_date,
            "availableDates": available_dates,
            "upcoming": [],
            "completed": _build_live_completed_boards(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "divination_live_basketball_feed",
        }

    schedule = full_schedule.loc[full_schedule["official_date"].map(_date_key) == resolved_selected_date].copy()
    schedule["game_id"] = schedule["game_id"].astype(str)
    team_logs = _load_league_table("team_game_logs", ["nba", "wnba"])
    expected_logs = _load_league_table("expected_rotation_game_logs", ["nba", "wnba"])
    player_logs = _load_league_table("player_game_logs", ["nba", "wnba"])

    games = prepare_games(schedule, pd.DataFrame(columns=["league", "game_id"]), require_completed=False)
    games = attach_pregame_team_features(games, team_logs)
    games = attach_expected_rotation_features(games, expected_logs)
    games = add_matchup_differentials(games)
    rotation_map = build_projected_rotation_map(games, player_logs) if not player_logs.empty else {}
    scored_games = _predict_games(games)

    return {
        "selectedDate": resolved_selected_date,
        "availableDates": available_dates,
        "upcoming": _build_upcoming_boards(scored_games, rotation_map),
        "completed": _build_live_completed_boards(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "divination_live_basketball_feed",
    }


def _build_live_completed_boards(calendar_year: int | None = None) -> list[dict[str, Any]]:
    client = BasketballStatsClient()
    schedule = _load_live_schedule(client, include_completed=True, calendar_year=calendar_year)
    if schedule.empty:
        return []

    team_logs = _load_league_table("team_game_logs", ["nba", "wnba"])
    expected_logs = _load_league_table("expected_rotation_game_logs", ["nba", "wnba"])
    player_boxscores = _load_league_table("player_boxscores", ["nba", "wnba"])

    games = prepare_games(schedule, pd.DataFrame(columns=["league", "game_id"]), require_completed=True)
    if games.empty:
        return []
    games = attach_pregame_team_features(games, team_logs)
    games = attach_expected_rotation_features(games, expected_logs)
    games = add_matchup_differentials(games)
    scored_games = _predict_games(games)

    boards: list[dict[str, Any]] = []
    for row in scored_games.itertuples(index=False):
        home_prob = _safe_float(getattr(row, "home_win_probability", np.nan)) or 0.5
        away_prob = 1.0 - home_prob
        predicted_winner = getattr(row, "home_team_name") if home_prob >= away_prob else getattr(row, "away_team_name")
        actual_winner = getattr(row, "home_team_name") if int(getattr(row, "home_win", 0) or 0) == 1 else getattr(row, "away_team_name")
        lineups = _historical_lineups(player_boxscores, str(getattr(row, "game_id")))
        away_availability = {
            "ilAdds14": _safe_int(getattr(row, "away_availability_likely_inactive_core_players", np.nan)),
            "ilActivations14": _safe_int(getattr(row, "away_availability_expected_available_players", np.nan)),
            "rosterMoves14": _safe_int(getattr(row, "away_availability_likely_absent_rotation_players", np.nan)),
        }
        home_availability = {
            "ilAdds14": _safe_int(getattr(row, "home_availability_likely_inactive_core_players", np.nan)),
            "ilActivations14": _safe_int(getattr(row, "home_availability_expected_available_players", np.nan)),
            "rosterMoves14": _safe_int(getattr(row, "home_availability_likely_absent_rotation_players", np.nan)),
        }
        boards.append(
            {
                "year": int(_to_datetime_mixed(getattr(row, "official_date")).year),
                "tournament": f"{getattr(row, 'away_team_name')} at {getattr(row, 'home_team_name')}",
                "tour": _display_tour(str(getattr(row, "league"))),
                "hitStatus": "Top Pick" if predicted_winner == actual_winner else "Miss",
                "predictedWinner": predicted_winner,
                "actualWinner": actual_winner,
                "prob": max(home_prob, away_prob),
                "venue": _optional_text(getattr(row, "arena_name", None)) or "Arena",
                "course": _optional_text(getattr(row, "arena_name", None)) or "Arena",
                "latestDate": _date_key_int(getattr(row, "official_date", None)),
                "scheduledDate": _date_key_int(getattr(row, "official_date", None)),
                "awayTeam": getattr(row, "away_team_name"),
                "homeTeam": getattr(row, "home_team_name"),
                "awayTeamDetails": _build_team_details(row, "away"),
                "homeTeamDetails": _build_team_details(row, "home"),
                "awayAvailability": away_availability,
                "homeAvailability": home_availability,
                "predictionSource": _optional_text(getattr(row, "prediction_source", None)),
                "awayLineup": lineups["away"],
                "homeLineup": lineups["home"],
                "awayFeaturedPlayer": _featured_player(lineups["away"]),
                "homeFeaturedPlayer": _featured_player(lineups["home"]),
            }
        )
    return sorted(boards, key=lambda item: (item.get("latestDate") or 0, item.get("tournament") or ""), reverse=True)


def _load_historical_sources() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    dataset = _load_table_optional(
        NORMALIZED_DIR / "basketball_training_dataset_latest.parquet",
        NORMALIZED_DIR / "basketball_training_dataset_latest.csv",
        NORMALIZED_DIR / "basketball_nba_training_dataset_latest.parquet",
        NORMALIZED_DIR / "basketball_nba_training_dataset_latest.csv",
    )
    if dataset.empty:
        dataset = _load_league_table("basketball", ["nba", "wnba"])
    player_boxscores = _load_league_table("player_boxscores", ["nba", "wnba"])

    for league in ("nba", "wnba"):
        predictions_path = ARTIFACTS_ROOT / league / "hist_gradient_boosting" / "home_win" / "validation_predictions.csv"
        if not predictions_path.exists():
            continue
        predictions = pd.read_csv(predictions_path)
        predictions["official_date"] = _to_datetime_mixed(predictions["official_date"])
        predictions["game_id"] = predictions["game_id"].astype(str)
        merged = predictions.merge(
            dataset.loc[dataset["league"] == league].drop_duplicates(subset=["game_id"]),
            on=["game_id", "league", "official_date", "away_team_name", "home_team_name"],
            how="left",
        )
        if not player_boxscores.empty:
            merged = merged.merge(
                player_boxscores.groupby("game_id").size().rename("player_rows"),
                on="game_id",
                how="left",
            )
        frames.append(merged)
    return frames


def _historical_lineups(player_boxscores: pd.DataFrame, game_id: str) -> dict[str, list[dict[str, Any]]]:
    if player_boxscores.empty:
        return {"away": [], "home": []}
    frame = player_boxscores.loc[player_boxscores["game_id"].astype(str) == str(game_id)].copy()
    result = {"away": [], "home": []}
    for side, side_name in (("away", "away"), ("home", "home")):
        side_frame = frame.loc[frame["team_side"] == side_name].copy()
        if side_frame.empty:
            continue
        side_frame["starter_numeric"] = side_frame["starter"].fillna(False).astype(int)
        side_frame["minutes_numeric"] = pd.to_numeric(side_frame["minutes"], errors="coerce").fillna(0.0)
        side_frame = side_frame.sort_values(["starter_numeric", "minutes_numeric", "points"], ascending=[False, False, False]).head(9)
        entries: list[dict[str, Any]] = []
        for index, row in enumerate(side_frame.itertuples(index=False), start=1):
            field_goal_pct = _safe_float(getattr(row, "field_goal_pct", None))
            entries.append(
                {
                    "playerId": _safe_int(getattr(row, "player_id", None)),
                    "playerName": getattr(row, "player_name"),
                    "lineupSlot": index,
                    "position": _optional_text(getattr(row, "position", None)),
                    "performanceSummary": f"{int(getattr(row, 'points', 0) or 0)} pts | {float(getattr(row, 'minutes', 0) or 0):.1f} min",
                    "profile": {
                        "imageUrl": _player_headshot_url(str(getattr(row, "league")), _safe_int(getattr(row, "player_id", None))),
                        "subtitle": "Starter" if bool(getattr(row, "starter", False)) else "Rotation",
                        "stats": [
                            {"label": "Pts", "value": str(int(getattr(row, "points", 0) or 0))},
                            {"label": "FG%", "value": f"{field_goal_pct * 100:.1f}%" if field_goal_pct is not None else "--"},
                            {"label": "Ast", "value": str(int(getattr(row, "assists", 0) or 0))},
                            {"label": "Reb", "value": str(int(getattr(row, "rebounds_total", 0) or 0))},
                        ],
                    },
                    "radarMetrics": _historical_player_radar(row),
                }
            )
        result[side] = entries
    return result


def _build_backtests() -> list[dict[str, Any]]:
    dataset = _load_table_optional(
        NORMALIZED_DIR / "basketball_training_dataset_latest.parquet",
        NORMALIZED_DIR / "basketball_training_dataset_latest.csv",
        NORMALIZED_DIR / "basketball_nba_training_dataset_latest.parquet",
        NORMALIZED_DIR / "basketball_nba_training_dataset_latest.csv",
    )
    if dataset.empty:
        dataset = _load_league_table("basketball", ["nba", "wnba"])
    player_boxscores = _load_league_table("player_boxscores", ["nba", "wnba"])

    boards: list[dict[str, Any]] = []
    for league in ("nba", "wnba"):
        predictions_path = ARTIFACTS_ROOT / league / "hist_gradient_boosting" / "home_win" / "validation_predictions.csv"
        if not predictions_path.exists():
            continue
        predictions = pd.read_csv(predictions_path)
        predictions["official_date"] = _to_datetime_mixed(predictions["official_date"])
        predictions["game_id"] = predictions["game_id"].astype(str)
        merged = predictions.merge(
            dataset.loc[dataset["league"] == league].drop_duplicates(subset=["game_id"]),
            on=["game_id", "league", "official_date", "away_team_name", "home_team_name"],
            how="left",
        )
        for row in merged.itertuples(index=False):
            home_prob = _safe_float(getattr(row, "home_win_probability", np.nan)) or 0.5
            away_prob = 1.0 - home_prob
            predicted_winner = getattr(row, "home_team_name") if home_prob >= away_prob else getattr(row, "away_team_name")
            actual_winner = getattr(row, "home_team_name") if int(getattr(row, "home_win", 0) or 0) == 1 else getattr(row, "away_team_name")
            lineups = _historical_lineups(player_boxscores, str(getattr(row, "game_id")))
            boards.append(
                {
                    "year": int(_to_datetime_mixed(getattr(row, "official_date")).year),
                    "tournament": f"{getattr(row, 'away_team_name')} at {getattr(row, 'home_team_name')}",
                    "tour": _display_tour(str(league)),
                    "hitStatus": "Top Pick" if predicted_winner == actual_winner else "Miss",
                    "predictedWinner": predicted_winner,
                    "actualWinner": actual_winner,
                    "prob": max(home_prob, away_prob),
                    "venue": _optional_text(getattr(row, "arena_name", None)) or "Arena",
                    "course": _optional_text(getattr(row, "arena_name", None)) or "Arena",
                    "latestDate": _date_key_int(getattr(row, "official_date", None)),
                    "scheduledDate": _date_key_int(getattr(row, "official_date", None)),
                    "awayTeam": getattr(row, "away_team_name"),
                    "homeTeam": getattr(row, "home_team_name"),
                    "awayTeamDetails": _build_team_details(row, "away"),
                    "homeTeamDetails": _build_team_details(row, "home"),
                    "awayLineup": lineups["away"],
                    "homeLineup": lineups["home"],
                    "awayFeaturedPlayer": _featured_player(lineups["away"]),
                    "homeFeaturedPlayer": _featured_player(lineups["home"]),
                }
            )
    return sorted(boards, key=lambda item: (item.get("latestDate") or 0, item.get("tournament") or ""), reverse=True)


def export_basketball_frontend_data() -> None:
    backtests = _build_backtests()
    upcoming_payload = build_live_upcoming_payload()
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DATA_DIR / "basketball_historical_backtests.json").write_text(json.dumps(backtests, indent=2))
    (FRONTEND_DATA_DIR / "basketball_upcoming_tournaments.json").write_text(json.dumps(upcoming_payload["upcoming"], indent=2))
    print(f"Exported {len(backtests)} Basketball historical boards")
    print(f"Exported {len(upcoming_payload['upcoming'])} Basketball upcoming boards")


if __name__ == "__main__":
    export_basketball_frontend_data()
