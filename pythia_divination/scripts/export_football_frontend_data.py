"""Football (NFL) board export: upcoming boards with markets, simulated history, and
the live-logged market record.

Headline win probability (predictions[].winProbability / predictedWinner) is the
de-vigged nflverse moneyline whenever a fresh line is posted ("basis": "market"). Our
own model is shown next to it as a labelled "model view" (a moneyline market with
basis "model"), and becomes the headline only when no line is posted -- then the board
says so ("lineStatus": "not_posted", "basis": "model"). No model here beats the market
(season walk-forward 2022-25: market Brier 0.2105 vs model view 0.2212), so nothing is
presented as a pick or an edge.

Outputs (FRONTEND_DATA_DIR):
  football_upcoming_tournaments.json  upcoming boards (+ gameId, gameStart, markets[])
  football_historical_backtests.json  "simulated" history: the served headline
                                      probability replayed out of sample (closing line,
                                      or the walk-forward model view where no line)
  football_market_history.json        graded market picks exactly as logged pre-game
  football_market_summary.json        per (market type, season) summary of those picks:
                                      recordKind "record" (W-L) only for the one-sided
                                      model-view moneyline, "calibration" otherwise
History rows whose headline is the market carry "prob": null (shipped apps label prob
as model confidence); model-basis rows keep it.
Every bake appends the published markets to data/sports/market_picks/football/ (an
append-only log, see sports/market_log.py); history is only ever graded from that log.
"""
from __future__ import annotations

import json
import os
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports import market_log
from sports.football import markets_nfl
from sports.football import model_view as mv
from sports.football.branding import get_team_branding
from sports.football.build_training_dataset import DEFAULT_FOOTBALL_DATA_ROOT
from sports.football.feature_engineering import (
    add_matchup_differentials,
    attach_pregame_qb_features,
    attach_pregame_roster_features,
    attach_pregame_team_features,
    build_qb_week_logs,
    build_team_game_logs,
    prepare_games,
)
from sports.pga.storage import read_preferred_table
from sports.table_dtypes import coerce_table_dtypes

PROJ_ROOT = Path(__file__).resolve().parents[2]
FOOTBALL_DATA_ROOT = DEFAULT_FOOTBALL_DATA_ROOT
NORMALIZED_DIR = FOOTBALL_DATA_ROOT / "normalized"
FRONTEND_DATA_DIR = PROJ_ROOT / "pythia_prophecy" / "frontend" / "src" / "data"
MARKET_PICKS_ROOT = Path(os.getenv("SPORTS_MARKET_PICKS_ROOT") or market_log.DEFAULT_ROOT)
MAX_AVAILABLE_DATES = 7
# Posted lines older than this are not shown as the market (nflverse lines move all
# week; the daily bake re-ingests games.csv right before the export, so a fresh run is
# minutes old). Past the limit the export first tries a live games.csv fetch.
LINES_MAX_AGE_HOURS = float(os.getenv("FOOTBALL_LINES_MAX_AGE_HOURS", "30"))
# History boards: the current season plus this many previous seasons.
HISTORY_PREVIOUS_SEASONS = int(os.getenv("FOOTBALL_HISTORY_PREVIOUS_SEASONS", "1"))
MARKET_HISTORY_CAP = 300
SPORT_KEY = markets_nfl.SPORT_KEY
logger = logging.getLogger(__name__)


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
    # Normalize the remaining key/id/date dtypes once at load (season/week -> Int64,
    # numeric ids -> float64, string ids stay str) so joins never hit a dtype clash.
    return coerce_table_dtypes(frame, table=stem)


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


def _load_model_params() -> dict[str, Any] | None:
    params = mv.load_params()
    if params is None:
        logger.warning("Football model view params missing (%s): headline = market only, no model view", mv.PARAMS_PATH)
    return params


def _model_version(params: dict[str, Any] | None) -> str:
    return str((params or {}).get("model_version") or "nfl-market-only")


def _lines_snapshot_time() -> datetime | None:
    """When games.csv (and its lines) was fetched: the ingest manifest's snapshot tag."""
    manifest = NORMALIZED_DIR / "football_history_manifest_latest.json"
    try:
        tag = json.loads(manifest.read_text()).get("snapshot_tag")
        return datetime.strptime(str(tag), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _load_games_with_lines() -> tuple[pd.DataFrame, datetime | None, bool]:
    """(games table, lines captured at, lines fresh?). Lines and results change daily,
    so if the local table is older than LINES_MAX_AGE_HOURS (e.g. a stale S3 sync with
    no ingest step) re-fetch nflverse games.csv live; never present stale lines."""
    games = _load_table_optional("football_games_latest")
    captured = _lines_snapshot_time()
    now = datetime.now(timezone.utc)
    if captured is not None and now - captured <= timedelta(hours=LINES_MAX_AGE_HOURS) and not games.empty:
        return games, captured, True
    try:
        from sports.football.client import FootballStatsClient

        live = FootballStatsClient().get_games()
        live["game_id"] = live["game_id"].astype(str)
        live["season"] = pd.to_numeric(live["season"], errors="coerce")
        if not games.empty:
            live = live[live["season"] >= pd.to_numeric(games["season"], errors="coerce").min()]
        logger.info("Football: local games table stale (%s); using a live nflverse games.csv fetch", captured)
        return coerce_table_dtypes(live, table="football_games_latest"), now, True
    except Exception as exc:  # noqa: BLE001 -- network failure must not kill the bake
        logger.warning("Football: live games.csv fetch failed (%s); lines treated as stale", exc)
        return games, captured, False


def _game_start_utc(gameday: object, gametime: object) -> str | None:
    """nflverse gameday (YYYY-MM-DD) + gametime (HH:MM, US Eastern) -> ISO-8601 UTC."""
    day = _optional_text(gameday)
    if not day:
        return None
    clock = _optional_text(gametime) or "13:00"
    try:
        local = datetime.strptime(f"{day[:10]} {clock[:5]}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    try:
        from zoneinfo import ZoneInfo

        aware = local.replace(tzinfo=ZoneInfo("America/New_York"))
    except Exception:  # noqa: BLE001 -- no tz database: US DST rule by hand
        year = local.year
        march = datetime(year, 3, 8)
        dst_start = march + timedelta(days=(6 - march.weekday()) % 7, hours=2)
        november = datetime(year, 11, 1)
        dst_end = november + timedelta(days=(6 - november.weekday()) % 7, hours=2)
        offset = -4 if dst_start <= local < dst_end else -5
        aware = local.replace(tzinfo=timezone(timedelta(hours=offset)))
    return aware.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _attach_model_view(frame: pd.DataFrame, all_games: pd.DataFrame, params: dict[str, Any] | None) -> pd.DataFrame:
    """Walk-forward ratings (from every completed game before each game's week) and the
    model-view outputs: model_home_win_probability, model_margin, model_total."""
    out = frame.copy()
    for column in ("model_home_win_probability", "model_margin", "model_total"):
        out[column] = np.nan
    if params is None or out.empty or all_games.empty:
        return out
    rating_params = params.get("rating_params") or {}
    ratings = mv.walkforward_ratings(
        all_games,
        half_life_days=float(rating_params.get("half_life_days", mv.DEFAULT_RATING_PARAMS["half_life_days"])),
        lam=float(rating_params.get("lam", mv.DEFAULT_RATING_PARAMS["lam"])),
    )
    ratings["game_id"] = all_games["game_id"].astype(str).values
    out["game_id"] = out["game_id"].astype(str)
    out = out.drop(columns=[c for c in ratings.columns if c != "game_id" and c in out.columns])
    out = out.merge(ratings.drop_duplicates("game_id"), on="game_id", how="left")
    predictions = mv.predict_model_view(params, out)
    for column in predictions.columns:
        out[column] = predictions[column].values
    return out


def _attach_headline(frame: pd.DataFrame, *, lines_fresh: bool, params: dict[str, Any] | None) -> pd.DataFrame:
    """home_win_probability = de-vigged moneyline when a fresh line is posted, else the
    model view (labelled), else an explicit 50/50 with basis "none"."""
    out = frame.copy()
    market = mv.market_home_probability(out) if lines_fresh else pd.Series(np.nan, index=out.index)
    out["market_home_win_probability"] = market
    model = pd.to_numeric(out.get("model_home_win_probability"), errors="coerce")
    out["headline_basis"] = np.where(market.notna(), "market", np.where(model.notna(), "model", "none"))
    out["home_win_probability"] = market.where(market.notna(), model).fillna(0.5)
    out["prediction_source"] = out["headline_basis"].map(
        {"market": "market_devig_nflverse", "model": _model_version(params), "none": "unavailable"})
    has_ml = out[["home_moneyline", "away_moneyline"]].notna().all(axis=1) if {"home_moneyline", "away_moneyline"} <= set(out.columns) else pd.Series(False, index=out.index)
    out["line_status"] = np.where(has_ml & lines_fresh, "posted", np.where(has_ml, "stale", "not_posted"))
    return out


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
        # merge_asof needs BOTH frames globally sorted on the `on` key (sort_key),
        # not grouped by player: sorting by player_id first raised "left keys must be
        # sorted", the map came back empty, and every board showed "Projected skill
        # players are still populating."
        roster = roster.sort_values(["sort_key", "player_id", "team"], kind="mergesort").reset_index(drop=True)
        history = player_trends.sort_values(["sort_key", "player_id", "game_id"], kind="mergesort").copy()
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
    # Prefer the season-to-date record computed by _attach_season_records (live boards);
    # the cumulative figure above is the model's feature, not what a fan expects to read.
    record = _optional_text(getattr(row, f"{side}_season_record", None)) or record

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


def _with_upcoming_qb_rows(player_week: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Seed the QB week stats with one empty row per (upcoming game, side) for the
    listed starter, so build_qb_week_logs' shift(1).rolling() lands the QB's PRIOR
    form on the upcoming game_id and attach_pregame_qb_features (a game_id join)
    finds it. Unplayed games have no stats row otherwise, and the join returned
    NaN for every QB feature (all-zero radars, "QB EPA +0.0")."""
    if player_week.empty or schedule.empty:
        return player_week
    existing = set(zip(player_week["game_id"].astype(str), player_week["player_id"].astype(str)))
    rows = []
    for game in schedule.to_dict(orient="records"):
        for side in ("away", "home"):
            qb_id = game.get(f"{side}_qb_id")
            if qb_id is None or (isinstance(qb_id, float) and np.isnan(qb_id)) or not str(qb_id).strip():
                continue
            if (str(game["game_id"]), str(qb_id)) in existing:
                continue
            rows.append({
                "season": game["season"], "week": game["week"], "game_id": game["game_id"],
                "player_id": str(qb_id), "player_name": game.get(f"{side}_qb_name"), "team": game[f"{side}_team"],
                "position": "QB", "position_group": "QB", "season_type": "REG",
            })
    if not rows:
        return player_week
    return pd.concat([player_week, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _fill_roster_features_asof(games: pd.DataFrame, roster_summaries: pd.DataFrame) -> pd.DataFrame:
    """nflverse publishes a week's rosters shortly before it is played, so games one
    or two weeks out have no (season, week, team) summary yet. Fill those sides from
    the team's latest earlier week instead of reporting "0 inactive | 0 reserve"."""
    if games.empty or roster_summaries.empty:
        return games
    feature_columns = [c for c in roster_summaries.columns if c not in {"season", "week", "team"}]
    summaries = roster_summaries.copy()
    summaries["season"] = pd.to_numeric(summaries["season"], errors="coerce")
    summaries["week"] = pd.to_numeric(summaries["week"], errors="coerce")
    summaries = summaries.sort_values(["team", "season", "week"])
    out = games.copy()
    for side in ("away", "home"):
        cols = [f"{side}_roster_{c}" for c in feature_columns if f"{side}_roster_{c}" in out.columns]
        if not cols:
            continue
        for idx, row in out.iterrows():
            if not out.loc[idx, cols].isna().all():
                continue
            team = str(row.get(f"{side}_team"))
            season, week = pd.to_numeric(row.get("season"), errors="coerce"), pd.to_numeric(row.get("week"), errors="coerce")
            candidates = summaries.loc[(summaries["team"].astype(str) == team)
                                       & ((summaries["season"] < season) | ((summaries["season"] == season) & (summaries["week"] < week)))]
            if candidates.empty:
                continue
            latest = candidates.iloc[-1]
            for c in feature_columns:
                col = f"{side}_roster_{c}"
                if col in out.columns:
                    out.loc[idx, col] = latest[c]
    return out


def _attach_season_records(games: pd.DataFrame, team_logs: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date W-L per side for DISPLAY. The model feature win_pct_prior is
    cumulative since 2020 (so a week-2 card read "32-70"); the card should say the
    current season's record before this game."""
    if games.empty or team_logs.empty or "won" not in team_logs.columns:
        return games
    logs = team_logs.loc[team_logs["won"].notna(), ["team_key", "season", "official_date", "won"]].copy()
    logs["season"] = pd.to_numeric(logs["season"], errors="coerce")
    out = games.copy()
    for side in ("away", "home"):
        records = []
        for row in out.itertuples(index=False):
            season = pd.to_numeric(getattr(row, "season"), errors="coerce")
            date = getattr(row, "official_date")
            team = str(getattr(row, f"{side}_team"))
            sub = logs.loc[(logs["team_key"] == team) & (logs["season"] == season) & (logs["official_date"] < date)]
            won = pd.to_numeric(sub["won"], errors="coerce")
            wins, losses, ties = int((won == 1).sum()), int((won == 0).sum()), int((won == 0.5).sum())
            records.append(f"{wins}-{losses}-{ties}" if ties else f"{wins}-{losses}")
        out[f"{side}_season_record"] = records
    return out


def _lineup_asof(lineup_map: dict[tuple[int, int, str], list[dict[str, Any]]], season: int, week: int, team: str) -> list[dict[str, Any]]:
    """Exact (season, week, team) lineup, else the team's latest earlier week."""
    exact = lineup_map.get((season, week, team))
    if exact:
        return exact
    earlier = [key for key in lineup_map if key[2] == team and (key[0], key[1]) < (season, week) and lineup_map[key]]
    if not earlier:
        return []
    return lineup_map[max(earlier)]


_LINE_FIELDS = ("home_moneyline", "away_moneyline", "spread_line", "home_spread_odds", "away_spread_odds",
                "total_line", "over_odds", "under_odds")


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def _has_started(game_start: str | None, now: datetime) -> bool:
    if not game_start:
        return False
    try:
        return datetime.fromisoformat(game_start.replace("Z", "+00:00")) <= now
    except ValueError:
        return False


def build_live_upcoming_payload(selected_date: str | None = None, *, published_at: str | None = None,
                                games_bundle: tuple[pd.DataFrame, datetime | None, bool] | None = None,
                                include_completed: bool = True) -> dict[str, Any]:
    games_df, captured, lines_fresh = games_bundle if games_bundle is not None else _load_games_with_lines()
    team_week = _load_table_optional("football_team_week_stats_latest")
    roster_summaries = _load_table_optional("football_roster_week_summaries_latest")
    player_week = _load_table_optional("football_player_week_stats_latest")
    weekly_rosters = _load_table_optional("football_weekly_rosters_latest")
    params = _load_model_params()
    model_version = _model_version(params)
    now = datetime.now(timezone.utc)
    published_at = published_at or now.isoformat()
    captured_at = _iso(captured)

    # Team form for an UNPLAYED game: build the game logs from the whole schedule
    # (played + upcoming) rather than loading the training-time table, which only
    # holds completed games. build_team_game_logs' shift(1).rolling() then puts each
    # team's prior-N form on the upcoming game_id, exactly the training semantics;
    # the old game_id join against completed-only logs returned NaN for every team
    # feature ("Record pending", "Recent form Pending", Win % 0%).
    all_games = prepare_games(games_df, require_completed=False) if not games_df.empty else pd.DataFrame()
    team_logs = build_team_game_logs(all_games, team_week) if not all_games.empty else pd.DataFrame()

    full_schedule = all_games.copy()
    if not full_schedule.empty:
        today = pd.Timestamp(now.date())
        full_schedule = full_schedule.loc[full_schedule["official_date"] >= today].copy()
        full_schedule = full_schedule.sort_values(["official_date", "season", "week", "game_id"]).reset_index(drop=True)

    available_dates = _build_available_dates(full_schedule)
    resolved_selected_date = _resolve_selected_date(available_dates, selected_date)
    if full_schedule.empty or not resolved_selected_date:
        return {
            "selectedDate": resolved_selected_date,
            "availableDates": available_dates,
            "upcoming": [],
            "completed": _build_live_completed_boards() if include_completed else [],
            "updated_at": now.isoformat(),
            "source": "divination_live_football_feed",
            "modelVersion": model_version,
        }

    # Bake the next FOOTBALL_UPCOMING_BOARD_DAYS game-days (an NFL week runs Thu to
    # Mon), not one: a single baked date empties as soon as UTC rolls past it. Same
    # fix as MLB's UPCOMING_BOARD_DAYS (edec4da).
    ordered_keys = [str(option["dateKey"]) for option in available_dates]
    start_idx = ordered_keys.index(resolved_selected_date) if resolved_selected_date in ordered_keys else 0
    board_days = max(1, int(os.getenv("FOOTBALL_UPCOMING_BOARD_DAYS", "5")))
    target_dates = set(ordered_keys[start_idx : start_idx + board_days])
    schedule = full_schedule.loc[full_schedule["official_date"].map(_date_key).isin(target_dates)].copy()
    qb_logs = build_qb_week_logs(_with_upcoming_qb_rows(player_week, schedule)) if not player_week.empty else pd.DataFrame()
    games = attach_pregame_team_features(schedule, team_logs)
    games = attach_pregame_qb_features(games, qb_logs)
    games = attach_pregame_roster_features(games, roster_summaries)
    games = _fill_roster_features_asof(games, roster_summaries)
    games = add_matchup_differentials(games)
    games = _attach_season_records(games, team_logs)
    games = _attach_model_view(games, all_games, params)
    scored_games = _attach_headline(games, lines_fresh=lines_fresh, params=params)
    ladder = (params or {}).get("ladder") or {}

    player_trends = _build_player_trends(player_week)
    try:
        roster_history_map = _build_roster_history_map(weekly_rosters, player_trends)
    except Exception as exc:
        logger.warning("Unable to build Football roster history map for completed boards: %s", exc)
        roster_history_map = {}

    boards: list[dict[str, Any]] = []
    for row in scored_games.itertuples(index=False):
        scheduled_date = _date_key_int(getattr(row, "official_date", None))
        away_lineup = _lineup_asof(roster_history_map, int(getattr(row, "season")), int(getattr(row, "week")), str(getattr(row, "away_team")))
        home_lineup = _lineup_asof(roster_history_map, int(getattr(row, "season")), int(getattr(row, "week")), str(getattr(row, "home_team")))
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
        game_id = str(getattr(row, "game_id"))
        board_id = f"football-{game_id}"
        game_start = _game_start_utc(getattr(row, "gameday", None), getattr(row, "gametime", None))
        basis = str(getattr(row, "headline_basis", "none"))
        line_status = str(getattr(row, "line_status", "not_posted"))
        market_game: dict[str, Any] = {"home_team": getattr(row, "home_team"), "away_team": getattr(row, "away_team")}
        for field in _LINE_FIELDS:
            # Stale lines are never priced: the board falls back to the model view.
            market_game[field] = _safe_float(getattr(row, field, None)) if lines_fresh else None
        model_payload = None
        if _value(row, "model_home_win_probability") is not None:
            model_payload = {"home_win_probability": _value(row, "model_home_win_probability"),
                             "margin": _value(row, "model_margin"), "total": _value(row, "model_total")}
        markets: list[dict[str, Any]] = []
        completed = _value(row, "home_score") is not None and _value(row, "away_score") is not None
        if not completed and not _has_started(game_start, now):
            markets = markets_nfl.build_board_markets(board_id=board_id, game=market_game, ladder=ladder,
                                                      model=model_payload, published_at=published_at,
                                                      captured_at=captured_at, model_version=model_version)
        line_posted = any(market_game.get(f) is not None for f in ("home_moneyline", "spread_line", "total_line"))
        boards.append(
            {
                "id": board_id,
                "gameId": game_id,
                "gameStart": game_start,
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
                # Where the headline probability comes from: "market" (de-vigged
                # moneyline), "model" (line not posted: our model view), "none".
                "basis": basis,
                "lineStatus": line_status,
                "lineCapturedAt": captured_at if line_posted else None,
                "attribution": markets_nfl.ATTRIBUTION if (basis == "market" or line_posted) else None,
                "modelVersion": model_version,
                "awayLineup": away_lineup,
                "homeLineup": home_lineup,
                "awayFeaturedPlayer": away_featured,
                "homeFeaturedPlayer": home_featured,
                "predictions": _build_live_predictions(row),
                "markets": markets,
            }
        )

    return {
        "selectedDate": resolved_selected_date,
        "availableDates": available_dates,
        "upcoming": boards,
        "completed": _build_live_completed_boards() if include_completed else [],
        "updated_at": now.isoformat(),
        "source": "divination_live_football_feed",
        "modelVersion": model_version,
    }


# ── simulated history: the served headline replayed out of sample ─────────────

def _history_frame(games_df: pd.DataFrame, params: dict[str, Any] | None, seasons: list[int]) -> pd.DataFrame:
    """Completed regular-season games of `seasons` with the probability the board WOULD
    have served: the de-vigged closing moneyline where posted, else the season
    walk-forward model view (fit on earlier seasons only). Never the in-sample model."""
    dataset = _load_table_optional("football_training_dataset_latest")
    if dataset.empty or games_df.empty or not seasons:
        return pd.DataFrame()
    first_oos = int((params or {}).get("first_oos_season") or 2021)
    frame = mv.prepare_training_frame(dataset, games_df, (params or {}).get("rating_params"))
    walk = mv.walk_forward(frame, test_seasons=[s for s in seasons if s > first_oos], first_oos_season=first_oos)
    frame = pd.concat([frame, walk], axis=1)
    frame = frame[frame["season"].isin(seasons) & frame["margin"].notna()].copy()
    market = frame["market_home_win_probability"]
    frame["home_win_probability"] = market.where(market.notna(), frame["model_home_win_probability"])
    frame["headline_basis"] = np.where(market.notna(), "market", "model")
    return frame


def _history_seasons(games_df: pd.DataFrame, only_current: bool = False) -> list[int]:
    seasons = pd.to_numeric(games_df.get("season"), errors="coerce").dropna()
    if seasons.empty:
        return []
    current = int(seasons.max())
    first = current if only_current else current - HISTORY_PREVIOUS_SEASONS
    return list(range(first, current + 1))


_HISTORY_MARKET_KEYS = ("marketId", "type", "period", "label", "side", "line", "modelLine", "modelProbability",
                        "pushProbability", "outcomeProbabilities", "market", "publishedAt", "basis", "modelVersion",
                        "attribution", "result", "unitReturn", "closingLine")


def _historical_boards(graded_by_game: dict[str, list[dict[str, Any]]] | None = None, *,
                       only_current_season: bool = False, games_df: pd.DataFrame | None = None,
                       params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if games_df is None:
        games_df, _, _ = _load_games_with_lines()
    if params is None:
        params = _load_model_params()
    seasons = _history_seasons(games_df, only_current=only_current_season)
    frame = _history_frame(games_df, params, seasons)
    if frame.empty:
        return []
    player_week = _load_table_optional("football_player_week_stats_latest")
    if not player_week.empty:
        player_week["game_id"] = player_week["game_id"].astype(str)

    boards: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        home_prob = _safe_float(getattr(row, "home_win_probability", None))
        home_win = _safe_float(getattr(row, "home_win", None))
        if home_prob is None or home_win is None or abs(home_prob - 0.5) < 1e-9:
            # Ties have no winner and a 50/50 line is no pick: neither is graded.
            continue
        away_prob = 1.0 - home_prob
        predicted_winner = getattr(row, "home_team") if home_prob > 0.5 else getattr(row, "away_team")
        actual_winner = getattr(row, "home_team") if home_win == 1.0 else getattr(row, "away_team")
        game_id = str(getattr(row, "game_id"))
        away_lineup, away_featured, away_starter_profile, away_starter_radar = _historical_featured_players(
            player_week, game_id, str(getattr(row, "away_team")), _optional_text(getattr(row, "away_qb_id", None)))
        home_lineup, home_featured, home_starter_profile, home_starter_radar = _historical_featured_players(
            player_week, game_id, str(getattr(row, "home_team")), _optional_text(getattr(row, "home_qb_id", None)))
        basis = str(getattr(row, "headline_basis", "market"))
        model_prob = _safe_float(getattr(row, "model_home_win_probability", None))
        logged = [{k: g.get(k) for k in _HISTORY_MARKET_KEYS if g.get(k) is not None}
                  for g in (graded_by_game or {}).get(game_id, [])]
        boards.append(
            {
                "year": int(_to_datetime_mixed(getattr(row, "official_date")).year),
                "tournament": f"{getattr(row, 'away_team')} at {getattr(row, 'home_team')}",
                "tournamentId": f"football-{game_id}",
                "gameId": game_id,
                "tour": "Football",
                "hitStatus": "Top Pick" if predicted_winner == actual_winner else "Miss",
                "predictedWinner": predicted_winner,
                "actualWinner": actual_winner,
                # Shipped iOS/Android builds render `prob` as "Model confidence: x%".
                # On a market-basis row it would be the de-vigged closing sportsbook
                # price, not our model, so it is withheld (null); the home/away
                # probabilities and predictedWinner still carry the headline.
                "prob": max(home_prob, away_prob) if basis != "market" else None,
                "homeWinProbability": round(home_prob, 4),
                "awayWinProbability": round(away_prob, 4),
                # The record is a replay, not a live log: "simulated" (closing line or
                # walk-forward model view). Live market picks carry their own record.
                "recordBasis": "simulated",
                "basis": basis,
                "predictionSource": "market_devig_nflverse_close" if basis == "market" else "model_view_walk_forward",
                "attribution": markets_nfl.ATTRIBUTION if basis == "market" else None,
                "modelHomeWinProbability": round(model_prob, 4) if model_prob is not None else None,
                "venue": _optional_text(getattr(row, "stadium", None)) or "Stadium",
                "course": _optional_text(getattr(row, "stadium", None)) or "Stadium",
                "latestDate": _date_key_int(getattr(row, "official_date", None)),
                "scheduledDate": _date_key_int(getattr(row, "official_date", None)),
                "awayTeam": getattr(row, "away_team"),
                "homeTeam": getattr(row, "home_team"),
                "homeScore": _safe_float(getattr(row, "home_score", None)),
                "awayScore": _safe_float(getattr(row, "away_score", None)),
                "awayStarter": _optional_text(getattr(row, "away_qb_name", None)),
                "homeStarter": _optional_text(getattr(row, "home_qb_name", None)),
                "awayStarterProfile": away_starter_profile or _build_live_qb_profile(row, "away"),
                "homeStarterProfile": home_starter_profile or _build_live_qb_profile(row, "home"),
                "awayStarterRadar": away_starter_radar or _build_live_qb_radar(row, "away"),
                "homeStarterRadar": home_starter_radar or _build_live_qb_radar(row, "home"),
                "awayTeamDetails": _build_live_team_details(row, "away"),
                "homeTeamDetails": _build_live_team_details(row, "home"),
                "awayLineup": away_lineup,
                "homeLineup": home_lineup,
                "awayFeaturedPlayer": away_featured,
                "homeFeaturedPlayer": home_featured,
                "markets": logged,
            }
        )
    return sorted(boards, key=lambda item: (item.get("latestDate") or 0, item.get("tournament") or ""), reverse=True)


def _build_live_completed_boards(calendar_year: int | None = None) -> list[dict[str, Any]]:
    """Completed boards for the live API: the current season of the simulated history
    (same out-of-sample headline as the exported file; `calendar_year` is ignored --
    NFL seasons span the new year)."""
    try:
        return _historical_boards(only_current_season=True)
    except Exception:  # noqa: BLE001 -- the live endpoint must still serve upcoming boards
        logger.exception("Football completed boards failed")
        return []


# ── market pick log: write, grade, summarize ──────────────────────────────────

def _final_scores(games_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if games_df.empty:
        return {}
    g = games_df.copy()
    g["home_score"] = pd.to_numeric(g["home_score"], errors="coerce")
    g["away_score"] = pd.to_numeric(g["away_score"], errors="coerce")
    g = g.dropna(subset=["home_score", "away_score"])
    return {str(r.game_id): (float(r.home_score), float(r.away_score)) for r in g.itertuples(index=False)}


def _attach_closing_lines(graded: list[dict[str, Any]], games_df: pd.DataFrame) -> None:
    """closingLine for posted-line spread/total picks: nflverse keeps the last pregame
    line on a completed game, i.e. the closing line (picked side's view)."""
    if games_df.empty or not graded:
        return
    lines = games_df.assign(game_id=games_df["game_id"].astype(str)).set_index("game_id")
    for g in graded:
        if ":alt" in str(g.get("marketId")) or g.get("type") not in ("spread", "total"):
            continue
        gid = str(g.get("gameId"))
        if gid not in lines.index:
            continue
        row = lines.loc[gid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if g["type"] == "spread":
            close = _safe_float(row.get("spread_line"))
            if close is not None:
                g["closingLine"] = -close if g.get("side") == "home" else close
        else:
            close = _safe_float(row.get("total_line"))
            if close is not None:
                g["closingLine"] = close


def _log_and_grade(upcoming_boards: list[dict[str, Any]], games_df: pd.DataFrame, model_version: str,
                   published_at: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = market_log.picks_from_boards(SPORT_KEY, upcoming_boards, season_of=markets_nfl.SEASON_OF,
                                           model_version=model_version, published_at=published_at)
    path = market_log.write_snapshot(SPORT_KEY, records, root=MARKET_PICKS_ROOT)
    print(f"Logged {len(records)} Football market picks -> {path}")
    chosen = market_log.pregame_picks(market_log.load_snapshots(SPORT_KEY, root=MARKET_PICKS_ROOT))
    graded = market_log.grade_picks(chosen, _final_scores(games_df))
    _attach_closing_lines(graded, games_df)
    for g in graded:
        markets_nfl.restore_labels(g)
        g["recordBasis"] = "live_logged_at_publish"
    graded.sort(key=lambda g: (str(g.get("gameStart") or g.get("gameDate") or ""), str(g.get("marketId") or "")),
                reverse=True)
    return graded, markets_nfl.summarize_graded(graded)


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def export_football_frontend_data() -> None:
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    published_at = datetime.now(timezone.utc).isoformat()
    params = _load_model_params()
    model_version = _model_version(params)
    games_bundle = _load_games_with_lines()
    games_df = games_bundle[0]
    upcoming_payload = build_live_upcoming_payload(published_at=published_at, games_bundle=games_bundle,
                                                   include_completed=False)
    upcoming = upcoming_payload["upcoming"]
    graded, summary = _log_and_grade(upcoming, games_df, model_version, published_at)
    graded_by_game: dict[str, list[dict[str, Any]]] = {}
    for g in graded:
        graded_by_game.setdefault(str(g.get("gameId")), []).append(g)
    historical = _historical_boards(graded_by_game, games_df=games_df, params=params)
    _write_json_atomic(FRONTEND_DATA_DIR / "football_historical_backtests.json", historical)
    _write_json_atomic(FRONTEND_DATA_DIR / "football_upcoming_tournaments.json", upcoming)
    _write_json_atomic(FRONTEND_DATA_DIR / "football_market_history.json", graded[:MARKET_HISTORY_CAP])
    _write_json_atomic(FRONTEND_DATA_DIR / "football_market_summary.json", summary)
    print(f"Exported {len(historical)} Football historical boards (simulated, seasons {_history_seasons(games_df)})")
    print(f"Exported {len(upcoming)} Football upcoming boards with {sum(len(b.get('markets') or []) for b in upcoming)} markets")
    print(f"Exported {min(len(graded), MARKET_HISTORY_CAP)} graded Football market picks; {len(summary)} summary rows")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    export_football_frontend_data()
