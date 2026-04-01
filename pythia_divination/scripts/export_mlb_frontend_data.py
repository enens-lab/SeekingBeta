from __future__ import annotations

import json
import sys
from datetime import timedelta
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
UPCOMING_LOOKAHEAD_DAYS = 6
UPCOMING_LIMIT = 18


def _optional_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _load_table(stem: str) -> pd.DataFrame:
    return read_preferred_table(
        NORMALIZED_DIR / f"{stem}.parquet",
        NORMALIZED_DIR / f"{stem}.csv",
    )


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
    dataset = pd.read_csv(
        DATASET_PATH,
        low_memory=False,
        usecols=[
            "game_pk",
            "season",
            "venue_name",
            "away_probable_pitcher_name",
            "home_probable_pitcher_name",
        ],
    )
    merged = predictions.merge(dataset.drop_duplicates(subset=["game_pk"]), on="game_pk", how="left")
    merged["official_date"] = pd.to_datetime(merged["official_date"], errors="coerce")
    merged = merged.dropna(subset=["game_pk", "official_date", "away_team_name", "home_team_name"])
    return merged.sort_values(["official_date", "game_pk"], ascending=[False, False]).reset_index(drop=True)


def _build_backtests(frame: pd.DataFrame) -> list[dict[str, Any]]:
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
                "tour": "MLB",
                "venue": row.venue_name or "MLB Venue",
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
                "course": _optional_text(row.venue_name) or "MLB Venue",
                "predictions": ranked,
                "homeStarter": _optional_text(row.home_probable_pitcher_name),
                "awayStarter": _optional_text(row.away_probable_pitcher_name),
            }
        )
    return backtests


def _load_upcoming_schedule(client: MLBStatsClient) -> pd.DataFrame:
    today = pd.Timestamp.utcnow().normalize()
    end_date = today + pd.Timedelta(days=UPCOMING_LOOKAHEAD_DAYS)
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
    now_utc = pd.Timestamp.now(tz="UTC")
    preview_mask = schedule["status_abstract"].isin(["Preview", "Live"])
    future_mask = schedule["game_date"].isna() | schedule["game_date"].gt(now_utc)
    probable_mask = schedule["away_team_name"].notna() & schedule["home_team_name"].notna()
    upcoming = schedule.loc[preview_mask & future_mask & probable_mask].copy()
    return upcoming.sort_values(["game_date", "game_pk"]).head(UPCOMING_LIMIT).reset_index(drop=True)


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


def _build_upcoming_dataset() -> pd.DataFrame:
    client = MLBStatsClient()
    schedule = _load_upcoming_schedule(client)
    if schedule.empty:
        return pd.DataFrame()

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
    if not batter_logs.empty:
        projected_lineups = build_projected_lineup_roster(active_roster, batter_logs, historical_lineups=historical_lineups)
        lineup_features = build_lineup_feature_frame(projected_lineups, batter_logs)
        games = merge_lineup_features(games, lineup_features)

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
    return games


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


def _build_upcoming_boards(frame: pd.DataFrame) -> list[dict[str, Any]]:
    boards: list[dict[str, Any]] = []
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
                "tour": "MLB",
                "course": _optional_text(row.venue_name) or "MLB Venue",
                "venue": _optional_text(row.venue_name) or "MLB Venue",
                "scheduledDate": date_key,
                "latestDate": date_key,
                "predictedWinner": ranked[0]["playerName"],
                "awayTeam": row.away_team_name,
                "homeTeam": row.home_team_name,
                "awayStarter": _optional_text(row.away_probable_pitcher_name),
                "homeStarter": _optional_text(row.home_probable_pitcher_name),
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
                "predictions": ranked,
            }
        )
    return boards


def export_mlb_frontend_data() -> None:
    historical_frame = _load_historical_source()
    backtests = _build_backtests(historical_frame)
    upcoming_dataset = _build_upcoming_dataset()
    upcoming_dataset = _predict_upcoming(upcoming_dataset)
    upcoming = _build_upcoming_boards(upcoming_dataset)

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DATA_DIR / "mlb_historical_backtests.json").write_text(json.dumps(backtests, indent=2))
    (FRONTEND_DATA_DIR / "mlb_upcoming_tournaments.json").write_text(json.dumps(upcoming, indent=2))

    print(f"Exported {len(backtests)} MLB historical boards")
    print(f"Exported {len(upcoming)} MLB upcoming boards")


if __name__ == "__main__":
    export_mlb_frontend_data()
