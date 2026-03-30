"""Build lineup and bullpen feature tables from saved MLB game payloads."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_int(value: Any) -> int | None:
    if value in (None, "", "-"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_divide(numerator: Any, denominator: Any) -> float | None:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den == 0:
        return None
    return num / den


def _innings_to_outs(value: Any) -> int | None:
    if value in (None, "", "-"):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "." not in text:
        try:
            return int(text) * 3
        except ValueError:
            return None
    whole, remainder = text.split(".", 1)
    try:
        return int(whole) * 3 + int((remainder or "0")[0])
    except ValueError:
        return None


def _parse_date(value: Any) -> datetime | None:
    if value in (None, "", "-"):
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _years_since(reference_date: Any, start_date: Any) -> float | None:
    ref = pd.to_datetime(reference_date, errors="coerce")
    start = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(ref) or pd.isna(start):
        return None
    delta_days = (ref - start).days
    if delta_days < 0:
        return None
    return float(delta_days / 365.25)


def _iter_raw_payload_paths(raw_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            continue
        paths.extend(sorted(raw_dir.glob("game_*.json")))
    return paths


def extract_roster_tables(raw_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse saved MLB raw payloads into lineup and bullpen tables plus player logs."""
    lineup_rows: list[dict[str, Any]] = []
    batter_logs: list[dict[str, Any]] = []
    bullpen_rows: list[dict[str, Any]] = []
    reliever_logs: list[dict[str, Any]] = []

    for payload_path in _iter_raw_payload_paths(raw_dirs):
        payload = json.loads(payload_path.read_text())
        live_feed = payload.get("live_feed", payload)
        boxscore = payload.get("boxscore") or live_feed.get("liveData", {}).get("boxscore", {})
        game_data = live_feed.get("gameData", {})
        game_meta = game_data.get("game", {})
        datetime_meta = game_data.get("datetime", {})
        teams_meta = game_data.get("teams", {})
        game_players = game_data.get("players", {})
        game_pk = _safe_int(live_feed.get("gamePk") or payload.get("game_pk"))
        official_date = datetime_meta.get("officialDate")
        season = _safe_int(game_meta.get("season"))
        if game_pk is None or not official_date:
            continue

        for side in ("away", "home"):
            team_box = boxscore.get("teams", {}).get(side, {})
            team_meta = teams_meta.get(side, {})
            team_id = _safe_int(team_meta.get("id") or team_box.get("team", {}).get("id"))
            team_name = team_meta.get("name") or team_box.get("team", {}).get("name")
            if team_id is None:
                continue

            players = team_box.get("players", {})
            lineup_ids: list[int] = []
            seen_lineup_ids: set[int] = set()
            for raw_player_id in team_box.get("battingOrder", []):
                player_id = _safe_int(raw_player_id)
                if player_id is None or player_id in seen_lineup_ids:
                    continue
                seen_lineup_ids.add(player_id)
                lineup_ids.append(player_id)

            for slot, player_id in enumerate(lineup_ids[:9], start=1):
                profile = game_players.get(f"ID{player_id}", {})
                years_since_debut = _years_since(official_date, profile.get("mlbDebutDate"))
                lineup_rows.append(
                    {
                        "game_pk": game_pk,
                        "official_date": official_date,
                        "season": season,
                        "team_side": side,
                        "team_id": team_id,
                        "team_name": team_name,
                        "lineup_slot": slot,
                        "batter_id": player_id,
                        "batter_name": profile.get("fullName"),
                        "batter_age": _safe_float(profile.get("currentAge")),
                        "batter_weight": _safe_float(profile.get("weight")),
                        "batter_bat_side": profile.get("batSide", {}).get("code"),
                        "batter_position": profile.get("primaryPosition", {}).get("abbreviation"),
                        "batter_mlb_debut_date": profile.get("mlbDebutDate"),
                        "batter_years_since_debut": years_since_debut,
                    }
                )

            seen_bullpen_ids: set[int] = set()
            for raw_pitcher_id in team_box.get("bullpen", []):
                pitcher_id = _safe_int(raw_pitcher_id)
                if pitcher_id is None or pitcher_id in seen_bullpen_ids:
                    continue
                seen_bullpen_ids.add(pitcher_id)
                profile = game_players.get(f"ID{pitcher_id}", {})
                years_since_debut = _years_since(official_date, profile.get("mlbDebutDate"))
                bullpen_rows.append(
                    {
                        "game_pk": game_pk,
                        "official_date": official_date,
                        "season": season,
                        "team_side": side,
                        "team_id": team_id,
                        "team_name": team_name,
                        "reliever_id": pitcher_id,
                        "reliever_name": profile.get("fullName"),
                        "reliever_age": _safe_float(profile.get("currentAge")),
                        "reliever_weight": _safe_float(profile.get("weight")),
                        "reliever_pitch_hand": profile.get("pitchHand", {}).get("code"),
                        "reliever_mlb_debut_date": profile.get("mlbDebutDate"),
                        "reliever_years_since_debut": years_since_debut,
                    }
                )

            for player in players.values():
                player_id = _safe_int(player.get("person", {}).get("id"))
                if player_id is None:
                    continue
                profile = game_players.get(f"ID{player_id}", {})
                years_since_debut = _years_since(official_date, profile.get("mlbDebutDate"))

                batting = player.get("stats", {}).get("batting", {})
                if batting:
                    plate_appearances = _safe_int(batting.get("plateAppearances")) or 0
                    at_bats = _safe_int(batting.get("atBats")) or 0
                    hits = _safe_int(batting.get("hits")) or 0
                    doubles = _safe_int(batting.get("doubles")) or 0
                    triples = _safe_int(batting.get("triples")) or 0
                    home_runs = _safe_int(batting.get("homeRuns")) or 0
                    walks = _safe_int(batting.get("baseOnBalls")) or 0
                    hit_by_pitch = _safe_int(batting.get("hitByPitch")) or 0
                    total_bases = _safe_int(batting.get("totalBases")) or 0
                    if plate_appearances > 0 or at_bats > 0:
                        obp_like = _safe_divide(hits + walks + hit_by_pitch, plate_appearances)
                        slg_like = _safe_divide(total_bases, at_bats)
                        ops_like = None
                        if obp_like is not None and slg_like is not None:
                            ops_like = obp_like + slg_like
                        batter_logs.append(
                            {
                                "game_pk": game_pk,
                                "official_date": official_date,
                                "season": season,
                                "team_side": side,
                                "team_id": team_id,
                                "team_name": team_name,
                                "batter_id": player_id,
                                "plate_appearances": plate_appearances,
                                "at_bats": at_bats,
                                "hits": hits,
                                "doubles": doubles,
                                "triples": triples,
                                "home_runs": home_runs,
                                "walks": walks,
                                "strikeouts": _safe_int(batting.get("strikeOuts")) or 0,
                                "stolen_bases": _safe_int(batting.get("stolenBases")) or 0,
                                "runs": _safe_int(batting.get("runs")) or 0,
                                "rbi": _safe_int(batting.get("rbi")) or 0,
                                "left_on_base": _safe_int(batting.get("leftOnBase")) or 0,
                                "total_bases": total_bases,
                                "obp_like": obp_like,
                                "slg_like": slg_like,
                                "ops_like": ops_like,
                                "current_age": _safe_float(profile.get("currentAge")),
                                "years_since_debut": years_since_debut,
                                "bat_side_code": profile.get("batSide", {}).get("code"),
                            }
                        )

                pitching = player.get("stats", {}).get("pitching", {})
                if pitching:
                    games_started = _safe_int(pitching.get("gamesStarted")) or 0
                    innings_pitched = pitching.get("inningsPitched")
                    outs_recorded = _innings_to_outs(innings_pitched)
                    pitches_thrown = _safe_int(pitching.get("pitchesThrown") or pitching.get("numberOfPitches")) or 0
                    if games_started <= 0 and (outs_recorded or 0) > 0:
                        hits_allowed = _safe_int(pitching.get("hits")) or 0
                        walks = _safe_int(pitching.get("baseOnBalls")) or 0
                        strikeouts = _safe_int(pitching.get("strikeOuts")) or 0
                        earned_runs = _safe_int(pitching.get("earnedRuns")) or 0
                        home_runs_allowed = _safe_int(pitching.get("homeRuns")) or 0
                        strikes = _safe_int(pitching.get("strikes")) or 0
                        innings_value = (outs_recorded or 0) / 3.0 if outs_recorded is not None else _safe_float(innings_pitched)
                        reliever_logs.append(
                            {
                                "game_pk": game_pk,
                                "official_date": official_date,
                                "season": season,
                                "team_side": side,
                                "team_id": team_id,
                                "team_name": team_name,
                                "reliever_id": player_id,
                                "outs_recorded": outs_recorded,
                                "innings_pitched": innings_value,
                                "earned_runs": earned_runs,
                                "hits_allowed": hits_allowed,
                                "walks": walks,
                                "strikeouts": strikeouts,
                                "home_runs_allowed": home_runs_allowed,
                                "pitches_thrown": pitches_thrown,
                                "strikes": strikes,
                                "batters_faced": _safe_int(pitching.get("battersFaced")) or 0,
                                "whip": _safe_divide(hits_allowed + walks, innings_value),
                                "era_like": _safe_divide(earned_runs * 9.0, innings_value),
                                "strike_pct": _safe_divide(strikes, pitches_thrown),
                                "current_age": _safe_float(profile.get("currentAge")),
                                "years_since_debut": years_since_debut,
                                "pitch_hand_code": profile.get("pitchHand", {}).get("code"),
                                "season_saves": _safe_int(player.get("seasonStats", {}).get("pitching", {}).get("saves")) or 0,
                                "season_holds": _safe_int(player.get("seasonStats", {}).get("pitching", {}).get("holds")) or 0,
                                "season_save_opportunities": _safe_int(player.get("seasonStats", {}).get("pitching", {}).get("saveOpportunities")) or 0,
                                "season_games_finished": _safe_int(player.get("seasonStats", {}).get("pitching", {}).get("gamesFinished")) or 0,
                            }
                        )

    return (
        pd.DataFrame(lineup_rows),
        pd.DataFrame(batter_logs),
        pd.DataFrame(bullpen_rows),
        pd.DataFrame(reliever_logs),
    )


def _build_lineup_continuity(lineup_roster: pd.DataFrame) -> pd.DataFrame:
    if lineup_roster.empty:
        return pd.DataFrame()
    frame = lineup_roster.copy()
    frame["official_date"] = pd.to_datetime(frame["official_date"], errors="coerce")
    frame = frame.sort_values(["team_id", "official_date", "game_pk", "lineup_slot"]).reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for team_id, group in frame.groupby("team_id", sort=False):
        previous_lineup: set[int] = set()
        previous_slots: dict[int, int] = {}
        for game_pk, game_group in group.groupby("game_pk", sort=False):
            current_players = set(game_group["batter_id"].dropna().astype(int).tolist())
            current_slots = {
                int(row.batter_id): int(row.lineup_slot)
                for row in game_group.itertuples(index=False)
                if pd.notna(row.batter_id) and pd.notna(row.lineup_slot)
            }
            overlap = len(current_players & previous_lineup) / max(len(current_players), 1) if previous_lineup else np.nan
            common_players = current_players & previous_lineup
            avg_slot_change = (
                float(np.mean([abs(current_slots[player] - previous_slots.get(player, current_slots[player])) for player in common_players]))
                if common_players
                else np.nan
            )
            records.append(
                {
                    "game_pk": game_pk,
                    "team_id": team_id,
                    "lineup_prev_game_overlap": overlap,
                    "lineup_prev_game_slot_change": avg_slot_change,
                }
            )
            previous_lineup = current_players
            previous_slots = current_slots
    return pd.DataFrame(records)


def build_batter_game_logs(batter_logs: pd.DataFrame) -> pd.DataFrame:
    if batter_logs.empty:
        return batter_logs
    frame = batter_logs.copy()
    frame["official_date"] = pd.to_datetime(frame["official_date"], errors="coerce")
    frame = frame.sort_values(["batter_id", "official_date", "game_pk"]).reset_index(drop=True)
    frame["days_rest"] = frame.groupby("batter_id")["official_date"].diff().dt.days
    frame["games_prior"] = frame.groupby("batter_id").cumcount()

    metrics = [
        "plate_appearances",
        "at_bats",
        "hits",
        "doubles",
        "triples",
        "home_runs",
        "walks",
        "strikeouts",
        "stolen_bases",
        "runs",
        "rbi",
        "left_on_base",
        "total_bases",
        "obp_like",
        "slg_like",
        "ops_like",
        "current_age",
        "years_since_debut",
    ]
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        grouped = frame.groupby("batter_id")[metric]
        for window in (3, 5, 10):
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return frame


def build_reliever_game_logs(reliever_logs: pd.DataFrame) -> pd.DataFrame:
    if reliever_logs.empty:
        return reliever_logs
    frame = reliever_logs.copy()
    frame["official_date"] = pd.to_datetime(frame["official_date"], errors="coerce")
    frame = frame.sort_values(["reliever_id", "official_date", "game_pk"]).reset_index(drop=True)
    frame["days_rest"] = frame.groupby("reliever_id")["official_date"].diff().dt.days
    frame["appearances_prior"] = frame.groupby("reliever_id").cumcount()

    metrics = [
        "outs_recorded",
        "innings_pitched",
        "earned_runs",
        "hits_allowed",
        "walks",
        "strikeouts",
        "home_runs_allowed",
        "pitches_thrown",
        "batters_faced",
        "whip",
        "era_like",
        "strike_pct",
        "current_age",
        "years_since_debut",
        "season_saves",
        "season_holds",
        "season_save_opportunities",
        "season_games_finished",
    ]
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        grouped = frame.groupby("reliever_id")[metric]
        for window in (3, 5, 10):
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return frame


def _merge_asof_roster_features(roster: pd.DataFrame, logs: pd.DataFrame, *, id_col: str) -> pd.DataFrame:
    if roster.empty or logs.empty:
        return pd.DataFrame()

    roster_frame = roster.copy()
    roster_frame["official_date"] = pd.to_datetime(roster_frame["official_date"], errors="coerce")
    logs_frame = logs.copy()
    logs_frame["official_date"] = pd.to_datetime(logs_frame["official_date"], errors="coerce")

    feature_columns = [column for column in logs_frame.columns if column not in {id_col, "game_pk", "official_date", "season", "team_side", "team_id", "team_name"}]
    roster_frame = roster_frame.dropna(subset=[id_col, "official_date"]).reset_index(drop=True)
    logs_frame = logs_frame.dropna(subset=[id_col, "official_date"]).reset_index(drop=True)

    merged_parts: list[pd.DataFrame] = []
    for entity_id, roster_group in roster_frame.groupby(id_col, sort=False):
        roster_group = roster_group.sort_values(["official_date", "game_pk"]).reset_index(drop=True)
        log_group = logs_frame.loc[logs_frame[id_col] == entity_id, ["official_date", *feature_columns]].sort_values("official_date")
        if log_group.empty:
            for column in feature_columns:
                roster_group[column] = np.nan
            merged_parts.append(roster_group)
            continue
        merged_parts.append(
            pd.merge_asof(
                roster_group,
                log_group,
                on="official_date",
                direction="backward",
                allow_exact_matches=False,
            )
        )

    if not merged_parts:
        return pd.DataFrame()
    return pd.concat(merged_parts, ignore_index=True)


def build_lineup_feature_frame(lineup_roster: pd.DataFrame, batter_logs: pd.DataFrame) -> pd.DataFrame:
    joined = _merge_asof_roster_features(lineup_roster, batter_logs, id_col="batter_id")
    if joined.empty:
        return joined
    continuity = _build_lineup_continuity(lineup_roster)
    joined["lineup_bucket"] = np.select(
        [joined["lineup_slot"].le(3), joined["lineup_slot"].between(4, 6)],
        ["top", "middle"],
        default="bottom",
    )
    joined["has_history"] = joined["games_prior"].notna().astype(float)
    joined["is_lefty_bat"] = joined["batter_bat_side"].eq("L").astype(float)
    joined["is_switch_bat"] = joined["batter_bat_side"].eq("S").astype(float)
    joined["is_inexperienced"] = (
        joined["games_prior"].fillna(0).lt(25)
        | joined["years_since_debut"].fillna(99).lt(2.0)
    ).astype(float)

    overall = joined.groupby(["game_pk", "team_side", "team_id", "team_name"], as_index=False).agg(
        lineup_player_count=("batter_id", "nunique"),
        lineup_history_coverage=("has_history", "mean"),
        lineup_avg_games_prior=("games_prior", "mean"),
        lineup_avg_days_rest=("days_rest", "mean"),
        lineup_avg_age=("current_age", "mean"),
        lineup_avg_years_since_debut=("years_since_debut", "mean"),
        lineup_lefty_share=("is_lefty_bat", "mean"),
        lineup_switch_share=("is_switch_bat", "mean"),
        lineup_inexperience_rate=("is_inexperienced", "mean"),
        lineup_avg_plate_appearances_avg_last_10=("plate_appearances_avg_last_10", "mean"),
        lineup_avg_hits_avg_last_5=("hits_avg_last_5", "mean"),
        lineup_avg_walks_avg_last_5=("walks_avg_last_5", "mean"),
        lineup_avg_strikeouts_avg_last_5=("strikeouts_avg_last_5", "mean"),
        lineup_avg_total_bases_avg_last_5=("total_bases_avg_last_5", "mean"),
        lineup_avg_home_runs_avg_last_10=("home_runs_avg_last_10", "mean"),
        lineup_avg_obp_like_avg_last_10=("obp_like_avg_last_10", "mean"),
        lineup_avg_slg_like_avg_last_10=("slg_like_avg_last_10", "mean"),
        lineup_avg_ops_like_avg_last_10=("ops_like_avg_last_10", "mean"),
        lineup_max_ops_like_avg_last_10=("ops_like_avg_last_10", "max"),
        lineup_avg_stolen_bases_avg_last_10=("stolen_bases_avg_last_10", "mean"),
        lineup_avg_rbi_avg_last_5=("rbi_avg_last_5", "mean"),
    )

    bucketed = (
        joined.groupby(["game_pk", "team_side", "team_id", "team_name", "lineup_bucket"], as_index=False)
        .agg(
            ops_like_avg_last_10=("ops_like_avg_last_10", "mean"),
            obp_like_avg_last_10=("obp_like_avg_last_10", "mean"),
            total_bases_avg_last_5=("total_bases_avg_last_5", "mean"),
            strikeouts_avg_last_5=("strikeouts_avg_last_5", "mean"),
        )
        .pivot(index=["game_pk", "team_side", "team_id", "team_name"], columns="lineup_bucket")
    )
    if not bucketed.empty:
        bucketed.columns = [f"lineup_{bucket}_{metric}" for metric, bucket in bucketed.columns]
        bucketed = bucketed.reset_index()
        overall = overall.merge(bucketed, on=["game_pk", "team_side", "team_id", "team_name"], how="left")
    if not continuity.empty:
        overall = overall.merge(continuity, on=["game_pk", "team_id"], how="left")
    return overall


def build_bullpen_feature_frame(bullpen_roster: pd.DataFrame, reliever_logs: pd.DataFrame) -> pd.DataFrame:
    joined = _merge_asof_roster_features(bullpen_roster, reliever_logs, id_col="reliever_id")
    if joined.empty:
        return joined
    joined["has_history"] = joined["appearances_prior"].notna().astype(float)
    joined["is_lefty_arm"] = joined["reliever_pitch_hand"].eq("L").astype(float)
    joined["is_short_rest"] = joined["days_rest"].fillna(99).le(1).astype(float)
    joined["is_two_day_rest_or_less"] = joined["days_rest"].fillna(99).le(2).astype(float)
    joined["high_leverage_score"] = (
        joined["season_saves"].fillna(0)
        + joined["season_holds"].fillna(0)
        + 0.25 * joined["season_games_finished"].fillna(0)
        + 0.25 * joined["season_save_opportunities"].fillna(0)
    )
    joined["is_high_leverage_arm"] = joined["high_leverage_score"].ge(5).astype(float)
    joined["high_leverage_short_rest"] = joined["is_high_leverage_arm"] * joined["is_short_rest"]
    joined["is_inexperienced_arm"] = (
        joined["appearances_prior"].fillna(0).lt(15)
        | joined["years_since_debut"].fillna(99).lt(2.0)
    ).astype(float)
    return joined.groupby(["game_pk", "team_side", "team_id", "team_name"], as_index=False).agg(
        bullpen_pitcher_count=("reliever_id", "nunique"),
        bullpen_history_coverage=("has_history", "mean"),
        bullpen_avg_appearances_prior=("appearances_prior", "mean"),
        bullpen_avg_days_rest=("days_rest", "mean"),
        bullpen_min_days_rest=("days_rest", "min"),
        bullpen_avg_age=("current_age", "mean"),
        bullpen_avg_years_since_debut=("years_since_debut", "mean"),
        bullpen_lefty_share=("is_lefty_arm", "mean"),
        bullpen_inexperience_rate=("is_inexperienced_arm", "mean"),
        bullpen_short_rest_count=("is_short_rest", "sum"),
        bullpen_two_day_rest_or_less_count=("is_two_day_rest_or_less", "sum"),
        bullpen_high_leverage_arms_count=("is_high_leverage_arm", "sum"),
        bullpen_high_leverage_short_rest_count=("high_leverage_short_rest", "sum"),
        bullpen_closer_saves_max=("season_saves", "max"),
        bullpen_holds_max=("season_holds", "max"),
        bullpen_games_finished_max=("season_games_finished", "max"),
        bullpen_high_leverage_score_avg=("high_leverage_score", "mean"),
        bullpen_avg_outs_recorded_avg_last_5=("outs_recorded_avg_last_5", "mean"),
        bullpen_avg_earned_runs_avg_last_5=("earned_runs_avg_last_5", "mean"),
        bullpen_avg_hits_allowed_avg_last_5=("hits_allowed_avg_last_5", "mean"),
        bullpen_avg_walks_avg_last_5=("walks_avg_last_5", "mean"),
        bullpen_avg_strikeouts_avg_last_5=("strikeouts_avg_last_5", "mean"),
        bullpen_max_strikeouts_avg_last_5=("strikeouts_avg_last_5", "max"),
        bullpen_avg_home_runs_allowed_avg_last_10=("home_runs_allowed_avg_last_10", "mean"),
        bullpen_avg_pitches_thrown_avg_last_3=("pitches_thrown_avg_last_3", "mean"),
        bullpen_recent_pitches_burden=("pitches_thrown_avg_last_3", "sum"),
        bullpen_avg_whip_avg_last_5=("whip_avg_last_5", "mean"),
        bullpen_avg_era_like_avg_last_5=("era_like_avg_last_5", "mean"),
        bullpen_max_era_like_avg_last_5=("era_like_avg_last_5", "max"),
        bullpen_avg_strike_pct_avg_last_5=("strike_pct_avg_last_5", "mean"),
    )


def merge_lineup_features(games: pd.DataFrame, lineup_features: pd.DataFrame) -> pd.DataFrame:
    if lineup_features.empty:
        return games
    away = lineup_features.loc[lineup_features["team_side"] == "away"].copy()
    away = away.rename(columns={column: f"away_{column}" for column in away.columns if column not in {"game_pk", "team_side", "team_id", "team_name"}})
    away = away.rename(columns={"team_id": "away_team_id", "team_name": "away_team_name"})
    home = lineup_features.loc[lineup_features["team_side"] == "home"].copy()
    home = home.rename(columns={column: f"home_{column}" for column in home.columns if column not in {"game_pk", "team_side", "team_id", "team_name"}})
    home = home.rename(columns={"team_id": "home_team_id", "team_name": "home_team_name"})
    merged = games.merge(away.drop(columns=["team_side"]), on=["game_pk", "away_team_id", "away_team_name"], how="left")
    merged = merged.merge(home.drop(columns=["team_side"]), on=["game_pk", "home_team_id", "home_team_name"], how="left")
    return merged


def merge_bullpen_features(games: pd.DataFrame, bullpen_features: pd.DataFrame) -> pd.DataFrame:
    if bullpen_features.empty:
        return games
    away = bullpen_features.loc[bullpen_features["team_side"] == "away"].copy()
    away = away.rename(columns={column: f"away_{column}" for column in away.columns if column not in {"game_pk", "team_side", "team_id", "team_name"}})
    away = away.rename(columns={"team_id": "away_team_id", "team_name": "away_team_name"})
    home = bullpen_features.loc[bullpen_features["team_side"] == "home"].copy()
    home = home.rename(columns={column: f"home_{column}" for column in home.columns if column not in {"game_pk", "team_side", "team_id", "team_name"}})
    home = home.rename(columns={"team_id": "home_team_id", "team_name": "home_team_name"})
    merged = games.merge(away.drop(columns=["team_side"]), on=["game_pk", "away_team_id", "away_team_name"], how="left")
    merged = merged.merge(home.drop(columns=["team_side"]), on=["game_pk", "home_team_id", "home_team_name"], how="left")
    return merged


def _is_active_status(status_code: Any, status_description: Any) -> bool:
    code = str(status_code or "").strip().upper()
    description = str(status_description or "").strip().lower()
    if code == "A":
        return True
    if not code and not description:
        return True
    return description == "active"


def build_projected_lineup_roster(
    active_roster: pd.DataFrame,
    batter_logs: pd.DataFrame,
    *,
    historical_lineups: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if active_roster.empty:
        return pd.DataFrame()

    hitters = active_roster.copy()
    hitters = hitters.loc[
        hitters.apply(lambda row: _is_active_status(row.get("status_code"), row.get("status_description")), axis=1)
    ].copy()
    hitters = hitters.loc[~hitters["position_abbreviation"].isin(["P", "TWP"])].copy()
    if hitters.empty:
        return pd.DataFrame()

    lineup_candidates = hitters.rename(
        columns={
            "player_id": "batter_id",
            "full_name": "batter_name",
            "current_age": "batter_age",
            "weight": "batter_weight",
            "bat_side": "batter_bat_side",
            "position_abbreviation": "batter_position",
            "mlb_debut_date": "batter_mlb_debut_date",
            "years_since_debut": "batter_years_since_debut",
        }
    )
    joined = _merge_asof_roster_features(lineup_candidates, batter_logs, id_col="batter_id")
    if joined.empty:
        joined = lineup_candidates.copy()

    historical = historical_lineups.copy() if historical_lineups is not None and not historical_lineups.empty else pd.DataFrame()
    if not historical.empty:
        historical["official_date"] = pd.to_datetime(historical["official_date"], errors="coerce")

    projected_rows: list[dict[str, Any]] = []
    for (game_pk, team_side, team_id, team_name), group in joined.groupby(["game_pk", "team_side", "team_id", "team_name"], sort=False):
        game_date = pd.to_datetime(group["official_date"].iloc[0], errors="coerce")
        previous_slot_map: dict[int, int] = {}
        if not historical.empty:
            historical_team = historical.loc[
                (historical["team_id"] == team_id) & (historical["official_date"] < game_date)
            ].sort_values(["official_date", "game_pk", "lineup_slot"])
            if not historical_team.empty:
                last_game_pk = historical_team["game_pk"].iloc[-1]
                last_lineup = historical_team.loc[historical_team["game_pk"] == last_game_pk]
                previous_slot_map = {
                    int(row.batter_id): int(row.lineup_slot)
                    for row in last_lineup.itertuples(index=False)
                    if pd.notna(row.batter_id) and pd.notna(row.lineup_slot)
                }

        selection = group.copy()
        selection["previous_lineup_slot"] = selection["batter_id"].map(previous_slot_map)
        selection["was_in_last_lineup"] = selection["previous_lineup_slot"].notna().astype(float)
        selection["projection_score"] = (
            selection["plate_appearances_avg_last_10"].fillna(0) * 1.5
            + selection["ops_like_avg_last_10"].fillna(0) * 80.0
            + selection["obp_like_avg_last_10"].fillna(0) * 40.0
            + selection["total_bases_avg_last_5"].fillna(0) * 2.5
            + selection["home_runs_avg_last_10"].fillna(0) * 4.0
            + selection["hits_avg_last_5"].fillna(0) * 1.5
            + selection["games_prior"].fillna(0) * 0.05
            + selection["was_in_last_lineup"] * 12.0
            + (10.0 - selection["previous_lineup_slot"].fillna(10.0))
            - selection["strikeouts_avg_last_5"].fillna(0) * 0.75
        )

        top_n = max(1, min(9, len(selection)))
        selected = selection.nlargest(top_n, "projection_score").copy()
        selected["_slot_rank"] = selected["previous_lineup_slot"].fillna(99)
        selected = selected.sort_values(["_slot_rank", "projection_score"], ascending=[True, False]).reset_index(drop=True)
        selected["lineup_slot"] = range(1, len(selected) + 1)

        projected_rows.extend(
            selected[
                [
                    "game_pk",
                    "official_date",
                    "season",
                    "team_side",
                    "team_id",
                    "team_name",
                    "lineup_slot",
                    "batter_id",
                    "batter_name",
                    "batter_age",
                    "batter_weight",
                    "batter_bat_side",
                    "batter_position",
                    "batter_mlb_debut_date",
                    "batter_years_since_debut",
                ]
            ].to_dict(orient="records")
        )

    return pd.DataFrame(projected_rows)


def build_projected_bullpen_roster(
    active_roster: pd.DataFrame,
    *,
    probable_pitchers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if active_roster.empty:
        return pd.DataFrame()

    pitchers = active_roster.copy()
    pitchers = pitchers.loc[
        pitchers.apply(lambda row: _is_active_status(row.get("status_code"), row.get("status_description")), axis=1)
    ].copy()
    pitchers = pitchers.loc[pitchers["position_abbreviation"].isin(["P", "TWP"])].copy()
    if pitchers.empty:
        return pd.DataFrame()

    probable_map: dict[tuple[int, str, int], int] = {}
    if probable_pitchers is not None and not probable_pitchers.empty:
        for row in probable_pitchers.itertuples(index=False):
            if pd.isna(row.probable_pitcher_id) or pd.isna(row.team_id) or pd.isna(row.game_pk):
                continue
            probable_map[(int(row.game_pk), str(row.team_side), int(row.team_id))] = int(row.probable_pitcher_id)

    bullpen_rows: list[dict[str, Any]] = []
    for row in pitchers.itertuples(index=False):
        probable_id = probable_map.get((int(row.game_pk), str(row.team_side), int(row.team_id)))
        if probable_id is not None and int(row.player_id) == probable_id:
            continue
        bullpen_rows.append(
            {
                "game_pk": row.game_pk,
                "official_date": row.official_date,
                "season": row.season,
                "team_side": row.team_side,
                "team_id": row.team_id,
                "team_name": row.team_name,
                "reliever_id": row.player_id,
                "reliever_name": row.full_name,
                "reliever_age": row.current_age,
                "reliever_weight": row.weight,
                "reliever_pitch_hand": row.pitch_hand,
                "reliever_mlb_debut_date": row.mlb_debut_date,
                "reliever_years_since_debut": row.years_since_debut,
            }
        )
    return pd.DataFrame(bullpen_rows)
