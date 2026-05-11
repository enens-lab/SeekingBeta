"""Official NHL API client helpers and flatteners for Hockey modeling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from .constants import API_BASE_URL, DEFAULT_TIMEOUT_SECONDS, DEFAULT_USER_AGENT, LEAGUE_KEY


@dataclass(frozen=True)
class HockeyGameBundle:
    game_id: str
    boxscore: dict[str, Any]


class HockeyStatsClient:
    """Thin wrapper around the public NHL API used for hockey modeling."""

    def __init__(
        self,
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "User-Agent": user_agent,
            }
        )

    def _get_json(self, path: str) -> Any:
        response = self.session.get(f"{API_BASE_URL}{path}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_schedule_for_date(self, game_date: str) -> dict[str, Any]:
        return self._get_json(f"/schedule/{game_date}")

    def get_club_schedule_season(self, team_code: str, season_id: int) -> dict[str, Any]:
        return self._get_json(f"/club-schedule-season/{team_code.upper()}/{int(season_id)}")

    def get_boxscore(self, game_id: str | int) -> dict[str, Any]:
        return self._get_json(f"/gamecenter/{int(game_id)}/boxscore")

    def get_boxscore_optional(self, game_id: str | int) -> dict[str, Any] | None:
        try:
            return self.get_boxscore(game_id)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {403, 404}:
                return None
            raise

    def get_game_bundle(self, game_id: str | int) -> HockeyGameBundle:
        return HockeyGameBundle(game_id=str(game_id), boxscore=self.get_boxscore(game_id))

    def get_game_bundle_optional(self, game_id: str | int) -> HockeyGameBundle | None:
        payload = self.get_boxscore_optional(game_id)
        if payload is None:
            return None
        return HockeyGameBundle(game_id=str(game_id), boxscore=payload)

    def get_club_stats(self, team_code: str, season_id: int, game_type: int = 2) -> dict[str, Any]:
        return self._get_json(f"/club-stats/{team_code.upper()}/{int(season_id)}/{int(game_type)}")

    def get_roster(self, team_code: str, season_id: int) -> dict[str, Any]:
        return self._get_json(f"/roster/{team_code.upper()}/{int(season_id)}")


def _safe_int(value: Any) -> int | None:
    if value in (None, "", "-", "--"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def _safe_float(value: Any) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _full_team_name(team: dict[str, Any] | None) -> str | None:
    if not isinstance(team, dict):
        return None
    place = ((team.get("placeName") or {}).get("default") or "").strip()
    common = ((team.get("commonName") or {}).get("default") or "").strip()
    name = f"{place} {common}".strip()
    return name or None



def _player_name(player: dict[str, Any] | None) -> str | None:
    if not isinstance(player, dict):
        return None
    if "name" in player and isinstance(player["name"], dict):
        return (player["name"].get("default") or "").strip() or None
    first = ((player.get("firstName") or {}).get("default") or "").strip()
    last = ((player.get("lastName") or {}).get("default") or "").strip()
    name = f"{first} {last}".strip()
    return name or None



def _parse_minutes(value: Any) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        try:
            minutes, seconds = text.split(":", 1)
            return float(int(minutes) + int(seconds) / 60.0)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None



def _weighted_pct(players: list[dict[str, Any]], pct_key: str, weight_key: str) -> float | None:
    weighted_total = 0.0
    weight_total = 0.0
    for player in players:
        pct = _safe_float(player.get(pct_key))
        weight = _safe_float(player.get(weight_key))
        if pct is None or weight is None or weight <= 0:
            continue
        weighted_total += pct * weight
        weight_total += weight
    if weight_total <= 0:
        return None
    return weighted_total / weight_total



def flatten_club_schedule_payload(payload: dict[str, Any], *, team_code: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in payload.get("games", []):
        away_team = game.get("awayTeam", {})
        home_team = game.get("homeTeam", {})
        season_value = _safe_int(game.get("season"))
        start_year = int(str(season_value)[:4]) if season_value else None
        rows.append(
            {
                "league": LEAGUE_KEY,
                "source_team": team_code.upper(),
                "season_id": season_value,
                "season_start": start_year,
                "season_display": f"{start_year}-{str(start_year + 1)[-2:]}" if start_year else None,
                "game_id": str(game.get("id") or ""),
                "game_type": _safe_int(game.get("gameType")),
                "game_date": game.get("gameDate"),
                "start_time_utc": game.get("startTimeUTC"),
                "game_state": game.get("gameState"),
                "game_schedule_state": game.get("gameScheduleState"),
                "venue_name": (game.get("venue") or {}).get("default"),
                "venue_location": (game.get("venueLocation") or {}).get("default"),
                "neutral_site": bool(game.get("neutralSite")),
                "away_team_id": _safe_int(away_team.get("id")),
                "away_team_key": away_team.get("abbrev"),
                "away_team_name": _full_team_name(away_team),
                "away_team_logo": away_team.get("logo"),
                "away_score": _safe_int(away_team.get("score")),
                "home_team_id": _safe_int(home_team.get("id")),
                "home_team_key": home_team.get("abbrev"),
                "home_team_name": _full_team_name(home_team),
                "home_team_logo": home_team.get("logo"),
                "home_score": _safe_int(home_team.get("score")),
                "gamecenter_link": game.get("gameCenterLink"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["official_date"] = pd.to_datetime(frame["game_date"], errors="coerce", format="mixed")
    frame["home_win"] = pd.NA
    frame["away_win"] = pd.NA
    completed_mask = frame["home_score"].notna() & frame["away_score"].notna()
    frame.loc[completed_mask, "home_win"] = (frame.loc[completed_mask, "home_score"] > frame.loc[completed_mask, "away_score"]).astype(float)
    frame.loc[completed_mask, "away_win"] = 1.0 - frame.loc[completed_mask, "home_win"].astype(float)
    return frame.sort_values(["official_date", "game_id"]).reset_index(drop=True)



def _aggregate_skaters(players: list[dict[str, Any]]) -> dict[str, Any]:
    if not players:
        return {
            "goals": None,
            "assists": None,
            "points": None,
            "hits": None,
            "blocked_shots": None,
            "pim": None,
            "power_play_goals": None,
            "shots": None,
            "giveaways": None,
            "takeaways": None,
            "toi_minutes": None,
            "faceoff_win_pct": None,
        }
    return {
        "goals": sum(_safe_int(player.get("goals")) or 0 for player in players),
        "assists": sum(_safe_int(player.get("assists")) or 0 for player in players),
        "points": sum(_safe_int(player.get("points")) or 0 for player in players),
        "hits": sum(_safe_int(player.get("hits")) or 0 for player in players),
        "blocked_shots": sum(_safe_int(player.get("blockedShots")) or 0 for player in players),
        "pim": sum(_safe_int(player.get("pim")) or 0 for player in players),
        "power_play_goals": sum(_safe_int(player.get("powerPlayGoals")) or 0 for player in players),
        "shots": sum(_safe_int(player.get("sog")) or 0 for player in players),
        "giveaways": sum(_safe_int(player.get("giveaways")) or 0 for player in players),
        "takeaways": sum(_safe_int(player.get("takeaways")) or 0 for player in players),
        "toi_minutes": sum(_parse_minutes(player.get("toi")) or 0.0 for player in players),
        "faceoff_win_pct": _weighted_pct(players, "faceoffWinningPctg", "sog"),
    }



def _select_goalie(goalies: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not goalies:
        return None
    starters = [goalie for goalie in goalies if goalie.get("starter")]
    candidates = starters or goalies
    return max(candidates, key=lambda goalie: (_parse_minutes(goalie.get("toi")) or 0.0, _safe_int(goalie.get("shotsAgainst")) or 0))



def flatten_game_bundle(bundle: HockeyGameBundle) -> pd.DataFrame:
    game = bundle.boxscore
    player_stats = game.get("playerByGameStats") or {}
    away_stats = player_stats.get("awayTeam") or {}
    home_stats = player_stats.get("homeTeam") or {}
    away_skaters = list(away_stats.get("forwards") or []) + list(away_stats.get("defense") or [])
    home_skaters = list(home_stats.get("forwards") or []) + list(home_stats.get("defense") or [])
    away_goalies = list(away_stats.get("goalies") or [])
    home_goalies = list(home_stats.get("goalies") or [])

    away_team = game.get("awayTeam") or {}
    home_team = game.get("homeTeam") or {}
    away_skater = _aggregate_skaters(away_skaters)
    home_skater = _aggregate_skaters(home_skaters)
    away_goalie = _select_goalie(away_goalies)
    home_goalie = _select_goalie(home_goalies)
    season_value = _safe_int(game.get("season"))
    start_year = int(str(season_value)[:4]) if season_value else None
    away_shots = _safe_int(away_team.get("sog"))
    home_shots = _safe_int(home_team.get("sog"))
    away_goals = _safe_int(away_team.get("score"))
    home_goals = _safe_int(home_team.get("score"))

    row = {
        "league": LEAGUE_KEY,
        "season_id": season_value,
        "season_start": start_year,
        "season_display": f"{start_year}-{str(start_year + 1)[-2:]}" if start_year else None,
        "game_id": bundle.game_id,
        "game_type": _safe_int(game.get("gameType")),
        "game_date": game.get("gameDate"),
        "start_time_utc": game.get("startTimeUTC"),
        "official_date": pd.to_datetime(game.get("gameDate"), errors="coerce", format="mixed"),
        "game_state": game.get("gameState"),
        "game_schedule_state": game.get("gameScheduleState"),
        "venue_name": (game.get("venue") or {}).get("default"),
        "venue_location": (game.get("venueLocation") or {}).get("default"),
        "away_team_id": _safe_int(away_team.get("id")),
        "away_team_key": away_team.get("abbrev"),
        "away_team_name": _full_team_name(away_team),
        "away_score": away_goals,
        "away_shots": away_shots,
        "away_goals_skaters": away_skater["goals"],
        "away_assists": away_skater["assists"],
        "away_points": away_skater["points"],
        "away_hits": away_skater["hits"],
        "away_blocked_shots": away_skater["blocked_shots"],
        "away_pim": away_skater["pim"],
        "away_power_play_goals": away_skater["power_play_goals"],
        "away_skaters_shots": away_skater["shots"],
        "away_giveaways": away_skater["giveaways"],
        "away_takeaways": away_skater["takeaways"],
        "away_faceoff_win_pct": away_skater["faceoff_win_pct"],
        "away_shooting_pct": (float(away_goals) / float(away_shots)) if away_goals is not None and away_shots not in (None, 0) else None,
        "away_starter_goalie_id": _safe_int((away_goalie or {}).get("playerId")),
        "away_starter_goalie_name": _player_name(away_goalie),
        "away_starter_goalie_saves": _safe_int((away_goalie or {}).get("saves")),
        "away_starter_goalie_goals_against": _safe_int((away_goalie or {}).get("goalsAgainst")),
        "away_starter_goalie_shots_against": _safe_int((away_goalie or {}).get("shotsAgainst")),
        "away_starter_goalie_save_pct": (
            float(_safe_int((away_goalie or {}).get("saves"))) / float(_safe_int((away_goalie or {}).get("shotsAgainst")))
            if _safe_int((away_goalie or {}).get("saves")) is not None and _safe_int((away_goalie or {}).get("shotsAgainst")) not in (None, 0)
            else None
        ),
        "away_starter_goalie_toi_minutes": _parse_minutes((away_goalie or {}).get("toi")),
        "home_team_id": _safe_int(home_team.get("id")),
        "home_team_key": home_team.get("abbrev"),
        "home_team_name": _full_team_name(home_team),
        "home_score": home_goals,
        "home_shots": home_shots,
        "home_goals_skaters": home_skater["goals"],
        "home_assists": home_skater["assists"],
        "home_points": home_skater["points"],
        "home_hits": home_skater["hits"],
        "home_blocked_shots": home_skater["blocked_shots"],
        "home_pim": home_skater["pim"],
        "home_power_play_goals": home_skater["power_play_goals"],
        "home_skaters_shots": home_skater["shots"],
        "home_giveaways": home_skater["giveaways"],
        "home_takeaways": home_skater["takeaways"],
        "home_faceoff_win_pct": home_skater["faceoff_win_pct"],
        "home_shooting_pct": (float(home_goals) / float(home_shots)) if home_goals is not None and home_shots not in (None, 0) else None,
        "home_starter_goalie_id": _safe_int((home_goalie or {}).get("playerId")),
        "home_starter_goalie_name": _player_name(home_goalie),
        "home_starter_goalie_saves": _safe_int((home_goalie or {}).get("saves")),
        "home_starter_goalie_goals_against": _safe_int((home_goalie or {}).get("goalsAgainst")),
        "home_starter_goalie_shots_against": _safe_int((home_goalie or {}).get("shotsAgainst")),
        "home_starter_goalie_save_pct": (
            float(_safe_int((home_goalie or {}).get("saves"))) / float(_safe_int((home_goalie or {}).get("shotsAgainst")))
            if _safe_int((home_goalie or {}).get("saves")) is not None and _safe_int((home_goalie or {}).get("shotsAgainst")) not in (None, 0)
            else None
        ),
        "home_starter_goalie_toi_minutes": _parse_minutes((home_goalie or {}).get("toi")),
    }
    return pd.DataFrame([row])



def flatten_player_boxscores(bundle: HockeyGameBundle) -> pd.DataFrame:
    game = bundle.boxscore
    player_stats = game.get("playerByGameStats") or {}
    rows: list[dict[str, Any]] = []
    season_value = _safe_int(game.get("season"))
    start_year = int(str(season_value)[:4]) if season_value else None
    official_date = pd.to_datetime(game.get("gameDate"), errors="coerce", format="mixed")
    game_type = _safe_int(game.get("gameType"))

    for side, opp in (("away", "home"), ("home", "away")):
        team = game.get(f"{side}Team") or {}
        opponent = game.get(f"{opp}Team") or {}
        team_stats = player_stats.get(f"{side}Team") or {}
        skaters = list(team_stats.get("forwards") or []) + list(team_stats.get("defense") or [])
        goalies = list(team_stats.get("goalies") or [])
        team_won = None
        if _safe_int(team.get("score")) is not None and _safe_int(opponent.get("score")) is not None:
            team_won = int((_safe_int(team.get("score")) or 0) > (_safe_int(opponent.get("score")) or 0))

        for player in skaters:
            rows.append(
                {
                    "league": LEAGUE_KEY,
                    "season_id": season_value,
                    "season_start": start_year,
                    "season_display": f"{start_year}-{str(start_year + 1)[-2:]}" if start_year else None,
                    "game_id": bundle.game_id,
                    "game_type": game_type,
                    "official_date": official_date,
                    "team_id": _safe_int(team.get("id")),
                    "team_key": team.get("abbrev"),
                    "opponent_team_id": _safe_int(opponent.get("id")),
                    "opponent_team_key": opponent.get("abbrev"),
                    "is_home": int(side == "home"),
                    "team_won": team_won,
                    "player_id": _safe_int(player.get("playerId")),
                    "player_name": _player_name(player),
                    "position": player.get("position"),
                    "player_type": "skater",
                    "starter": int(False),
                    "goals": _safe_int(player.get("goals")),
                    "assists": _safe_int(player.get("assists")),
                    "points": _safe_int(player.get("points")),
                    "plus_minus": _safe_int(player.get("plusMinus")),
                    "pim": _safe_int(player.get("pim")),
                    "hits": _safe_int(player.get("hits")),
                    "power_play_goals": _safe_int(player.get("powerPlayGoals")),
                    "sog": _safe_int(player.get("sog")),
                    "blocked_shots": _safe_int(player.get("blockedShots")),
                    "shifts": _safe_int(player.get("shifts")),
                    "giveaways": _safe_int(player.get("giveaways")),
                    "takeaways": _safe_int(player.get("takeaways")),
                    "faceoff_win_pct": _safe_float(player.get("faceoffWinningPctg")),
                    "toi_minutes": _parse_minutes(player.get("toi")),
                    "shots_against": None,
                    "saves": None,
                    "goals_against": None,
                    "save_pct": None,
                }
            )

        for player in goalies:
            shots_against = _safe_int(player.get("shotsAgainst"))
            saves = _safe_int(player.get("saves"))
            rows.append(
                {
                    "league": LEAGUE_KEY,
                    "season_id": season_value,
                    "season_start": start_year,
                    "season_display": f"{start_year}-{str(start_year + 1)[-2:]}" if start_year else None,
                    "game_id": bundle.game_id,
                    "game_type": game_type,
                    "official_date": official_date,
                    "team_id": _safe_int(team.get("id")),
                    "team_key": team.get("abbrev"),
                    "opponent_team_id": _safe_int(opponent.get("id")),
                    "opponent_team_key": opponent.get("abbrev"),
                    "is_home": int(side == "home"),
                    "team_won": team_won,
                    "player_id": _safe_int(player.get("playerId")),
                    "player_name": _player_name(player),
                    "position": player.get("position"),
                    "player_type": "goalie",
                    "starter": int(bool(player.get("starter"))),
                    "goals": _safe_int(player.get("goals")),
                    "assists": _safe_int(player.get("assists")),
                    "points": _safe_int(player.get("points")),
                    "plus_minus": None,
                    "pim": _safe_int(player.get("pim")),
                    "hits": None,
                    "power_play_goals": None,
                    "sog": None,
                    "blocked_shots": None,
                    "shifts": None,
                    "giveaways": None,
                    "takeaways": None,
                    "faceoff_win_pct": None,
                    "toi_minutes": _parse_minutes(player.get("toi")),
                    "shots_against": shots_against,
                    "saves": saves,
                    "goals_against": _safe_int(player.get("goalsAgainst")),
                    "save_pct": (float(saves) / float(shots_against)) if saves is not None and shots_against not in (None, 0) else None,
                }
            )
    return pd.DataFrame(rows)



def flatten_club_stats(payload: dict[str, Any], *, team_code: str, season_id: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for section, player_type in (("skaters", "skater"), ("goalies", "goalie")):
        for player in payload.get(section, []):
            rows.append(
                {
                    "league": LEAGUE_KEY,
                    "season_id": season_id,
                    "season_start": int(str(season_id)[:4]),
                    "season_display": f"{str(season_id)[:4]}-{str(int(str(season_id)[:4]) + 1)[-2:]}",
                    "team_key": team_code.upper(),
                    "player_id": _safe_int(player.get("playerId")),
                    "player_name": _player_name(player),
                    "player_type": player_type,
                    "position": player.get("positionCode"),
                    "headshot": player.get("headshot"),
                    "games_played": _safe_int(player.get("gamesPlayed")),
                    "games_started": _safe_int(player.get("gamesStarted")),
                    "wins": _safe_int(player.get("wins")),
                    "losses": _safe_int(player.get("losses")),
                    "overtime_losses": _safe_int(player.get("overtimeLosses")),
                    "goals": _safe_int(player.get("goals")),
                    "assists": _safe_int(player.get("assists")),
                    "points": _safe_int(player.get("points")),
                    "plus_minus": _safe_int(player.get("plusMinus")),
                    "penalty_minutes": _safe_int(player.get("penaltyMinutes")),
                    "power_play_goals": _safe_int(player.get("powerPlayGoals")),
                    "shots": _safe_int(player.get("shots")),
                    "shooting_pct": _safe_float(player.get("shootingPctg")),
                    "avg_toi_per_game_seconds": _safe_float(player.get("avgTimeOnIcePerGame")),
                    "avg_shifts_per_game": _safe_float(player.get("avgShiftsPerGame")),
                    "faceoff_win_pct": _safe_float(player.get("faceoffWinPctg")),
                    "goals_against_average": _safe_float(player.get("goalsAgainstAverage")),
                    "save_percentage": _safe_float(player.get("savePercentage")),
                    "shots_against": _safe_int(player.get("shotsAgainst")),
                    "saves": _safe_int(player.get("saves")),
                    "goals_against": _safe_int(player.get("goalsAgainst")),
                    "shutouts": _safe_int(player.get("shutouts")),
                    "time_on_ice_seconds": _safe_float(player.get("timeOnIce")),
                }
            )
    return pd.DataFrame(rows)



def flatten_roster(payload: dict[str, Any], *, team_code: str, season_id: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for section in ("forwards", "defensemen", "goalies"):
        for player in payload.get(section, []):
            rows.append(
                {
                    "league": LEAGUE_KEY,
                    "season_id": season_id,
                    "season_start": int(str(season_id)[:4]),
                    "season_display": f"{str(season_id)[:4]}-{str(int(str(season_id)[:4]) + 1)[-2:]}",
                    "team_key": team_code.upper(),
                    "roster_section": section,
                    "player_id": _safe_int(player.get("id")),
                    "player_name": _player_name(player),
                    "position": player.get("positionCode"),
                    "shoots_catches": player.get("shootsCatches"),
                    "headshot": player.get("headshot"),
                    "sweater_number": _safe_int(player.get("sweaterNumber")),
                    "height_in_inches": _safe_int(player.get("heightInInches")),
                    "weight_in_pounds": _safe_int(player.get("weightInPounds")),
                    "birth_date": player.get("birthDate"),
                    "birth_country": player.get("birthCountry"),
                }
            )
    return pd.DataFrame(rows)
