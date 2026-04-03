"""Official NBA/WNBA live-data client and flatteners for basketball modeling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from .constants import (
    BOXSCORE_TEMPLATE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    LEAGUE_CONFIGS,
    SCHEDULE_PATH,
    TODAY_SCOREBOARD_TEMPLATE,
    LeagueConfig,
)


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


def _safe_bool(value: Any) -> bool | None:
    if value in (None, "", "-", "--"):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _duration_minutes(value: Any) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("PT") and text.endswith("S"):
        try:
            minutes = 0.0
            remainder = text[2:-1]
            if "M" in remainder:
                minute_part, second_part = remainder.split("M", 1)
                minutes += float(minute_part or 0)
                remainder = second_part
            if remainder:
                minutes += float(remainder or 0) / 60.0
            return minutes
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class BasketballGameBundle:
    """Convenience wrapper for a fetched basketball game's core payloads."""

    league: str
    game_id: str
    boxscore: dict[str, Any]


class BasketballStatsClient:
    """Small wrapper around official NBA/WNBA CDN live-data feeds."""

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

    def _league_config(self, league: str) -> LeagueConfig:
        try:
            return LEAGUE_CONFIGS[league.lower()]
        except KeyError as exc:  # pragma: no cover - caller validation
            raise ValueError(f"Unsupported basketball league: {league}") from exc

    def _get_json(self, url: str) -> Any:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_schedule(self, league: str, *, season: str | None = None) -> dict[str, Any]:
        config = self._league_config(league)
        payload = self._get_json(f"{config.domain}{SCHEDULE_PATH}")
        schedule = payload.get("leagueSchedule", {})
        if season and str(schedule.get("seasonYear")) != str(season):
            raise ValueError(
                f"Requested season {season} is not available from the live feed; current feed serves {schedule.get('seasonYear')}"
            )
        return payload

    def get_today_scoreboard(self, league: str) -> dict[str, Any]:
        config = self._league_config(league)
        return self._get_json(f"{config.domain}{TODAY_SCOREBOARD_TEMPLATE.format(league_id=config.league_id)}")

    def get_boxscore(self, league: str, game_id: str) -> dict[str, Any]:
        config = self._league_config(league)
        return self._get_json(f"{config.domain}{BOXSCORE_TEMPLATE.format(game_id=game_id)}")

    def get_boxscore_optional(self, league: str, game_id: str) -> dict[str, Any] | None:
        try:
            return self.get_boxscore(league, game_id)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {403, 404}:
                return None
            raise

    def get_game_bundle(self, league: str, game_id: str) -> BasketballGameBundle:
        return BasketballGameBundle(league=league, game_id=game_id, boxscore=self.get_boxscore(league, game_id))

    def get_game_bundle_optional(self, league: str, game_id: str) -> BasketballGameBundle | None:
        payload = self.get_boxscore_optional(league, game_id)
        if payload is None:
            return None
        return BasketballGameBundle(league=league, game_id=game_id, boxscore=payload)


def _parse_schedule_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, ""):
        return pd.NaT
    text = str(value).strip()
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return pd.to_datetime(text, errors="coerce")


def flatten_schedule(payload: dict[str, Any], *, league: str) -> pd.DataFrame:
    schedule = payload.get("leagueSchedule", {})
    season_year = str(schedule.get("seasonYear") or "")
    rows: list[dict[str, Any]] = []
    for game_date in schedule.get("gameDates", []):
        game_date_value = game_date.get("gameDate")
        for game in game_date.get("games", []):
            home_team = game.get("homeTeam", {})
            away_team = game.get("awayTeam", {})
            game_id = str(game.get("gameId") or "")
            rows.append(
                {
                    "league": league,
                    "season_display": season_year,
                    "game_id": game_id,
                    "game_code": game.get("gameCode"),
                    "game_date": game_date_value,
                    "game_date_est": game.get("gameDateEst"),
                    "game_date_time_est": game.get("gameDateTimeEst"),
                    "game_date_time_utc": game.get("gameDateTimeUTC") or game.get("gameDateUTC"),
                    "game_time_utc": game.get("gameTimeUTC"),
                    "status_code": _safe_int(game.get("gameStatus")),
                    "status_text": game.get("gameStatusText"),
                    "game_label": game.get("gameLabel"),
                    "game_sub_label": game.get("gameSubLabel"),
                    "game_subtype": game.get("gameSubtype"),
                    "week_name": game.get("weekName"),
                    "week_number": _safe_int(game.get("weekNumber")),
                    "series_text": game.get("seriesText"),
                    "series_game_number": _safe_int(game.get("seriesGameNumber")),
                    "is_neutral": bool(game.get("isNeutral")),
                    "if_necessary": bool(game.get("ifNecessary")),
                    "arena_name": game.get("arenaName"),
                    "arena_city": game.get("arenaCity"),
                    "arena_state": game.get("arenaState"),
                    "away_team_id": _safe_int(away_team.get("teamId")),
                    "away_team_name": away_team.get("teamName"),
                    "away_team_city": away_team.get("teamCity"),
                    "away_team_tricode": away_team.get("teamTricode"),
                    "away_team_wins": _safe_int(away_team.get("wins")),
                    "away_team_losses": _safe_int(away_team.get("losses")),
                    "away_score": _safe_int(away_team.get("score")),
                    "home_team_id": _safe_int(home_team.get("teamId")),
                    "home_team_name": home_team.get("teamName"),
                    "home_team_city": home_team.get("teamCity"),
                    "home_team_tricode": home_team.get("teamTricode"),
                    "home_team_wins": _safe_int(home_team.get("wins")),
                    "home_team_losses": _safe_int(home_team.get("losses")),
                    "home_score": _safe_int(home_team.get("score")),
                    "game_id_prefix": game_id[:3],
                    "is_regular_season": game_id.startswith(LEAGUE_CONFIGS[league].regular_season_game_prefix),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["official_date"] = frame["game_date"].map(_parse_schedule_date)
    frame["away_is_winner"] = (pd.to_numeric(frame["away_score"], errors="coerce") > pd.to_numeric(frame["home_score"], errors="coerce")).astype(float)
    frame["home_is_winner"] = (pd.to_numeric(frame["home_score"], errors="coerce") > pd.to_numeric(frame["away_score"], errors="coerce")).astype(float)
    frame.loc[frame["status_code"] != 3, ["away_is_winner", "home_is_winner"]] = pd.NA
    frame["winner_team_id"] = pd.NA
    frame.loc[frame["away_is_winner"] == 1.0, "winner_team_id"] = frame["away_team_id"]
    frame.loc[frame["home_is_winner"] == 1.0, "winner_team_id"] = frame["home_team_id"]
    return frame.sort_values(["official_date", "game_id"]).reset_index(drop=True)


def flatten_schedule_from_game_bundle(bundle: BasketballGameBundle, *, season_display: str) -> pd.DataFrame:
    game = bundle.boxscore.get("game", {})
    home_team = game.get("homeTeam", {})
    away_team = game.get("awayTeam", {})
    home_score = _safe_int(home_team.get("score"))
    away_score = _safe_int(away_team.get("score"))
    status_code = _safe_int(game.get("gameStatus"))
    row = {
        "league": bundle.league,
        "season_display": season_display,
        "game_id": bundle.game_id,
        "game_code": game.get("gameCode"),
        "game_date": game.get("gameEt") or game.get("gameTimeUTC"),
        "game_date_est": game.get("gameEt"),
        "game_date_time_est": game.get("gameEt"),
        "game_date_time_utc": game.get("gameTimeUTC"),
        "game_time_utc": game.get("gameTimeUTC"),
        "status_code": status_code,
        "status_text": game.get("gameStatusText"),
        "game_label": game.get("gameLabel"),
        "game_sub_label": game.get("gameSubLabel"),
        "game_subtype": game.get("gameSubtype"),
        "week_name": game.get("weekName"),
        "week_number": _safe_int(game.get("weekNumber")),
        "series_text": game.get("seriesText"),
        "series_game_number": _safe_int(game.get("seriesGameNumber")),
        "is_neutral": bool(game.get("isNeutral")),
        "if_necessary": bool(game.get("ifNecessary")),
        "arena_name": game.get("arena", {}).get("arenaName"),
        "arena_city": game.get("arena", {}).get("arenaCity"),
        "arena_state": game.get("arena", {}).get("arenaState"),
        "away_team_id": _safe_int(away_team.get("teamId")),
        "away_team_name": away_team.get("teamName"),
        "away_team_city": away_team.get("teamCity"),
        "away_team_tricode": away_team.get("teamTricode"),
        "away_team_wins": _safe_int(away_team.get("wins")),
        "away_team_losses": _safe_int(away_team.get("losses")),
        "away_score": away_score,
        "home_team_id": _safe_int(home_team.get("teamId")),
        "home_team_name": home_team.get("teamName"),
        "home_team_city": home_team.get("teamCity"),
        "home_team_tricode": home_team.get("teamTricode"),
        "home_team_wins": _safe_int(home_team.get("wins")),
        "home_team_losses": _safe_int(home_team.get("losses")),
        "home_score": home_score,
        "game_id_prefix": str(bundle.game_id)[:3],
        "is_regular_season": True,
    }
    frame = pd.DataFrame([row])
    frame["official_date"] = pd.to_datetime(frame["game_date_time_utc"], errors="coerce").dt.tz_localize(None)
    frame["away_is_winner"] = (pd.to_numeric(frame["away_score"], errors="coerce") > pd.to_numeric(frame["home_score"], errors="coerce")).astype(float)
    frame["home_is_winner"] = (pd.to_numeric(frame["home_score"], errors="coerce") > pd.to_numeric(frame["away_score"], errors="coerce")).astype(float)
    frame.loc[frame["status_code"] != 3, ["away_is_winner", "home_is_winner"]] = pd.NA
    frame["winner_team_id"] = pd.NA
    frame.loc[frame["away_is_winner"] == 1.0, "winner_team_id"] = frame["away_team_id"]
    frame.loc[frame["home_is_winner"] == 1.0, "winner_team_id"] = frame["home_team_id"]
    return frame


def _extract_team_statistics(team: dict[str, Any] | None, *, side: str) -> dict[str, Any]:
    if not team:
        return {}
    stats = team.get("statistics", {})
    return {
        f"{side}_points": _safe_int(stats.get("points")),
        f"{side}_assists": _safe_int(stats.get("assists")),
        f"{side}_rebounds_total": _safe_int(stats.get("reboundsTotal")),
        f"{side}_rebounds_offensive": _safe_int(stats.get("reboundsOffensive")),
        f"{side}_rebounds_defensive": _safe_int(stats.get("reboundsDefensive")),
        f"{side}_turnovers": _safe_int(stats.get("turnovers")),
        f"{side}_steals": _safe_int(stats.get("steals")),
        f"{side}_blocks": _safe_int(stats.get("blocks")),
        f"{side}_fouls_personal": _safe_int(stats.get("foulsPersonal")),
        f"{side}_field_goals_made": _safe_int(stats.get("fieldGoalsMade")),
        f"{side}_field_goals_attempted": _safe_int(stats.get("fieldGoalsAttempted")),
        f"{side}_field_goal_pct": _safe_float(stats.get("fieldGoalsPercentage")),
        f"{side}_three_pointers_made": _safe_int(stats.get("threePointersMade")),
        f"{side}_three_pointers_attempted": _safe_int(stats.get("threePointersAttempted")),
        f"{side}_three_point_pct": _safe_float(stats.get("threePointersPercentage")),
        f"{side}_free_throws_made": _safe_int(stats.get("freeThrowsMade")),
        f"{side}_free_throws_attempted": _safe_int(stats.get("freeThrowsAttempted")),
        f"{side}_free_throw_pct": _safe_float(stats.get("freeThrowsPercentage")),
        f"{side}_bench_points": _safe_int(stats.get("benchPoints")),
        f"{side}_fast_break_points": _safe_int(stats.get("pointsFastBreak")),
        f"{side}_paint_points": _safe_int(stats.get("pointsInThePaint")),
        f"{side}_second_chance_points": _safe_int(stats.get("pointsSecondChance")),
        f"{side}_true_shooting_pct": _safe_float(stats.get("trueShootingPercentage")),
        f"{side}_effective_fg_pct": _safe_float(stats.get("fieldGoalsEffectiveAdjusted")),
        f"{side}_time_leading": stats.get("timeLeading"),
        f"{side}_largest_lead": _safe_int(stats.get("biggestLead")),
        f"{side}_points_off_turnovers": _safe_int(stats.get("pointsFromTurnovers")),
    }


def flatten_game_bundle(bundle: BasketballGameBundle) -> pd.DataFrame:
    game = bundle.boxscore.get("game", {})
    home_team = game.get("homeTeam", {})
    away_team = game.get("awayTeam", {})
    row = {
        "league": bundle.league,
        "game_id": bundle.game_id,
        "game_date_time_utc": game.get("gameTimeUTC"),
        "status_text": game.get("gameStatusText"),
        "status_code": _safe_int(game.get("gameStatus")),
        "attendance": _safe_int(game.get("attendance")),
        "arena_name": game.get("arena", {}).get("arenaName"),
        "arena_city": game.get("arena", {}).get("arenaCity"),
        "arena_state": game.get("arena", {}).get("arenaState"),
        "home_team_id": _safe_int(home_team.get("teamId")),
        "home_team_name": home_team.get("teamName"),
        "home_team_tricode": home_team.get("teamTricode"),
        "away_team_id": _safe_int(away_team.get("teamId")),
        "away_team_name": away_team.get("teamName"),
        "away_team_tricode": away_team.get("teamTricode"),
    }
    row.update(_extract_team_statistics(home_team, side="home"))
    row.update(_extract_team_statistics(away_team, side="away"))
    return pd.DataFrame([row])


def flatten_player_boxscores(bundle: BasketballGameBundle) -> pd.DataFrame:
    game = bundle.boxscore.get("game", {})
    rows: list[dict[str, Any]] = []
    for side_key, side_name in (("awayTeam", "away"), ("homeTeam", "home")):
        team = game.get(side_key, {})
        for player in team.get("players", []):
            stats = player.get("statistics", {})
            rows.append(
                {
                    "league": bundle.league,
                    "game_id": bundle.game_id,
                    "game_date_time_utc": game.get("gameTimeUTC"),
                    "team_side": side_name,
                    "team_id": _safe_int(team.get("teamId")),
                    "team_name": team.get("teamName"),
                    "team_tricode": team.get("teamTricode"),
                    "player_id": _safe_int(player.get("personId")),
                    "player_name": player.get("name") or " ".join(
                        part for part in [player.get("firstName"), player.get("familyName")] if part
                    ).strip(),
                    "first_name": player.get("firstName"),
                    "last_name": player.get("familyName"),
                    "position": player.get("position"),
                    "starter": _safe_bool(player.get("starter")),
                    "played": _safe_bool(player.get("played")),
                    "status": player.get("status"),
                    "minutes": _duration_minutes(stats.get("minutes") or stats.get("minutesCalculated")),
                    "points": _safe_int(stats.get("points")),
                    "assists": _safe_int(stats.get("assists")),
                    "rebounds_total": _safe_int(stats.get("reboundsTotal")),
                    "rebounds_offensive": _safe_int(stats.get("reboundsOffensive")),
                    "rebounds_defensive": _safe_int(stats.get("reboundsDefensive")),
                    "steals": _safe_int(stats.get("steals")),
                    "blocks": _safe_int(stats.get("blocks")),
                    "turnovers": _safe_int(stats.get("turnovers")),
                    "field_goals_made": _safe_int(stats.get("fieldGoalsMade")),
                    "field_goals_attempted": _safe_int(stats.get("fieldGoalsAttempted")),
                    "field_goal_pct": _safe_float(stats.get("fieldGoalsPercentage")),
                    "three_pointers_made": _safe_int(stats.get("threePointersMade")),
                    "three_pointers_attempted": _safe_int(stats.get("threePointersAttempted")),
                    "three_point_pct": _safe_float(stats.get("threePointersPercentage")),
                    "free_throws_made": _safe_int(stats.get("freeThrowsMade")),
                    "free_throws_attempted": _safe_int(stats.get("freeThrowsAttempted")),
                    "free_throw_pct": _safe_float(stats.get("freeThrowsPercentage")),
                    "plus_minus_points": _safe_float(stats.get("plusMinusPoints")),
                }
            )
    return pd.DataFrame(rows)
