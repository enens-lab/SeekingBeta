"""Official MLB Stats API client and flatteners for initial model ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

from .constants import (
    DEFAULT_SPORT_ID,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    MLB_GAME_BOXSCORE_PATH,
    MLB_GAME_CONTEXT_METRICS_PATH,
    MLB_GAME_FEED_PATH,
    MLB_GAME_PLAY_BY_PLAY_PATH,
    MLB_GAME_WIN_PROBABILITY_PATH,
    MLB_PEOPLE_PATH,
    MLB_PLAYER_STATS_PATH,
    MLB_SCHEDULE_PATH,
    MLB_STATS_API_BASE_URL,
    MLB_TEAMS_PATH,
    MLB_TEAM_STATS_PATH,
)


def _iso(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


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
        whole_int = int(whole)
        remainder_int = int(remainder[:1] or 0)
    except ValueError:
        return None
    return whole_int * 3 + remainder_int


def _outs_to_innings(outs: int | None, fallback_value: Any = None) -> float | None:
    if outs is not None:
        return outs / 3.0
    return _safe_float(fallback_value)


def _extract_pitcher_line(
    boxscore: dict[str, Any] | None,
    *,
    side: str,
    pitcher_id: int | None,
) -> dict[str, Any]:
    if not boxscore or pitcher_id is None:
        return {}
    player = boxscore.get("teams", {}).get(side, {}).get("players", {}).get(f"ID{pitcher_id}", {})
    pitching = player.get("stats", {}).get("pitching", {})
    innings_pitched = pitching.get("inningsPitched")
    outs = _innings_to_outs(innings_pitched)
    return {
        "starter_summary": pitching.get("summary"),
        "starter_games_started": _safe_int(pitching.get("gamesStarted")),
        "starter_innings_pitched": _outs_to_innings(outs, innings_pitched),
        "starter_outs_recorded": outs,
        "starter_hits_allowed": _safe_int(pitching.get("hits")),
        "starter_runs_allowed": _safe_int(pitching.get("runs")),
        "starter_earned_runs": _safe_int(pitching.get("earnedRuns")),
        "starter_walks": _safe_int(pitching.get("baseOnBalls")),
        "starter_strikeouts": _safe_int(pitching.get("strikeOuts")),
        "starter_home_runs_allowed": _safe_int(pitching.get("homeRuns")),
        "starter_pitches_thrown": _safe_int(pitching.get("pitchesThrown") or pitching.get("numberOfPitches")),
        "starter_strikes": _safe_int(pitching.get("strikes")),
        "starter_balls": _safe_int(pitching.get("balls")),
        "starter_batters_faced": _safe_int(pitching.get("battersFaced")),
    }


def _extract_team_boxscore_stats(boxscore: dict[str, Any] | None, *, side: str) -> dict[str, Any]:
    if not boxscore:
        return {}
    team_stats = boxscore.get("teams", {}).get(side, {}).get("teamStats", {})
    batting = team_stats.get("batting", {})
    pitching = team_stats.get("pitching", {})
    pitching_outs = _safe_int(pitching.get("outs"))
    return {
        "batting_hits": _safe_int(batting.get("hits")),
        "batting_runs": _safe_int(batting.get("runs")),
        "batting_home_runs": _safe_int(batting.get("homeRuns")),
        "batting_walks": _safe_int(batting.get("baseOnBalls")),
        "batting_strikeouts": _safe_int(batting.get("strikeOuts")),
        "batting_total_bases": _safe_int(batting.get("totalBases")),
        "batting_plate_appearances": _safe_int(batting.get("plateAppearances")),
        "batting_left_on_base": _safe_int(batting.get("leftOnBase")),
        "batting_stolen_bases": _safe_int(batting.get("stolenBases")),
        "batting_obp": _safe_float(batting.get("obp")),
        "batting_slg": _safe_float(batting.get("slg")),
        "batting_ops": _safe_float(batting.get("ops")),
        "pitching_hits_allowed": _safe_int(pitching.get("hits")),
        "pitching_runs_allowed": _safe_int(pitching.get("runs")),
        "pitching_earned_runs": _safe_int(pitching.get("earnedRuns")),
        "pitching_walks": _safe_int(pitching.get("baseOnBalls")),
        "pitching_strikeouts": _safe_int(pitching.get("strikeOuts")),
        "pitching_home_runs_allowed": _safe_int(pitching.get("homeRuns")),
        "pitching_outs": pitching_outs,
        "pitching_innings_pitched": _outs_to_innings(pitching_outs, pitching.get("inningsPitched")),
        "pitching_whip": _safe_float(pitching.get("whip")),
        "pitching_pitches_thrown": _safe_int(pitching.get("pitchesThrown") or pitching.get("numberOfPitches")),
        "pitching_strikes": _safe_int(pitching.get("strikes")),
        "pitching_balls": _safe_int(pitching.get("balls")),
        "pitching_batters_faced": _safe_int(pitching.get("battersFaced")),
    }


@dataclass(frozen=True)
class GameFeedBundle:
    """Convenience wrapper for a fetched game's core payloads."""

    game_pk: int
    live_feed: dict[str, Any]
    boxscore: dict[str, Any] | None = None
    play_by_play: dict[str, Any] | None = None
    context_metrics: dict[str, Any] | None = None
    win_probability: list[dict[str, Any]] | None = None


class MLBStatsClient:
    """Small wrapper around the official MLB Stats API."""

    def __init__(
        self,
        *,
        base_url: str = MLB_STATS_API_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": user_agent,
            }
        )

    def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(
            f"{self.base_url}{path}",
            params={key: value for key, value in (params or {}).items() if value is not None},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_schedule(
        self,
        *,
        game_date: date | datetime | str | None = None,
        start_date: date | datetime | str | None = None,
        end_date: date | datetime | str | None = None,
        season: int | None = None,
        team_id: int | None = None,
        sport_id: int = DEFAULT_SPORT_ID,
        game_type: str | None = None,
        hydrate: str | None = None,
    ) -> dict[str, Any]:
        return self._get_json(
            MLB_SCHEDULE_PATH,
            params={
                "sportId": sport_id,
                "date": _iso(game_date),
                "startDate": _iso(start_date),
                "endDate": _iso(end_date),
                "season": season,
                "teamId": team_id,
                "gameType": game_type,
                "hydrate": hydrate,
            },
        )

    def get_teams(
        self,
        *,
        season: int | None = None,
        sport_id: int = DEFAULT_SPORT_ID,
    ) -> dict[str, Any]:
        return self._get_json(
            MLB_TEAMS_PATH,
            params={"sportId": sport_id, "season": season},
        )

    def get_person(self, person_id: int) -> dict[str, Any]:
        return self._get_json(MLB_PEOPLE_PATH.format(person_id=person_id))

    def get_team_stats(
        self,
        team_id: int,
        *,
        season: int | None = None,
        stats: str = "season",
        group: str = "hitting",
        game_type: str | None = None,
        sit_codes: str | None = None,
    ) -> dict[str, Any]:
        return self._get_json(
            MLB_TEAM_STATS_PATH.format(team_id=team_id),
            params={
                "season": season,
                "stats": stats,
                "group": group,
                "gameType": game_type,
                "sitCodes": sit_codes,
            },
        )

    def get_player_stats(
        self,
        person_id: int,
        *,
        season: int | None = None,
        stats: str = "season",
        group: str = "hitting",
        game_type: str | None = None,
        sit_codes: str | None = None,
    ) -> dict[str, Any]:
        return self._get_json(
            MLB_PLAYER_STATS_PATH.format(person_id=person_id),
            params={
                "season": season,
                "stats": stats,
                "group": group,
                "gameType": game_type,
                "sitCodes": sit_codes,
            },
        )

    def get_game_feed(self, game_pk: int) -> dict[str, Any]:
        return self._get_json(MLB_GAME_FEED_PATH.format(game_pk=game_pk))

    def get_game_boxscore(self, game_pk: int) -> dict[str, Any]:
        return self._get_json(MLB_GAME_BOXSCORE_PATH.format(game_pk=game_pk))

    def get_game_play_by_play(self, game_pk: int) -> dict[str, Any]:
        return self._get_json(MLB_GAME_PLAY_BY_PLAY_PATH.format(game_pk=game_pk))

    def get_game_context_metrics(self, game_pk: int) -> dict[str, Any]:
        return self._get_json(MLB_GAME_CONTEXT_METRICS_PATH.format(game_pk=game_pk))

    def get_game_win_probability(self, game_pk: int) -> list[dict[str, Any]]:
        return self._get_json(MLB_GAME_WIN_PROBABILITY_PATH.format(game_pk=game_pk))

    def get_game_bundle(
        self,
        game_pk: int,
        *,
        include_boxscore: bool = True,
        include_play_by_play: bool = False,
        include_context_metrics: bool = True,
        include_win_probability: bool = False,
    ) -> GameFeedBundle:
        live_feed = self.get_game_feed(game_pk)
        return GameFeedBundle(
            game_pk=game_pk,
            live_feed=live_feed,
            boxscore=self.get_game_boxscore(game_pk) if include_boxscore else None,
            play_by_play=self.get_game_play_by_play(game_pk) if include_play_by_play else None,
            context_metrics=self.get_game_context_metrics(game_pk) if include_context_metrics else None,
            win_probability=self.get_game_win_probability(game_pk) if include_win_probability else None,
        )


def flatten_schedule(schedule_payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten a schedule payload into one row per game."""
    rows: list[dict[str, Any]] = []
    for day in schedule_payload.get("dates", []):
        for game in day.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            away_record = away.get("leagueRecord", {})
            home_record = home.get("leagueRecord", {})
            venue = game.get("venue", {})
            content = game.get("content", {})
            row = {
                "game_pk": game.get("gamePk"),
                "game_guid": game.get("gameGuid"),
                "link": game.get("link"),
                "game_type": game.get("gameType"),
                "season": game.get("season"),
                "game_date": game.get("gameDate"),
                "official_date": game.get("officialDate"),
                "status_abstract": game.get("status", {}).get("abstractGameState"),
                "status_detailed": game.get("status", {}).get("detailedState"),
                "status_code": game.get("status", {}).get("statusCode"),
                "away_team_id": away.get("team", {}).get("id"),
                "away_team_name": away.get("team", {}).get("name"),
                "away_score": away.get("score"),
                "away_is_winner": away.get("isWinner"),
                "away_wins": _safe_int(away_record.get("wins")),
                "away_losses": _safe_int(away_record.get("losses")),
                "home_team_id": home.get("team", {}).get("id"),
                "home_team_name": home.get("team", {}).get("name"),
                "home_score": home.get("score"),
                "home_is_winner": home.get("isWinner"),
                "home_wins": _safe_int(home_record.get("wins")),
                "home_losses": _safe_int(home_record.get("losses")),
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),
                "day_night": game.get("dayNight"),
                "double_header": game.get("doubleHeader"),
                "series_description": game.get("seriesDescription"),
                "games_in_series": game.get("gamesInSeries"),
                "series_game_number": game.get("seriesGameNumber"),
                "scheduled_innings": game.get("scheduledInnings"),
                "if_necessary": game.get("ifNecessary"),
                "if_necessary_description": game.get("ifNecessaryDescription"),
                "calendar_event_id": game.get("calendarEventID"),
                "season_display": game.get("seasonDisplay"),
                "ticket_link": content.get("link"),
            }
            if row["home_is_winner"] is True:
                row["winner_team_id"] = row["home_team_id"]
                row["winner_team_name"] = row["home_team_name"]
            elif row["away_is_winner"] is True:
                row["winner_team_id"] = row["away_team_id"]
                row["winner_team_name"] = row["away_team_name"]
            else:
                row["winner_team_id"] = None
                row["winner_team_name"] = None
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["official_date", "game_pk"], ascending=[True, True]).reset_index(drop=True)
    return frame


def flatten_teams(teams_payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for team in teams_payload.get("teams", []):
        venue = team.get("venue", {})
        league = team.get("league", {})
        division = team.get("division", {})
        rows.append(
            {
                "team_id": team.get("id"),
                "season": team.get("season"),
                "team_name": team.get("name"),
                "team_code": team.get("teamCode"),
                "file_code": team.get("fileCode"),
                "abbreviation": team.get("abbreviation"),
                "location_name": team.get("locationName"),
                "franchise_name": team.get("franchiseName"),
                "club_name": team.get("clubName"),
                "short_name": team.get("shortName"),
                "first_year_of_play": team.get("firstYearOfPlay"),
                "active": team.get("active"),
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "division_id": division.get("id"),
                "division_name": division.get("name"),
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),
            }
        )
    return pd.DataFrame(rows)


def flatten_person(person_payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for person in person_payload.get("people", []):
        primary_position = person.get("primaryPosition", {})
        bat_side = person.get("batSide", {})
        pitch_hand = person.get("pitchHand", {})
        rows.append(
            {
                "player_id": person.get("id"),
                "full_name": person.get("fullName"),
                "first_name": person.get("firstName"),
                "last_name": person.get("lastName"),
                "birth_date": person.get("birthDate"),
                "current_age": person.get("currentAge"),
                "birth_city": person.get("birthCity"),
                "birth_country": person.get("birthCountry"),
                "height": person.get("height"),
                "weight": person.get("weight"),
                "active": person.get("active"),
                "mlb_debut_date": person.get("mlbDebutDate"),
                "position_code": primary_position.get("code"),
                "position_name": primary_position.get("name"),
                "bat_side": bat_side.get("code"),
                "pitch_hand": pitch_hand.get("code"),
            }
        )
    return pd.DataFrame(rows)


def _flatten_stats_splits(payload: dict[str, Any], *, entity_type: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stat_group in payload.get("stats", []):
        stat_type = stat_group.get("type", {}).get("displayName")
        stat_group_name = stat_group.get("group", {}).get("displayName")
        for split in stat_group.get("splits", []):
            base = {
                "entity_type": entity_type,
                "stat_type": stat_type,
                "stat_group": stat_group_name,
                "season": split.get("season"),
                "game_type": split.get("gameType"),
                "split_code": split.get("split", {}).get("code"),
                "split_name": split.get("split", {}).get("description"),
            }
            if entity_type == "team":
                base["team_id"] = split.get("team", {}).get("id")
                base["team_name"] = split.get("team", {}).get("name")
            else:
                base["player_id"] = split.get("player", {}).get("id")
                base["player_name"] = split.get("player", {}).get("fullName")
                base["team_id"] = split.get("team", {}).get("id")
                base["team_name"] = split.get("team", {}).get("name")
            stats = split.get("stat", {})
            row = {**base}
            for key, value in stats.items():
                row[key] = value
            rows.append(row)
    return pd.DataFrame(rows)


def flatten_team_stats(payload: dict[str, Any]) -> pd.DataFrame:
    return _flatten_stats_splits(payload, entity_type="team")


def flatten_player_stats(payload: dict[str, Any]) -> pd.DataFrame:
    return _flatten_stats_splits(payload, entity_type="player")


def flatten_game_bundle(bundle: GameFeedBundle) -> pd.DataFrame:
    """Flatten the core pregame/live metadata for one game into a single-row frame."""
    live_feed = bundle.live_feed
    game_data = live_feed.get("gameData", {})
    live_data = live_feed.get("liveData", {})
    datetime_info = game_data.get("datetime", {})
    teams = game_data.get("teams", {})
    venue = game_data.get("venue", {})
    venue_location = venue.get("location", {})
    field_info = venue.get("fieldInfo", {})
    weather = game_data.get("weather", {})
    probable_pitchers = game_data.get("probablePitchers", {})
    away = teams.get("away", {})
    home = teams.get("home", {})
    decisions = live_data.get("decisions", {})
    away_pitcher_id = probable_pitchers.get("away", {}).get("id")
    home_pitcher_id = probable_pitchers.get("home", {}).get("id")
    away_pitcher_line = _extract_pitcher_line(bundle.boxscore, side="away", pitcher_id=away_pitcher_id)
    home_pitcher_line = _extract_pitcher_line(bundle.boxscore, side="home", pitcher_id=home_pitcher_id)
    away_team_stats = _extract_team_boxscore_stats(bundle.boxscore, side="away")
    home_team_stats = _extract_team_boxscore_stats(bundle.boxscore, side="home")
    rows = [
        {
            "game_pk": bundle.game_pk,
            "official_date": datetime_info.get("officialDate"),
            "game_datetime": datetime_info.get("dateTime"),
            "day_night": datetime_info.get("dayNight"),
            "away_team_id": away.get("id"),
            "away_team_name": away.get("name"),
            "home_team_id": home.get("id"),
            "home_team_name": home.get("name"),
            "venue_id": venue.get("id"),
            "venue_name": venue.get("name"),
            "venue_city": venue_location.get("city"),
            "venue_state": venue_location.get("stateAbbrev"),
            "venue_turf_type": field_info.get("turfType"),
            "venue_roof_type": field_info.get("roofType"),
            "venue_left_line": _safe_int(field_info.get("leftLine")),
            "venue_left": _safe_int(field_info.get("left")),
            "venue_left_center": _safe_int(field_info.get("leftCenter")),
            "venue_center": _safe_int(field_info.get("center")),
            "venue_right_center": _safe_int(field_info.get("rightCenter")),
            "venue_right_line": _safe_int(field_info.get("rightLine")),
            "weather_condition": weather.get("condition"),
            "weather_temp_f": _safe_float(weather.get("temp")),
            "weather_wind": weather.get("wind"),
            "away_probable_pitcher_id": away_pitcher_id,
            "away_probable_pitcher_name": probable_pitchers.get("away", {}).get("fullName"),
            "home_probable_pitcher_id": home_pitcher_id,
            "home_probable_pitcher_name": probable_pitchers.get("home", {}).get("fullName"),
            "winning_pitcher_id": decisions.get("winner", {}).get("id"),
            "losing_pitcher_id": decisions.get("loser", {}).get("id"),
            "save_pitcher_id": decisions.get("save", {}).get("id"),
            **{f"away_{key}": value for key, value in away_pitcher_line.items()},
            **{f"home_{key}": value for key, value in home_pitcher_line.items()},
            **{f"away_{key}": value for key, value in away_team_stats.items()},
            **{f"home_{key}": value for key, value in home_team_stats.items()},
        }
    ]
    return pd.DataFrame(rows)


def flatten_context_metrics(game_pk: int, payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_pk": game_pk,
                "away_win_probability": payload.get("awayWinProbability"),
                "home_win_probability": payload.get("homeWinProbability"),
                "left_field_sac_fly_probability": payload.get("leftFieldSacFlyProbability"),
                "center_field_sac_fly_probability": payload.get("centerFieldSacFlyProbability"),
                "right_field_sac_fly_probability": payload.get("rightFieldSacFlyProbability"),
            }
        ]
    )


def flatten_win_probability(game_pk: int, payload: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in payload:
        about = entry.get("about", {})
        result = entry.get("result", {})
        matchup = entry.get("matchup", {})
        home_prob = entry.get("homeWinProbability")
        away_prob = entry.get("awayWinProbability")
        leverage = entry.get("homeTeamWinProbabilityAdded")
        rows.append(
            {
                "game_pk": game_pk,
                "at_bat_index": about.get("atBatIndex"),
                "inning": about.get("inning"),
                "is_top_inning": about.get("isTopInning"),
                "half_inning": about.get("halfInning"),
                "start_time": about.get("startTime"),
                "end_time": about.get("endTime"),
                "event_type": result.get("eventType"),
                "event": result.get("event"),
                "description": result.get("description"),
                "away_score": result.get("awayScore"),
                "home_score": result.get("homeScore"),
                "batter_id": matchup.get("batter", {}).get("id"),
                "pitcher_id": matchup.get("pitcher", {}).get("id"),
                "bat_side": matchup.get("batSide", {}).get("code"),
                "pitch_hand": matchup.get("pitchHand", {}).get("code"),
                "home_win_probability": home_prob,
                "away_win_probability": away_prob,
                "home_team_win_probability_added": leverage,
            }
        )
    return pd.DataFrame(rows)


def to_json_serializable(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2)
