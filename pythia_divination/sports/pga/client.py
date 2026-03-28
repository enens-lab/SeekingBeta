"""Client utilities for extracting PGA Tour stats from the public stats site."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

import pandas as pd
import requests

from .constants import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TRACKED_STATS,
    DEFAULT_TOUR_CODE,
    DEFAULT_USER_AGENT,
    PGA_PLAYER_PROFILE_PATH,
    PGA_SCHEDULE_PATH,
    PGA_SCHEDULE_SEASON_PATH,
    PGA_STAT_DETAIL_PATH,
    PGA_TOURNAMENT_LEADERBOARD_PATH,
    PGA_STATS_PATH,
    PGA_TOUR_BASE_URL,
    TrackedPGAStat,
)

logger = logging.getLogger(__name__)

_NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_POSITION_NUMERIC_PATTERN = re.compile(r"^T?(\d+)$")
_NON_FINISH_POSITIONS = {"CUT", "WD", "DQ", "MDF"}
_HEIGHT_PATTERN = re.compile(r"(?P<feet>\d+)'\s*(?P<inches>\d+)")


def _slugify(value: str) -> str:
    normalized = _NON_ALNUM_PATTERN.sub("_", (value or "").strip().lower()).strip("_")
    return normalized or "value"


def _slugify_path_segment(value: str) -> str:
    text = (value or "").strip().lower()
    text = text.replace("&", "")
    text = text.replace("'", "")
    text = _NON_ALNUM_PATTERN.sub("-", text).strip("-")
    return text or "tournament"


def _tournament_slug_candidates(name: str) -> list[str]:
    base = (name or "").strip().lower()
    candidates = [
        _slugify_path_segment(base),
        _slugify_path_segment(base.replace("&", " and ")),
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered or ["tournament"]


def _coerce_numeric(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    text = str(value).strip()
    if not text or text in {"--", "-"}:
        return None

    cleaned = text.replace(",", "").replace("%", "")
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]

    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return text


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.date()


def _coerce_year(value: Any) -> int | None:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return None
    try:
        return int(numeric)
    except (TypeError, ValueError):
        return None


def _parse_height_inches(value: Any) -> int | None:
    if value is None:
        return None
    match = _HEIGHT_PATTERN.search(str(value))
    if not match:
        return None
    return int(match.group("feet")) * 12 + int(match.group("inches"))


def _parse_position_numeric(position: Any) -> int | None:
    if position is None:
        return None
    match = _POSITION_NUMERIC_PATTERN.match(str(position).strip())
    if not match:
        return None
    return int(match.group(1))


def flatten_stat_catalog(stat_overview: dict[str, Any]) -> pd.DataFrame:
    """Flatten the stat catalog exposed on /stats for discovery and auditing."""
    rows: list[dict[str, Any]] = []
    for category in stat_overview.get("categories", []):
        category_name = category.get("displayName")
        category_code = category.get("category")
        category_type = category.get("categoryType")
        for sub_category in category.get("subCategories", []):
            sub_category_name = sub_category.get("displayName")
            for stat in sub_category.get("stats", []):
                rows.append(
                    {
                        "category_code": category_code,
                        "category_name": category_name,
                        "category_type": category_type,
                        "sub_category_name": sub_category_name,
                        "stat_id": str(stat.get("statId")),
                        "stat_title": stat.get("statTitle"),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "category_code",
                "category_name",
                "category_type",
                "sub_category_name",
                "stat_id",
                "stat_title",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        by=["category_name", "sub_category_name", "stat_title", "stat_id"]
    ).reset_index(drop=True)


def flatten_current_leaders(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten the current tournament leaderboard snapshot."""
    rows: list[dict[str, Any]] = []
    for player in payload.get("players", []):
        rows.append(
            {
                "tournament_id": payload.get("id"),
                "tournament_name": payload.get("tournamentName"),
                "round_display": payload.get("roundDisplay"),
                "round_status_display": payload.get("roundStatusDisplay"),
                "player_id": player.get("id"),
                "player_name": player.get("displayName"),
                "country": player.get("country"),
                "position": player.get("position"),
                "thru": player.get("thru"),
                "total_score": player.get("totalScore"),
                "round_score": player.get("roundScore"),
                "player_state": player.get("playerState"),
            }
        )
    return pd.DataFrame(rows)


def flatten_schedule(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten season schedule data from /schedule/{season}."""
    rows: list[dict[str, Any]] = []
    for tournament in payload.get("tournaments", []):
        champions = tournament.get("champions") or []
        champion = champions[0] if champions else {}
        course_data = tournament.get("courseData") or {}
        standings = tournament.get("standings") or {}
        rows.append(
            {
                "season": payload.get("season"),
                "tour_code": payload.get("tourCode"),
                "tournament_id": tournament.get("tournamentId"),
                "tournament_name": tournament.get("name"),
                "display_date": tournament.get("displayDate"),
                "month": tournament.get("month"),
                "status": tournament.get("status"),
                "champion_name": champion.get("displayName"),
                "champion_player_id": champion.get("playerId"),
                "champion_earnings": tournament.get("championEarnings"),
                "purse": tournament.get("purse"),
                "standings_heading": standings.get("heading"),
                "standings_value": standings.get("value"),
                "course_name": course_data.get("name"),
                "course_city": course_data.get("city"),
                "course_state_code": course_data.get("stateCode"),
                "course_country": course_data.get("country"),
                "use_tournament_site_url": tournament.get("useTournamentSiteUrl"),
                "tournament_site_url": tournament.get("tournamentSiteUrl"),
                "leaderboard_path_guess": PGA_TOURNAMENT_LEADERBOARD_PATH.format(
                    season=tournament.get("year") or payload.get("season"),
                    slug=_slugify_path_segment(tournament.get("name", "")),
                    tournament_id=tournament.get("tournamentId"),
                ),
            }
        )
    return pd.DataFrame(rows)


def flatten_stat_detail(
    stat_detail: dict[str, Any],
    tracked_stat: TrackedPGAStat | None = None,
) -> pd.DataFrame:
    """Flatten a single PGA stat detail page into one row per player."""
    stat_slug = tracked_stat.slug if tracked_stat else _slugify(stat_detail.get("statTitle", "stat"))
    rows: list[dict[str, Any]] = []
    for player in stat_detail.get("rows", []):
        record: dict[str, Any] = {
            "tour_code": stat_detail.get("tourCode"),
            "season_year": stat_detail.get("year"),
            "display_season": stat_detail.get("displaySeason"),
            "stat_id": str(stat_detail.get("statId")),
            "stat_title": stat_detail.get("statTitle"),
            "stat_description": stat_detail.get("statDescription"),
            "stat_slug": stat_slug,
            "player_id": player.get("playerId"),
            "player_name": player.get("playerName"),
            "country": player.get("country"),
            f"{stat_slug}__rank": _coerce_numeric(player.get("rank")),
            f"{stat_slug}__rank_diff": _coerce_numeric(player.get("rankDiff")),
            f"{stat_slug}__rank_change_tendency": player.get("rankChangeTendency"),
        }
        for stat in player.get("stats", []):
            stat_name = _slugify(stat.get("statName", "value"))
            record[f"{stat_slug}__{stat_name}"] = _coerce_numeric(stat.get("statValue"))
        rows.append(record)

    return pd.DataFrame(rows)


def flatten_tournament_results(
    leaderboard_payload: dict[str, Any],
    tournament_payload: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Flatten a tournament leaderboard into player-level result rows and labels."""
    tournament_meta = tournament_payload or {}
    courses = tournament_meta.get("courses") or []
    host_course = next((course for course in courses if course.get("hostCourse")), courses[0] if courses else {})
    rows: list[dict[str, Any]] = []

    if isinstance(leaderboard_payload.get("leaderboard"), list):
        for team_row in leaderboard_payload.get("leaderboard", []):
            players = list(team_row.get("players") or [])
            position = team_row.get("position")
            position_numeric = _parse_position_numeric(position)
            rounds = list(team_row.get("rounds") or [])
            for player in players:
                teammate_names = [
                    teammate.get("displayName")
                    for teammate in players
                    if str(teammate.get("id")) != str(player.get("id"))
                ]
                teammate_ids = [
                    teammate.get("id")
                    for teammate in players
                    if str(teammate.get("id")) != str(player.get("id"))
                ]
                rows.append(
                    {
                        "season_year": tournament_meta.get("seasonYear"),
                        "tournament_id": leaderboard_payload.get("id") or tournament_meta.get("id"),
                        "tournament_name": tournament_meta.get("tournamentName"),
                        "course_id": team_row.get("courseId") or host_course.get("id"),
                        "course_code": host_course.get("courseCode"),
                        "course_name_live": host_course.get("courseName"),
                        "tournament_status": leaderboard_payload.get("tournamentStatus") or tournament_meta.get("tournamentStatus"),
                        "round_header": leaderboard_payload.get("currentRoundScoringFormat") or leaderboard_payload.get("leaderboardRoundHeader"),
                        "format_type": leaderboard_payload.get("formatType") or tournament_meta.get("formatType"),
                        "team_event": True,
                        "team_id": team_row.get("teamId"),
                        "team_name": team_row.get("teamName"),
                        "teammate_ids": "|".join(str(value) for value in teammate_ids if value is not None) or None,
                        "teammate_names": " / ".join(name for name in teammate_names if name) or None,
                        "player_id": player.get("id"),
                        "player_name": player.get("displayName"),
                        "country": player.get("country"),
                        "position": position,
                        "position_numeric": position_numeric,
                        "total_score": team_row.get("total"),
                        "total_score_sort": _coerce_numeric(team_row.get("totalSort")),
                        "total_strokes": _coerce_numeric(team_row.get("totalStrokes")),
                        "thru": team_row.get("thru"),
                        "round_score": team_row.get("score"),
                        "player_state": team_row.get("status"),
                        "current_round": _coerce_numeric(team_row.get("currentRound")),
                        "official_position": player.get("official"),
                        "official_position_sort": _coerce_numeric(player.get("official")),
                        "projected_fedex_rank": _coerce_numeric(player.get("projected")),
                        "projected_fedex_rank_sort": _coerce_numeric(player.get("projected")),
                        "won": bool(position_numeric == 1),
                        "top_5": bool(position_numeric is not None and position_numeric <= 5),
                        "top_10": bool(position_numeric is not None and position_numeric <= 10),
                        "made_cut": True,
                        "withdrawn": False,
                        "round_1_score": rounds[0] if len(rounds) > 0 else None,
                        "round_2_score": rounds[1] if len(rounds) > 1 else None,
                        "round_3_score": rounds[2] if len(rounds) > 2 else None,
                        "round_4_score": rounds[3] if len(rounds) > 3 else None,
                    }
                )
        return pd.DataFrame(rows)

    if isinstance(leaderboard_payload.get("matches"), list):
        raise RuntimeError("Unsupported cup/match-play leaderboard format")

    for player_row in leaderboard_payload.get("players", []):
        player = player_row.get("player")
        scoring = player_row.get("scoringData")
        if not player or not scoring:
            continue

        position = scoring.get("position")
        position_numeric = _parse_position_numeric(position)
        player_state = str(scoring.get("playerState") or "").strip().upper()
        position_code = str(position or "").strip().upper()
        made_cut = position_code not in _NON_FINISH_POSITIONS and player_state != "WITHDRAWN"
        withdrawn = position_code == "WD" or player_state == "WITHDRAWN"
        rounds = list(scoring.get("rounds") or [])

        rows.append(
            {
                "season_year": tournament_meta.get("seasonYear"),
                "tournament_id": leaderboard_payload.get("tournamentId") or tournament_meta.get("id"),
                "tournament_name": tournament_meta.get("tournamentName"),
                "course_id": host_course.get("id"),
                "course_code": host_course.get("courseCode"),
                "course_name_live": host_course.get("courseName"),
                "tournament_status": leaderboard_payload.get("tournamentStatus") or tournament_meta.get("tournamentStatus"),
                "round_header": leaderboard_payload.get("leaderboardRoundHeader"),
                "format_type": leaderboard_payload.get("formatType") or tournament_meta.get("formatType"),
                "team_event": False,
                "team_id": None,
                "team_name": None,
                "teammate_ids": None,
                "teammate_names": None,
                "player_id": player.get("id"),
                "player_name": player.get("displayName"),
                "country": player.get("country"),
                "position": position,
                "position_numeric": position_numeric,
                "total_score": scoring.get("total"),
                "total_score_sort": _coerce_numeric(scoring.get("totalSort")),
                "total_strokes": _coerce_numeric(scoring.get("totalStrokes")),
                "thru": scoring.get("thru"),
                "round_score": scoring.get("score"),
                "player_state": scoring.get("playerState"),
                "current_round": _coerce_numeric(scoring.get("currentRound")),
                "official_position": scoring.get("official"),
                "official_position_sort": _coerce_numeric(scoring.get("officialSort")),
                "projected_fedex_rank": _coerce_numeric(scoring.get("projected")),
                "projected_fedex_rank_sort": _coerce_numeric(scoring.get("projectedSort")),
                "won": bool(position_numeric == 1),
                "top_5": bool(position_numeric is not None and position_numeric <= 5),
                "top_10": bool(position_numeric is not None and position_numeric <= 10),
                "made_cut": bool(made_cut),
                "withdrawn": bool(withdrawn),
                "round_1_score": rounds[0] if len(rounds) > 0 else None,
                "round_2_score": rounds[1] if len(rounds) > 1 else None,
                "round_3_score": rounds[2] if len(rounds) > 2 else None,
                "round_4_score": rounds[3] if len(rounds) > 3 else None,
            }
        )
    return pd.DataFrame(rows)


def _extract_profile_summary(profile_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = profile_payload or {}
    summary_data = payload.get("summaryData") or {}
    if isinstance(summary_data, dict) and "summaryData" in summary_data:
        nested = summary_data.get("summaryData")
        if isinstance(nested, dict):
            return nested
    if isinstance(summary_data, dict):
        return summary_data
    return {}


def _extract_profile_overview_elements(profile_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = profile_payload or {}
    overview_sections = payload.get("overview") or []
    fields: dict[str, Any] = {}
    for section in overview_sections:
        if section.get("type") != "OVERVIEW_STATS":
            continue
        for item in section.get("items", []):
            item_title = str(item.get("title") or "").strip().lower()
            if item_title != "bio":
                continue
            for element in item.get("elements", []):
                title = str(element.get("title") or "").strip()
                if title:
                    fields[title] = element.get("data")
    return fields


def flatten_player_profile(
    *,
    player_id: str,
    player_name: str | None,
    profile_payload: dict[str, Any] | None,
    sponsors_payload: dict[str, Any] | None,
    source_path: str,
) -> dict[str, Any]:
    """Flatten the public PGA player page into stable profile features."""
    summary = _extract_profile_summary(profile_payload)
    bio_fields = _extract_profile_overview_elements(profile_payload)
    sponsors = (sponsors_payload or {}).get("sponsors") or []
    equipment_sponsors = [
        str(sponsor.get("sponsor") or "").strip().upper()
        for sponsor in sponsors
        if str(sponsor.get("type") or "").strip().upper() == "EQUIPMENT"
        and str(sponsor.get("sponsor") or "").strip()
    ]
    socials = summary.get("socials") or []
    social_types = {str(social.get("type") or "").strip().lower() for social in socials if social}

    born_date = _coerce_date(summary.get("born"))
    age_current = _coerce_numeric(summary.get("age"))
    turned_pro_year = _coerce_year(summary.get("turnedPro") or bio_fields.get("Turned Pro"))
    height_text = bio_fields.get("Height")
    return {
        "player_id": str(player_id),
        "player_name": player_name or " ".join(
            part for part in [summary.get("firstName"), summary.get("lastName")] if part
        ).strip(),
        "profile_source_path": source_path,
        "profile_first_name": summary.get("firstName"),
        "profile_last_name": summary.get("lastName"),
        "profile_country": summary.get("country"),
        "profile_country_code": summary.get("countryCode"),
        "profile_born_date": born_date.isoformat() if born_date else None,
        "profile_born_year": born_date.year if born_date else None,
        "profile_age_current": age_current,
        "profile_turned_pro_year": turned_pro_year,
        "profile_birthplace": summary.get("birthplace"),
        "profile_college": summary.get("college"),
        "profile_height_text": height_text,
        "profile_height_inches": _parse_height_inches(height_text),
        "profile_equipment_sponsor_primary": equipment_sponsors[0] if equipment_sponsors else None,
        "profile_equipment_sponsor_count": len(equipment_sponsors),
        "profile_has_equipment_sponsor": bool(equipment_sponsors),
        "profile_social_count": len(social_types),
        "profile_has_instagram": "instagram" in social_types,
        "profile_has_x": "x" in social_types,
        "profile_has_twitter": "twitter" in social_types,
        "profile_has_facebook": "facebook" in social_types,
        "profile_has_tiktok": "tiktok" in social_types,
        "profile_has_youtube": "youtube" in social_types,
    }


def build_player_feature_snapshot(stat_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple stat detail frames into a single player feature table."""
    non_empty_frames = [frame.copy() for frame in stat_frames if frame is not None and not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()

    base_columns = ["player_id", "player_name", "country"]
    feature_frame = non_empty_frames[0]
    duplicate_meta = ["tour_code", "season_year", "display_season", "stat_id", "stat_title", "stat_description", "stat_slug"]
    feature_frame = feature_frame.drop(columns=[c for c in duplicate_meta if c in feature_frame.columns], errors="ignore")

    for frame in non_empty_frames[1:]:
        frame_to_merge = frame.drop(columns=[c for c in duplicate_meta if c in frame.columns], errors="ignore")
        feature_frame = feature_frame.merge(frame_to_merge, how="outer", on=base_columns)

    ordered_columns = base_columns + [c for c in feature_frame.columns if c not in base_columns]
    return feature_frame[ordered_columns].sort_values(by=["player_name", "player_id"]).reset_index(drop=True)


class PGATourStatsClient:
    """HTTP client for PGA Tour stats pages backed by Next.js payload extraction."""

    def __init__(
        self,
        *,
        base_url: str = PGA_TOUR_BASE_URL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )

    def _fetch_next_data(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()

        match = _NEXT_DATA_PATTERN.search(response.text)
        if not match:
            raise RuntimeError(f"Unable to locate __NEXT_DATA__ payload for {response.url}")
        return json.loads(match.group(1))

    @staticmethod
    def _find_query_data(next_data: dict[str, Any], query_name: str) -> list[tuple[list[Any], Any]]:
        dehydrated = (
            next_data.get("props", {})
            .get("pageProps", {})
            .get("dehydratedState", {})
            .get("queries", [])
        )
        matches: list[tuple[list[Any], Any]] = []
        for query in dehydrated:
            key = query.get("queryKey") or []
            if key and key[0] == query_name:
                matches.append((key, query.get("state", {}).get("data")))
        return matches

    def get_stats_index(self) -> dict[str, Any]:
        """Fetch /stats and return the key dehydrated queries we care about."""
        next_data = self._fetch_next_data(PGA_STATS_PATH)
        bundle: dict[str, Any] = {"page_path": PGA_STATS_PATH}
        for query_name in ("tournament", "currentLeaders", "statOverview", "courseStatsOverview"):
            matches = self._find_query_data(next_data, query_name)
            if matches:
                bundle[query_name] = matches[0][1]
        return bundle

    def get_schedule(self, season: int | None = None) -> dict[str, Any]:
        """Fetch the PGA Tour schedule payload for a season."""
        path = PGA_SCHEDULE_SEASON_PATH.format(season=season) if season is not None else PGA_SCHEDULE_PATH
        next_data = self._fetch_next_data(path)
        matches = self._find_query_data(next_data, "schedule")
        if not matches:
            raise RuntimeError(f"No schedule payload found for path={path}")
        return matches[0][1]

    def build_tournament_leaderboard_path(self, season: int | str, tournament_name: str, tournament_id: str) -> str:
        """Build the best-guess public leaderboard path for a PGA tournament."""
        return PGA_TOURNAMENT_LEADERBOARD_PATH.format(
            season=season,
            slug=_slugify_path_segment(tournament_name),
            tournament_id=tournament_id,
        )

    def get_tournament_leaderboard(
        self,
        *,
        season: int | str,
        tournament_id: str,
        tournament_name: str,
    ) -> dict[str, Any]:
        """Fetch a tournament leaderboard page by trying likely slug variants."""
        last_error: Exception | None = None
        for slug in _tournament_slug_candidates(tournament_name):
            path = PGA_TOURNAMENT_LEADERBOARD_PATH.format(
                season=season,
                slug=slug,
                tournament_id=tournament_id,
            )
            try:
                next_data = self._fetch_next_data(path)
                leaderboard_matches: list[tuple[list[Any], Any]] = []
                leaderboard_query_name: str | None = None
                for query_name in ("leaderboard", "teamStrokePlayLeaderboard", "cupTournamentLeaderboard"):
                    leaderboard_matches = self._find_query_data(next_data, query_name)
                    if leaderboard_matches:
                        leaderboard_query_name = query_name
                        break
                tournament_matches = self._find_query_data(next_data, "tournament")
                if not leaderboard_matches:
                    continue
                return {
                    "path": path,
                    "leaderboard": leaderboard_matches[0][1],
                    "leaderboard_query_name": leaderboard_query_name,
                    "tournament": tournament_matches[0][1] if tournament_matches else {},
                }
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.debug(
                    "Failed fetching leaderboard for %s using slug '%s': %s",
                    tournament_id,
                    slug,
                    exc,
                )
        raise RuntimeError(
            f"Unable to fetch leaderboard for tournament_id={tournament_id}, season={season}"
        ) from last_error

    def get_player_profile(
        self,
        *,
        player_id: str,
        player_name: str | None = None,
    ) -> dict[str, Any]:
        """Fetch the public player page and extract stable profile payloads."""
        path = PGA_PLAYER_PROFILE_PATH.format(player_id=player_id)
        next_data = self._fetch_next_data(path)
        profile_matches = self._find_query_data(next_data, "playerProfileOverview")
        summary_matches = self._find_query_data(next_data, "playerProfileSummaryData")
        sponsor_matches = self._find_query_data(next_data, "additionalSponsors")
        profile_payload = profile_matches[0][1] if profile_matches else {}
        summary_payload = summary_matches[0][1] if summary_matches else {}
        sponsors_payload = sponsor_matches[0][1] if sponsor_matches else {}
        flattened = flatten_player_profile(
            player_id=player_id,
            player_name=player_name,
            profile_payload=profile_payload or summary_payload,
            sponsors_payload=sponsors_payload,
            source_path=path,
        )
        return {
            "path": path,
            "profile": profile_payload,
            "summary": summary_payload,
            "sponsors": sponsors_payload,
            "flattened": flattened,
        }

    def list_available_stats(self) -> pd.DataFrame:
        """Return the full stat catalog as a DataFrame."""
        bundle = self.get_stats_index()
        stat_overview = bundle.get("statOverview") or {}
        return flatten_stat_catalog(stat_overview)

    def get_stat_detail(
        self,
        stat_id: str,
        *,
        year: int | None = None,
        tour_code: str = DEFAULT_TOUR_CODE,
        event_query: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a single stat detail payload from /stats/detail/{stat_id}."""
        params: dict[str, Any] = {}
        if year is not None:
            params["year"] = year
        if tour_code:
            params["tourCode"] = tour_code
        if event_query:
            params["eventQuery"] = event_query

        next_data = self._fetch_next_data(PGA_STAT_DETAIL_PATH.format(stat_id=stat_id), params=params or None)
        matches = self._find_query_data(next_data, "statDetails")
        for query_key, payload in matches:
            key_params = query_key[1] if len(query_key) > 1 and isinstance(query_key[1], dict) else {}
            if str(key_params.get("statId")) == str(stat_id):
                return payload
        if matches:
            return matches[0][1]
        raise RuntimeError(f"No statDetails payload found for stat_id={stat_id}")

    def fetch_tracked_stat_details(
        self,
        tracked_stats: Iterable[TrackedPGAStat] | None = None,
        *,
        year: int | None = None,
        tour_code: str = DEFAULT_TOUR_CODE,
    ) -> dict[str, dict[str, Any]]:
        """Fetch a curated set of tracked stat detail payloads keyed by stat ID."""
        tracked = tuple(tracked_stats or DEFAULT_TRACKED_STATS)
        results: dict[str, dict[str, Any]] = {}
        for tracked_stat in tracked:
            logger.info("Fetching PGA stat detail %s (%s)", tracked_stat.stat_id, tracked_stat.title)
            results[tracked_stat.stat_id] = self.get_stat_detail(
                tracked_stat.stat_id,
                year=year,
                tour_code=tour_code,
            )
        return results
