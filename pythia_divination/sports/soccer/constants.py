"""Static soccer constants for ingestion and modeling.

Data sources (all free, matching the existing scrape/public-API posture):
  * football-data.co.uk  -> deep historical results + goals + bookmaker odds
                            (one CSV per league per season). Training backbone.
  * ESPN hidden soccer API -> live/upcoming fixtures, team names, logos.
"""

from __future__ import annotations

import re

DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36 SeekingBetaAI/1.0"
)

SPORT_KEY = "soccer"

FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# Current European season start year (2025/26). football-data season codes are
# the last two digits of each year, e.g. 2025/26 -> "2526".
CURRENT_SEASON_START = 2025
DEFAULT_HISTORY_SEASONS = 3  # seasons of history to fit Dixon-Coles on


def football_data_season_code(start_year: int) -> str:
    """2025 -> '2526' (football-data.co.uk per-season CSV code)."""
    a = int(start_year) % 100
    b = (int(start_year) + 1) % 100
    return f"{a:02d}{b:02d}"


# Each supported competition. ``fd_code`` is the football-data.co.uk league file
# code (None for competitions without a clean per-league history file, e.g. UCL,
# which is live-fixtures-only in Phase 1). ``espn_slug`` drives the live feed.
LEAGUE_CONFIGS: dict[str, dict[str, str | None]] = {
    "eng.1": {"tour": "Premier League", "fd_code": "E0", "espn_slug": "eng.1", "country": "England"},
    "esp.1": {"tour": "La Liga", "fd_code": "SP1", "espn_slug": "esp.1", "country": "Spain"},
    "ita.1": {"tour": "Serie A", "fd_code": "I1", "espn_slug": "ita.1", "country": "Italy"},
    "ger.1": {"tour": "Bundesliga", "fd_code": "D1", "espn_slug": "ger.1", "country": "Germany"},
    "fra.1": {"tour": "Ligue 1", "fd_code": "F1", "espn_slug": "fra.1", "country": "France"},
    # Live-only in Phase 1 (cross-league strength rating is a follow-on):
    "uefa.champions": {"tour": "UEFA Champions League", "fd_code": None, "espn_slug": "uefa.champions", "country": "Europe"},
}

# Competitions we can predict today (have a fitted Dixon-Coles model).
PREDICTABLE_LEAGUE_KEYS = tuple(k for k, v in LEAGUE_CONFIGS.items() if v["fd_code"])


def tour_for_league(league_key: str) -> str:
    cfg = LEAGUE_CONFIGS.get(league_key)
    return str(cfg["tour"]) if cfg else league_key


# --- Team name normalization -------------------------------------------------
# football-data.co.uk and ESPN spell teams differently ("Man City" vs
# "Manchester City"). Canonicalize so a live ESPN fixture can be matched to the
# historical strengths fitted from football-data.co.uk.

_TEAM_ALIASES: dict[str, str] = {
    # England
    "manchester city": "man city",
    "manchester united": "man united",
    "manchester utd": "man united",
    "newcastle united": "newcastle",
    "tottenham hotspur": "tottenham",
    "spurs": "tottenham",
    "wolverhampton wanderers": "wolves",
    "west ham united": "west ham",
    "brighton hove albion": "brighton",
    "brighton and hove albion": "brighton",
    "nottingham forest": "nottm forest",
    "afc bournemouth": "bournemouth",
    "leicester city": "leicester",
    "leeds united": "leeds",
    "sheffield united": "sheffield united",
    "ipswich town": "ipswich",
    # Spain
    "atletico madrid": "ath madrid",
    "atletico de madrid": "ath madrid",
    "athletic club": "ath bilbao",
    "athletic bilbao": "ath bilbao",
    "real betis": "betis",
    "real sociedad": "sociedad",
    "celta vigo": "celta",
    "deportivo alaves": "alaves",
    "rayo vallecano": "vallecano",
    "real valladolid": "valladolid",
    # Italy
    "internazionale": "inter",
    "inter milan": "inter",
    "ac milan": "milan",
    "as roma": "roma",
    "ssc napoli": "napoli",
    "hellas verona": "verona",
    # Germany
    "bayern munich": "bayern munich",
    "bayer leverkusen": "leverkusen",
    "borussia dortmund": "dortmund",
    "borussia monchengladbach": "m'gladbach",
    "eintracht frankfurt": "ein frankfurt",
    "vfb stuttgart": "stuttgart",
    "rb leipzig": "rb leipzig",
    "fc koln": "fc koln",
    "1 fc koln": "fc koln",
    # France
    "paris saint germain": "paris sg",
    "paris saint-germain": "paris sg",
    "psg": "paris sg",
    "olympique marseille": "marseille",
    "olympique lyonnais": "lyon",
    "as monaco": "monaco",
    "lille osc": "lille",
}

_STRIP_TOKENS = {
    "fc", "cf", "afc", "ac", "as", "ssc", "sc", "club", "calcio", "de",
    "cd", "ud", "rc", "1", "vfb", "vfl", "tsg", "sv", "fsv",
}


def canonical_team_name(name: str | None) -> str:
    """Normalize a club name to a comparable key across data sources."""
    if not name:
        return ""
    text = name.strip().lower()
    text = text.replace("&", "and").replace(".", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in _TEAM_ALIASES:
        return _TEAM_ALIASES[text]
    # Drop common org-type tokens, then re-check aliases on the stripped form.
    tokens = [t for t in text.split(" ") if t and t not in _STRIP_TOKENS]
    stripped = " ".join(tokens)
    return _TEAM_ALIASES.get(stripped, stripped or text)


# --- National-team normalization (World Cup / internationals) ---------------
# Align names across martj42 (training), OpenFootball + ESPN (fixtures).
_NATION_ALIASES: dict[str, str] = {
    "czech republic": "czechia",
    "usa": "united states",
    "us": "united states",
    "korea republic": "south korea",
    "korea dpr": "north korea",
    "ir iran": "iran",
    "china pr": "china",
    "cote d'ivoire": "ivory coast",
    "cabo verde": "cape verde",
    "the gambia": "gambia",
    "republic of ireland": "ireland",
    "bosnia and herzegovina": "bosnia herzegovina",
    "north macedonia": "macedonia",
    "turkiye": "turkey",
    "curacao": "curacao",
}


def canonical_national_name(name: str | None) -> str:
    """Normalize a national-team name to a comparable key across sources."""
    if not name:
        return ""
    text = name.strip().lower()
    text = text.replace("&", "and").replace(".", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _NATION_ALIASES.get(text, text)
