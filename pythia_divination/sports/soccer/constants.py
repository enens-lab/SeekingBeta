"""Static soccer constants for ingestion and modeling.

Data sources (all free, matching the existing scrape/public-API posture):
  * football-data.co.uk  -> deep historical results + goals + bookmaker odds
                            (one CSV per league per season). Training backbone.
  * ESPN hidden soccer API -> live/upcoming fixtures, team names, logos.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36 SeekingBetaAI/1.0"
)

SPORT_KEY = "soccer"

FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# European seasons run August -> May; the new season starts in July, when the
# football-data per-season CSV for it appears. football-data season codes are the
# last two digits of each year, e.g. 2025/26 -> "2526".
SEASON_START_MONTH = 7


def current_season_start(today: date | datetime | None = None, start_month: int = SEASON_START_MONTH) -> int:
    """Start year of the European season that `today` falls in (July rollover):
    2026-06-30 -> 2025 (2025/26), 2026-07-01 -> 2026 (2026/27).

    Derived from the date rather than hard-coded: a fixed 2025 meant the 2026/27
    results were never loaded and every promoted team was predicted blind."""
    today = today or datetime.now(timezone.utc).date()
    return today.year if today.month >= start_month else today.year - 1


def season_label(start_year: int) -> str:
    """2025 -> '2025-26' (same convention as sports/market_log.season_cross_year)."""
    return f"{int(start_year)}-{str(int(start_year) + 1)[-2:]}"


# Evaluated at import (each export run is a fresh process). Prefer calling
# current_season_start() in new code.
CURRENT_SEASON_START = current_season_start()
DEFAULT_HISTORY_SEASONS = 6  # current season + the 5-season fitting window behind the 2025-26 record

MODEL_PARAMS_PATH = Path(__file__).resolve().with_name("model_params.json")


def load_model_params(path: Path | None = None) -> dict[str, Any]:
    """Tuned Dixon-Coles hyper-parameters + measured acceptance (committed JSON)."""
    return json.loads(Path(path or MODEL_PARAMS_PATH).read_text())


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
    "nottingham forest": "nott'm forest",
    "nottm forest": "nott'm forest",
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
    "espanyol": "espanol",
    "rcd espanyol": "espanol",
    "deportivo": "la coruna",
    "deportivo la coruna": "la coruna",
    "racing santander": "santander",
    "real oviedo": "oviedo",
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
    "fc cologne": "fc koln",
    "cologne": "fc koln",
    "koln": "fc koln",
    # France
    "paris saint germain": "paris sg",
    "paris saint-germain": "paris sg",
    "psg": "paris sg",
    "olympique marseille": "marseille",
    "olympique lyonnais": "lyon",
    "as monaco": "monaco",
    "lille osc": "lille",
    "stade rennais": "rennes",
    "saint etienne": "st etienne",
}

_STRIP_TOKENS = {
    "fc", "cf", "afc", "ac", "as", "ssc", "sc", "club", "calcio", "de",
    "cd", "ud", "rc", "1", "vfb", "vfl", "tsg", "sv", "fsv",
}


def _ascii_fold(text: str) -> str:
    """'Alavés' -> 'Alaves', 'Mönchengladbach' -> 'Monchengladbach' (ESPN uses
    accents, football-data does not; the old regex turned 'é' into a space)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def canonical_team_name(name: str | None) -> str:
    """Normalize a club name to a comparable key across data sources."""
    if not isinstance(name, str) or not name:
        return ""
    text = _ascii_fold(name).strip().lower()
    text = text.replace("&", "and").replace(".", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in _TEAM_ALIASES:
        return _TEAM_ALIASES[text]
    # Drop common org-type tokens, then re-check aliases on the stripped form.
    tokens = [t for t in text.split(" ") if t and t not in _STRIP_TOKENS]
    stripped = " ".join(tokens)
    return _TEAM_ALIASES.get(stripped, stripped or text)


def resolve_team(name: str | None, known: Iterable[str]) -> str:
    """Map a fixture team name (ESPN) onto the names the model was fitted on
    (football-data, canonicalised). Falls back to a unique token-subset match
    ('Coventry City' -> 'coventry', 'SC Paderborn 07' -> 'paderborn'); returns the
    canonical name unchanged when nothing matches uniquely, which the model then
    treats as an unseen, league-average team."""
    canon = canonical_team_name(name)
    known_set = set(known)
    if not canon or canon in known_set:
        return canon
    tokens = set(canon.split())
    # Only "every token of the known name appears in the fixture name": the fixture
    # side carries the longer official name. The reverse direction would map an
    # unseen 'Paris' onto 'paris sg'.
    matches = [k for k in known_set if set(k.split()) and set(k.split()) <= tokens]
    return matches[0] if len(matches) == 1 else canon


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
    if not isinstance(name, str) or not name:
        return ""
    text = _ascii_fold(name).strip().lower()
    text = text.replace("&", "and").replace(".", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _NATION_ALIASES.get(text, text)
