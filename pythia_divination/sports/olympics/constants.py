"""Static Summer Olympics constants.

Historical data: rgriff23's mirror of the "120 years of Olympic history"
dataset (athlete-event rows, Athens 1896 -> Rio 2016), free on GitHub raw.
"""

from __future__ import annotations

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36 SeekingBetaAI/1.0"
)

SPORT_KEY = "olympics"
TOUR_NAME = "Summer Olympics"

ATHLETE_EVENTS_URL = (
    "https://raw.githubusercontent.com/rgriff23/Olympic_history/master/data/athlete_events.csv"
)

# Next Summer Games to project. (Historical training data ends at Rio 2016;
# the projection is trained on all available editions and rolled forward.)
NEXT_SUMMER_GAMES = {"year": 2028, "city": "Los Angeles", "host_noc": "USA"}

# NOC -> display country name for the medal-table board.
NOC_DISPLAY: dict[str, str] = {
    "USA": "United States", "GBR": "Great Britain", "CHN": "China", "RUS": "Russia",
    "GER": "Germany", "FRA": "France", "ITA": "Italy", "AUS": "Australia",
    "JPN": "Japan", "NED": "Netherlands", "KOR": "South Korea", "CAN": "Canada",
    "BRA": "Brazil", "ESP": "Spain", "HUN": "Hungary", "SWE": "Sweden",
    "URS": "Soviet Union", "GDR": "East Germany", "CUB": "Cuba", "KEN": "Kenya",
    "JAM": "Jamaica", "NZL": "New Zealand", "NOR": "Norway", "UKR": "Ukraine",
}


def noc_display_name(noc: str) -> str:
    return NOC_DISPLAY.get(noc, noc)
