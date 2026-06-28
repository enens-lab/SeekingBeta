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

ATHLETE_EVENTS_URL = (
    "https://raw.githubusercontent.com/rgriff23/Olympic_history/master/data/athlete_events.csv"
)

# One olympics key serves both seasons as separate "tours" (like soccer leagues).
# Each NEXT_*_GAMES is the real next edition the medal table is projected toward.
# (Historical training data ends at Rio 2016 / Sochi 2014; projections roll the
# model forward from the latest available editions.)
# start_date / end_date (YYYYMMDD) let the export tell when an edition has ended, so a
# finished Games never lingers under "Upcoming". Winter rolls forward once its Games
# end: the 2026 Milan-Cortina Winter Games are now over -> next is 2030 French Alps
# (dates approximate until the IOC confirms the 2030 schedule).
NEXT_SUMMER_GAMES = {"season": "Summer", "year": 2028, "city": "Los Angeles", "host_noc": "USA",
                     "tour": "Summer Olympics 2028", "start_date": 20280714, "end_date": 20280730}
NEXT_WINTER_GAMES = {"season": "Winter", "year": 2030, "city": "French Alps", "host_noc": "FRA",
                     "tour": "Winter Olympics 2030", "start_date": 20300208, "end_date": 20300224}
OLYMPIC_EDITIONS = (NEXT_SUMMER_GAMES, NEXT_WINTER_GAMES)

# Back-compat alias (older imports referenced TOUR_NAME = "Summer Olympics").
TOUR_NAME = "Summer Olympics"

# NOC -> display country name for the medal-table board.
NOC_DISPLAY: dict[str, str] = {
    "USA": "United States", "GBR": "Great Britain", "CHN": "China", "RUS": "Russia",
    "GER": "Germany", "FRA": "France", "ITA": "Italy", "AUS": "Australia",
    "JPN": "Japan", "NED": "Netherlands", "KOR": "South Korea", "CAN": "Canada",
    "BRA": "Brazil", "ESP": "Spain", "HUN": "Hungary", "SWE": "Sweden",
    "URS": "Soviet Union", "GDR": "East Germany", "CUB": "Cuba", "KEN": "Kenya",
    "JAM": "Jamaica", "NZL": "New Zealand", "NOR": "Norway", "UKR": "Ukraine",
    # Winter-strong nations
    "AUT": "Austria", "SUI": "Switzerland", "FIN": "Finland", "SWZ": "Switzerland",
    "TCH": "Czechoslovakia", "CZE": "Czechia", "POL": "Poland", "EUN": "Unified Team",
}


def noc_display_name(noc: str) -> str:
    return NOC_DISPLAY.get(noc, noc)
