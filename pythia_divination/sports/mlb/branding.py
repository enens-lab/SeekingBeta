"""Static MLB branding assets keyed by MLB team id."""

from __future__ import annotations

from typing import Any


MLB_TEAM_BRANDING: dict[int, dict[str, str]] = {
    108: {"abbreviation": "LAA", "logoUrl": "https://www.mlbstatic.com/team-logos/108.svg", "primaryColor": "#BA0021", "secondaryColor": "#003263"},
    109: {"abbreviation": "AZ", "logoUrl": "https://www.mlbstatic.com/team-logos/109.svg", "primaryColor": "#A71930", "secondaryColor": "#000000"},
    110: {"abbreviation": "BAL", "logoUrl": "https://www.mlbstatic.com/team-logos/110.svg", "primaryColor": "#DF4601", "secondaryColor": "#000000"},
    111: {"abbreviation": "BOS", "logoUrl": "https://www.mlbstatic.com/team-logos/111.svg", "primaryColor": "#BD3039", "secondaryColor": "#0C2340"},
    112: {"abbreviation": "CHC", "logoUrl": "https://www.mlbstatic.com/team-logos/112.svg", "primaryColor": "#0E3386", "secondaryColor": "#CC3433"},
    113: {"abbreviation": "CIN", "logoUrl": "https://www.mlbstatic.com/team-logos/113.svg", "primaryColor": "#C6011F", "secondaryColor": "#000000"},
    114: {"abbreviation": "CLE", "logoUrl": "https://www.mlbstatic.com/team-logos/114.svg", "primaryColor": "#0C2340", "secondaryColor": "#E31937"},
    115: {"abbreviation": "COL", "logoUrl": "https://www.mlbstatic.com/team-logos/115.svg", "primaryColor": "#33006F", "secondaryColor": "#C4CED4"},
    116: {"abbreviation": "DET", "logoUrl": "https://www.mlbstatic.com/team-logos/116.svg", "primaryColor": "#0C2340", "secondaryColor": "#FA4616"},
    117: {"abbreviation": "HOU", "logoUrl": "https://www.mlbstatic.com/team-logos/117.svg", "primaryColor": "#002D62", "secondaryColor": "#EB6E1F"},
    118: {"abbreviation": "KC", "logoUrl": "https://www.mlbstatic.com/team-logos/118.svg", "primaryColor": "#004687", "secondaryColor": "#BD9B60"},
    119: {"abbreviation": "LAD", "logoUrl": "https://www.mlbstatic.com/team-logos/119.svg", "primaryColor": "#005A9C", "secondaryColor": "#EF3E42"},
    120: {"abbreviation": "WSH", "logoUrl": "https://www.mlbstatic.com/team-logos/120.svg", "primaryColor": "#AB0003", "secondaryColor": "#14225A"},
    121: {"abbreviation": "NYM", "logoUrl": "https://www.mlbstatic.com/team-logos/121.svg", "primaryColor": "#002D72", "secondaryColor": "#FF5910"},
    133: {"abbreviation": "ATH", "logoUrl": "https://www.mlbstatic.com/team-logos/133.svg", "primaryColor": "#003831", "secondaryColor": "#EFB21E"},
    134: {"abbreviation": "PIT", "logoUrl": "https://www.mlbstatic.com/team-logos/134.svg", "primaryColor": "#FDB827", "secondaryColor": "#27251F"},
    135: {"abbreviation": "SD", "logoUrl": "https://www.mlbstatic.com/team-logos/135.svg", "primaryColor": "#2F241D", "secondaryColor": "#FFC425"},
    136: {"abbreviation": "SEA", "logoUrl": "https://www.mlbstatic.com/team-logos/136.svg", "primaryColor": "#0C2C56", "secondaryColor": "#005C5C"},
    137: {"abbreviation": "SF", "logoUrl": "https://www.mlbstatic.com/team-logos/137.svg", "primaryColor": "#FD5A1E", "secondaryColor": "#27251F"},
    138: {"abbreviation": "STL", "logoUrl": "https://www.mlbstatic.com/team-logos/138.svg", "primaryColor": "#C41E3A", "secondaryColor": "#0C2340"},
    139: {"abbreviation": "TB", "logoUrl": "https://www.mlbstatic.com/team-logos/139.svg", "primaryColor": "#092C5C", "secondaryColor": "#8FBCE6"},
    140: {"abbreviation": "TEX", "logoUrl": "https://www.mlbstatic.com/team-logos/140.svg", "primaryColor": "#003278", "secondaryColor": "#C0111F"},
    141: {"abbreviation": "TOR", "logoUrl": "https://www.mlbstatic.com/team-logos/141.svg", "primaryColor": "#134A8E", "secondaryColor": "#E8291C"},
    142: {"abbreviation": "MIN", "logoUrl": "https://www.mlbstatic.com/team-logos/142.svg", "primaryColor": "#002B5C", "secondaryColor": "#D31145"},
    143: {"abbreviation": "PHI", "logoUrl": "https://www.mlbstatic.com/team-logos/143.svg", "primaryColor": "#E81828", "secondaryColor": "#002D72"},
    144: {"abbreviation": "ATL", "logoUrl": "https://www.mlbstatic.com/team-logos/144.svg", "primaryColor": "#CE1141", "secondaryColor": "#13274F"},
    145: {"abbreviation": "CWS", "logoUrl": "https://www.mlbstatic.com/team-logos/145.svg", "primaryColor": "#27251F", "secondaryColor": "#C4CED4"},
    146: {"abbreviation": "MIA", "logoUrl": "https://www.mlbstatic.com/team-logos/146.svg", "primaryColor": "#00A3E0", "secondaryColor": "#EF3340"},
    147: {"abbreviation": "NYY", "logoUrl": "https://www.mlbstatic.com/team-logos/147.svg", "primaryColor": "#0C2340", "secondaryColor": "#C4CED4"},
    158: {"abbreviation": "MIL", "logoUrl": "https://www.mlbstatic.com/team-logos/158.svg", "primaryColor": "#12284B", "secondaryColor": "#FFC52F"},
}


def get_team_branding(team_id: int | None, abbreviation: str | None = None) -> dict[str, Any]:
    if team_id is None:
        return {
            "teamId": None,
            "abbreviation": abbreviation,
            "logoUrl": None,
            "primaryColor": None,
            "secondaryColor": None,
        }

    team_id_int = int(team_id)
    branding = MLB_TEAM_BRANDING.get(team_id_int, {})
    return {
        "teamId": team_id_int,
        "abbreviation": abbreviation or branding.get("abbreviation"),
        "logoUrl": branding.get("logoUrl"),
        "primaryColor": branding.get("primaryColor"),
        "secondaryColor": branding.get("secondaryColor"),
    }
