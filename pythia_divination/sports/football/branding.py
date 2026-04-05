"""Static branding helpers for Football teams."""

from __future__ import annotations

from typing import Any


_TEAM_BRANDING: dict[str, dict[str, str]] = {
    "ARI": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/ari.png", "primaryColor": "#97233f", "secondaryColor": "#000000"},
    "ATL": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/atl.png", "primaryColor": "#a71930", "secondaryColor": "#000000"},
    "BAL": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/bal.png", "primaryColor": "#241773", "secondaryColor": "#000000"},
    "BUF": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png", "primaryColor": "#00338d", "secondaryColor": "#c60c30"},
    "CAR": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/car.png", "primaryColor": "#0085ca", "secondaryColor": "#101820"},
    "CHI": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/chi.png", "primaryColor": "#0b162a", "secondaryColor": "#c83803"},
    "CIN": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/cin.png", "primaryColor": "#fb4f14", "secondaryColor": "#000000"},
    "CLE": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/cle.png", "primaryColor": "#311d00", "secondaryColor": "#ff3c00"},
    "DAL": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png", "primaryColor": "#003594", "secondaryColor": "#869397"},
    "DEN": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/den.png", "primaryColor": "#fb4f14", "secondaryColor": "#002244"},
    "DET": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/det.png", "primaryColor": "#0076b6", "secondaryColor": "#b0b7bc"},
    "GB": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png", "primaryColor": "#203731", "secondaryColor": "#ffb612"},
    "HOU": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png", "primaryColor": "#03202f", "secondaryColor": "#a71930"},
    "IND": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/ind.png", "primaryColor": "#002c5f", "secondaryColor": "#a2aaad"},
    "JAX": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/jax.png", "primaryColor": "#006778", "secondaryColor": "#9f792c"},
    "KC": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png", "primaryColor": "#e31837", "secondaryColor": "#ffb81c"},
    "LA": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png", "primaryColor": "#003594", "secondaryColor": "#ffa300"},
    "LAC": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/lac.png", "primaryColor": "#0080c6", "secondaryColor": "#ffc20e"},
    "LAR": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png", "primaryColor": "#003594", "secondaryColor": "#ffa300"},
    "LV": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/lv.png", "primaryColor": "#000000", "secondaryColor": "#a5acaf"},
    "MIA": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/mia.png", "primaryColor": "#008e97", "secondaryColor": "#fc4c02"},
    "MIN": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/min.png", "primaryColor": "#4f2683", "secondaryColor": "#ffc62f"},
    "NE": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/ne.png", "primaryColor": "#002244", "secondaryColor": "#c60c30"},
    "NO": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/no.png", "primaryColor": "#d3bc8d", "secondaryColor": "#101820"},
    "NYG": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png", "primaryColor": "#0b2265", "secondaryColor": "#a71930"},
    "NYJ": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png", "primaryColor": "#125740", "secondaryColor": "#000000"},
    "PHI": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png", "primaryColor": "#004c54", "secondaryColor": "#a5acaf"},
    "PIT": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/pit.png", "primaryColor": "#ffb612", "secondaryColor": "#101820"},
    "SEA": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/sea.png", "primaryColor": "#002244", "secondaryColor": "#69be28"},
    "SF": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png", "primaryColor": "#aa0000", "secondaryColor": "#b3995d"},
    "TB": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/tb.png", "primaryColor": "#d50a0a", "secondaryColor": "#34302b"},
    "TEN": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/ten.png", "primaryColor": "#0c2340", "secondaryColor": "#4b92db"},
    "WAS": {"logoUrl": "https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png", "primaryColor": "#5a1414", "secondaryColor": "#ffb612"},
}


def get_team_branding(team_abbr: str | None) -> dict[str, Any]:
    key = str(team_abbr or "").strip().upper()
    branding = _TEAM_BRANDING.get(key, {})
    return {
        "abbreviation": key or None,
        "logoUrl": branding.get("logoUrl"),
        "primaryColor": branding.get("primaryColor"),
        "secondaryColor": branding.get("secondaryColor"),
    }
