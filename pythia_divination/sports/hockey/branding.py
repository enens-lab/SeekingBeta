"""Static branding helpers for Hockey teams.

TODO: populate `_TEAM_BRANDING` with NHL team logo URLs and primary/secondary
colors before the `/api/sports/hockey/boards` route and iOS / web rendering
land. Mirror the football package's shape — each team code maps to a dict
with `logoUrl`, `primaryColor`, `secondaryColor`. The 32 team codes are
already declared in `sports.hockey.constants.TEAM_CODES`. ESPN exposes NHL
logos at `https://a.espncdn.com/i/teamlogos/nhl/500/<code>.png` if a free
mirror is acceptable.
"""

from __future__ import annotations

from typing import Any


_TEAM_BRANDING: dict[str, dict[str, str]] = {
    # Intentionally empty for now. See module docstring.
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
