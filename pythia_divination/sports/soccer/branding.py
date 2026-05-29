"""Branding helpers for soccer clubs.

Unlike the US sports (which ship a static team->logo map), soccer spans
hundreds of clubs across competitions, so we read branding directly from the
ESPN fixture payload (each competitor carries ``team.logo`` / ``team.color``).
"""

from __future__ import annotations

from typing import Any

DEFAULT_PRIMARY_COLOR = "#2a6f4f"  # pitch green fallback


def _hexify(color: str | None) -> str | None:
    if not color:
        return None
    color = color.strip()
    if not color:
        return None
    return color if color.startswith("#") else f"#{color}"


def branding_from_espn_team(team: dict[str, Any] | None) -> dict[str, Any]:
    """Build a SportsTeamDetails-compatible dict from an ESPN team object."""
    if not isinstance(team, dict):
        return {"primaryColor": DEFAULT_PRIMARY_COLOR}

    logo_url = team.get("logo")
    if not logo_url:
        logos = team.get("logos")
        if isinstance(logos, list) and logos and isinstance(logos[0], dict):
            logo_url = logos[0].get("href")

    return {
        "abbreviation": team.get("abbreviation") or team.get("shortDisplayName"),
        "logoUrl": logo_url,
        "primaryColor": _hexify(team.get("color")) or DEFAULT_PRIMARY_COLOR,
        "secondaryColor": _hexify(team.get("alternateColor")),
        "venue": (team.get("venue") or {}).get("fullName") if isinstance(team.get("venue"), dict) else None,
    }
