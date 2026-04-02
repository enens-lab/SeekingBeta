from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote


SPORTS_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sports"
HEADSHOT_REGISTRY_PATH = SPORTS_DATA_DIR / "player_headshots.json"


def seeded_avatar_url(name: str, sport: str, tour: str | None = None) -> str:
    key = (tour or sport).upper()
    palette = {
        "GOLF": "0f766e,14b8a6,99f6e4",
        "PGA": "0f766e,14b8a6,99f6e4",
        "LPGA": "0f766e,14b8a6,99f6e4",
        "TENNIS": "2563eb,38bdf8,fdba74",
        "ATP": "2563eb,38bdf8,fdba74",
        "WTA": "ec4899,f97316,f9a8d4",
        "MLB": "1d4ed8,38bdf8,94a3b8",
    }.get(key, "2563eb,38bdf8,fdba74")
    return (
        "https://api.dicebear.com/9.x/adventurer-neutral/svg"
        f"?seed={quote(name)}&backgroundColor={palette}"
    )


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Any]:
    if not HEADSHOT_REGISTRY_PATH.exists():
        return {}
    try:
        payload = json.loads(HEADSHOT_REGISTRY_PATH.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def resolve_player_image(
    *,
    name: str,
    sport: str,
    tour: str | None = None,
    player_id: str | int | None = None,
    player_key: str | None = None,
) -> str:
    registry = _load_registry()
    sport_entries = registry.get(sport.lower(), {}) if isinstance(registry.get(sport.lower(), {}), dict) else {}

    candidate_keys = [
        str(player_id) if player_id is not None else None,
        player_key,
        name.strip().lower(),
    ]
    for key in candidate_keys:
        if not key:
            continue
        value = sport_entries.get(str(key))
        if isinstance(value, str) and value.strip():
            return value.strip()

    return seeded_avatar_url(name, sport, tour)
