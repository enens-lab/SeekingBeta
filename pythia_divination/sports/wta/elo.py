"""Walk-forward tennis Elo on a stable player identity.

Two fixes over the original ``feature_engineering.process_match_history`` Elo:

1. Identity. The 2025+ ATP files (tennismylife, Sackmann format) use ATP's own
   alphanumeric player codes ("SU55") instead of Sackmann's numeric ids, so
   keying on the raw id silently reset every ATP player to 1500 on 1 Jan 2025.
   ``assign_player_keys`` keeps a legacy numeric id where one exists and maps
   any other id (new-scheme or ``fallback-*``) onto the legacy id with the same
   normalised name; only players never seen under a legacy id get a name key.
2. Method. FiveThirtyEight-style decaying K, ``k = k_numerator / (n + k_offset)
   ** k_shape`` with n the player's prior matches, kept separately for an
   overall rating and a per-surface rating; the match probability uses
   ``overall_weight * overall + (1 - overall_weight) * surface``. The committed
   values (200, 10, 0.4, 0.7) were chosen on 2021-22 only; the plan's literal
   250/(n+5)^0.4 with a 50/50 blend scores 64.04% / 0.6292 on 2023-25 and
   0.6330 on ATP 2025 with this data, short of the acceptance bar.

Every number this module produces for a match uses only matches processed
before it (walk-forward). Parameters live in ``model_params.json`` next to this
file, committed with the code, so the RunPod exporter needs no artifact.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PARAMS_PATH = Path(__file__).with_name("model_params.json")

ROUND_ORDER = {
    "Q1": -3.0,
    "Q2": -2.0,
    "Q3": -1.0,
    "ER": 0.0,
    "RR": 0.0,
    "R128": 1.0,
    "R64": 2.0,
    "R32": 3.0,
    "R16": 4.0,
    "QF": 5.0,
    "SF": 6.0,
    "BR": 6.5,
    "F": 7.0,
}

# Team competitions: their "events" are country-vs-country ties or group
# stages, not individual knockout fields, so they never become boards.
TEAM_EVENT_PATTERNS = (
    "davis cup",
    "billie jean king",
    "bjk cup",
    "fed cup",
    "united cup",
    "atp cup",
    "laver cup",
    "hopman cup",
)
TEAM_EVENT_LEVELS = {"D"}

# Short-name repair (see _repair_short_name_keys): with several namesake
# candidates, take one only if she clearly dominates the two seasons before.
REPAIR_MIN_RECENT = 5
REPAIR_DOMINANCE = 5

_WALKOVER_RE = re.compile(r"w/o|walkover|\bdef\b", re.IGNORECASE)
_LEGACY_ID_RE = re.compile(r"^\d+$")
_SHORT_NAME_RE = re.compile(r"^[^\s]+(?:[\s-][^\s]+)*\s+[A-Z]\.(?:[A-Z]\.)*$")
_SHORT_PARTS_RE = re.compile(r"^(?P<last>.+?)\s+(?P<initials>(?:[A-Z]\.)+)$")


def load_params(path: Path | None = None) -> dict[str, Any]:
    with (path or PARAMS_PATH).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_name(name: Any) -> str:
    """ASCII, lower-case, letters only: "Félix Auger-Aliassime" -> "felixaugeraliassime"."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", text)


def normalize_identifier(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def is_walkover(score: Any) -> bool:
    return bool(_WALKOVER_RE.search(str(score or "")))


def is_team_event(level: Any, name: Any) -> bool:
    if str(level or "").strip().upper() in TEAM_EVENT_LEVELS:
        return True
    lowered = str(name or "").lower()
    return any(pattern in lowered for pattern in TEAM_EVENT_PATTERNS)


def canonical_surface(surface: Any, params: dict[str, Any]) -> str:
    text = str(surface or "").strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return params.get("default_surface", "Hard")
    return params.get("surface_aliases", {}).get(text, text)


def round_rank(value: Any) -> float:
    return ROUND_ORDER.get(str(value or "").strip().upper(), 0.0)


def sort_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Chronological order: date, then event, then round, then match number."""
    frame = matches.copy()
    frame["tour"] = frame["tour"].astype(str).str.upper().str.strip()
    frame["tourney_date"] = pd.to_numeric(frame["tourney_date"], errors="coerce")
    frame = frame.dropna(subset=["tourney_date"])
    frame["tourney_date"] = frame["tourney_date"].astype(int)
    frame["_round_rank"] = frame.get("round", pd.Series("", index=frame.index)).map(round_rank)
    frame["_match_num"] = pd.to_numeric(frame.get("match_num"), errors="coerce").fillna(0)
    frame["tourney_id"] = frame["tourney_id"].astype(str)
    frame = frame.sort_values(
        ["tourney_date", "tour", "tourney_id", "_round_rank", "_match_num"],
        kind="mergesort",
    ).reset_index(drop=True)
    return frame.drop(columns=["_round_rank", "_match_num"])


def _looks_like_short_name(name: str) -> bool:
    return bool(_SHORT_NAME_RE.match(name.strip()))


def assign_player_keys(matches: pd.DataFrame) -> pd.DataFrame:
    """Add ``winner_key``/``loser_key`` (stable across id schemes) and display names.

    A legacy numeric id is the key wherever one exists. Any other id is mapped
    to the legacy id that carries the same normalised name on the same tour, if
    exactly one does; otherwise the player is keyed on the normalised name.
    """
    frame = matches.copy()
    frame["tour"] = frame["tour"].astype(str).str.upper().str.strip()

    name_to_legacy: dict[tuple[str, str], set[str]] = {}
    for side in ("winner", "loser"):
        ids = frame[f"{side}_id"].map(normalize_identifier)
        names = frame[f"{side}_name"].map(normalize_name)
        for tour, pid, name in zip(frame["tour"], ids, names):
            if pid and _LEGACY_ID_RE.match(pid) and name:
                name_to_legacy.setdefault((tour, name), set()).add(pid)

    def resolve(tour: str, raw_id: Any, raw_name: Any) -> str | None:
        pid = normalize_identifier(raw_id)
        name = normalize_name(raw_name)
        if pid and _LEGACY_ID_RE.match(pid):
            return f"{tour}:{pid}"
        legacy = name_to_legacy.get((tour, name))
        if legacy and len(legacy) == 1:
            return f"{tour}:{next(iter(legacy))}"
        if name:
            return f"{tour}:name:{name}"
        if pid:
            return f"{tour}:id:{pid}"
        return None

    for side in ("winner", "loser"):
        frame[f"{side}_key"] = [
            resolve(tour, pid, name)
            for tour, pid, name in zip(frame["tour"], frame[f"{side}_id"], frame[f"{side}_name"])
        ]
    return _repair_short_name_keys(frame)


def _short_name_parts(name: str) -> tuple[str, str] | None:
    match = _SHORT_PARTS_RE.match(str(name or "").strip())
    if not match:
        return None
    return normalize_name(match.group("last")), match.group("initials")[0].lower()


def _repair_short_name_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-key tennis-data rows ("Navarro E.") whose converted id has no history.

    The 2025 WTA conversion matched "Last F." names to ids by surname + first
    initial and sometimes landed on a namesake with no matches (Emma Navarro's
    2025 results went to an empty id, restarting her at 1500). A short-name row
    keeps its key when that key's full name agrees with it; otherwise it moves to
    the one full-name player (same tour, same surname ending and initial, active
    within two seasons) if exactly one exists. With several, it moves to the one
    that clearly dominates the two seasons before the row (at least
    ``REPAIR_MIN_RECENT`` matches and ``REPAIR_DOMINANCE`` times the runner-up's:
    "Fernandez L.A." is Leylah Fernandez, 100 matches, not Lya Fernandez, one),
    and is left alone otherwise.
    """
    if not any(frame[f"{side}_name"].astype(str).str.match(_SHORT_PARTS_RE).any() for side in ("winner", "loser")):
        return frame
    full_name: dict[str, str] = {}
    last_seen: dict[str, int] = {}
    played: dict[str, list[int]] = {}
    for side in ("winner", "loser"):
        for key, name, day in zip(frame[f"{side}_key"], frame[f"{side}_name"], frame["tourney_date"]):
            if not key:
                continue
            last_seen[key] = max(last_seen.get(key, 0), int(day))
            played.setdefault(key, []).append(int(day))
            if _short_name_parts(name) is None and normalize_name(name):
                full_name[key] = normalize_name(name)

    def recent_matches(key: str, start: int, end: int) -> int:
        return sum(1 for value in played.get(key, ()) if start <= value < end)

    by_tour: dict[str, list[tuple[str, str]]] = {}
    for key, name in full_name.items():
        by_tour.setdefault(key.split(":", 1)[0], []).append((key, name))

    remap: dict[tuple[str, str], str] = {}
    for side in ("winner", "loser"):
        pairs = frame[[f"{side}_key", f"{side}_name", "tourney_date", "tour"]].drop_duplicates(subset=[f"{side}_key", f"{side}_name"])
        for key, name, day, tour in pairs.itertuples(index=False, name=None):
            parts = _short_name_parts(name)
            if parts is None or not key or (key, name) in remap:
                continue
            surname, initial = parts

            def agrees(full: str) -> bool:
                return full.endswith(surname) and full.startswith(initial)

            if key in full_name and agrees(full_name[key]):
                continue
            horizon = (int(day) // 10000 - 2) * 10000
            candidates = [
                other for other, full in by_tour.get(tour, [])
                if agrees(full) and last_seen.get(other, 0) >= horizon
            ]
            if len(candidates) > 1:
                activity = sorted(((recent_matches(other, horizon, int(day)), other) for other in candidates), reverse=True)
                best, runner_up = activity[0][0], activity[1][0]
                if best >= REPAIR_MIN_RECENT and best >= REPAIR_DOMINANCE * max(runner_up, 1):
                    candidates = [activity[0][1]]
            if len(candidates) == 1:
                remap[(key, name)] = candidates[0]
            elif key in full_name:
                # The id belongs to someone else entirely: do not merge histories.
                remap[(key, name)] = f"{tour}:name:{normalize_name(name)}"
    if not remap:
        return frame
    for side in ("winner", "loser"):
        frame[f"{side}_key"] = [
            remap.get((key, name), key) for key, name in zip(frame[f"{side}_key"], frame[f"{side}_name"])
        ]
    return frame


def readable_name(name: str) -> str:
    """"Fernandez L.A." -> "L.A. Fernandez" (tennis-data short form only)."""
    match = _SHORT_PARTS_RE.match(str(name or "").strip())
    if not match:
        return str(name or "").strip()
    return f"{match.group('initials')} {match.group('last').strip()}"


def display_names(matches: pd.DataFrame) -> dict[str, str]:
    """Most recent full-form name per player key.

    tennis-data conversions write "Osaka N."; prefer the latest name that is not
    in that short form so boards show "Naomi Osaka" throughout.
    """
    names: dict[str, str] = {}
    short: dict[str, str] = {}
    for side in ("winner", "loser"):
        for key, name in zip(matches[f"{side}_key"], matches[f"{side}_name"]):
            if not key or name is None or (isinstance(name, float) and np.isnan(name)):
                continue
            text = re.sub(r"\s+", " ", str(name)).strip()
            if not text:
                continue
            if _looks_like_short_name(text):
                short[key] = text
            else:
                names[key] = text
    for key, text in short.items():
        names.setdefault(key, readable_name(text))
    return names


@dataclass
class EloState:
    """Mutable Elo state. ``expected`` and ``update`` are the whole model."""

    params: dict[str, Any]
    overall: dict[str, float] = field(default_factory=dict)
    surface: dict[tuple[str, str], float] = field(default_factory=dict)
    matches: dict[str, int] = field(default_factory=dict)
    surface_matches: dict[tuple[str, str], int] = field(default_factory=dict)
    last_played: dict[str, int] = field(default_factory=dict)

    def k(self, played: int) -> float:
        p = self.params
        return float(p["k_numerator"]) / ((played + float(p["k_offset"])) ** float(p["k_shape"]))

    def ratings(self, key: str, surface: str) -> tuple[float, float, float]:
        init = float(self.params["initial_rating"])
        overall = self.overall.get(key, init)
        surf = self.surface.get((key, surface), init)
        weight = float(self.params["overall_weight"])
        return overall, surf, weight * overall + (1.0 - weight) * surf

    def expected(self, a: str, b: str, surface: str) -> float:
        _, _, blend_a = self.ratings(a, surface)
        _, _, blend_b = self.ratings(b, surface)
        return 1.0 / (1.0 + 10.0 ** ((blend_b - blend_a) / 400.0))

    def update(self, winner: str, loser: str, surface: str, date_key: int | None = None) -> None:
        init = float(self.params["initial_rating"])
        rw, rl = self.overall.get(winner, init), self.overall.get(loser, init)
        sw, sl = self.surface.get((winner, surface), init), self.surface.get((loser, surface), init)
        e_overall = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))
        e_surface = 1.0 / (1.0 + 10.0 ** ((sl - sw) / 400.0))
        nw, nl = self.matches.get(winner, 0), self.matches.get(loser, 0)
        nsw, nsl = self.surface_matches.get((winner, surface), 0), self.surface_matches.get((loser, surface), 0)
        self.overall[winner] = rw + self.k(nw) * (1.0 - e_overall)
        self.overall[loser] = rl - self.k(nl) * (1.0 - e_overall)
        self.surface[(winner, surface)] = sw + self.k(nsw) * (1.0 - e_surface)
        self.surface[(loser, surface)] = sl - self.k(nsl) * (1.0 - e_surface)
        self.matches[winner] = nw + 1
        self.matches[loser] = nl + 1
        self.surface_matches[(winner, surface)] = nsw + 1
        self.surface_matches[(loser, surface)] = nsl + 1
        if date_key is not None:
            self.last_played[winner] = max(self.last_played.get(winner, 0), int(date_key))
            self.last_played[loser] = max(self.last_played.get(loser, 0), int(date_key))

    def copy(self) -> "EloState":
        return EloState(
            params=self.params,
            overall=dict(self.overall),
            surface=dict(self.surface),
            matches=dict(self.matches),
            surface_matches=dict(self.surface_matches),
            last_played=dict(self.last_played),
        )


def prepare_matches(matches: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Sort, key and flag a raw combined-matches frame (any id scheme)."""
    params = params or load_params()
    frame = sort_matches(matches)
    frame = assign_player_keys(frame)
    frame = frame.dropna(subset=["winner_key", "loser_key"]).reset_index(drop=True)
    frame = frame[frame["winner_key"] != frame["loser_key"]].reset_index(drop=True)
    frame["surface_canon"] = [canonical_surface(value, params) for value in frame.get("surface", "")]
    frame["is_walkover"] = frame.get("score", pd.Series("", index=frame.index)).map(is_walkover)
    frame["is_team_event"] = [
        is_team_event(level, name)
        for level, name in zip(frame.get("tourney_level", ""), frame.get("tourney_name", ""))
    ]
    frame["event_id"] = frame["tour"] + ":" + frame["tourney_id"].astype(str)
    return frame


def run_elo(
    prepared: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    snapshot_dates: Iterable[int] = (),
) -> tuple[pd.DataFrame, EloState, dict[int, EloState]]:
    """Walk-forward pass over prepared matches.

    Returns per-match pre-match values (winner/loser overall, surface and blend
    ratings, prior match counts and the blended P(winner)), the final state, and
    copies of the state taken just before the first match dated on or after each
    requested snapshot date.
    """
    params = params or load_params()
    state = EloState(params=params)
    update_walkovers = bool(params.get("update_on_walkovers", False))
    update_team = bool(params.get("update_on_team_events", True))
    pending = sorted({int(value) for value in snapshot_dates})
    snapshots: dict[int, EloState] = {}

    count = len(prepared)
    out = {
        name: np.empty(count, dtype=float)
        for name in ("w_elo", "l_elo", "w_surf_elo", "l_surf_elo", "w_blend_elo", "l_blend_elo", "p_winner")
    }
    out_n = {name: np.empty(count, dtype=int) for name in ("w_matches", "l_matches")}

    rows = zip(
        prepared["winner_key"].values,
        prepared["loser_key"].values,
        prepared["surface_canon"].values,
        prepared["tourney_date"].values,
        prepared["is_walkover"].values,
        prepared["is_team_event"].values,
    )
    for index, (winner, loser, surface, date_key, walkover, team) in enumerate(rows):
        while pending and int(date_key) >= pending[0]:
            snapshots[pending.pop(0)] = state.copy()
        wo, ws, wb = state.ratings(winner, surface)
        lo, ls, lb = state.ratings(loser, surface)
        out["w_elo"][index], out["l_elo"][index] = wo, lo
        out["w_surf_elo"][index], out["l_surf_elo"][index] = ws, ls
        out["w_blend_elo"][index], out["l_blend_elo"][index] = wb, lb
        out["p_winner"][index] = 1.0 / (1.0 + 10.0 ** ((lb - wb) / 400.0))
        out_n["w_matches"][index] = state.matches.get(winner, 0)
        out_n["l_matches"][index] = state.matches.get(loser, 0)
        if walkover and not update_walkovers:
            continue
        if team and not update_team:
            continue
        state.update(winner, loser, surface, int(date_key))

    for date_key in pending:
        snapshots[date_key] = state.copy()

    result = pd.DataFrame(out, index=prepared.index)
    for name, values in out_n.items():
        result[name] = values
    return result, state, snapshots


def match_metrics(p_winner: np.ndarray) -> dict[str, float]:
    """Accuracy / Brier / log-loss of winner-oriented probabilities (ties count half)."""
    p = np.clip(np.asarray(p_winner, dtype=float), 1e-9, 1 - 1e-9)
    if len(p) == 0:
        return {"n": 0, "accuracy": float("nan"), "brier": float("nan"), "log_loss": float("nan")}
    accuracy = float(np.mean(np.where(np.isclose(p, 0.5), 0.5, (p > 0.5).astype(float))))
    return {
        "n": int(len(p)),
        "accuracy": accuracy,
        "brier": float(np.mean((1.0 - p) ** 2)),
        "log_loss": float(-np.mean(np.log(p))),
    }


def evaluation_mask(prepared: pd.DataFrame, start: int, end: int) -> pd.Series:
    """Matches scored in [start, end): Davis Cup / BJK Cup ties (level D) and
    walkovers excluded. Same definition as the 2026-10 markets plan, so the
    numbers in model_params.json compare like for like."""
    level = prepared.get("tourney_level", pd.Series("", index=prepared.index)).astype(str).str.strip().str.upper()
    return (
        (prepared["tourney_date"] >= start)
        & (prepared["tourney_date"] < end)
        & (level != "D")
        & ~prepared["is_walkover"]
    )
