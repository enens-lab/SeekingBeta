"""Tennis (ATP + WTA) boards for the web/iOS/Android clients.

Every probability on these boards comes from the walk-forward Elo in
``sports/wta/elo.py`` (stable player identity, decaying K, overall/surface blend)
fed through a seeded knockout simulation (``sports/wta/tournament_sim.py``):

* Historical boards: one per completed individual event with a known champion
  and a field of at least ``min_backtest_field`` (16) players, from
  ``backtest_start_year``. Ratings are taken before each player's first match of
  the event, so every board is an honest out-of-sample prediction. Davis Cup /
  BJK Cup / United Cup / Laver Cup ties are never boards. Each board also names
  the ranking favourite (best-ranked player in the field) as the baseline; WTA
  seasons from ESPN (2026+, ``sports/wta/espn_results.py``) carry seeds but no
  rankings, so there the top seed is the baseline (``rankingFavoriteBasis``).
* Upcoming boards: the published main draw when ESPN has it
  (``fieldBasis: "current_draw"``, simulated bracket seeded with the draw's
  seeds), otherwise this season's entrants seen so far (``"observed_entrants"``)
  or last season's field re-rated with today's ratings (``"previous_edition"``).
  Ratings for an event already under way are frozen at its start date.

Also writes ``tennis_model_metrics.json``: match-level walk-forward accuracy,
log-loss and Brier against the previous Elo and the ranking favourite, plus
tournament top-pick rates beside the ranking favourite (fields of 16+ only).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.player_media import resolve_player_image  # noqa: E402
from sports.wta.elo import EloState, canonical_surface, load_params, normalize_name, prepare_matches, run_elo  # noqa: E402
from sports.wta.evaluate_elo import evaluate as evaluate_match_level  # noqa: E402
from sports.wta.feature_engineering import (  # noqa: E402
    add_rolling_features,
    build_player_event_features,
    process_match_history,
)
from sports.wta.ingest import load_combined_matches  # noqa: E402
from sports.wta.live_fields import fetch_espn_events, match_espn_event  # noqa: E402
from sports.wta.tournament_sim import event_rng, simulate_knockout  # noqa: E402

DIV_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DATA_DIR = PROJ_ROOT / "pythia_prophecy" / "frontend" / "src" / "data"
UPCOMING_PER_TOUR = {"ATP": 12, "WTA": 12}
UPCOMING_MIN_FIELD = 8
ACTIVE_WINDOW_DAYS = 365
SLAM_NAMES = ("roland garros", "french open", "wimbledon", "us open", "australian open")
PREDICTION_SOURCE = "elo_knockout_simulation"

FIELD_NOTES = {
    "current_draw": "Field is the published main draw.",
    "observed_entrants": "Field is the players seen in this event's results so far; the full draw was not available.",
    "previous_edition": "Field is last year's entry list re-rated with current ratings; this year's draw is not out yet.",
}


# --------------------------------------------------------------------------- helpers


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


def _append_stat(stats: list[dict[str, str]], label: str, value: str | None) -> None:
    if value:
        stats.append({"label": label, "value": value})


def _clamp_score(value: float | None, low: float, high: float, *, inverse: bool = False) -> float | None:
    if value is None or high <= low:
        return None
    clipped = min(max(value, low), high)
    ratio = (clipped - low) / (high - low)
    if inverse:
        ratio = 1.0 - ratio
    return round(ratio * 100.0, 1)


def _build_tennis_radar(row: dict[str, Any]) -> list[dict[str, float]]:
    metrics = [
        ("Overall Level", _clamp_score(_safe_float(row.get("elo")), 1400.0, 2350.0)),
        ("Surface Fit", _clamp_score(_safe_float(row.get("surf_elo")), 1400.0, 2350.0)),
        ("Serve", _clamp_score(_safe_float(row.get("serve_won")), 0.48, 0.74)),
        ("Return", _clamp_score(_safe_float(row.get("return_won")), 0.28, 0.46)),
        ("Field Edge", _clamp_score(_safe_float(row.get("elo_field_percentile")), 0.05, 0.98)),
        ("Recent Form", _clamp_score(_safe_float(row.get("player_elo_diff_5")), -120.0, 120.0)),
    ]
    return [{"label": label, "value": value or 0.0} for label, value in metrics if value is not None]


def _build_tennis_profile(row: dict[str, Any]) -> dict:
    stats: list[dict[str, str]] = []
    elo = _safe_float(row.get("elo"))
    surf_elo = _safe_float(row.get("surf_elo"))
    serve_won = _safe_float(row.get("serve_won"))
    return_won = _safe_float(row.get("return_won"))
    age = _safe_float(row.get("age"))
    height = _safe_float(row.get("height"))
    tour = str(row.get("tour") or "Tennis")
    surface = str(row.get("surface") or "Unknown")

    _append_stat(stats, "Elo", f"{elo:.0f}" if elo is not None else None)
    _append_stat(stats, "Surface Elo", f"{surf_elo:.0f}" if surf_elo is not None else None)
    _append_stat(stats, "Serve Won", f"{serve_won * 100:.1f}%" if serve_won is not None else None)
    _append_stat(stats, "Return Won", f"{return_won * 100:.1f}%" if return_won is not None else None)
    if len(stats) < 4:
        _append_stat(stats, "Age", f"{age:.1f}" if age is not None else None)
    if len(stats) < 4:
        _append_stat(stats, "Height", f"{height:.0f} cm" if height is not None else None)

    subtitle_parts = [tour, surface]
    if age is not None:
        subtitle_parts.append(f"Age {age:.1f}")

    return {
        "imageUrl": resolve_player_image(
            name=str(row.get("player_name") or ""),
            sport="tennis",
            tour=tour,
            player_id=row.get("player_id"),
            player_key=row.get("player_key"),
        ),
        "subtitle": " | ".join(subtitle_parts),
        "country": None,
        "stats": stats[:4],
    }


def _seed_order(ranks: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """Official ranking first (as the tours seed), rating breaks ties / fills gaps."""
    rank_key = np.where(np.isfinite(ranks), ranks, np.inf)
    return np.lexsort((-ratings, rank_key))


def win_probabilities(event_key: str, ratings: np.ndarray, ranks: np.ndarray, params: dict[str, Any],
                      extra_ratings: np.ndarray | None = None) -> np.ndarray:
    """P(title) per player. ``extra_ratings`` are unnamed placeholder entrants
    (qualifiers not yet known): simulated, then dropped from the output."""
    ratings = np.asarray(ratings, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    if extra_ratings is not None and len(extra_ratings):
        ratings = np.concatenate([ratings, np.asarray(extra_ratings, dtype=float)])
        ranks = np.concatenate([ranks, np.full(len(extra_ratings), np.nan)])
    probs = simulate_knockout(
        ratings,
        seed_order=_seed_order(ranks, ratings),
        simulations=int(params.get("simulations", 4000)),
        seed_fraction=float(params.get("seed_fraction", 0.25)),
        rng=event_rng(event_key),
    )
    return probs[: len(probs) - (len(extra_ratings) if extra_ratings is not None else 0)]


def ranking_favourite(group: pd.DataFrame) -> tuple[str | None, str | None]:
    """Baseline pick for a field: the best-ranked player; for events with no
    rankings (ESPN-sourced WTA seasons) the top seed, i.e. the best-ranked
    entrant at the entry deadline. Returns (name, "ranking" | "top_seed")."""
    for column, basis in (("rank", "ranking"), ("seed", "top_seed")):
        if column not in group.columns:
            continue
        values = pd.to_numeric(group[column], errors="coerce")
        if values.notna().any():
            return str(group.at[values.idxmin(), "player_name"]), basis
    return None, None


def _hit_status(actual: str | None, ordered_names: list[str]) -> str:
    if actual and ordered_names and actual == ordered_names[0]:
        return "Top Pick"
    if actual in ordered_names[:3]:
        return "Top 3"
    if actual in ordered_names[:5]:
        return "Top 5"
    return "Miss"


# --------------------------------------------------------------------------- data


def load_tennis_tables(params: dict[str, Any]) -> dict[str, Any]:
    matches = load_combined_matches()
    history = process_match_history(matches, params)
    events = add_rolling_features(build_player_event_features(history))
    return {"matches": matches, "history": history, "events": events}


# --------------------------------------------------------------------------- backtests


def build_backtests(events: pd.DataFrame, params: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Historical boards plus one scoring row per board (for the metrics file)."""
    min_field = int(params.get("min_backtest_field", 16))
    start_year = int(params.get("backtest_start_year", 2023))
    eligible = events[
        events["has_champion"]
        & (events["field_size"] >= min_field)
        & ((events["date"] // 10000) >= start_year)
    ]
    boards: list[dict] = []
    scoring: list[dict] = []
    for event_id, group in eligible.groupby("tournament_id", sort=False):
        group = group.reset_index(drop=True)
        champion_rows = group[group["won_tournament"] == 1]
        if len(champion_rows) != 1:
            continue
        probs = win_probabilities(
            event_id,
            group["blend_elo"].to_numpy(dtype=float),
            pd.to_numeric(group["rank"], errors="coerce").to_numpy(dtype=float),
            params,
        )
        order = np.argsort(-probs, kind="mergesort")
        names = [str(group.at[index, "player_name"]) for index in order]
        actual = str(champion_rows.iloc[0]["player_name"])
        favourite, favourite_basis = ranking_favourite(group)

        full_field = []
        for position, index in enumerate(order, start=1):
            row = group.iloc[index].to_dict()
            full_field.append(
                {
                    "rank": position,
                    "playerName": str(row["player_name"]),
                    "winProbability": round(float(probs[index] * 100.0), 2),
                    "actualWinner": bool(row["won_tournament"] == 1),
                    "profile": _build_tennis_profile(row),
                    "radarMetrics": _build_tennis_radar(row),
                }
            )
        first_date = int(group["date"].iloc[0])
        boards.append(
            {
                "year": first_date // 10000,
                "tournament": str(group["tournament_name"].iloc[0]),
                "tour": str(group["tour"].iloc[0]),
                "surface": str(group["surface"].iloc[0]) if pd.notna(group["surface"].iloc[0]) else "Unknown",
                "predictedWinner": names[0],
                "predictedTop3": names[:3],
                "predictedTop5": names[:5],
                "actualWinner": actual,
                "hitStatus": _hit_status(actual, names),
                "prob": round(float(probs[order[0]]), 4),
                "fullField": full_field,
                "firstDate": first_date,
                "latestDate": int(group["end_date"].iloc[0]),
                "tournamentId": str(event_id),
                "fieldSize": int(len(group)),
                "rankingFavorite": favourite,
                "rankingFavoriteWon": bool(favourite is not None and favourite == actual),
                "rankingFavoriteBasis": favourite_basis,
                "modelVersion": params.get("version"),
            }
        )
        champion_index = int(champion_rows.index[0])
        scoring.append(
            {
                "tour": boards[-1]["tour"],
                "year": boards[-1]["year"],
                "field": int(len(group)),
                "top_pick": names[0] == actual,
                "top3": actual in names[:3],
                "favourite_known": favourite is not None,
                "favourite_hit": favourite is not None and favourite == actual,
                "favourite_basis": favourite_basis,
                "champion_probability": float(probs[champion_index]),
            }
        )
    boards.sort(key=lambda row: (-row["year"], -row["firstDate"], row["tour"], row["tournament"]))
    return boards, scoring


def tournament_metrics(scoring: list[dict]) -> list[dict]:
    if not scoring:
        return []
    frame = pd.DataFrame(scoring)
    rows = []
    for (tour, year), group in frame.groupby(["tour", "year"]):
        known = group[group["favourite_known"]]
        rows.append(
            {
                "tour": tour,
                "season": int(year),
                "events": int(len(group)),
                "minField": int(group["field"].min()),
                "topPickHits": int(group["top_pick"].sum()),
                "topPickRate": round(float(group["top_pick"].mean()), 4),
                "top3Rate": round(float(group["top3"].mean()), 4),
                "rankingFavoriteEvents": int(len(known)),
                "rankingFavoriteHits": int(known["favourite_hit"].sum()),
                "rankingFavoriteRate": round(float(known["favourite_hit"].mean()), 4) if len(known) else None,
                "rankingFavoriteBasis": sorted({str(basis) for basis in known.get("favourite_basis", []) if basis}),
                "championLogLoss": round(float(-np.log(np.clip(group["champion_probability"], 1e-6, 1)).mean()), 4),
                "uniformLogLoss": round(float(np.log(group["field"]).mean()), 4),
            }
        )
    return rows


# --------------------------------------------------------------------------- upcoming


def _player_snapshots(history: pd.DataFrame, events: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Latest per-player facts: name, id, rank, last match date, last-10 serve/return, height, form."""
    frames = []
    for side, short in (("winner", "w"), ("loser", "l")):
        frames.append(
            pd.DataFrame(
                {
                    "key": history[f"{side}_key"].values,
                    "id": history[f"{side}_id"].values,
                    "rank": pd.to_numeric(history.get(f"{side}_rank"), errors="coerce").values,
                    "height": pd.to_numeric(history.get(f"{side}_ht"), errors="coerce").values,
                    "date": history["tourney_date"].values,
                    "serve": history[f"{short}_serve_share"].values,
                    "ret": history[f"{short}_return_share"].values,
                    "order": np.arange(len(history)),
                }
            )
        )
    appearances = pd.concat(frames, ignore_index=True).sort_values("order", kind="mergesort")
    names = dict(zip(events["player_key"], events["player_name"]))
    snapshot: dict[str, dict[str, Any]] = {}
    for key, group in appearances.groupby("key", sort=False):
        ranks = group["rank"].dropna()
        heights = group["height"].dropna()
        tail = group.tail(10)
        snapshot[key] = {
            "player_key": key,
            "player_id": group["id"].iloc[-1],
            "player_name": names.get(key),
            "rank": float(ranks.iloc[-1]) if len(ranks) else np.nan,
            "last_played": int(group["date"].iloc[-1]),
            "serve_won": float(tail["serve"].mean()),
            "return_won": float(tail["ret"].mean()),
            "height": float(heights.iloc[-1]) if len(heights) else None,
        }
    elo_by_player = events.sort_values(["player_key", "date"]).groupby("player_key")["elo"].apply(list)
    for key, values in elo_by_player.items():
        if key in snapshot and len(values) >= 5:
            snapshot[key]["elo_5_events_ago"] = float(values[-5])
    return snapshot


class EntrantResolver:
    """Map a draw name (ESPN spelling) to our player key.

    Tries the exact name, then the reversed order (ESPN writes Chinese names
    surname-first: "Zhang Zhizhen" vs "Zhizhen Zhang"), then surname plus first
    initial, but only for players we know solely by a tennis-data short name
    ("Tagger L."): anyone whose first name we know must match it exactly.
    Unknown -> None (the player is then rated as a newcomer).
    """

    def __init__(self, snapshots: dict[str, dict[str, Any]], history: pd.DataFrame):
        self.exact: dict[tuple[str, str], str] = {}
        initials: dict[tuple[str, str, str], set[str]] = {}
        full_name_keys: set[str] = set()
        for side in ("winner", "loser"):
            for tour, key, name in zip(history["tour"], history[f"{side}_key"], history[f"{side}_name"]):
                if not key:
                    continue
                normalized = normalize_name(name)
                if normalized:
                    self.exact.setdefault((tour, normalized), key)
                short = _SHORT_FORM.match(str(name or "").strip())
                if short:
                    initials.setdefault((tour, normalize_name(short.group("last")), short.group("init")[0].lower()), set()).add(key)
                elif normalized:
                    full_name_keys.add(key)
        for key, info in snapshots.items():
            if info.get("player_name"):
                self.exact[(key.split(":")[0], normalize_name(info["player_name"]))] = key
        self.initials = {
            parts: next(iter(keys))
            for parts, keys in initials.items()
            if len(keys) == 1 and not keys & full_name_keys
        }

    def resolve(self, tour: str, name: str) -> str | None:
        tokens = str(name or "").split()
        candidates = [name]
        if len(tokens) >= 2:
            candidates.append(" ".join(tokens[1:] + tokens[:1]))
            candidates.append(" ".join(tokens[-1:] + tokens[:-1]))
        for candidate in candidates:
            key = self.exact.get((tour, normalize_name(candidate)))
            if key:
                return key
        parts = _surname_initial(name)
        return self.initials.get((tour, *parts)) if parts else None


_SHORT_FORM = re.compile(r"^(?P<last>.+?)\s+(?P<init>(?:[A-Z]\.)+)$")


def _surname_initial(name: str) -> tuple[str, str] | None:
    """("tagger", "l") from "Lilli Tagger" (draw spelling)."""
    tokens = str(name or "").split()
    if len(tokens) < 2:
        return None
    return normalize_name(tokens[-1]), normalize_name(tokens[0])[:1]


def _projected_window(edition_start: int, edition_end: int, name: str, year: int) -> tuple[int, int] | None:
    try:
        start = date(year, (edition_start // 100) % 100, edition_start % 100)
        original_start = date(edition_start // 10000, (edition_start // 100) % 100, edition_start % 100)
        original_end = date(edition_end // 10000, (edition_end // 100) % 100, edition_end % 100)
    except ValueError:
        return None
    minimum = 13 if any(slam in name.lower() for slam in SLAM_NAMES) else 6
    duration = max((original_end - original_start).days, minimum)
    end = start + timedelta(days=duration)
    return int(start.strftime("%Y%m%d")), int(end.strftime("%Y%m%d"))


def _board_rows(
    keys: list[str | None],
    display: list[str],
    state: EloState,
    surface: str,
    tour: str,
    snapshots: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for key, name in zip(keys, display):
        info = snapshots.get(key or "", {})
        lookup = key or f"{tour}:name:{normalize_name(name)}"
        overall, surf, blend = state.ratings(lookup, surface)
        ago = info.get("elo_5_events_ago")
        rows.append(
            {
                "player_key": key,
                "player_id": info.get("player_id"),
                "player_name": info.get("player_name") or name,
                "tour": tour,
                "surface": surface,
                "elo": overall,
                "surf_elo": surf,
                "blend_elo": blend,
                "rank": info.get("rank", np.nan),
                "serve_won": info.get("serve_won"),
                "return_won": info.get("return_won"),
                "height": info.get("height"),
                "player_elo_diff_5": (overall - ago) if ago is not None else None,
                "matches_played": state.matches.get(lookup, 0),
            }
        )
    return rows


def build_upcoming(
    tables: dict[str, Any],
    params: dict[str, Any],
    *,
    today: date | None = None,
    espn_events: list[dict[str, Any]] | None = None,
) -> list[dict]:
    today = today or datetime.now(timezone.utc).date()
    today_key = int(today.strftime("%Y%m%d"))
    year = today.year
    history: pd.DataFrame = tables["history"]
    events: pd.DataFrame = tables["events"]
    espn_events = espn_events if espn_events is not None else fetch_espn_events(today=today)

    snapshots = _player_snapshots(history, events)
    resolver = EntrantResolver(snapshots, history)
    data_end = {tour: int(group["tourney_date"].max()) for tour, group in history.groupby("tour")}

    # Latest edition of every individual event, last season or this one.
    editions = (
        events[events["field_size"] >= UPCOMING_MIN_FIELD]
        .groupby(["tour", "tournament_name", "tournament_id"], sort=False)
        .agg(start=("date", "first"), end=("end_date", "first"), surface=("surface", "first"), has_champion=("has_champion", "first"))
        .reset_index()
        .sort_values("start")
    )
    editions = editions[(editions["start"] // 10000) >= year - 1]
    latest = editions.groupby(["tour", "tournament_name"], sort=False).tail(1)

    candidates = []
    for row in latest.itertuples(index=False):
        if int(row.start) // 10000 == year:
            if row.has_champion:
                continue  # already played this season: it is a historical board
            window = (int(row.start), max(int(row.end), int(row.start)))
        else:
            window = _projected_window(int(row.start), int(row.end), str(row.tournament_name), year)
        if window is None:
            continue
        candidates.append((row, window))

    # Ratings frozen at the start of events already under way.
    matched = []
    for row, window in candidates:
        live = match_espn_event(str(row.tour), str(row.tournament_name), window[0], espn_events)
        start, end = (int(live["startKey"]), int(live["endKey"])) if live else window
        if end < today_key:
            continue
        matched.append((row, start, end, live))
    snapshot_dates = sorted({start for _, start, _, _ in matched if start <= today_key})
    prepared = prepare_matches(tables["matches"], params)
    _, final_state, frozen = run_elo(prepared, params, snapshot_dates=snapshot_dates)

    boards: list[dict] = []
    seen_live: set[str] = set()
    per_tour = {tour: 0 for tour in UPCOMING_PER_TOUR}
    for row, start, end, live in sorted(matched, key=lambda item: (item[1], item[0].tour, item[0].tournament_name)):
        tour = str(row.tour)
        if per_tour.get(tour, 0) >= UPCOMING_PER_TOUR.get(tour, 10):
            continue
        if live is not None:
            # ESPN lists combined ATP+WTA events once per tour scoreboard under one id.
            live_key = f"{tour}:{live['espnId']}"
            if live_key in seen_live:
                continue
            seen_live.add(live_key)
        state = frozen.get(start, final_state) if start <= today_key else final_state
        surface = canonical_surface(row.surface, params)
        edition_field = events[events["tournament_id"] == row.tournament_id]
        min_field = UPCOMING_MIN_FIELD if len(edition_field) < 16 else 16

        open_slots = 0
        if live is not None and len(live["entrants"]) >= min_field:
            basis = "current_draw"
            keys, display, seen_keys = [], [], set()
            for name in live["entrants"]:
                key = resolver.resolve(tour, name)
                if key is not None and key in seen_keys:
                    continue  # two spellings of one player
                seen_keys.add(key)
                keys.append(key)
                display.append(name)
            open_slots = int(live.get("openSlots") or 0)
        elif int(row.start) // 10000 == year:
            basis = "observed_entrants"
            keys = edition_field["player_key"].tolist()
            display = edition_field["player_name"].tolist()
        else:
            basis = "previous_edition"
            cutoff = date(data_end[tour] // 10000, (data_end[tour] // 100) % 100, data_end[tour] % 100) - timedelta(days=ACTIVE_WINDOW_DAYS)
            cutoff_key = int(cutoff.strftime("%Y%m%d"))
            active = edition_field[[snapshots.get(key, {}).get("last_played", 0) >= cutoff_key for key in edition_field["player_key"]]]
            keys = active["player_key"].tolist()
            display = active["player_name"].tolist()
        if len(keys) < UPCOMING_MIN_FIELD:
            continue

        rows = _board_rows(keys, display, state, surface, tour, snapshots)
        frame = pd.DataFrame(rows)
        frame["elo_field_percentile"] = frame["blend_elo"].rank(pct=True)
        placeholders = None
        if open_slots:
            placeholders = np.full(open_slots, float(np.percentile(frame["blend_elo"], 25)))
        # Seed the simulated bracket like the real one: the published draw's seeds
        # when ESPN shows them, else the latest ranking we hold.
        seeding = pd.to_numeric(frame["rank"], errors="coerce").to_numpy(dtype=float)
        draw_seeds = ((live or {}).get("seeds") or {}) if basis == "current_draw" else {}
        if len(draw_seeds) >= 2:
            seeding = np.array([float(draw_seeds.get(name, np.nan)) for name in display], dtype=float)
        board_key = f"{tour}:{row.tournament_name}:{start}:{basis}"
        probs = win_probabilities(
            board_key,
            frame["blend_elo"].to_numpy(dtype=float),
            seeding,
            params,
            extra_ratings=placeholders,
        )
        order = np.argsort(-probs, kind="mergesort")
        predictions = []
        for position, index in enumerate(order, start=1):
            entry = frame.iloc[index].to_dict()
            predictions.append(
                {
                    "rank": position,
                    "playerName": str(entry["player_name"]),
                    "winProbability": round(float(probs[index] * 100.0), 2),
                    "profile": _build_tennis_profile(entry),
                    "radarMetrics": _build_tennis_radar(entry),
                }
            )
        rated_through = data_end.get(tour)
        if rated_through and start <= today_key:
            # frozen at the event start: nothing dated on or after it is in the ratings
            day_before = datetime.strptime(str(start), "%Y%m%d") - timedelta(days=1)
            rated_through = min(rated_through, int(day_before.strftime("%Y%m%d")))
        tournament = str(row.tournament_name)
        boards.append(
            {
                "id": f"{tour.lower()}-{tournament.replace(' ', '-').lower()}",
                "name": f"{year} {tournament}",
                "tour": tour,
                "course": surface,
                "surface": surface,
                "scheduledDate": int(start),
                "latestDate": int(end),
                "predictedWinner": predictions[0]["playerName"] if predictions else None,
                "predictions": predictions,
                "predictionSource": PREDICTION_SOURCE,
                "fieldBasis": basis,
                "fieldNote": FIELD_NOTES[basis],
                "fieldSize": int(len(predictions) + open_slots),
                "openDrawSlots": int(open_slots),
                "ratingsAsOf": rated_through,
                "modelVersion": params.get("version"),
            }
        )
        per_tour[tour] = per_tour.get(tour, 0) + 1

    return sorted(boards, key=lambda board: (board["scheduledDate"], board["tour"], board["name"]))


# --------------------------------------------------------------------------- main


def build_metrics(tables: dict[str, Any], params: dict[str, Any], scoring: list[dict]) -> dict[str, Any]:
    this_year = datetime.now(timezone.utc).year
    match_level = evaluate_match_level(tables["matches"], params, start=20230101, end=20260101)
    current = evaluate_match_level(tables["matches"], params, start=this_year * 10000 + 101, end=(this_year + 1) * 10000 + 101)
    history = tables["history"]
    return {
        "sport": "tennis",
        "modelVersion": params.get("version"),
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ratingsAsOf": {tour: int(group["tourney_date"].max()) for tour, group in history.groupby("tour")},
        "method": params.get("description"),
        "matchLevel": {
            "window": "2023-2025",
            "definition": params.get("evaluation", {}).get("definition"),
            "model": match_level["all"]["elo"],
            "previousElo": match_level["all"]["legacy_elo"],
            "rankingFavorite": match_level["all"]["ranking_favourite"],
            "byTourSeason": {key: value["elo"] for key, value in match_level["by_tour_year"].items()},
            "currentSeason": {key: value["elo"] for key, value in current["by_tour"].items()},
        },
        "tournamentLevel": {
            "definition": f"Champion pick on individual events with fields of {params.get('min_backtest_field', 16)}+ players; baseline is the best-ranked player in the field.",
            "bySeason": tournament_metrics(scoring),
        },
        "notes": [
            "All probabilities are walk-forward: each uses only matches dated before it.",
            "Market odds are not used or shown for tennis.",
            "ATP results 2025+ come from tennismylife; WTA results 2026+ from ESPN's scoreboard (tour-level events only), "
            "where the ranking-favourite baseline is the top seed because ESPN publishes no rankings.",
        ],
    }


def export_wta_frontend_data() -> None:
    params = load_params()
    tables = load_tennis_tables(params)
    backtests, scoring = build_backtests(tables["events"], params)
    upcoming = build_upcoming(tables, params)
    metrics = build_metrics(tables, params, scoring)

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DATA_DIR / "wta_historical_backtests.json").write_text(json.dumps(backtests, separators=(",", ":")))
    (FRONTEND_DATA_DIR / "wta_upcoming_tournaments.json").write_text(json.dumps(upcoming, indent=2))
    (FRONTEND_DATA_DIR / "tennis_model_metrics.json").write_text(json.dumps(metrics, indent=2))

    backtest_tours = pd.Series([row["tour"] for row in backtests]).value_counts().to_dict()
    upcoming_tours = pd.Series([row["tour"] for row in upcoming]).value_counts().to_dict()
    bases = pd.Series([row["fieldBasis"] for row in upcoming]).value_counts().to_dict()
    print(f"Exported {len(backtests)} tennis backtests by tour: {backtest_tours}")
    print(f"Exported {len(upcoming)} tennis upcoming events by tour: {upcoming_tours} field basis: {bases}")
    model = metrics["matchLevel"]["model"]
    print(
        "Match-level 2023-25 walk-forward: n=%d acc=%.4f log-loss=%.4f brier=%.4f"
        % (model["n"], model["accuracy"], model["log_loss"], model["brier"])
    )
    for row in metrics["tournamentLevel"]["bySeason"]:
        print(
            f"  {row['tour']} {row['season']}: top pick {row['topPickHits']}/{row['events']} ({row['topPickRate']:.1%})"
            f" vs ranking favourite {row['rankingFavoriteHits']}/{row['rankingFavoriteEvents']}"
        )


if __name__ == "__main__":
    export_wta_frontend_data()
