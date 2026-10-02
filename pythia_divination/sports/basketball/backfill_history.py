"""Append-only backfill of NBA/WNBA games from the official CDN boxscore archive.

Why this exists next to ingest_history.py: ingest_history rewrites the
``*_<league>_latest`` tables with ONLY the seasons named on its command line and
enumerates regular-season ids only. The honest walk-forward model needs more
seasons (NBA 2022-23/2023-24, WNBA 2024/2026) plus the play-in, playoff and cup
games (Elo and rest must see every game a team played), without dropping what
is already in the tables. This module:

  * enumerates game ids per (league, season, game type) -- regular season
    sequentially until a run of misses, brackets (playoffs / play-in) round by
    round, series by series, game by game until a series ends;
  * skips ids already present in the normalized tables (so re-runs are cheap
    and a rate-limited miss can simply be retried);
  * keeps only FINAL games (gameStatus 3);
  * appends rows to schedule_/game_details_/player_boxscores_<league>_latest.csv
    (CSV only, the format the tables already use), de-duplicated by canonical
    10-digit game id with the newest row winning.

The exporter also uses :func:`fetch_final_games` to pull boxscores of completed
current-season games the (possibly months-old) S3 snapshot does not have yet.

The CDN answers 403 for an id that does not exist, the same code it uses when
it rate-limits, so requests are throttled (BASKETBALL_INGEST_SLEEP_MS, default
300 ms) and the caller should compare counts with the known season sizes.

CLI (cwd = pythia_divination):
    python -m sports.basketball.backfill_history --league nba --seasons 2022-23 2023-24
    python -m sports.basketball.backfill_history --league wnba --seasons 2024 2026 --kinds regular playoff cup_final
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.basketball.client import (  # noqa: E402
    BasketballGameBundle,
    BasketballStatsClient,
    flatten_game_bundle,
    flatten_player_boxscores,
    flatten_schedule_from_game_bundle,
)
from sports.basketball.constants import normalize_season_label, season_start_year_from_label  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_BASKETBALL_DATA_ROOT = ROOT / "data" / "sports" / "basketball"
_SLEEP_SECONDS = max(0.0, float(os.getenv("BASKETBALL_INGEST_SLEEP_MS", "300")) / 1000.0)

# League digit + game-type digits of the 10-digit id: NBA "00" + type, WNBA "10" + type.
LEAGUE_ID_PREFIX = {"nba": "00", "wnba": "10"}
GAME_TYPES = {"1": "preseason", "2": "regular", "3": "all_star", "4": "playoff", "5": "play_in", "6": "cup_final"}
# Competitive games the honest model trains on / rates teams with. Preseason and
# All-Star games are exhibitions (rested starters, mixed rosters).
COMPETITIVE_TYPES = ("regular", "play_in", "playoff", "cup_final")
# Regular-season sizes, used only for the post-run completeness check.
EXPECTED_REGULAR_GAMES = {
    ("nba", "2021-22"): 1230, ("nba", "2022-23"): 1230, ("nba", "2023-24"): 1230,
    ("nba", "2024-25"): 1230, ("nba", "2025-26"): 1230,
    ("wnba", "2024"): 240, ("wnba", "2025"): 286, ("wnba", "2026"): 330,
}


def canonical_game_id(value: Any) -> str | None:
    """'22100001', 22100001, '0022100001' -> '0022100001' (CSV round trips drop the zeros)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        return None
    return text.zfill(10)


def game_type_from_id(game_id: Any) -> str | None:
    gid = canonical_game_id(game_id)
    if gid is None:
        return None
    # WNBA '105' is the Commissioner's Cup final; NBA '005' the play-in, '006' the NBA Cup final.
    kind = GAME_TYPES.get(gid[2])
    if gid[:2] == "10" and kind == "play_in":
        return "cup_final"
    return kind


def _season_code(league: str, season_display: str) -> str:
    return str(season_start_year_from_label(league, season_display))[-2:]


def _game_id(league: str, type_digit: str, season_display: str, tail: str) -> str:
    return f"{LEAGUE_ID_PREFIX[league]}{type_digit}{_season_code(league, season_display)}{tail}"


class _Fetcher:
    """Throttled CDN boxscore fetcher. Returns the payload of a FINAL game, None otherwise."""

    def __init__(self, sleep_seconds: float = _SLEEP_SECONDS, retries: int = 2) -> None:
        self.client = BasketballStatsClient()
        self.sleep_seconds = sleep_seconds
        self.retries = retries
        self.requests = 0

    def final_boxscore(self, league: str, game_id: str) -> dict[str, Any] | None:
        for attempt in range(self.retries + 1):
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
            self.requests += 1
            try:
                payload = self.client.get_boxscore(league, game_id)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in (403, 404):
                    return None  # missing id (the CDN answers 403 for those)
                if attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                logger.warning("boxscore %s %s failed: %s", league, game_id, exc)
                return None
            except (requests.RequestException, ValueError) as exc:
                if attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                logger.warning("boxscore %s %s failed: %s", league, game_id, exc)
                return None
            game = payload.get("game", {}) if isinstance(payload, dict) else {}
            try:
                status_code = int(game.get("gameStatus"))
            except (TypeError, ValueError):
                status_code = None
            return payload if status_code == 3 else None
        return None


def _enumerate_regular(fetcher: _Fetcher, league: str, season: str, existing: set[str], *,
                       max_number: int = 1300, miss_limit: int = 25) -> list[BasketballGameBundle]:
    found: list[BasketballGameBundle] = []
    misses = 0
    for number in range(1, max_number + 1):
        gid = _game_id(league, "2", season, f"{number:05d}")
        if gid in existing:
            misses = 0
            continue
        payload = fetcher.final_boxscore(league, gid)
        if payload is None:
            misses += 1
            if misses >= miss_limit:
                break
            continue
        misses = 0
        found.append(BasketballGameBundle(league=league, game_id=gid, boxscore=payload))
    return found


def _enumerate_bracket(fetcher: _Fetcher, league: str, season: str, type_digit: str, existing: set[str], *,
                       rounds: int = 4, series: int = 8, games: int = 7) -> list[BasketballGameBundle]:
    """Ids <prefix><yy>00<round><series><game>: walk each series until it ends."""
    found: list[BasketballGameBundle] = []
    for rnd in range(1, rounds + 1):
        round_had_games = False
        for ser in range(series):
            series_had_games = False
            for game in range(1, games + 1):
                gid = _game_id(league, type_digit, season, f"00{rnd}{ser}{game}")
                if gid in existing:
                    series_had_games = True
                    continue
                payload = fetcher.final_boxscore(league, gid)
                if payload is None:
                    break
                series_had_games = True
                found.append(BasketballGameBundle(league=league, game_id=gid, boxscore=payload))
            if not series_had_games:
                break
            round_had_games = True
        if not round_had_games:
            break
    return found


def _enumerate_cup_final(fetcher: _Fetcher, league: str, season: str, existing: set[str]) -> list[BasketballGameBundle]:
    digit = "6" if league == "nba" else "5"
    gid = _game_id(league, digit, season, "00001")
    if gid in existing:
        return []
    payload = fetcher.final_boxscore(league, gid)
    return [BasketballGameBundle(league=league, game_id=gid, boxscore=payload)] if payload else []


def bundles_to_frames(bundles: Iterable[BasketballGameBundle], *, season_display: str | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flatten FINAL boxscores into (schedule, game_details, player_boxscores) rows."""
    schedule_rows, detail_rows, player_rows = [], [], []
    for bundle in bundles:
        season = season_display or _season_from_bundle(bundle)
        sched = flatten_schedule_from_game_bundle(bundle, season_display=season)
        kind = game_type_from_id(bundle.game_id)
        sched["is_regular_season"] = kind == "regular"
        sched["game_id"] = canonical_game_id(bundle.game_id)
        schedule_rows.append(sched)
        detail = flatten_game_bundle(bundle)
        detail["game_id"] = canonical_game_id(bundle.game_id)
        detail["season_display"] = season
        detail_rows.append(detail)
        players = flatten_player_boxscores(bundle)
        if not players.empty:
            players["game_id"] = canonical_game_id(bundle.game_id)
            players["season_display"] = season
            player_rows.append(players)
    schedule = pd.concat(schedule_rows, ignore_index=True) if schedule_rows else pd.DataFrame()
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    players = pd.concat(player_rows, ignore_index=True) if player_rows else pd.DataFrame()
    return schedule, details, players


def _season_from_bundle(bundle: BasketballGameBundle) -> str:
    gid = canonical_game_id(bundle.game_id) or ""
    start_year = 2000 + int(gid[3:5]) if len(gid) >= 5 and gid[3:5].isdigit() else None
    if start_year is None:
        return ""
    return normalize_season_label(bundle.league, start_year)


def fetch_final_games(league: str, game_ids: Iterable[str], *, max_workers: int = 2,
                      max_games: int | None = None, time_budget_seconds: float | None = None,
                      sleep_seconds: float = _SLEEP_SECONDS) -> list[BasketballGameBundle]:
    """Boxscores of specific (known) game ids; used by the exporter to refresh the
    current season. Stops submitting new work once the time budget is spent."""
    ids = [gid for gid in (canonical_game_id(g) for g in game_ids) if gid]
    if max_games is not None:
        ids = ids[: max(0, int(max_games))]
    if not ids:
        return []
    started = time.time()
    fetcher = _Fetcher(sleep_seconds=sleep_seconds, retries=1)
    bundles: list[BasketballGameBundle] = []

    def _one(gid: str) -> BasketballGameBundle | None:
        if time_budget_seconds is not None and time.time() - started > time_budget_seconds:
            return None
        payload = fetcher.final_boxscore(league, gid)
        return BasketballGameBundle(league=league, game_id=gid, boxscore=payload) if payload else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        for result in pool.map(_one, ids):
            if result is not None:
                bundles.append(result)
    logger.info("basketball refresh: %d/%d %s boxscores fetched in %.0fs", len(bundles), len(ids), league.upper(),
                time.time() - started)
    return bundles


# ── table IO ──────────────────────────────────────────────────────────────────

def _table_path(normalized_dir: Path, stem: str, league: str) -> Path:
    return normalized_dir / f"{stem}_{league}_latest.csv"


def read_table(normalized_dir: Path, stem: str, league: str) -> pd.DataFrame:
    path = _table_path(normalized_dir, stem, league)
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        frame = pd.read_parquet(parquet)
    elif path.exists():
        frame = pd.read_csv(path, low_memory=False)
    else:
        return pd.DataFrame()
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].map(canonical_game_id)
    return frame


def existing_game_ids(normalized_dir: Path, league: str) -> set[str]:
    """Ids with a FINAL schedule row AND a game-detail row (both are needed downstream)."""
    sched = read_table(normalized_dir, "schedule", league)
    details = read_table(normalized_dir, "game_details", league)
    if sched.empty or details.empty:
        return set()
    final = sched.loc[pd.to_numeric(sched.get("status_code"), errors="coerce") == 3, "game_id"]
    return set(final.dropna()) & set(details["game_id"].dropna())


def append_to_tables(normalized_dir: Path, league: str, schedule: pd.DataFrame, details: pd.DataFrame,
                     players: pd.DataFrame) -> dict[str, int]:
    """Append rows (newest wins per game id) and rewrite the league CSVs. Returns row counts."""
    counts: dict[str, int] = {}
    for stem, new, keys in (("schedule", schedule, ["game_id"]), ("game_details", details, ["game_id"]),
                            ("player_boxscores", players, ["game_id", "team_id", "player_id"])):
        if new is None or new.empty:
            continue
        old = read_table(normalized_dir, stem, league)
        combined = pd.concat([old, new], ignore_index=True) if not old.empty else new.copy()
        combined["game_id"] = combined["game_id"].map(canonical_game_id)
        combined = combined.drop_duplicates(subset=keys, keep="last")
        sort_cols = [c for c in ("official_date", "game_date_time_utc", "game_id") if c in combined.columns]
        combined = combined.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        path = _table_path(normalized_dir, stem, league)
        tmp = path.with_suffix(".csv.tmp")
        combined.to_csv(tmp, index=False)
        os.replace(tmp, path)
        counts[stem] = int(len(combined))
    return counts


def backfill_season(league: str, season: str, *, kinds: Iterable[str], normalized_dir: Path,
                    existing: set[str]) -> list[BasketballGameBundle]:
    fetcher = _Fetcher()
    bundles: list[BasketballGameBundle] = []
    kinds = set(kinds)
    if "regular" in kinds:
        bundles += _enumerate_regular(fetcher, league, season, existing,
                                      max_number=1300 if league == "nba" else 400)
    if "play_in" in kinds and league == "nba":
        bundles += _enumerate_bracket(fetcher, league, season, "5", existing, rounds=2, series=4, games=1)
    if "playoff" in kinds:
        bundles += _enumerate_bracket(fetcher, league, season, "4", existing,
                                      rounds=4 if league == "nba" else 3, series=8 if league == "nba" else 4,
                                      games=7 if league == "nba" else 5)
    if "cup_final" in kinds:
        bundles += _enumerate_cup_final(fetcher, league, season, existing)
    logger.info("%s %s: %d new final games (%d requests)", league.upper(), season, len(bundles), fetcher.requests)
    return bundles


def run(league: str, seasons: list[str], kinds: list[str], output_root: Path, max_workers: int = 2) -> dict[str, Any]:
    normalized_dir = Path(output_root) / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    existing = existing_game_ids(normalized_dir, league)
    labels = [normalize_season_label(league, s) for s in seasons]
    results: dict[str, list[BasketballGameBundle]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = {pool.submit(backfill_season, league, season, kinds=kinds, normalized_dir=normalized_dir,
                               existing=existing): season for season in labels}
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    frames = [bundles_to_frames(results[s], season_display=s) for s in labels if results.get(s)]
    if not frames:
        return {"league": league, "new_games": 0}
    schedule = pd.concat([f[0] for f in frames], ignore_index=True)
    details = pd.concat([f[1] for f in frames], ignore_index=True)
    players = pd.concat([f[2] for f in frames], ignore_index=True)
    counts = append_to_tables(normalized_dir, league, schedule, details, players)
    after = read_table(normalized_dir, "schedule", league)
    after["kind"] = after["game_id"].map(game_type_from_id)
    completeness = {}
    for season in labels:
        rows = after.loc[(after["season_display"].astype(str) == season) & (after["kind"] == "regular")
                         & (pd.to_numeric(after["status_code"], errors="coerce") == 3)]
        expected = EXPECTED_REGULAR_GAMES.get((league, season))
        completeness[season] = {"regular_final": int(rows["game_id"].nunique()), "expected": expected}
    summary = {"league": league, "new_games": int(len(details)), "tables": counts, "completeness": completeness}
    logger.info("backfill summary: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Append archived NBA/WNBA games to the normalized tables.")
    parser.add_argument("--league", required=True, choices=["nba", "wnba"])
    parser.add_argument("--seasons", nargs="+", required=True)
    parser.add_argument("--kinds", nargs="+", default=["regular", "play_in", "playoff", "cup_final"],
                        choices=["regular", "play_in", "playoff", "cup_final"])
    parser.add_argument("--output-root", default=str(DEFAULT_BASKETBALL_DATA_ROOT))
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(run(args.league, args.seasons, args.kinds, Path(args.output_root), args.max_workers))


if __name__ == "__main__":
    main()
