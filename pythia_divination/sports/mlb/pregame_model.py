"""Honest pregame MLB model: moneyline, run line and totals from pregame information only.

Why this exists: the HistGradientBoosting model that used to be served (68.8% "accuracy")
learned the result from post-game boxscore fields -- team_box["bullpen"] lists the
relievers who did NOT pitch, and team_box["battingOrder"] is the END-of-game order
(pinch hitters and defensive subs included). Without them the repo recipe scored a
Brier of 0.2585 in 2024, worse than always picking the home team (0.2497).

This module replaces it with two small, regularized models whose every input is
known before first pitch:

  * Win model -- logistic regression on Elo (K=4, home edge 24, margin-of-victory
    multiplier, 1/3 regression to the mean between seasons; every game type,
    postseason included), decayed + shrunk starter FIP-like and K-BB% (half-life 15
    starts), starter expected innings and experience, park factor and weather.
  * Runs model -- Poisson GLM for each team's runs (offense vs the opposing starter,
    park, weather) with a negative-binomial dispersion r (~3.8) fitted on the training
    games. The two teams' runs are treated as independent (measured residual
    correlation -0.014 / 0.005), which gives a full score matrix: run line +/-1.5,
    game total and team totals at any line.

Data sources, all pregame: results/schedule (MLB Stats API, with local normalized
tables as fallback), starter pitching lines (local starter_game_logs topped up live
from the Stats API people gameLog hydrate for starts the local tables do not have).

Evaluation is MONTHLY WALK-FORWARD: for every month, both models are refit on all
completed games before the first day of that month and predict every game in it. The
exporter uses the identical procedure for upcoming games, so an in-app history entry
is the prediction the board showed (modulo late lineup/weather changes).

Fixed numbers (Elo constants, half-lives, shrinkage priors, league constants from
2020-2023, feature lists, regularization) live in model_params.json next to this
file, with the latest fitted coefficients and the walk-forward evaluation.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).with_name("model_params.json")
DIV_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = DIV_ROOT / "data" / "sports" / "mlb" / "normalized"

GAME_TYPES = ("R", "F", "D", "L", "W")
POSTSEASON_TYPES = {"F", "D", "L", "W"}
COMPLETED_STATES = {"Final", "Completed Early", "Game Over"}
# detailedState prefixes: the Stats API also emits variants such as "Completed Early: Rain"
# and "Final: Tied"; a prefix match keeps those from being treated as unplayed.
_COMPLETED_PREFIXES = ("Final", "Completed Early", "Game Over")
_NOT_PLAYED_PREFIXES = ("Postponed", "Cancelled", "Canceled")


def is_completed_state(state: Any) -> bool:
    text = str(state or "")
    return text in COMPLETED_STATES or text.startswith(_COMPLETED_PREFIXES)


def result_status(state: Any) -> str:
    """Grading status of a completed game: "shortened" for an official game called
    before 9 innings ("Completed Early"), whose run lines and totals are void by the
    standard sportsbook rule while the moneyline stands; "final" otherwise."""
    return "shortened" if str(state or "").startswith("Completed Early") else "final"


FIRST_SEASON = 2020
MAX_RUNS = 30  # score-matrix support: P(team scores > 30) is ~1e-9 at r=3.8, mu=6

DEFAULT_PARAMS: dict[str, Any] = {
    "version": "mlb-pregame-v1",
    "elo": {"k": 4.0, "home_edge": 24.0, "season_regress": 1.0 / 3.0, "initial": 1500.0},
    "decay": {"team_half_life_games": 40.0, "team_prior_games": 15.0,
              "starter_half_life_starts": 15.0, "starter_ip_half_life_starts": 4.0,
              "starter_prior_ip": 40.0, "starter_prior_bf": 150.0, "starter_replacement": 1.10,
              "starter_ip_prior_starts": 3.0, "venue_half_life_games": 250.0, "venue_prior_games": 120.0},
    # League constants, measured on the 2020-2023 regular seasons (fixed: they only set
    # the shrinkage targets, and fixing them keeps every walk-forward month honest).
    "league": {"fip_num_per_ip": 0.9629, "kbb_per_bf": 0.1438, "ip_per_start": 5.088,
               "runs_per_team_game": 4.4951, "total_runs_per_game": 8.9902},
    "win_model": {"features": ["elo_diff", "h_sp_fip", "a_sp_fip", "h_sp_kbb", "a_sp_kbb", "h_sp_ip", "a_sp_ip",
                               "h_sp_log_starts", "a_sp_log_starts", "park_pf", "temp", "wind_out"],
                  "C": 0.05},
    # elo_diff_side (this team's Elo edge incl. home field) lets the runs model see team
    # strength: without it the favourite's -1.5 cover was under-predicted by ~3 points in
    # held-out seasons (2022: 0.399 vs 0.427 actual; 0.428 with it).
    "runs_model": {"features": ["off_rs_pg", "def_ra_pg", "opp_sp_fip", "opp_sp_kbb", "opp_sp_ip", "opp_sp_log_starts",
                                "log_pf", "temp", "wind_out", "roof_closed", "is_night", "is_home", "elo_diff_side"],
                   "alpha": 0.001},
}


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """DEFAULT_PARAMS overlaid with model_params.json (missing file = defaults)."""
    params = json.loads(json.dumps(DEFAULT_PARAMS))
    if path.exists():
        stored = json.loads(path.read_text())
        for key, value in stored.items():
            if isinstance(value, dict) and isinstance(params.get(key), dict):
                params[key] = {**params[key], **value}
            else:
                params[key] = value
    return params


# ── weather ──────────────────────────────────────────────────────────────────

def parse_wind(value: Any) -> tuple[Optional[float], Optional[str]]:
    from sports.mlb.feature_engineering import _parse_wind  # single regex, fixed there
    return _parse_wind(value)


def wind_out_mph(wind: Any, condition: Any) -> float:
    """Signed wind toward the outfield: +mph blowing out, -mph blowing in, 0 otherwise
    (cross winds, calm, domes and closed roofs)."""
    if str(condition or "") in ("Dome", "Roof Closed"):
        return 0.0
    mph, direction = parse_wind(wind)
    if mph is None:
        return 0.0
    d = (direction or "").strip().lower()
    if d.startswith("out"):
        return float(mph)
    if d.startswith("in"):
        return -float(mph)
    return 0.0


# ── results / schedule ───────────────────────────────────────────────────────

_SCHEDULE_HYDRATE = "probablePitcher,weather"


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return None if math.isnan(out) else out
    except (TypeError, ValueError):
        return None


def flatten_results(payload: dict[str, Any]) -> pd.DataFrame:
    """One row per game from a Stats API schedule payload (scores, probables, weather)."""
    rows: list[dict[str, Any]] = []
    for day in payload.get("dates", []) or []:
        for game in day.get("games", []) or []:
            teams = game.get("teams", {}) or {}
            away, home = teams.get("away", {}) or {}, teams.get("home", {}) or {}
            weather = game.get("weather", {}) or {}
            status = game.get("status", {}) or {}
            rows.append({
                "game_pk": game.get("gamePk"),
                "game_type": game.get("gameType"),
                "season": game.get("season"),
                "game_date": game.get("gameDate"),
                "official_date": game.get("officialDate"),
                "game_number": game.get("gameNumber"),
                "status_abstract": status.get("abstractGameState"),
                "status_detailed": status.get("detailedState"),
                "away_team_id": (away.get("team") or {}).get("id"),
                "away_team_name": (away.get("team") or {}).get("name"),
                "home_team_id": (home.get("team") or {}).get("id"),
                "home_team_name": (home.get("team") or {}).get("name"),
                "away_score": away.get("score"),
                "home_score": home.get("score"),
                "away_probable_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
                "away_probable_pitcher_name": (away.get("probablePitcher") or {}).get("fullName"),
                "home_probable_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
                "home_probable_pitcher_name": (home.get("probablePitcher") or {}).get("fullName"),
                "venue_id": (game.get("venue") or {}).get("id"),
                "venue_name": (game.get("venue") or {}).get("name"),
                "day_night": game.get("dayNight"),
                "double_header": game.get("doubleHeader"),
                "scheduled_innings": game.get("scheduledInnings"),
                "series_description": game.get("seriesDescription"),
                "weather_condition": weather.get("condition"),
                "weather_temp_f": _to_float(weather.get("temp")),
                "weather_wind": weather.get("wind"),
            })
    return pd.DataFrame(rows)


def fetch_season_results(client: Any, season: int, *, game_types: Iterable[str] = GAME_TYPES) -> pd.DataFrame:
    """Every game of a season (regular season + postseason) with final scores."""
    from sports.mlb.constants import MLB_SCHEDULE_PATH

    payload = client._get_json(MLB_SCHEDULE_PATH, params={
        "sportId": 1, "season": int(season), "gameType": ",".join(game_types), "hydrate": _SCHEDULE_HYDRATE,
    })
    return flatten_results(payload)


def _read_table(path_stem: Path) -> pd.DataFrame:
    for suffix in (".parquet", ".csv"):
        candidate = path_stem.with_suffix(suffix)
        if candidate.exists():
            return pd.read_parquet(candidate) if suffix == ".parquet" else pd.read_csv(candidate, low_memory=False)
    raise FileNotFoundError(str(path_stem))


def load_local_season(season: int, normalized_dir: Path = NORMALIZED_DIR) -> pd.DataFrame:
    """Local (S3-synced) schedule + game details for one season, in flatten_results shape.
    Regular season only (that is all the local ingest keeps)."""
    sched = _read_table(normalized_dir / f"schedule_{season}_latest")
    try:
        det = _read_table(normalized_dir / f"game_details_{season}_latest")
    except FileNotFoundError:
        det = pd.DataFrame(columns=["game_pk"])
    keep_det = [c for c in ["game_pk", "game_datetime", "weather_condition", "weather_temp_f", "weather_wind",
                            "away_probable_pitcher_id", "away_probable_pitcher_name",
                            "home_probable_pitcher_id", "home_probable_pitcher_name"] if c in det.columns]
    det = det[keep_det].drop_duplicates("game_pk")
    sched = sched.drop(columns=[c for c in keep_det if c != "game_pk" and c in sched.columns], errors="ignore")
    out = sched.merge(det, on="game_pk", how="left")
    if "game_datetime" in out.columns:
        out["game_date"] = out["game_datetime"].combine_first(out.get("game_date"))
    cols = ["game_pk", "game_type", "season", "game_date", "official_date", "game_number", "status_abstract",
            "status_detailed", "away_team_id", "away_team_name", "home_team_id", "home_team_name", "away_score",
            "home_score", "away_probable_pitcher_id", "away_probable_pitcher_name", "home_probable_pitcher_id",
            "home_probable_pitcher_name", "venue_id", "venue_name", "day_night", "double_header", "scheduled_innings",
            "series_description", "weather_condition", "weather_temp_f", "weather_wind"]
    for col in cols:
        if col not in out.columns:
            out[col] = None
    return out[cols]


def _normalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.copy()
    g = g[g["game_type"].isin(GAME_TYPES)]
    for col in ("game_pk", "away_team_id", "home_team_id", "venue_id", "season"):
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.dropna(subset=["game_pk", "away_team_id", "home_team_id"])
    g["game_pk"] = g["game_pk"].astype(int)
    for col in ("away_team_id", "home_team_id"):
        g[col] = g[col].astype(int)
    for col in ("away_score", "home_score", "weather_temp_f", "away_probable_pitcher_id", "home_probable_pitcher_id",
                "scheduled_innings", "game_number"):
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g["official_date"] = pd.to_datetime(g["official_date"], errors="coerce").dt.normalize()
    g["dt"] = pd.to_datetime(g["game_date"], utc=True, errors="coerce").dt.tz_localize(None)
    g["dt"] = g["dt"].fillna(g["official_date"] + pd.Timedelta(hours=23))
    g["season"] = g["season"].fillna(g["official_date"].dt.year).astype(int)
    g["is_final"] = (g["status_detailed"].map(is_completed_state).astype(bool)
                     & g["home_score"].notna() & g["away_score"].notna())
    # A rescheduled game keeps its gamePk and shows up twice (once "Postponed"): keep the
    # played one, else the latest listing.
    g = g.sort_values(["game_pk", "is_final", "dt"]).drop_duplicates("game_pk", keep="last")
    not_played = g["status_detailed"].astype(str).str.startswith(_NOT_PLAYED_PREFIXES)
    g = g[~not_played | g["is_final"]]
    tie = g["is_final"] & (g["home_score"] == g["away_score"])
    g["home_win"] = np.where(g["is_final"] & ~tie, (g["home_score"] > g["away_score"]).astype(float), np.nan)
    g["is_postseason"] = g["game_type"].isin(POSTSEASON_TYPES).astype(float)
    g["game_number"] = g["game_number"].fillna(1)
    g = g.sort_values(["dt", "game_number", "game_pk"]).reset_index(drop=True)
    g["order"] = np.arange(len(g), dtype=np.int64)
    return g


def load_results(seasons: Iterable[int], *, client: Any = None, normalized_dir: Path = NORMALIZED_DIR,
                 allow_network: bool = True) -> pd.DataFrame:
    """All games (completed and scheduled) for `seasons`, live from the Stats API when
    reachable, else from the local normalized tables. Local probable-pitcher IDs and
    weather fill gaps in the live rows (the live schedule occasionally drops them)."""
    frames = []
    for season in seasons:
        live = None
        if allow_network and client is not None:
            try:
                live = fetch_season_results(client, season)
            except Exception as exc:  # network variability; the local table is the fallback
                logger.warning("MLB results %s: live fetch failed (%s); using local tables", season, exc)
        local = None
        try:
            local = load_local_season(season, normalized_dir)
        except FileNotFoundError:
            pass
        if live is not None and not live.empty:
            if local is not None and not local.empty:
                fill_cols = ["away_probable_pitcher_id", "away_probable_pitcher_name", "home_probable_pitcher_id",
                             "home_probable_pitcher_name", "weather_condition", "weather_temp_f", "weather_wind"]
                lk = local.drop_duplicates("game_pk").set_index("game_pk")[fill_cols]
                live = live.set_index("game_pk")
                for col in fill_cols:
                    live[col] = live[col].where(live[col].notna(), lk[col].reindex(live.index))
                live = live.reset_index()
            frames.append(live)
        elif local is not None and not local.empty:
            frames.append(local)
        else:
            logger.warning("MLB results %s: no live or local data", season)
    if not frames:
        raise RuntimeError("no MLB results available (live and local both failed)")
    return _normalize_results(pd.concat(frames, ignore_index=True))


# ── starter pitching lines ───────────────────────────────────────────────────

LINE_COLUMNS = ["game_pk", "team_id", "starter_id", "ip", "er", "bb", "so", "hr", "bf", "h", "pitches", "strikes"]
PROFILE_COLUMNS = ["player_id", "full_name", "birth_date", "height", "birth_country", "pitch_hand"]


def load_local_starter_lines(normalized_dir: Path = NORMALIZED_DIR) -> pd.DataFrame:
    try:
        sl = _read_table(normalized_dir / "starter_game_logs_latest")
    except FileNotFoundError:
        return pd.DataFrame(columns=LINE_COLUMNS)
    out = pd.DataFrame({
        "game_pk": pd.to_numeric(sl["game_pk"], errors="coerce"),
        "team_id": pd.to_numeric(sl["team_id"], errors="coerce"),
        "starter_id": pd.to_numeric(sl["starter_id"], errors="coerce"),
        "ip": pd.to_numeric(sl["innings_pitched"], errors="coerce"),
        "er": pd.to_numeric(sl["earned_runs"], errors="coerce"),
        "bb": pd.to_numeric(sl["walks"], errors="coerce"),
        "so": pd.to_numeric(sl["strikeouts"], errors="coerce"),
        "hr": pd.to_numeric(sl["home_runs_allowed"], errors="coerce"),
        "bf": pd.to_numeric(sl["batters_faced"], errors="coerce"),
        "h": pd.to_numeric(sl.get("hits_allowed"), errors="coerce"),
        "pitches": pd.to_numeric(sl.get("pitches_thrown"), errors="coerce"),
        "strikes": pd.to_numeric(sl.get("strikes"), errors="coerce"),
    })
    out = out.dropna(subset=["game_pk", "team_id", "starter_id", "ip"])
    for col in ("game_pk", "team_id", "starter_id"):
        out[col] = out[col].astype(int)
    return out.drop_duplicates(["game_pk", "team_id"], keep="last").reset_index(drop=True)


def _outs(stat: dict[str, Any]) -> Optional[float]:
    if stat.get("outs") is not None:
        return _to_float(stat.get("outs"))
    text = str(stat.get("inningsPitched") or "")
    if not text:
        return None
    whole, _, frac = text.partition(".")
    try:
        return int(whole) * 3 + int((frac or "0")[0])
    except ValueError:
        return None


def flatten_pitcher_profiles(payload: dict[str, Any]) -> pd.DataFrame:
    """Display profile fields from a /people payload (hand, birth date, height, country)."""
    rows = [{
        "player_id": person.get("id"), "full_name": person.get("fullName"), "birth_date": person.get("birthDate"),
        "height": person.get("height"), "birth_country": person.get("birthCountry"),
        "pitch_hand": (person.get("pitchHand") or {}).get("code"),
    } for person in payload.get("people", []) or []]
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def flatten_pitcher_game_logs(payload: dict[str, Any]) -> pd.DataFrame:
    """Starts (gamesStarted == 1) from a /people?hydrate=stats(type=[gameLog]) payload."""
    rows = []
    for person in payload.get("people", []) or []:
        pid = person.get("id")
        for block in person.get("stats", []) or []:
            for split in block.get("splits", []) or []:
                stat = split.get("stat", {}) or {}
                if int(stat.get("gamesStarted") or 0) != 1:
                    continue
                outs = _outs(stat)
                rows.append({
                    "game_pk": (split.get("game") or {}).get("gamePk"),
                    "team_id": (split.get("team") or {}).get("id"),
                    "starter_id": pid,
                    "ip": None if outs is None else outs / 3.0,
                    "er": _to_float(stat.get("earnedRuns")),
                    "bb": _to_float(stat.get("baseOnBalls")),
                    "so": _to_float(stat.get("strikeOuts")),
                    "hr": _to_float(stat.get("homeRuns")),
                    "bf": _to_float(stat.get("battersFaced")),
                    "h": _to_float(stat.get("hits")),
                    "pitches": _to_float(stat.get("numberOfPitches")),
                    "strikes": _to_float(stat.get("strikes")),
                })
    frame = pd.DataFrame(rows, columns=LINE_COLUMNS)
    frame = frame.dropna(subset=["game_pk", "team_id", "starter_id", "ip"])
    for col in ("game_pk", "team_id", "starter_id"):
        frame[col] = frame[col].astype(int)
    return frame


def fetch_starter_lines(client: Any, pitcher_seasons: dict[int, set[int]], *, chunk: int = 40,
                        profiles_sink: Optional[list] = None) -> pd.DataFrame:
    """Live starts for each (season -> pitcher ids), in bulk: one people call per chunk.
    Pitcher profile rows from the same payloads are appended to `profiles_sink`."""
    frames = []
    types = ",".join(GAME_TYPES)
    for season, ids in sorted(pitcher_seasons.items()):
        ids = sorted(int(i) for i in ids if i is not None and not pd.isna(i))
        for start in range(0, len(ids), chunk):
            part = ids[start:start + chunk]
            try:
                payload = client._get_json("/api/v1/people", params={
                    "personIds": ",".join(str(i) for i in part),
                    "hydrate": f"stats(group=[pitching],type=[gameLog],season={int(season)},gameType=[{types}])",
                })
            except Exception as exc:
                logger.warning("MLB starter logs %s (%d ids): fetch failed (%s)", season, len(part), exc)
                continue
            frames.append(flatten_pitcher_game_logs(payload))
            if profiles_sink is not None:
                profiles_sink.append(flatten_pitcher_profiles(payload))
    if not frames:
        return pd.DataFrame(columns=LINE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def assemble_starter_lines(games: pd.DataFrame, *, client: Any = None, normalized_dir: Path = NORMALIZED_DIR,
                           allow_network: bool = True, local: Optional[pd.DataFrame] = None,
                           profiles_sink: Optional[list] = None) -> pd.DataFrame:
    """Local starter lines + live top-up for completed games the local tables lack
    (everything after the last S3 sync, and every postseason game)."""
    local = load_local_starter_lines(normalized_dir) if local is None else local
    lines = local[local["game_pk"].isin(games["game_pk"])].copy()
    if allow_network and client is not None:
        have = set(zip(lines["game_pk"], lines["team_id"]))
        need: dict[int, set[int]] = {}
        done = games[games["is_final"]]
        for side in ("home", "away"):
            pid = done[f"{side}_probable_pitcher_id"]
            miss = [(int(s), int(p)) for gp, tid, p, s in zip(done["game_pk"], done[f"{side}_team_id"], pid, done["season"])
                    if (gp, tid) not in have and not pd.isna(p)]
            for season, p in miss:
                need.setdefault(season, set()).add(p)
        # Upcoming probables: their latest starts may also post-date the local tables.
        upcoming = games[~games["is_final"]]
        for side in ("home", "away"):
            for p, s in zip(upcoming[f"{side}_probable_pitcher_id"], upcoming["season"]):
                if not pd.isna(p):
                    need.setdefault(int(s), set()).add(int(p))
        if need:
            logger.info("MLB starter logs: fetching %d pitcher-seasons live",
                        sum(len(v) for v in need.values()))
            live = fetch_starter_lines(client, need, profiles_sink=profiles_sink)
            live = live[live["game_pk"].isin(games["game_pk"])]
            lines = pd.concat([lines, live], ignore_index=True)
    return lines.drop_duplicates(["game_pk", "team_id"], keep="last").reset_index(drop=True)


# ── features ─────────────────────────────────────────────────────────────────

def compute_elo(games: pd.DataFrame, *, k: float, home_edge: float, season_regress: float,
                initial: float = 1500.0) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """Pregame Elo for every row of `games` (sorted by `order`). Completed games update the
    ratings; scheduled games only read them."""
    elo: dict[int, float] = {}
    last_season: dict[int, int] = {}
    pre_h = np.zeros(len(games)); pre_a = np.zeros(len(games))
    cols = games[["home_team_id", "away_team_id", "season", "home_win", "home_score", "away_score"]].itertuples(index=False)
    for i, r in enumerate(cols):
        for t in (r.home_team_id, r.away_team_id):
            if t not in elo:
                elo[t] = initial; last_season[t] = r.season
            elif last_season[t] != r.season:
                elo[t] = initial + (elo[t] - initial) * (1.0 - season_regress); last_season[t] = r.season
        eh, ea = elo[r.home_team_id], elo[r.away_team_id]
        pre_h[i], pre_a[i] = eh, ea
        if not np.isnan(r.home_win):
            p = 1.0 / (1.0 + 10 ** (-(eh + home_edge - ea) / 400.0))
            mov = math.log(abs(r.home_score - r.away_score) + 1.0)
            delta = k * mov * (r.home_win - p)
            elo[r.home_team_id] += delta; elo[r.away_team_id] -= delta
    return pre_h, pre_a, elo


def _decayed_state(frame: pd.DataFrame, key: str, values: list[str], half_life: float) -> pd.DataFrame:
    """Per `key`, exponentially decayed sums of `values` (and of the count, column 'n')
    INCLUDING each row -- the state after that row. Rows whose first value is NaN do not
    update (no decay, nothing added). `frame` must be sorted by order within key."""
    a = 0.5 ** (1.0 / half_life)
    arr = frame[values].to_numpy(dtype=float)
    out = np.zeros((len(frame), len(values) + 1))
    for _, idx in frame.groupby(key, sort=False).indices.items():
        acc = np.zeros(len(values) + 1)
        for i in idx:
            row = arr[i]
            if not np.isnan(row[0]):
                acc = acc * a
                acc[:-1] += np.nan_to_num(row)
                acc[-1] += 1.0
            out[i] = acc
    res = pd.DataFrame(out, columns=[f"d_{v}" for v in values] + ["d_n"], index=frame.index)
    return pd.concat([frame[[key, "order"]], res], axis=1)


def _asof(target: pd.DataFrame, state: pd.DataFrame, key_target: str, key_state: str) -> pd.DataFrame:
    """For each target row, the state of its key after the last row STRICTLY before it."""
    t = target[["order", key_target]].rename(columns={key_target: "_key"}).copy()
    s = state.rename(columns={key_state: "_key"})
    t["_key"] = pd.to_numeric(t["_key"], errors="coerce")
    s = s.assign(_key=pd.to_numeric(s["_key"], errors="coerce")).dropna(subset=["_key"])
    t["_row"] = np.arange(len(t))
    tt = t.dropna(subset=["_key"]).sort_values("order")
    ss = s.sort_values("order")
    tt["_key"] = tt["_key"].astype(np.int64); ss["_key"] = ss["_key"].astype(np.int64)
    m = pd.merge_asof(tt, ss, on="order", by="_key", direction="backward", allow_exact_matches=False)
    m = m.set_index("_row").reindex(np.arange(len(t)))
    return m.drop(columns=["order", "_key"], errors="ignore").set_index(target.index)


def build_features(games: pd.DataFrame, lines: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """One row per game (completed and scheduled) with every pregame feature both models use."""
    P, D, L = params["elo"], params["decay"], params["league"]
    g = games.sort_values("order").reset_index(drop=True).copy()

    # weather / context
    g["temp"] = pd.to_numeric(g["weather_temp_f"], errors="coerce")
    g["wind_out"] = [wind_out_mph(w, c) for w, c in zip(g["weather_wind"], g["weather_condition"])]
    g["roof_closed"] = g["weather_condition"].isin(["Dome", "Roof Closed"]).astype(float)
    g["is_night"] = (g["day_night"] == "night").astype(float)

    # Elo
    g["home_elo"], g["away_elo"], _ = compute_elo(g, k=P["k"], home_edge=P["home_edge"],
                                                  season_regress=P["season_regress"], initial=P["initial"])
    g["elo_diff"] = g["home_elo"] + P["home_edge"] - g["away_elo"]
    g["elo_p"] = 1.0 / (1.0 + 10 ** (-g["elo_diff"] / 400.0))

    # team offense/defense: decayed runs scored/allowed per game, shrunk to the league
    long = []
    for side, opp in (("home", "away"), ("away", "home")):
        t = pd.DataFrame({"order": g["order"], "team_id": g[f"{side}_team_id"],
                          "rs": np.where(g["is_final"], g[f"{side}_score"], np.nan),
                          "ra": np.where(g["is_final"], g[f"{opp}_score"], np.nan)})
        long.append(t)
    long = pd.concat(long, ignore_index=True).sort_values(["team_id", "order"]).reset_index(drop=True)
    tstate = _decayed_state(long, "team_id", ["rs", "ra"], D["team_half_life_games"])
    kprior, lrpg = D["team_prior_games"], L["runs_per_team_game"]
    for side in ("home", "away"):
        s = _asof(g, tstate, f"{side}_team_id", "team_id")
        g[f"{side[0]}_rs_pg"] = (s["d_rs"].fillna(0) + kprior * lrpg) / (s["d_n"].fillna(0) + kprior)
        g[f"{side[0]}_ra_pg"] = (s["d_ra"].fillna(0) + kprior * lrpg) / (s["d_n"].fillna(0) + kprior)
        g[f"{side[0]}_games_prior"] = s["d_n"].fillna(0)

    # park factor: decayed total runs at the venue, shrunk to the league total
    v = pd.DataFrame({"order": g["order"], "venue_id": g["venue_id"],
                      "tot": np.where(g["is_final"], g["home_score"] + g["away_score"], np.nan)})
    v = v.dropna(subset=["venue_id"]).sort_values(["venue_id", "order"])
    vstate = _decayed_state(v, "venue_id", ["tot"], D["venue_half_life_games"])
    vs = _asof(g, vstate, "venue_id", "venue_id")
    ltot, vprior = L["total_runs_per_game"], D["venue_prior_games"]
    g["park_pf"] = ((vs["d_tot"].fillna(0) + vprior * ltot) / (vs["d_n"].fillna(0) + vprior)) / ltot
    g["log_pf"] = np.log(g["park_pf"])

    # starters: who starts each game (actual starter line when known, else the probable)
    lines = lines.copy()
    starter_of = {(int(a), int(b)): int(c) for a, b, c in zip(lines["game_pk"], lines["team_id"], lines["starter_id"])}
    for side in ("home", "away"):
        g[f"{side}_starter_id"] = [starter_of.get((int(pk), int(tid)), p) for pk, tid, p in
                                   zip(g["game_pk"], g[f"{side}_team_id"], g[f"{side}_probable_pitcher_id"])]
        g[f"{side}_starter_id"] = pd.to_numeric(g[f"{side}_starter_id"], errors="coerce")
    lines = lines.merge(g[["game_pk", "order"]], on="game_pk", how="inner")
    lines["fip_num"] = 13 * lines["hr"] + 3 * lines["bb"] - 2 * lines["so"]
    lines["kbb"] = lines["so"] - lines["bb"]
    lines = lines.sort_values(["starter_id", "order"]).reset_index(drop=True)
    sstate = _decayed_state(lines, "starter_id", ["ip", "fip_num", "kbb", "bf"], D["starter_half_life_starts"])
    ipstate = _decayed_state(lines.assign(ip2=lines["ip"]), "starter_id", ["ip2"], D["starter_ip_half_life_starts"])
    sstate["d_ip_short"] = ipstate["d_ip2"].to_numpy(); sstate["d_n_short"] = ipstate["d_n"].to_numpy()
    sstate["starts"] = lines.groupby("starter_id").cumcount().to_numpy() + 1
    kip, kbf, repl = D["starter_prior_ip"], D["starter_prior_bf"], D["starter_replacement"]
    lfip, lkbb, lip, kips = L["fip_num_per_ip"], L["kbb_per_bf"], L["ip_per_start"], D["starter_ip_prior_starts"]
    for side in ("home", "away"):
        s = _asof(g, sstate, f"{side}_starter_id", "starter_id")
        has = g[f"{side}_starter_id"].notna()
        fip = (s["d_fip_num"].fillna(0) + kip * lfip * repl) / (s["d_ip"].fillna(0) + kip)
        kbb = (s["d_kbb"].fillna(0) + kbf * lkbb / repl) / (s["d_bf"].fillna(0) + kbf)
        ip = (s["d_ip_short"].fillna(0) + kips * lip * 0.9) / (s["d_n_short"].fillna(0) + kips)
        starts = s["starts"].fillna(0)
        p = side[0]
        g[f"{p}_sp_fip"] = fip.where(has)
        g[f"{p}_sp_kbb"] = kbb.where(has)
        g[f"{p}_sp_ip"] = ip.where(has)
        g[f"{p}_sp_starts"] = starts.where(has)
        g[f"{p}_sp_log_starts"] = np.log1p(starts).where(has)
    return g


def team_game_frame(F: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (each team's offense vs the other's starter) for the runs model."""
    parts = []
    for s, o, is_home in (("h", "a", 1.0), ("a", "h", 0.0)):
        side = "home" if s == "h" else "away"
        parts.append(pd.DataFrame({
            "game_pk": F["game_pk"].to_numpy(), "order": F["order"].to_numpy(),
            "official_date": F["official_date"].to_numpy(), "season": F["season"].to_numpy(),
            "is_final": F["is_final"].to_numpy(), "is_home": is_home,
            "runs": F[f"{side}_score"].where(F["is_final"]).to_numpy(),
            "off_rs_pg": F[f"{s}_rs_pg"].to_numpy(), "def_ra_pg": F[f"{o}_ra_pg"].to_numpy(),
            "opp_sp_fip": F[f"{o}_sp_fip"].to_numpy(), "opp_sp_kbb": F[f"{o}_sp_kbb"].to_numpy(),
            "opp_sp_ip": F[f"{o}_sp_ip"].to_numpy(), "opp_sp_log_starts": F[f"{o}_sp_log_starts"].to_numpy(),
            "log_pf": F["log_pf"].to_numpy(), "park_pf": F["park_pf"].to_numpy(), "temp": F["temp"].to_numpy(),
            "wind_out": F["wind_out"].to_numpy(), "roof_closed": F["roof_closed"].to_numpy(),
            "is_night": F["is_night"].to_numpy(), "elo_diff_side": (F["elo_diff"] if s == "h" else -F["elo_diff"]).to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


# ── models (fit with sklearn, stored and applied as plain numbers) ───────────

@dataclass
class LinearModel:
    """Median-impute -> standardize -> linear predictor. kind: 'logistic' | 'poisson'."""
    kind: str
    features: list[str]
    medians: list[float]
    means: list[float]
    scales: list[float]
    coef: list[float]
    intercept: float
    extra: dict[str, Any]

    def linear(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        X = np.where(np.isnan(X), np.asarray(self.medians), X)
        Z = (X - np.asarray(self.means)) / np.asarray(self.scales)
        return Z @ np.asarray(self.coef) + self.intercept

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        eta = self.linear(frame)
        if self.kind == "logistic":
            return 1.0 / (1.0 + np.exp(-eta))
        return np.exp(eta)

    def to_dict(self) -> dict[str, Any]:
        r = lambda xs: [round(float(x), 8) for x in xs]
        return {"kind": self.kind, "features": self.features, "medians": r(self.medians), "means": r(self.means),
                "scales": r(self.scales), "coef": r(self.coef), "intercept": round(float(self.intercept), 8),
                "extra": self.extra}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LinearModel":
        return cls(d["kind"], list(d["features"]), list(d["medians"]), list(d["means"]), list(d["scales"]),
                   list(d["coef"]), float(d["intercept"]), dict(d.get("extra") or {}))


def _design(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    X = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X = np.where(np.isnan(X), med, X)
    mean = X.mean(axis=0); scale = X.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (X - mean) / scale, med.tolist(), mean.tolist(), scale.tolist()


def fit_win_model(train: pd.DataFrame, params: dict[str, Any]) -> LinearModel:
    from sklearn.linear_model import LogisticRegression

    feats = list(params["win_model"]["features"])
    rows = train[train["home_win"].notna()]
    Z, med, mean, scale = _design(rows, feats)
    m = LogisticRegression(C=float(params["win_model"]["C"]), max_iter=2000).fit(Z, rows["home_win"].astype(int))
    return LinearModel("logistic", feats, med, mean, scale, m.coef_[0].tolist(), float(m.intercept_[0]),
                       {"n_train": int(len(rows)), "home_rate": round(float(rows["home_win"].mean()), 5)})


def _nb_logpmf(y: np.ndarray, r: float, mu: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def fit_nb_r(y: np.ndarray, mu: np.ndarray) -> float:
    from scipy import optimize
    f = lambda lr: -_nb_logpmf(y, math.exp(lr), mu).sum()
    res = optimize.minimize_scalar(f, bounds=(-2.0, 5.0), method="bounded")
    return float(math.exp(res.x))


def fit_runs_model(train_long: pd.DataFrame, params: dict[str, Any]) -> LinearModel:
    from sklearn.linear_model import PoissonRegressor

    feats = list(params["runs_model"]["features"])
    rows = train_long[train_long["runs"].notna()]
    Z, med, mean, scale = _design(rows, feats)
    m = PoissonRegressor(alpha=float(params["runs_model"]["alpha"]), max_iter=1000).fit(Z, rows["runs"].to_numpy())
    model = LinearModel("poisson", feats, med, mean, scale, m.coef_.tolist(), float(m.intercept_),
                        {"n_train": int(len(rows))})
    mu = model.predict(rows)
    model.extra["nb_r"] = round(fit_nb_r(rows["runs"].to_numpy(dtype=float), mu), 5)
    return model


# ── score distributions ──────────────────────────────────────────────────────

def nb_pmf(mu: float, r: float, max_runs: int = MAX_RUNS) -> np.ndarray:
    """P(runs = 0..max_runs) for a negative binomial with mean mu and size r (renormalized)."""
    ks = np.arange(max_runs + 1, dtype=float)
    pmf = np.exp(_nb_logpmf(ks, float(r), np.full_like(ks, max(float(mu), 1e-6))))
    return pmf / pmf.sum()


def score_matrix(mu_home: float, mu_away: float, r: float, max_runs: int = MAX_RUNS) -> np.ndarray:
    """P[home_runs = i, away_runs = j] with independent NB margins."""
    return np.outer(nb_pmf(mu_home, r, max_runs), nb_pmf(mu_away, r, max_runs))


def nb_pmf_rows(mu: Iterable[float], r: float, max_runs: int = MAX_RUNS) -> np.ndarray:
    """nb_pmf for many means at once: shape (len(mu), max_runs + 1), rows renormalized."""
    ks = np.arange(max_runs + 1, dtype=float)[None, :]
    mu = np.maximum(np.asarray(list(mu), dtype=float), 1e-6)[:, None]
    pmf = np.exp(_nb_logpmf(ks, float(r), mu))
    return pmf / pmf.sum(axis=1, keepdims=True)


def _fair_half_lines(pmf: np.ndarray) -> np.ndarray:
    """Row-wise x.5 line whose P(value > line) is closest to 0.5 (markets_mlb.fair_half_line)."""
    sf = 1.0 - np.cumsum(pmf, axis=1)
    return np.argmin(np.abs(sf - 0.5), axis=1) + 0.5


def batch_probabilities(mu_home: Iterable[float], mu_away: Iterable[float], r: Iterable[float] | float,
                        *, total_lines: Iterable[float] = (), team_lines: Iterable[float] = ()) -> dict[str, np.ndarray]:
    """Per game (vectorized) P(home by 2+), P(away by 2+), P(tied after the modelled
    runs), P(total > L) and P(total < L) for each L in total_lines, P(team runs > L) for
    each L in team_lines, and the served fair x.5 lines with P(over) at them
    (total_fair_line / over_fair, {home,away}_fair_line / {home,away}_over_fair) -- the
    numbers behind the run line / total / team total markets, for evaluation over
    thousands of games."""
    mu_home = np.asarray(list(mu_home), dtype=float); mu_away = np.asarray(list(mu_away), dtype=float)
    rs = np.broadcast_to(np.asarray(r, dtype=float), mu_home.shape)
    out: dict[str, list] = {}
    i, j = np.indices((MAX_RUNS + 1, MAX_RUNS + 1))
    for r_value in np.unique(rs):
        idx = np.flatnonzero(rs == r_value)
        ph, pa = nb_pmf_rows(mu_home[idx], r_value), nb_pmf_rows(mu_away[idx], r_value)
        m = ph[:, :, None] * pa[:, None, :]
        parts = {"home_by_2": (m * (i - j >= 2)).sum((1, 2)), "away_by_2": (m * (j - i >= 2)).sum((1, 2)),
                 "tie": (m * (i == j)).sum((1, 2))}
        for line in total_lines:
            parts[f"over_{line:g}"] = (m * ((i + j) > line)).sum((1, 2))
            parts[f"under_{line:g}"] = (m * ((i + j) < line)).sum((1, 2))
        ks = np.arange(MAX_RUNS + 1)
        for line in team_lines:
            parts[f"home_over_{line:g}"] = ph[:, ks > line].sum(axis=1)
            parts[f"away_over_{line:g}"] = pa[:, ks > line].sum(axis=1)
        tot = np.stack([(m * ((i + j) == k)).sum((1, 2)) for k in range(2 * MAX_RUNS + 1)], axis=1)
        parts["total_fair_line"] = _fair_half_lines(tot)
        parts["over_fair"] = (tot * (np.arange(tot.shape[1])[None, :] > parts["total_fair_line"][:, None])).sum(axis=1)
        for name, pmf in (("home", ph), ("away", pa)):
            line = _fair_half_lines(pmf)
            parts[f"{name}_fair_line"] = line
            parts[f"{name}_over_fair"] = (pmf * (ks[None, :] > line[:, None])).sum(axis=1)
        for key, values in parts.items():
            out.setdefault(key, []).append((idx, values))
    result = {}
    for key, chunks in out.items():
        arr = np.empty(len(mu_home))
        for idx, values in chunks:
            arr[idx] = values
        result[key] = arr
    return result


def matrix_probabilities(m: np.ndarray, *, total_line: Optional[float] = None) -> dict[str, float]:
    i, j = np.indices(m.shape)
    out = {
        "home_by_2": float(m[i - j >= 2].sum()), "away_by_2": float(m[j - i >= 2].sum()),
        "tie_after_9": float(m[i == j].sum()),
        "exp_home": float((m.sum(axis=1) * np.arange(m.shape[0])).sum()),
        "exp_away": float((m.sum(axis=0) * np.arange(m.shape[1])).sum()),
    }
    if total_line is not None:
        out["over"] = float(m[(i + j) > total_line].sum())
    return out


# ── walk-forward ─────────────────────────────────────────────────────────────

def month_starts(dates: pd.Series) -> list[pd.Timestamp]:
    months = pd.to_datetime(dates).dt.to_period("M").dropna().unique()
    return [p.to_timestamp() for p in sorted(months)]


def walk_forward(F: pd.DataFrame, params: dict[str, Any], *, start: pd.Timestamp,
                 end: Optional[pd.Timestamp] = None, refit: str = "monthly",
                 runs: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict every game (completed or scheduled) dated in [start, end] with models fit
    on completed games strictly before the first day of the game's month.

    Returns (predictions, last_models) where predictions has game_pk, p_home, mu_home,
    mu_away, nb_r, train_cutoff, train_n."""
    target = F[F["official_date"] >= start]
    if end is not None:
        target = target[target["official_date"] <= end]
    if target.empty:
        return pd.DataFrame(columns=["game_pk", "p_home", "mu_home", "mu_away", "nb_r", "train_cutoff", "train_n"]), {}
    long = team_game_frame(F)
    if refit == "monthly":
        cutoffs = month_starts(target["official_date"])
        bucket = target["official_date"].dt.to_period("M").dt.to_timestamp()
    elif refit == "season":
        cutoffs = sorted({pd.Timestamp(f"{s}-01-01") for s in target["season"]})
        bucket = pd.to_datetime(target["season"].astype(str) + "-01-01")
    else:
        raise ValueError(refit)
    out = []
    models: dict[str, Any] = {}
    for cutoff in cutoffs:
        rows = target[bucket == cutoff]
        if rows.empty:
            continue
        train = F[F["is_final"] & (F["official_date"] < cutoff)]
        win = fit_win_model(train, params)
        res = pd.DataFrame({"game_pk": rows["game_pk"].to_numpy(), "p_home": win.predict(rows),
                            "train_cutoff": cutoff.strftime("%Y-%m-%d"), "train_n": len(train)})
        if runs:
            tl = long[long["is_final"] & (long["official_date"] < cutoff)]
            rm = fit_runs_model(tl, params)
            lt = long[long["game_pk"].isin(rows["game_pk"])]
            mu = pd.Series(rm.predict(lt), index=lt.index)
            mh = dict(zip(lt.loc[lt["is_home"] == 1, "game_pk"], mu[lt["is_home"] == 1]))
            ma = dict(zip(lt.loc[lt["is_home"] == 0, "game_pk"], mu[lt["is_home"] == 0]))
            res["mu_home"] = res["game_pk"].map(mh)
            res["mu_away"] = res["game_pk"].map(ma)
            res["nb_r"] = rm.extra["nb_r"]
            models["runs"] = rm
        models["win"] = win
        models["cutoff"] = cutoff
        out.append(res)
    return pd.concat(out, ignore_index=True), models


@dataclass
class PregameInputs:
    frame: pd.DataFrame       # build_features output: one row per game, completed and scheduled
    lines: pd.DataFrame       # starter pitching lines (local + live top-up)
    profiles: pd.DataFrame    # pitcher display profiles (local table + live people payloads)


def load_pitcher_profiles(normalized_dir: Path = NORMALIZED_DIR) -> pd.DataFrame:
    try:
        pp = _read_table(normalized_dir / "pitcher_profiles_latest")
    except FileNotFoundError:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    out = pd.DataFrame({c: pp[c] if c in pp.columns else None for c in PROFILE_COLUMNS})
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    return out.dropna(subset=["player_id"])


def prepare_inputs(seasons: Iterable[int], *, client: Any = None, params: Optional[dict[str, Any]] = None,
                   normalized_dir: Path = NORMALIZED_DIR, allow_network: bool = True) -> PregameInputs:
    """Results + starter lines + features in one call (what the exporter needs)."""
    params = params or load_params()
    games = load_results(seasons, client=client, normalized_dir=normalized_dir, allow_network=allow_network)
    sink: list[pd.DataFrame] = []
    lines = assemble_starter_lines(games, client=client, normalized_dir=normalized_dir,
                                   allow_network=allow_network, profiles_sink=sink)
    profiles = pd.concat([load_pitcher_profiles(normalized_dir), *sink], ignore_index=True)
    profiles["player_id"] = pd.to_numeric(profiles["player_id"], errors="coerce")
    profiles = profiles.dropna(subset=["player_id"]).drop_duplicates("player_id", keep="last")
    logger.info("MLB pregame frame: %d games (%d final), %d starter lines", len(games),
                int(games["is_final"].sum()), len(lines))
    return PregameInputs(build_features(games, lines, params), lines, profiles)


# ── display context (pregame, for board cards; never model inputs) ───────────

def team_display_context(F: pd.DataFrame) -> pd.DataFrame:
    """Per game: each team's record and recent form before first pitch, named the way the
    exporter's card builders read them ({side}_team_games_played_prior, ...)."""
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({
            "game_pk": F["game_pk"], "order": F["order"], "season": F["season"], "team_id": F[f"{side}_team_id"],
            "side": side, "final": F["is_final"] & F["home_win"].notna(),
            "won": np.where(F["home_win"].notna(), (F["home_win"] == (1.0 if side == "home" else 0.0)).astype(float), np.nan),
            "rs": F[f"{side}_score"].where(F["is_final"]), "ra": F[f"{opp}_score"].where(F["is_final"]),
            "is_reg": F["game_type"] == "R",
        }))
    t = pd.concat(rows, ignore_index=True).sort_values(["team_id", "order"]).reset_index(drop=True)
    done = t[t["final"]].copy()
    done["rd"] = done["rs"] - done["ra"]
    g = done.groupby("team_id")
    done["w_last5"] = g["won"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    done["rd_last10"] = g["rd"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    done["rs_last5"] = g["rs"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    reg = done[done["is_reg"]].copy()
    rg = reg.groupby(["team_id", "season"])
    reg["gp"] = rg.cumcount() + 1
    reg["wins"] = rg["won"].cumsum()
    state = done[["team_id", "order", "w_last5", "rd_last10", "rs_last5"]]
    rec = reg[["team_id", "order", "season", "gp", "wins"]].rename(columns={"season": "rec_season"})
    out = pd.DataFrame({"game_pk": F["game_pk"]})
    for side in ("home", "away"):
        s = _asof(F, state, f"{side}_team_id", "team_id")
        r = _asof(F, rec, f"{side}_team_id", "team_id")
        same = r["rec_season"] == F["season"]
        out[f"{side}_team_games_played_prior"] = r["gp"].where(same, 0.0).to_numpy()
        out[f"{side}_team_win_pct_prior"] = (r["wins"] / r["gp"]).where(same).to_numpy()
        out[f"{side}_team_won_avg_last_5"] = s["w_last5"].to_numpy()
        out[f"{side}_team_run_diff_avg_last_10"] = s["rd_last10"].to_numpy()
        out[f"{side}_team_runs_scored_avg_last_5"] = s["rs_last5"].to_numpy()
    return out


def starter_display_context(F: pd.DataFrame, lines: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Per game: each starter's last-3/last-5 averages and profile before first pitch, named
    the way the exporter's starter card/radar builders read them."""
    ln = lines.merge(F[["game_pk", "order", "official_date"]], on="game_pk", how="inner")
    ln = ln.sort_values(["starter_id", "order"]).reset_index(drop=True)
    ip = ln["ip"].where(ln["ip"] > 0)
    ln["era_like"] = 9.0 * ln["er"] / ip
    ln["whip"] = (ln["h"] + ln["bb"]) / ip
    ln["strike_pct"] = ln["strikes"] / ln["pitches"].where(ln["pitches"] > 0)
    metrics = {"era_like": "era_like", "whip": "whip", "so": "strikeouts", "bb": "walks", "h": "hits_allowed",
               "hr": "home_runs_allowed", "ip": "innings_pitched", "bf": "batters_faced", "pitches": "pitches_thrown",
               "strike_pct": "strike_pct", "er": "earned_runs"}
    g = ln.groupby("starter_id")
    cols = []
    for src, name in metrics.items():
        for w in (3, 5):
            col = f"{name}_avg_last_{w}"
            ln[col] = g[src].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            cols.append(col)
    ln["starts_prior"] = g.cumcount() + 1
    ln["last_start_date"] = ln["official_date"]
    state = ln[["starter_id", "order", "starts_prior", "last_start_date", *cols]]
    prof = profiles.drop_duplicates("player_id").set_index("player_id") if not profiles.empty else pd.DataFrame()
    out = pd.DataFrame({"game_pk": F["game_pk"]})
    for side in ("home", "away"):
        s = _asof(F, state, f"{side}_starter_id", "starter_id")
        for col in cols + ["starts_prior"]:
            out[f"{side}_starter_{col}"] = s[col].to_numpy()
        rest = (F["official_date"] - pd.to_datetime(s["last_start_date"])).dt.days
        out[f"{side}_starter_days_rest"] = rest.where(rest <= 30).to_numpy()
        sid = F[f"{side}_starter_id"]
        if not prof.empty:
            born = pd.to_datetime(sid.map(prof["birth_date"]), errors="coerce")
            out[f"{side}_starter_profile_current_age"] = ((F["official_date"] - born).dt.days / 365.25).to_numpy()
            out[f"{side}_starter_profile_pitch_hand"] = sid.map(prof["pitch_hand"]).to_numpy()
            out[f"{side}_starter_profile_height"] = sid.map(prof["height"]).to_numpy()
            out[f"{side}_starter_profile_birth_country"] = sid.map(prof["birth_country"]).to_numpy()
    return out
