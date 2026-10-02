"""Export the NBA/WNBA boards (upcoming + honest history + model-only markets).

What changed (2026-10-02, plan items P1-4 and P1-12):

* Served model: Elo-MOV prior + L1-regularized logistic regression over box-score
  form, true rest/back-to-back and postseason flags (sports/basketball/
  honest_model.py), refit at every export on every completed game before today.
  The torch MLP / HistGradientBoosting artifacts are no longer read.
* The served probability goes through a Platt map fitted only on earlier
  out-of-sample (walk-forward) predictions and shrunk toward the identity.
* History boards come from a 14-day walk-forward: each graded game was predicted by
  a model (and calibrator) fitted only on games from earlier dates, and the board
  records that cutoff (``modelTrainedThrough``) plus ``recordBasis: "simulated"``.
  Seasons whose games tuned the hyper-parameters (NBA 2022-24, WNBA 2024) are
  predicted too but never published. The old backtests were the HGB validation file
  and the "completed" boards regraded calendar-year games with the current model,
  including 167 games inside its training window (the published 72.6%).
* Everything that changes daily (results, Elo, rest, form) is rebuilt from data this
  export fetches: the live CDN schedule (whole current season with scores) plus the
  boxscores of completed games missing from the S3-synced tables.
* Upcoming boards carry a ``markets`` block (model fair spread both sides, total,
  team totals, moneyline; ``basis: "model"``, ``market: null``, never an edge). Each
  bake appends them to the market pick log and regrades everything logged so far
  into basketball_market_history.json / basketball_market_summary.json. Market
  results never touch hitStatus / seasonSummary.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports import market_log, markets  # noqa: E402
from sports.basketball import honest_model as hm  # noqa: E402
from sports.basketball.backfill_history import (  # noqa: E402
    COMPETITIVE_TYPES,
    append_to_tables,
    bundles_to_frames,
    canonical_game_id,
    fetch_final_games,
    game_type_from_id,
)
from sports.basketball.build_training_dataset import DEFAULT_BASKETBALL_DATA_ROOT  # noqa: E402
from sports.basketball.client import BasketballStatsClient, flatten_schedule  # noqa: E402
from sports.basketball.constants import LEAGUE_CONFIGS  # noqa: E402
from sports.basketball.results import LOCAL_TZ, local_dates  # noqa: E402
from sports.basketball.rotation_features import build_projected_rotation_map  # noqa: E402

PROJ_ROOT = Path(__file__).resolve().parents[2]
BASKETBALL_DATA_ROOT = DEFAULT_BASKETBALL_DATA_ROOT
NORMALIZED_DIR = BASKETBALL_DATA_ROOT / "normalized"
FRONTEND_DATA_DIR = PROJ_ROOT / "pythia_prophecy" / "frontend" / "src" / "data"
UPCOMING_LOOKAHEAD_DAYS = 7
MAX_AVAILABLE_DATES = 7
LEAGUES = ("nba", "wnba")
MARKET_SPORT_KEY = "basketball"
MARKET_HISTORY_CAP = 300
# History boards keep lineups only for the newest N (the BFF serves 150); older
# boards keep the keys with empty lineups so the file stays a few MB, not ~90.
HISTORY_LINEUP_BOARDS = int(os.getenv("BASKETBALL_HISTORY_LINEUP_BOARDS", "200"))
logger = logging.getLogger(__name__)

SEASON_OF = {"nba": market_log.season_cross_year(9), "wnba": market_log.season_calendar}


# ── small helpers (unchanged board formatting) ───────────────────────────────

def _to_datetime_mixed(values: object) -> pd.Series | pd.Timestamp:
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _display_tour(league: str) -> str:
    return "Women's Basketball" if league == "wnba" else "Basketball"


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _date_key(value: object) -> str | None:
    timestamp = _to_datetime_mixed(value)
    if pd.isna(timestamp):
        return None
    return timestamp.normalize().strftime("%Y-%m-%d")


def _date_key_int(value: object) -> int | None:
    timestamp = _to_datetime_mixed(value)
    if pd.isna(timestamp):
        return None
    return int(timestamp.normalize().strftime("%Y%m%d"))


def _date_label(value: str) -> str:
    timestamp = _to_datetime_mixed(value)
    if pd.isna(timestamp):
        return value
    return timestamp.strftime("%a, %b %d").replace(" 0", " ")


def _iso_utc(value: object) -> str | None:
    ts = _to_datetime_mixed(value)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _team_logo_url(league: str, team_id: int | None) -> str | None:
    if not team_id:
        return None
    if league == "wnba":
        return f"https://cdn.wnba.com/logos/wnba/{int(team_id)}/global/L/logo.svg"
    return f"https://cdn.nba.com/logos/nba/{int(team_id)}/global/L/logo.svg"


def _player_headshot_url(league: str, player_id: int | None) -> str | None:
    if not player_id:
        return None
    if league == "wnba":
        return f"https://cdn.wnba.com/headshots/wnba/latest/1040x760/{int(player_id)}.png"
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png"


def _format_record(wins: float | None, games: float | None) -> str | None:
    if games is None or wins is None or games <= 0:
        return None
    w = int(round(wins))
    return f"{w}-{int(round(games)) - w}"


def _format_recent_form(row: Any, side: str) -> str | None:
    wins_rate = _safe_float(getattr(row, f"{side}_team_won_avg_last_5", np.nan))
    net_rating = _safe_float(getattr(row, f"{side}_team_net_rating_est_avg_last_10", np.nan))
    points = _safe_float(getattr(row, f"{side}_team_points_scored_avg_last_5", np.nan))
    games = _safe_float(getattr(row, f"{side}_team_games_played_prior", np.nan)) or 0
    parts: list[str] = []
    if wins_rate is not None:
        n = int(min(5, games)) or 5
        wins = max(0, min(n, int(round(wins_rate * n))))
        parts.append(f"{wins}-{n - wins} last {n}")
    if net_rating is not None:
        parts.append(f"{net_rating:+.1f} net")
    if points is not None:
        parts.append(f"{points:.1f} pts")
    return " | ".join(parts) or None


def _format_availability_summary(row: Any, side: str) -> str | None:
    inactive_core = _safe_int(getattr(row, f"{side}_availability_likely_inactive_core_players", np.nan)) or 0
    absent_rotation = _safe_int(getattr(row, f"{side}_availability_likely_absent_rotation_players", np.nan)) or 0
    core_rating = _safe_float(getattr(row, f"{side}_availability_core_availability_rating", np.nan))
    parts: list[str] = []
    if inactive_core:
        parts.append(f"{inactive_core} core out")
    if absent_rotation:
        parts.append(f"{absent_rotation} rotation out")
    if core_rating is not None:
        parts.append(f"{core_rating * 100:.0f}% core ready")
    return " | ".join(parts) or "Rotation mostly intact"


def _format_lineup_continuity(row: Any, side: str) -> str | None:
    continuity = _safe_float(getattr(row, f"{side}_availability_expected_lineup_continuity_last_game", np.nan))
    starters = _safe_float(getattr(row, f"{side}_availability_expected_starter_continuity_last_game", np.nan))
    if continuity is not None:
        return f"{continuity * 100:.0f}% rotation overlap"
    if starters is not None:
        return f"{starters * 100:.0f}% starter overlap"
    return None


def _build_team_details(row: Any, side: str) -> dict[str, Any]:
    league = str(getattr(row, "league"))
    team_id = _safe_int(getattr(row, f"{side}_team_id"))
    return {
        "teamId": team_id,
        "abbreviation": _optional_text(getattr(row, f"{side}_tricode", None)),
        "logoUrl": _team_logo_url(league, team_id),
        # Same-season record before tip (from results, so it counts every game played).
        "recordPrior": _format_record(_safe_float(getattr(row, f"{side}_wins_before", np.nan)),
                                      _safe_float(getattr(row, f"{side}_games_before", np.nan))),
        "recentForm": _format_recent_form(row, side),
        "availabilitySummary": _format_availability_summary(row, side),
        "lineupContinuity": _format_lineup_continuity(row, side),
        "venue": _optional_text(getattr(row, "arena_name", None)),
    }


def _build_prediction_team_profile(row: Any, side: str) -> dict[str, Any]:
    wins = _safe_float(getattr(row, f"{side}_wins_before", np.nan))
    games = _safe_float(getattr(row, f"{side}_games_before", np.nan))
    win_pct = (wins / games) if wins is not None and games else None
    net = _safe_float(getattr(row, f"{side}_team_net_rating_est_avg_last_10", np.nan))
    core = _safe_float(getattr(row, f"{side}_availability_core_availability_rating", np.nan))
    # "--" when there is no same-season history yet (season openers), not a fake 0.
    return {
        "subtitle": _format_recent_form(row, side),
        "stats": [
            {"label": "Win %", "value": f"{win_pct * 100:.0f}%" if win_pct is not None else "--"},
            {"label": "Net L10", "value": f"{net:+.1f}" if net is not None else "--"},
            {"label": "Core Ready", "value": f"{core * 100:.0f}%" if core is not None else "--"},
        ],
    }


def _featured_player(lineup: list[dict[str, Any]]) -> dict[str, Any] | None:
    return lineup[0] if lineup else None


def _historical_player_radar(row: Any) -> list[dict[str, float]]:
    points = _safe_float(getattr(row, "points", None))
    assists = _safe_float(getattr(row, "assists", None))
    rebounds = _safe_float(getattr(row, "rebounds_total", None))
    steals = _safe_float(getattr(row, "steals", None))
    blocks = _safe_float(getattr(row, "blocks", None))
    minutes = _safe_float(getattr(row, "minutes", None))
    fg_pct = _safe_float(getattr(row, "field_goal_pct", None))
    defense = (steals or 0.0) + 1.35 * (blocks or 0.0)
    return [
        {"label": "Scoring", "value": float(max(0.0, min(100.0, ((points or 0.0) / 30.0) * 100.0)))},
        {"label": "Playmaking", "value": float(max(0.0, min(100.0, ((assists or 0.0) / 10.0) * 100.0)))},
        {"label": "Rebounding", "value": float(max(0.0, min(100.0, ((rebounds or 0.0) / 14.0) * 100.0)))},
        {"label": "Defense", "value": float(max(0.0, min(100.0, (defense / 4.0) * 100.0)))},
        {"label": "Shooting", "value": float(max(0.0, min(100.0, ((fg_pct or 0.0) * 100.0))))},
        {"label": "Minutes", "value": float(max(0.0, min(100.0, ((minutes or 0.0) / 38.0) * 100.0)))},
    ]


def _historical_lineup(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    side_frame = frame.copy()
    side_frame["starter_numeric"] = side_frame["starter"].fillna(False).astype(bool).astype(int)
    side_frame["minutes_numeric"] = pd.to_numeric(side_frame["minutes"], errors="coerce").fillna(0.0)
    side_frame["points_numeric"] = pd.to_numeric(side_frame["points"], errors="coerce").fillna(0.0)
    side_frame = side_frame.sort_values(["starter_numeric", "minutes_numeric", "points_numeric"], ascending=[False, False, False]).head(9)
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(side_frame.itertuples(index=False), start=1):
        field_goal_pct = _safe_float(getattr(row, "field_goal_pct", None))
        entries.append({
            "playerId": _safe_int(getattr(row, "player_id", None)),
            "playerName": getattr(row, "player_name"),
            "lineupSlot": index,
            "position": _optional_text(getattr(row, "position", None)),
            "performanceSummary": f"{int(_safe_float(getattr(row, 'points', 0)) or 0)} pts | {(_safe_float(getattr(row, 'minutes', 0)) or 0):.1f} min",
            "profile": {
                "imageUrl": _player_headshot_url(str(getattr(row, "league")), _safe_int(getattr(row, "player_id", None))),
                "subtitle": "Starter" if bool(getattr(row, "starter", False)) else "Rotation",
                "stats": [
                    {"label": "Pts", "value": str(int(_safe_float(getattr(row, "points", 0)) or 0))},
                    {"label": "FG%", "value": f"{field_goal_pct * 100:.1f}%" if field_goal_pct is not None else "--"},
                    {"label": "Ast", "value": str(int(_safe_float(getattr(row, "assists", 0)) or 0))},
                    {"label": "Reb", "value": str(int(_safe_float(getattr(row, "rebounds_total", 0)) or 0))},
                ],
            },
            "radarMetrics": _historical_player_radar(row),
        })
    return entries


# ── inputs: tables + live schedule + fresh boxscores ─────────────────────────

def _today_local() -> pd.Timestamp:
    """Bake date in US Eastern (games are scheduled on ET dates). BASKETBALL_EXPORT_TODAY
    (YYYY-MM-DD) overrides it for dry runs / tests."""
    override = os.getenv("BASKETBALL_EXPORT_TODAY")
    if override:
        return pd.Timestamp(override).normalize()
    return pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).normalize()


def _cached_schedule_payload(league: str) -> dict[str, Any] | None:
    pattern = f"schedule_{league}_{LEAGUE_CONFIGS[league].current_season}.json"
    for candidate in sorted((BASKETBALL_DATA_ROOT / "raw").glob(f"*/{pattern}"), reverse=True):
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _live_schedule(client: BasketballStatsClient, league: str) -> pd.DataFrame:
    try:
        payload = client.get_schedule(league)
    except Exception as exc:  # network / CDN block: fall back to the newest raw snapshot
        payload = _cached_schedule_payload(league)
        logger.warning("Live %s schedule unavailable (%s); %s", league.upper(), exc,
                       "using cached snapshot" if payload else "no snapshot")
        if payload is None:
            return pd.DataFrame()
    schedule = flatten_schedule(payload, league=league)
    if schedule.empty:
        return schedule
    schedule["game_id"] = schedule["game_id"].map(canonical_game_id)
    return schedule


def _refresh_boxscores(league: str, live: pd.DataFrame, tables: hm.LeagueTables) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Boxscores of completed live-schedule games the (possibly stale) tables lack."""
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if live.empty or os.getenv("BASKETBALL_BOXSCORE_REFRESH", "1") == "0":
        return empty
    have = set(tables.details["game_id"].dropna()) if "game_id" in tables.details.columns else set()
    kinds = live["game_id"].map(game_type_from_id)
    done = live.loc[(pd.to_numeric(live["status_code"], errors="coerce") == 3) & kinds.isin(COMPETITIVE_TYPES)]
    missing = [gid for gid in done.sort_values("official_date")["game_id"] if gid not in have]
    if not missing:
        return empty
    max_games = int(os.getenv("BASKETBALL_BOXSCORE_REFRESH_MAX", "1500"))
    budget = float(os.getenv("BASKETBALL_BOXSCORE_REFRESH_SECONDS", "900"))
    logger.info("basketball refresh: %d completed %s games missing from the tables", len(missing), league.upper())
    bundles = fetch_final_games(league, missing[-max_games:], max_workers=2, time_budget_seconds=budget)
    if not bundles:
        return empty
    sched, details, players = bundles_to_frames(bundles)
    if os.getenv("BASKETBALL_PERSIST_REFRESH", "1") != "0":
        try:
            append_to_tables(NORMALIZED_DIR, league, sched, details, players)
        except Exception:  # never fail the export over a cache write
            logger.exception("Could not persist refreshed %s boxscores", league.upper())
    return sched, details, players


@dataclass
class ExportContext:
    today: pd.Timestamp
    params: dict[str, Any]
    runs: dict[str, hm.LeagueRun] = field(default_factory=dict)
    live: dict[str, pd.DataFrame] = field(default_factory=dict)
    upcoming_ids: dict[str, set[str]] = field(default_factory=dict)
    tbd_tip_ids: set[str] = field(default_factory=set)   # date known, tip time not yet set
    players: dict[str, pd.DataFrame] = field(default_factory=dict)
    published_at: str = ""


def _upcoming_live_games(live: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    if live.empty:
        return live
    frame = live.copy()
    frame["kind"] = frame["game_id"].map(game_type_from_id)
    frame["tip_utc"] = pd.to_datetime(frame["game_date_time_utc"], errors="coerce", utc=True).dt.tz_localize(None)
    frame["local_date"] = local_dates(frame["tip_utc"])
    known = (pd.to_numeric(frame["home_team_id"], errors="coerce").fillna(0) > 0) & (
        pd.to_numeric(frame["away_team_id"], errors="coerce").fillna(0) > 0)
    window = (frame["local_date"] >= today) & (frame["local_date"] <= today + pd.Timedelta(days=UPCOMING_LOOKAHEAD_DAYS))
    # Regular season, play-in, playoffs and cup finals. Preseason and All-Star games
    # stay out: rested starters make a model trained on real games mislead.
    keep = known & window & frame["kind"].isin(COMPETITIVE_TYPES) & (pd.to_numeric(frame["status_code"], errors="coerce") != 3)
    return frame.loc[keep]


def tbd_tip_game_ids(live: pd.DataFrame) -> set[str]:
    """Games whose tip time is a placeholder: the CDN lists them as "TBD" with a
    midnight-Eastern (04:00/05:00 UTC) timestamp. Their date is right but the time is
    not, so boards publish no gameStart for them (the pick log then cuts off at the
    conservative noon-UTC fallback instead of a fake midnight tip)."""
    if live.empty:
        return set()
    status = live.get("status_text", pd.Series("", index=live.index)).fillna("").astype(str).str.strip().str.upper()
    tip = pd.to_datetime(live.get("game_date_time_utc"), errors="coerce", utc=True)
    local = tip.dt.tz_convert(LOCAL_TZ)
    midnight = (local.dt.hour == 0) & (local.dt.minute == 0)
    return set(live.loc[status.str.startswith("TBD") | midnight.fillna(False), "game_id"].astype(str))


def build_context(*, with_history: bool = True, today: pd.Timestamp | None = None) -> ExportContext:
    ctx = ExportContext(today=today if today is not None else _today_local(), params=hm.load_params())
    ctx.published_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    client = BasketballStatsClient()
    for league in LEAGUES:
        live = _live_schedule(client, league)
        ctx.live[league] = live
        tables = hm.load_league_tables(NORMALIZED_DIR, league, extra_schedules=[live] if not live.empty else None)
        sched, details, players = _refresh_boxscores(league, live, tables)
        if not details.empty:
            tables = hm.load_league_tables(NORMALIZED_DIR, league,
                                           extra_schedules=[s for s in (sched, live) if not s.empty],
                                           extra_details=[details], extra_players=[players])
        upcoming = _upcoming_live_games(live, ctx.today)
        ctx.upcoming_ids[league] = set(upcoming["game_id"])
        ctx.tbd_tip_ids |= tbd_tip_game_ids(upcoming)
        ctx.players[league] = tables.players
        try:
            run = hm.run_league(tables, ctx.params, today=ctx.today, upcoming_ids=ctx.upcoming_ids[league],
                                with_history=with_history)
        except Exception:
            logger.exception("Basketball %s model run failed", league.upper())
            run = None
        if run is not None:
            ctx.runs[league] = run
            logger.info("basketball %s: %d games, trained through %s, %d history, %d upcoming, timings %s", league.upper(),
                        int(run.design["final"].sum()), run.trained_through, len(run.history), len(run.upcoming),
                        {k: round(v, 1) for k, v in run.timings.items()})
    return ctx


# ── upcoming boards ───────────────────────────────────────────────────────────

def _board_rows(run: hm.LeagueRun, game_ids: set[str]) -> pd.DataFrame:
    """Design rows (results + pregame features) for the given games, with the column
    names the board formatters read."""
    design = run.design.loc[run.design["game_id"].isin(game_ids)].copy()
    feats = run.bundle.features
    extra_cols = [c for c in feats.columns if c.startswith(("home_availability_", "away_availability_"))
                  and c not in design.columns]
    if extra_cols:
        design = design.join(feats[extra_cols], on="game_id")
    for side in ("home", "away"):
        design[f"{side}_team_name"] = design[f"{side}_name"]
    return design


def _current_season_player_logs(player_logs: pd.DataFrame, season: str) -> pd.DataFrame:
    if player_logs.empty or "season_display" not in player_logs.columns:
        return player_logs
    same = player_logs.loc[player_logs["season_display"].astype(str) == str(season)]
    teams_with_games = set(same["team_key"])
    # Season openers have no same-season rows yet: fall back to the newest earlier season.
    earlier = player_logs.loc[(player_logs["season_display"].astype(str) < str(season)) & ~player_logs["team_key"].isin(teams_with_games)]
    if not earlier.empty:
        newest = earlier.groupby("team_key")["season_display"].transform("max")
        earlier = earlier.loc[earlier["season_display"] == newest]
    return pd.concat([same, earlier], ignore_index=True)


def _rotation_map(run: hm.LeagueRun, rows: pd.DataFrame) -> dict[str, dict[str, list[dict[str, Any]]]]:
    logs = run.bundle.player_logs
    if logs.empty or rows.empty:
        return {}
    games = pd.DataFrame({
        "game_id": rows["game_id"].values,
        "official_date": pd.to_datetime(rows["tip_utc"]).values,
        "home_team_key": rows["league"].astype(str).values + ":" + rows["home_team_id"].astype(str).values,
        "away_team_key": rows["league"].astype(str).values + ":" + rows["away_team_id"].astype(str).values,
    })
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for season, idx in rows.groupby("season").groups.items():
        sub = games.loc[games["game_id"].isin(rows.loc[idx, "game_id"])]
        try:
            out.update(build_projected_rotation_map(sub, _current_season_player_logs(logs, str(season))))
        except Exception:
            logger.exception("projected rotations failed for %s", season)
    return out


def _spread_label(tricode: str, line: float) -> str:
    return f"{tricode} {line:+.1f}"


def build_board_markets(board_id: str, row: Any, pred: Any, sigmas: dict[str, float], *, published_at: str,
                        version: str) -> list[dict[str, Any]]:
    """Model-only markets for one upcoming game (P1-12). No market line, no edge."""
    out: list[dict[str, Any]] = []
    home, away = str(getattr(row, "home_tricode") or "HOME"), str(getattr(row, "away_tricode") or "AWAY")

    def add(entry: dict[str, Any], suffix: str | None) -> None:
        if suffix:
            entry["marketId"] = f"{entry['marketId']}:{suffix}"
        entry["basis"] = "model"
        entry["modelVersion"] = version
        # No tier: tiers need per-market calibrated cut points we do not have yet
        # (plan review C7). Drop the default-cut tier the shared helper adds.
        entry.pop("confidenceTier", None)
        out.append(entry)

    p_home = _safe_float(getattr(pred, "p_home", None))
    if p_home is not None:
        side, p, team = ("home", p_home, home) if p_home >= 0.5 else ("away", 1.0 - p_home, away)
        prices = markets.LinePrices(win=p, half_win=0.0, push=0.0, half_loss=0.0, loss=1.0 - p)
        # One moneyline pick per game (no side suffix): if the favourite flips between
        # bakes, the last pregame snapshot is the one graded.
        add(markets.market_pick(board_id=board_id, market_type="moneyline", side=side, label=f"{team} to win",
                                prices=prices, line=None, model_line=None, published_at=published_at,
                                show_edge=False), None)
    margin = _safe_float(getattr(pred, "margin_pred", None))
    total = _safe_float(getattr(pred, "total_pred", None))
    if margin is None or total is None:
        return out
    s_m, s_t, s_tt = float(sigmas["margin"]), float(sigmas["total"]), float(sigmas["team_total"])
    # Fair spread: both sides at the model's x.5 line (no push), home view = -line.
    home_line = -markets.fair_line_from_mean(margin)
    for side, team, line, mean in (("home", home, home_line, margin), ("away", away, -home_line, -margin)):
        prices = markets.spread_prices_normal(mean, s_m, line, integer_scores=True)
        add(markets.market_pick(board_id=board_id, market_type="spread", side=side, label=_spread_label(team, line),
                                prices=prices, line=line, model_line=round(-mean, 1), published_at=published_at,
                                show_edge=False), side)
    total_line = markets.fair_line_from_mean(total)
    for side in ("over", "under"):
        prices = markets.total_prices_normal(total, s_t, total_line, side, integer_scores=True)
        add(markets.market_pick(board_id=board_id, market_type="total", side=side,
                                label=f"{side.title()} {total_line:.1f}", prices=prices, line=total_line,
                                model_line=round(total, 1), published_at=published_at, show_edge=False), side)
    for kind, team, mean in (("team_total_home", home, (total + margin) / 2.0), ("team_total_away", away, (total - margin) / 2.0)):
        line = markets.fair_line_from_mean(mean)
        for side in ("over", "under"):
            prices = markets.total_prices_normal(mean, s_tt, line, side, integer_scores=True)
            add(markets.market_pick(board_id=board_id, market_type=kind, side=side,
                                    label=f"{team} {side.title()} {line:.1f}", prices=prices, line=line,
                                    model_line=round(mean, 1), published_at=published_at, show_edge=False), side)
    return out


def _build_upcoming_boards(ctx: ExportContext, league: str, game_ids: set[str]) -> list[dict[str, Any]]:
    run = ctx.runs.get(league)
    if run is None or run.upcoming.empty:
        return []
    preds = run.upcoming.loc[run.upcoming["game_id"].isin(game_ids)].set_index("game_id")
    rows = _board_rows(run, set(preds.index))
    rotation_map = _rotation_map(run, rows)
    version = hm.model_version(ctx.params)
    trained = run.trained_through.strftime("%Y-%m-%d") if run.trained_through is not None else None
    boards: list[dict[str, Any]] = []
    for row in rows.sort_values(["tip_utc", "game_id"]).itertuples(index=False):
        game_id = str(row.game_id)
        pred = preds.loc[game_id]
        home_prob = float(pred["p_home"])
        away_prob = 1.0 - home_prob
        scheduled_date = _date_key_int(row.local_date)
        predicted_winner = row.home_name if home_prob >= away_prob else row.away_name
        predictions = sorted([
            {"rank": 0, "playerName": row.away_name, "winProbability": away_prob * 100.0, "side": "away",
             "profile": _build_prediction_team_profile(row, "away")},
            {"rank": 0, "playerName": row.home_name, "winProbability": home_prob * 100.0, "side": "home",
             "profile": _build_prediction_team_profile(row, "home")},
        ], key=lambda item: item["winProbability"], reverse=True)
        for index, item in enumerate(predictions, start=1):
            item["rank"] = index
        away_lineup = rotation_map.get(game_id, {}).get("away", [])
        home_lineup = rotation_map.get(game_id, {}).get("home", [])
        board_id = f"basketball-{league}-{game_id}"
        boards.append({
            "id": board_id,
            "gameId": game_id,
            "gameStart": None if game_id in ctx.tbd_tip_ids else _iso_utc(row.tip_utc),
            "name": f"{row.away_name} at {row.home_name}",
            "tour": _display_tour(league),
            "course": _optional_text(row.arena_name) or "Arena",
            "venue": _optional_text(row.arena_name) or "Arena",
            "scheduledDate": scheduled_date,
            "latestDate": scheduled_date,
            "predictedWinner": predicted_winner,
            "awayTeam": row.away_name,
            "homeTeam": row.home_name,
            "awayTeamDetails": _build_team_details(row, "away"),
            "homeTeamDetails": _build_team_details(row, "home"),
            "awayAvailability": {
                "ilAdds14": _safe_int(getattr(row, "away_availability_likely_inactive_core_players", np.nan)) or 0,
                "ilActivations14": _safe_int(getattr(row, "away_availability_expected_starters", np.nan)) or 0,
                "rosterMoves14": _safe_int(getattr(row, "away_availability_expected_rotation_players", np.nan)) or 0,
            },
            "homeAvailability": {
                "ilAdds14": _safe_int(getattr(row, "home_availability_likely_inactive_core_players", np.nan)) or 0,
                "ilActivations14": _safe_int(getattr(row, "home_availability_expected_starters", np.nan)) or 0,
                "rosterMoves14": _safe_int(getattr(row, "home_availability_expected_rotation_players", np.nan)) or 0,
            },
            "predictionSource": f"{league}_elo_l1_logistic",
            "basis": "model",
            "modelVersion": version,
            "modelTrainedThrough": trained,
            "gameType": str(row.kind),
            "awayLineup": away_lineup,
            "homeLineup": home_lineup,
            "awayFeaturedPlayer": _featured_player(away_lineup),
            "homeFeaturedPlayer": _featured_player(home_lineup),
            "predictions": predictions,
            "markets": build_board_markets(board_id, row, pred, run.sigmas, published_at=ctx.published_at,
                                           version=version),
        })
    return boards


def _available_dates(ctx: ExportContext) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for league in LEAGUES:
        run = ctx.runs.get(league)
        if run is None or run.upcoming.empty:
            continue
        rows = run.design.loc[run.design["game_id"].isin(set(run.upcoming["game_id"]))]
        for key in rows["local_date"].map(_date_key).dropna():
            counts[key] = counts.get(key, 0) + 1
    return [{"dateKey": k, "label": _date_label(k), "gameCount": int(v)} for k, v in sorted(counts.items())[:MAX_AVAILABLE_DATES]]


def build_upcoming(ctx: ExportContext, selected_date: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    available_dates = _available_dates(ctx)
    if not available_dates:
        return [], available_dates, None
    keys = [d["dateKey"] for d in available_dates]
    resolved = selected_date if selected_date in keys else keys[0]
    # Bake the next BASKETBALL_UPCOMING_BOARD_DAYS game-days (a single day empties as
    # soon as UTC rolls past it; the BFF derives availableDates from the boards).
    board_days = max(1, int(os.getenv("BASKETBALL_UPCOMING_BOARD_DAYS", "4")))
    start = keys.index(resolved)
    target = set(keys[start:start + board_days])
    boards: list[dict[str, Any]] = []
    for league in LEAGUES:
        run = ctx.runs.get(league)
        if run is None or run.upcoming.empty:
            continue
        rows = run.design.loc[run.design["game_id"].isin(set(run.upcoming["game_id"]))]
        ids = set(rows.loc[rows["local_date"].map(_date_key).isin(target), "game_id"])
        boards.extend(_build_upcoming_boards(ctx, league, ids))
    # Date first: a TBD tip has no gameStart and must not jump ahead of today's games.
    boards.sort(key=lambda b: (b.get("scheduledDate") or 0, b.get("gameStart") or "~", b["id"]))
    return boards, available_dates, resolved


def build_live_upcoming_payload(selected_date: str | None = None) -> dict[str, Any]:
    """Live API path (divination /api/sports/basketball/boards): upcoming boards only.
    History is the exported walk-forward file; regrading completed games here with the
    current model is exactly what produced the in-sample 72.6% figure. The walk-forward
    still runs (not returned): the served calibrator and the spread/total sigmas are
    fitted on its out-of-sample predictions, so live and baked boards agree."""
    ctx = build_context(with_history=True)
    boards, available_dates, resolved = build_upcoming(ctx, selected_date)
    return {
        "selectedDate": resolved,
        "availableDates": available_dates,
        "upcoming": boards,
        "completed": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "divination_live_basketball_feed",
    }


# ── honest history boards ─────────────────────────────────────────────────────

def _player_groups(players: pd.DataFrame, game_ids: set[str]) -> dict[tuple[str, str], pd.DataFrame]:
    if players is None or players.empty or not game_ids:
        return {}
    p = players.loc[players["game_id"].isin(game_ids)]
    return {(str(g), str(s)): frame for (g, s), frame in p.groupby(["game_id", "team_side"])}


def build_history_boards(ctx: ExportContext) -> list[dict[str, Any]]:
    version = hm.model_version(ctx.params)
    entries: list[tuple[Any, ...]] = []
    for league in LEAGUES:
        run = ctx.runs.get(league)
        if run is None or run.history.empty or "p_home" not in run.history.columns:
            continue
        hist = _published_history(run)
        rows = _board_rows(run, set(hist["game_id"])).set_index("game_id")
        for rec in hist.itertuples(index=False):
            entries.append((league, rec, rows.loc[rec.game_id]))
    entries.sort(key=lambda e: (e[1].local_date, e[1].game_id), reverse=True)
    lineup_ids = {str(e[1].game_id) for e in entries[:HISTORY_LINEUP_BOARDS]}
    groups: dict[tuple[str, str], pd.DataFrame] = {}
    for league in LEAGUES:
        groups.update(_player_groups(ctx.players.get(league, pd.DataFrame()), lineup_ids))
    boards: list[dict[str, Any]] = []
    for league, rec, row in entries:
        home_prob = float(rec.p_home)
        away_prob = 1.0 - home_prob
        home_name, away_name = str(row["home_name"]), str(row["away_name"])
        predicted = home_name if home_prob >= away_prob else away_name
        actual = home_name if float(rec.home_win) == 1.0 else away_name
        gid = str(rec.game_id)
        away_lineup = _historical_lineup(groups.get((gid, "away"))) if gid in lineup_ids else []
        home_lineup = _historical_lineup(groups.get((gid, "home"))) if gid in lineup_ids else []
        row_obj = pd.Series({**row.to_dict(), "league": league, "game_id": gid})
        row_ns = _RowView(row_obj)
        date_int = _date_key_int(rec.local_date)
        boards.append({
            "year": int(pd.Timestamp(rec.local_date).year),
            "tournament": f"{away_name} at {home_name}",
            "tournamentId": f"basketball-{league}-{gid}",
            "tour": _display_tour(league),
            # Walk-forward grade: the model that made this pick never saw this game.
            "hitStatus": "Top Pick" if predicted == actual else "Miss",
            "predictedWinner": predicted,
            "actualWinner": actual,
            "prob": max(home_prob, away_prob),
            "homeWinProbability": round(home_prob, 4),
            "awayWinProbability": round(away_prob, 4),
            # A replay, not a live log: "simulated" walk-forward record (plan review D3).
            # Live market picks carry their own record (basketball_market_history.json).
            "recordBasis": "simulated",
            "basis": "model",
            "venue": _optional_text(row.get("arena_name")) or "Arena",
            "course": _optional_text(row.get("arena_name")) or "Arena",
            "latestDate": date_int,
            "scheduledDate": date_int,
            "awayTeam": away_name,
            "homeTeam": home_name,
            "awayTeamDetails": _build_team_details(row_ns, "away"),
            "homeTeamDetails": _build_team_details(row_ns, "home"),
            "awayLineup": away_lineup,
            "homeLineup": home_lineup,
            "awayFeaturedPlayer": _featured_player(away_lineup),
            "homeFeaturedPlayer": _featured_player(home_lineup),
            "gameId": gid,
            "gameType": str(row.get("kind")),
            "homeScore": _safe_int(rec.home_score),
            "awayScore": _safe_int(rec.away_score),
            "predictionSource": f"{league}_elo_l1_logistic_walk_forward",
            "modelVersion": version,
            "modelTrainedThrough": pd.Timestamp(rec.train_cutoff).strftime("%Y-%m-%d"),
        })
    return boards


def _published_history(run: hm.LeagueRun) -> pd.DataFrame:
    """Walk-forward predictions users may see: published seasons only (warm-up seasons
    tuned the hyper-parameters and only seed the calibrator / sigma), and only games
    the model actually picked (a probability of exactly 0.5 is no pick)."""
    if run.history.empty or "p_home" not in run.history.columns:
        return pd.DataFrame()
    hist = run.history.dropna(subset=["p_home", "home_win"])
    if "published" in hist.columns:
        hist = hist.loc[hist["published"].astype(bool)]
    return hist.loc[(hist["p_home"] - 0.5).abs() > 1e-9]


class _RowView:
    """Attribute access over a Series (the board formatters use getattr)."""

    def __init__(self, series: pd.Series) -> None:
        self._s = series

    def __getattr__(self, name: str) -> Any:
        return self._s.get(name, np.nan)


def live_model_line_accuracy(graded: list[dict[str, Any]]) -> dict[str, Any]:
    """P1-12 grading for the LIVE log: mean absolute error of the published model
    line (spread / total / team totals) against the final score, per type and season,
    one row per game (both sides of a fair line carry the same model line). Win rates
    of both-sided fair lines are 50% by construction, so MAE is the honest metric."""
    seen: dict[tuple[str, str, str], float] = {}
    for g in graded:
        kind, line = str(g.get("type")), g.get("modelLine")
        if kind not in ("spread", "total", "team_total_home", "team_total_away") or line is None:
            continue
        h, a = float(g["homeScore"]), float(g["awayScore"])
        if kind == "spread":
            # modelLine is the picked side's handicap: -(expected margin of that side)
            actual = -(h - a) if g.get("side") == "home" else -(a - h)
        else:
            actual = {"total": h + a, "team_total_home": h, "team_total_away": a}[kind]
        seen[(kind, str(g.get("season")), str(g.get("gameId")))] = abs(float(line) - actual)
    out: dict[str, Any] = {}
    for (kind, season, _), err in seen.items():
        row = out.setdefault(kind, {}).setdefault(season, {"n": 0, "_sum": 0.0})
        row["n"] += 1
        row["_sum"] += err
    for kind in out.values():
        for row in kind.values():
            row["mae"] = round(row.pop("_sum") / row["n"], 2)
    return out


def build_model_record(ctx: ExportContext, graded: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Per league/season honest record with baselines, for the app's track-record copy."""
    metrics_path = DIV_ROOT / "sports" / "basketball" / "walk_forward_metrics.json"
    offline = {}
    if metrics_path.exists():
        try:
            offline = json.loads(metrics_path.read_text()).get("leagues", {})
        except Exception:
            offline = {}
    out: dict[str, Any] = {"modelVersion": hm.model_version(ctx.params), "generatedAt": ctx.published_at,
                           "recordBasis": "simulated",
                           "method": ("14-day walk-forward: each game graded with a model (and calibrator) fitted only on "
                                      "games from earlier dates. A probability of exactly 0.5 is no pick."),
                           "leagues": {}}
    for league, run in ctx.runs.items():
        hist = _published_history(run)
        seasons = {}
        for season, g in hist.groupby("season"):
            y = g["home_win"].to_numpy(float)
            model = hm.win_metrics(y, g["p_home"].to_numpy(float))
            elo = hm.win_metrics(y, g["p_elo"].to_numpy(float))
            offline_market = (((offline.get(league) or {}).get("winner") or {}).get(str(season)) or {}).get("market_matched")
            seasons[str(season)] = {
                "recordBasis": "simulated",
                "n": model.get("n"),
                "nPicks": model.get("n_picks"),
                "accuracy": round(model.get("accuracy", float("nan")), 4),
                "brier": round(model.get("brier", float("nan")), 4),
                "logLoss": round(model.get("log_loss", float("nan")), 4),
                "baselines": {
                    "alwaysHomeAccuracy": round(float(np.mean(y == 1)), 4),
                    "eloAccuracy": round(elo.get("accuracy", float("nan")), 4),
                    "eloBrier": round(elo.get("brier", float("nan")), 4),
                    "closingMarket": ({"n": offline_market["n"], "accuracy": offline_market["market"]["accuracy"],
                                       "brier": offline_market["market"]["brier"],
                                       "source": "closing moneyline, de-vigged (offline QA; aggregate only, no lines shown)"}
                                      if offline_market else None),
                },
                "firstGame": str(g["local_date"].min().date()),
                "lastGame": str(g["local_date"].max().date()),
            }
        out["leagues"][league] = {"trainedThrough": str(run.trained_through.date()) if run.trained_through is not None else None,
                                  "walkForwardLines": _walk_forward_line_accuracy(run),
                                  "sigmas": {k: round(v, 2) for k, v in run.sigmas.items()},
                                  "calibration": run.calibration, "seasons": seasons}
    out["liveModelLines"] = {"recordBasis": "live, logged at publish",
                             "byType": live_model_line_accuracy(graded or [])}
    return out


def _walk_forward_line_accuracy(run: hm.LeagueRun) -> dict[str, Any]:
    """Simulated (walk-forward) margin / total MAE and 80% interval coverage per season."""
    hist = run.history
    if hist.empty or "margin_pred" not in hist.columns:
        return {}
    hist = hist.loc[hist["published"].astype(bool)] if "published" in hist.columns else hist
    out: dict[str, Any] = {}
    for season, g in hist.dropna(subset=["margin_pred", "total_pred"]).groupby("season"):
        out[str(season)] = {
            "n": int(len(g)),
            "marginMae": round(float((g["margin"] - g["margin_pred"]).abs().mean()), 2),
            "totalMae": round(float((g["total"] - g["total_pred"]).abs().mean()), 2),
            "marginCoverage80": round(hm.interval_coverage(g["resid_margin"], g["sigma_margin"]).get("cov80", float("nan")), 3),
            "totalCoverage80": round(hm.interval_coverage(g["resid_total"], g["sigma_total"]).get("cov80", float("nan")), 3),
        }
    return out


# ── market pick log ───────────────────────────────────────────────────────────

def _market_root() -> Path:
    return Path(os.getenv("MARKET_PICKS_ROOT") or market_log.DEFAULT_ROOT)


def log_and_grade_markets(ctx: ExportContext, upcoming: list[dict[str, Any]]
                          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Append this bake's picks to the log, then regrade everything logged so far.
    Returns (all graded, newest-first history capped for the app, summary)."""
    root = _market_root()
    version = hm.model_version(ctx.params)
    records: list[dict[str, Any]] = []
    for league in LEAGUES:
        boards = [b for b in upcoming if b["id"].startswith(f"basketball-{league}-")]
        records += market_log.picks_from_boards(MARKET_SPORT_KEY, boards, season_of=SEASON_OF[league],
                                                model_version=version, published_at=ctx.published_at)
    market_log.write_snapshot(MARKET_SPORT_KEY, records, root=root)
    chosen = market_log.pregame_picks(market_log.load_snapshots(MARKET_SPORT_KEY, root=root))
    results: dict[str, tuple[float, float]] = {}
    for run in ctx.runs.values():
        finals = run.design.loc[run.design["final"]]
        results.update({str(g): (float(h), float(a)) for g, h, a in zip(finals["game_id"], finals["home_score"], finals["away_score"])})
    graded = market_log.grade_picks(chosen, results)
    graded.sort(key=lambda g: (str(g.get("gameDate") or ""), str(g.get("gameStart") or ""), str(g.get("marketId"))),
                reverse=True)
    return graded, graded[:MARKET_HISTORY_CAP], market_log.summarize(graded)


# ── main ──────────────────────────────────────────────────────────────────────

def _write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")) if compact else json.dumps(payload, indent=2))
    os.replace(tmp, path)


def export_basketball_frontend_data() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ctx = build_context(with_history=True)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    backtests = build_history_boards(ctx)
    if backtests:
        _write_json(FRONTEND_DATA_DIR / "basketball_historical_backtests.json", backtests, compact=True)
    upcoming, _, _ = build_upcoming(ctx)
    _write_json(FRONTEND_DATA_DIR / "basketball_upcoming_tournaments.json", upcoming)
    graded: list[dict[str, Any]] = []
    try:
        graded, history, summary = log_and_grade_markets(ctx, upcoming)
        _write_json(FRONTEND_DATA_DIR / f"{MARKET_SPORT_KEY}_market_history.json", history)
        _write_json(FRONTEND_DATA_DIR / f"{MARKET_SPORT_KEY}_market_summary.json", summary)
    except Exception:
        logger.exception("Basketball market log / grading failed")
    _write_json(FRONTEND_DATA_DIR / "basketball_model_record.json", build_model_record(ctx, graded))
    print(f"Exported {len(backtests)} Basketball historical boards (walk-forward)")
    print(f"Exported {len(upcoming)} Basketball upcoming boards")


if __name__ == "__main__":
    export_basketball_frontend_data()
