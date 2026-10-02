"""Canonical NBA/WNBA results table: one row per competitive game, from any source.

Everything the honest model computes from RESULTS (Elo, rest, opponent-adjusted
ratings, labels) reads this table, so it can be rebuilt at export time from data
the RunPod worker can always get: the S3-synced normalized tables (older seasons,
possibly months stale) plus the live CDN schedule (the whole current season with
scores) plus any boxscores fetched during the export. Sources are merged by the
canonical 10-digit game id; a FINAL row beats a scheduled one and later sources
win ties.

Dates: ``tip_utc`` is the naive-UTC tip time; ``local_date`` is the US-Eastern
calendar date of the tip, which is what rest / back-to-back mean (a 7:30 pm
Pacific tip is 02:30 UTC the next day; using the UTC date there mislabels every
West-coast back-to-back). The repo's old ``days_rest`` diffed UTC timestamps of
the PREVIOUS game's row and matched true rest 34.4% of the time.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from sports.basketball.backfill_history import COMPETITIVE_TYPES, canonical_game_id, game_type_from_id
from sports.basketball.constants import normalize_season_label

LOCAL_TZ = "America/New_York"
POSTSEASON_TYPES = ("play_in", "playoff")

RESULT_COLUMNS = [
    "league", "season", "game_id", "kind", "tip_utc", "local_date", "final",
    "home_team_id", "away_team_id", "home_tricode", "away_tricode", "home_name", "away_name",
    "home_city", "away_city", "home_score", "away_score", "neutral", "arena_name", "series_text",
]


def season_from_game_id(league: str, game_id: str) -> str | None:
    gid = canonical_game_id(game_id)
    if gid is None or not gid[3:5].isdigit():
        return None
    return normalize_season_label(league, 2000 + int(gid[3:5]))


def _to_utc_naive(values: pd.Series) -> pd.Series:
    out = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    return out.dt.tz_localize(None)


def local_dates(tip_utc: pd.Series) -> pd.Series:
    ts = pd.to_datetime(tip_utc, errors="coerce")
    return ts.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()


def _normalize_source(frame: pd.DataFrame, league: str, priority: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS + ["_priority"])
    f = frame.copy()
    if "league" in f.columns:
        f = f.loc[f["league"].astype(str).str.lower() == league]
    if f.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS + ["_priority"])

    def col(name: str) -> pd.Series:
        return f[name] if name in f.columns else pd.Series(np.nan, index=f.index, dtype=object)

    out = pd.DataFrame(index=f.index)
    out["league"] = league
    out["game_id"] = f["game_id"].map(canonical_game_id)
    out["kind"] = out["game_id"].map(game_type_from_id)
    out["season"] = out["game_id"].map(lambda g: season_from_game_id(league, g))
    tip = _to_utc_naive(col("game_date_time_utc"))
    if "official_date" in f.columns:
        tip = tip.fillna(pd.to_datetime(f["official_date"], errors="coerce", format="mixed"))
    out["tip_utc"] = tip
    status = pd.to_numeric(col("status_code"), errors="coerce")
    hs = pd.to_numeric(col("home_score"), errors="coerce")
    as_ = pd.to_numeric(col("away_score"), errors="coerce")
    final = (status == 3) & hs.notna() & as_.notna() & (hs != as_) & ((hs + as_) > 0)
    out["final"] = final.fillna(False).astype(bool)
    out["home_score"] = hs.where(out["final"])
    out["away_score"] = as_.where(out["final"])
    for side in ("home", "away"):
        out[f"{side}_team_id"] = pd.to_numeric(col(f"{side}_team_id"), errors="coerce")
        out[f"{side}_tricode"] = col(f"{side}_team_tricode")
        out[f"{side}_name"] = col(f"{side}_team_name")
        out[f"{side}_city"] = col(f"{side}_team_city")
    out["neutral"] = col("is_neutral").astype(str).str.lower().isin(["true", "1", "1.0"])
    out["arena_name"] = col("arena_name")
    out["series_text"] = col("series_text")
    out["_priority"] = priority
    return out


def build_results(league: str, sources: Iterable[pd.DataFrame], *, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Merge schedule-shaped frames (lowest priority first) into one row per game.

    Keeps competitive games (regular season, play-in, playoffs, cup finals) whose two
    teams are known. A non-final game dated before `today` (postponed / abandoned /
    stale status) is dropped: it neither has a result nor is upcoming."""
    parts = [_normalize_source(frame, league, priority) for priority, frame in enumerate(sources)]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    allrows = pd.concat(parts, ignore_index=True)
    allrows = allrows.loc[allrows["game_id"].notna() & allrows["kind"].isin(COMPETITIVE_TYPES)]
    allrows = allrows.loc[(allrows["home_team_id"].fillna(0) > 0) & (allrows["away_team_id"].fillna(0) > 0)]
    allrows = allrows.loc[allrows["tip_utc"].notna()]
    # final rows first, then the latest source
    allrows = allrows.sort_values(["game_id", "final", "_priority"], ascending=[True, False, False])
    res = allrows.drop_duplicates("game_id", keep="first").drop(columns=["_priority"])
    res["home_team_id"] = res["home_team_id"].astype("int64")
    res["away_team_id"] = res["away_team_id"].astype("int64")
    res["local_date"] = local_dates(res["tip_utc"])
    if today is not None:
        today_local = pd.Timestamp(today).normalize()
        res = res.loc[res["final"] | (res["local_date"] >= today_local)]
    res = res.sort_values(["tip_utc", "game_id"]).reset_index(drop=True)
    return res[RESULT_COLUMNS]


def add_rest_features(results: pd.DataFrame, *, cap: int = 4) -> pd.DataFrame:
    """Days since each team's previous game of the SAME season (local dates), capped.

    rest 1 = back-to-back. A season opener has no previous game and gets the cap
    (fully rested), as does any rest above it -- the old feature reached 928 days
    across the 2021-22 -> 2024-25 data gap. Scheduled (not yet played) games count
    as previous games, so a Tuesday/Wednesday pair two days ahead is still a
    back-to-back."""
    out = results.copy()
    long = pd.concat([
        out[["game_id", "season", "local_date", "home_team_id"]].rename(columns={"home_team_id": "team_id"}).assign(side="home"),
        out[["game_id", "season", "local_date", "away_team_id"]].rename(columns={"away_team_id": "team_id"}).assign(side="away"),
    ], ignore_index=True).sort_values(["team_id", "season", "local_date", "game_id"])
    long["prev_date"] = long.groupby(["team_id", "season"])["local_date"].shift(1)
    long["rest"] = (long["local_date"] - long["prev_date"]).dt.days
    for side in ("home", "away"):
        part = long.loc[long["side"] == side, ["game_id", "rest"]].set_index("game_id")["rest"]
        raw = out["game_id"].map(part)
        out[f"rest_{side}_days"] = raw
        out[f"rest_{side}_c"] = raw.clip(upper=cap).fillna(cap).clip(lower=0)
        out[f"b2b_{side}"] = (raw == 1).astype(float)
    out["rest_diff"] = out["rest_home_c"] - out["rest_away_c"]
    out["b2b_diff"] = out["b2b_home"] - out["b2b_away"]
    out["is_postseason"] = out["kind"].isin(POSTSEASON_TYPES).astype(float)
    return out


def season_record_before(results: pd.DataFrame) -> pd.DataFrame:
    """Wins/losses of each team in the same season before each game (finals only)."""
    out = results[["game_id"]].copy()
    long = pd.concat([
        results[["game_id", "season", "tip_utc", "home_team_id", "final"]].rename(columns={"home_team_id": "team_id"})
        .assign(side="home", won=np.where(results["final"], (results["home_score"] > results["away_score"]).astype(float), np.nan)),
        results[["game_id", "season", "tip_utc", "away_team_id", "final"]].rename(columns={"away_team_id": "team_id"})
        .assign(side="away", won=np.where(results["final"], (results["away_score"] > results["home_score"]).astype(float), np.nan)),
    ], ignore_index=True).sort_values(["team_id", "season", "tip_utc", "game_id"])
    long["w"] = long["won"].fillna(0.0)
    long["g"] = long["won"].notna().astype(float)
    grp = long.groupby(["team_id", "season"])
    long["wins_before"] = grp["w"].cumsum() - long["w"]
    long["games_before"] = grp["g"].cumsum() - long["g"]
    for side in ("home", "away"):
        part = long.loc[long["side"] == side].set_index("game_id")
        out[f"{side}_wins_before"] = out["game_id"].map(part["wins_before"])
        out[f"{side}_games_before"] = out["game_id"].map(part["games_before"])
    return out
