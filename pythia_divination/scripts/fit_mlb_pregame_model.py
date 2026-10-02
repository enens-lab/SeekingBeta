"""Fit + evaluate the honest pregame MLB model and write sports/mlb/model_params.json.

    cd pythia_divination && python scripts/fit_mlb_pregame_model.py [--market-csv bf_mlb_2025.csv]

What it does (all from pregame information; see sports/mlb/pregame_model.py):
  1. Loads every game 2020 -> today (live Stats API, local tables as fallback) and the
     starter pitching lines (local + live top-up).
  2. Measures the league constants used as shrinkage targets on the 2020-2023 regular
     seasons.
  3. Runs the MONTHLY WALK-FORWARD from 2024: each month, both models are refit on all
     games before the month and predict every game in it -- exactly what the exporter
     serves -- and scores them against always-home, Elo and constant baselines.
  4. Optionally compares with a de-vigged closing moneyline file (2025 ESPN BET lines).
  5. Stores the constants, the fit the exporter will use this month, and the evaluation
     (with the plan's P1-1/P1-11 acceptance checks) in model_params.json.

The exporter refits at run time with the same procedure, so this file documents the
served model and its evidence; it is not a gitignored artifact that must be uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports.mlb import pregame_model as pm  # noqa: E402
from sports.mlb.client import MLBStatsClient  # noqa: E402

ESPN_ABBR = {"ARI": 109, "AZ": 109, "ATH": 133, "OAK": 133, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CHW": 145,
             "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119,
             "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "PHI": 143, "PIT": 134, "SD": 135, "SEA": 136,
             "SF": 137, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120}


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def _binary(y: np.ndarray, p: np.ndarray) -> dict:
    return {"n": int(len(y)), "accuracy": round(float(((p >= 0.5) == (y == 1)).mean()), 4),
            "brier": round(_brier(y, p), 5), "logloss": round(_logloss(y, p), 5), "auc": round(_auc(y, p), 4)}


def _quintile_gap(p: np.ndarray, y: np.ndarray) -> tuple[float, list]:
    q = pd.qcut(p, 5, labels=False, duplicates="drop")
    cal = pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return round(float((cal["pred"] - cal["actual"]).abs().max()), 4), cal.round(4).reset_index().to_dict(orient="records")


def _rate_line(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    gap, cal = _quintile_gap(p, y)
    return {"brier": round(_brier(y, p), 5), "constant_brier": round(_brier(y, np.full(len(y), base_rate)), 5),
            "constant_rate_from_training": round(float(base_rate), 4), "mean_predicted": round(float(p.mean()), 4),
            "actual_rate": round(float(y.mean()), 4), "max_quintile_gap": gap, "quintiles": cal}


def league_constants(games: pd.DataFrame, lines: pd.DataFrame, seasons: range) -> dict:
    lg = games[games["season"].isin(list(seasons)) & (games["game_type"] == "R") & games["is_final"]]
    ln = lines.merge(lg[["game_pk"]], on="game_pk")
    return {
        "fip_num_per_ip": round(float((13 * ln["hr"] + 3 * ln["bb"] - 2 * ln["so"]).sum() / ln["ip"].sum()), 4),
        "kbb_per_bf": round(float((ln["so"] - ln["bb"]).sum() / ln["bf"].sum()), 4),
        "ip_per_start": round(float(ln["ip"].mean()), 3),
        "runs_per_team_game": round(float((lg["home_score"].sum() + lg["away_score"].sum()) / (2 * len(lg))), 4),
        "total_runs_per_game": round(float((lg["home_score"] + lg["away_score"]).mean()), 4),
        "measured_on": f"{seasons.start}-{seasons.stop - 1} regular seasons (n={len(lg)} games)",
    }


def evaluate(F: pd.DataFrame, pred: pd.DataFrame, seasons: list[int]) -> dict:
    D = F.merge(pred, on="game_pk")
    D = D[D["is_final"] & D["home_win"].notna()]
    out = {}
    for season in seasons:
        for scope, mask in (("regular_season", D["game_type"] == "R"), ("all_games", D["game_type"].notna())):
            d = D[(D["season"] == season) & mask]
            if d.empty:
                continue
            train = F[F["is_final"] & F["home_win"].notna() & (F["season"] < season)]
            y = d["home_win"].to_numpy(float)
            res = {
                "games": int(len(d)), "from": str(d["official_date"].min().date()), "through": str(d["official_date"].max().date()),
                "model": _binary(y, d["p_home"].to_numpy(float)),
                "elo": _binary(y, d["elo_p"].to_numpy(float)),
                "always_home": {**_binary(y, np.full(len(y), train["home_win"].mean())),
                                "accuracy": round(float(y.mean()), 4)},
            }
            ph, pa, po, pth, pta = [], [], [], [], []
            for mh, ma, r in zip(d["mu_home"], d["mu_away"], d["nb_r"]):
                m = pm.score_matrix(mh, ma, r)
                i, j = np.indices(m.shape)
                ph.append(m[i - j >= 2].sum()); pa.append(m[j - i >= 2].sum()); po.append(m[(i + j) > 8.5].sum())
                pth.append(m[i > 4.5].sum()); pta.append(m[j > 4.5].sum())
            ph, pa, po, pth, pta = map(np.asarray, (ph, pa, po, pth, pta))
            margin = (d["home_score"] - d["away_score"]).to_numpy(float)
            total = (d["home_score"] + d["away_score"]).to_numpy(float)
            tm = (train["home_score"] - train["away_score"]).to_numpy(float)
            tt = (train["home_score"] + train["away_score"]).to_numpy(float)
            fav_home = d["p_home"].to_numpy(float) >= 0.5
            y_h2, y_a2 = (margin >= 2).astype(float), (margin <= -2).astype(float)
            train_fav_home = train["elo_p"].to_numpy(float) >= 0.5  # no walk-forward model for training rows; Elo picks the favourite
            fav_base = float(np.where(train_fav_home, tm >= 2, tm <= -2).mean())
            res["run_line"] = {
                "home_minus_1_5": _rate_line(y_h2, ph, float((tm >= 2).mean())),
                "away_minus_1_5": _rate_line(y_a2, pa, float((tm <= -2).mean())),
                "favorite_minus_1_5": _rate_line(np.where(fav_home, y_h2, y_a2), np.where(fav_home, ph, pa), fav_base),
            }
            res["total_over_8_5"] = _rate_line((total > 8.5).astype(float), po, float((tt > 8.5).mean()))
            th = train["home_score"].to_numpy(float); ta = train["away_score"].to_numpy(float)
            res["team_total_over_4_5"] = {
                "home": _rate_line((d["home_score"].to_numpy(float) > 4.5).astype(float), pth, float((th > 4.5).mean())),
                "away": _rate_line((d["away_score"].to_numpy(float) > 4.5).astype(float), pta, float((ta > 4.5).mean())),
            }
            for block in (res["run_line"].values(), [res["total_over_8_5"]], res["team_total_over_4_5"].values()):
                for item in block:
                    item.pop("quintiles", None) if scope == "all_games" else None
            out.setdefault(str(season), {})[scope] = res
    return out


def acceptance(ev: dict) -> dict:
    checks = {}
    for season in ("2024", "2025"):
        r = ev.get(season, {}).get("regular_season")
        if not r:
            continue
        checks[f"{season}_brier_le_0.2430"] = r["model"]["brier"] <= 0.2430
        checks[f"{season}_brier_le_elo"] = r["model"]["brier"] <= r["elo"]["brier"]
        rl = r["run_line"]["home_minus_1_5"]
        checks[f"{season}_home_minus_1_5_brier_below_constant"] = rl["brier"] < rl["constant_brier"]
        checks[f"{season}_home_minus_1_5_quintile_gap_le_0.05"] = rl["max_quintile_gap"] <= 0.05
        fl = r["run_line"]["favorite_minus_1_5"]
        checks[f"{season}_favorite_minus_1_5_brier_below_constant"] = fl["brier"] < fl["constant_brier"]
        checks[f"{season}_favorite_minus_1_5_quintile_gap_le_0.05"] = fl["max_quintile_gap"] <= 0.05
    return checks


def market_comparison(F: pd.DataFrame, pred: pd.DataFrame, csv_path: Path) -> dict:
    """De-vigged closing moneyline vs the walk-forward model on the same games."""
    b = pd.read_csv(csv_path)
    b = b[b["season_type"] == 2].copy()
    b["home_team_id"] = b["home"].map(ESPN_ABBR); b["away_team_id"] = b["away"].map(ESPN_ABBR)
    b["edt"] = pd.to_datetime(b["date"], utc=True).dt.tz_localize(None)
    D = F.merge(pred, on="game_pk")
    D = D[D["is_final"] & D["home_win"].notna() & (D["game_type"] == "R")]
    m = b.merge(D[["game_pk", "home_team_id", "away_team_id", "home_score", "away_score", "dt", "home_win", "p_home", "elo_p"]],
                on=["home_team_id", "away_team_id"], suffixes=("_e", ""))
    m = m[(m["edt"] - m["dt"]).abs() <= pd.Timedelta(hours=6)]
    m = m.sort_values("game_pk").drop_duplicates("event_id").drop_duplicates("game_pk")
    m = m[(m["home_score_e"] == m["home_score"]) & (m["away_score_e"] == m["away_score"])]

    def implied(ml: pd.Series) -> np.ndarray:
        ml = ml.astype(float).to_numpy()
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))

    ih, ia = implied(m["p0_home_ml"]), implied(m["p0_away_ml"])
    m["market_p_home"] = ih / (ih + ia)
    pickem = (m["p0_home_ml"] == m["p0_away_ml"]).to_numpy()
    y = m["home_win"].to_numpy(float)
    fav_home = m["market_p_home"].to_numpy() > 0.5
    fav_won = np.where(fav_home, y == 1, y == 0)[~pickem]
    model_p = m["p_home"].to_numpy(float)
    return {
        "season": int(m["dt"].dt.year.mode().iloc[0]),
        "source": f"{csv_path.name}: ESPN BET closing moneyline (ESPN public odds API), de-vigged proportionally; "
                  "regular season, matched by teams + start time + final score",
        "games_matched": int(len(m)),
        "favoriteWinRate": round(float(fav_won.mean()), 4),
        "favoriteGames": int((~pickem).sum()),
        "pickemGamesExcluded": int(pickem.sum()),
        "marketBrier": round(_brier(y, m["market_p_home"].to_numpy()), 5),
        "marketLogloss": round(_logloss(y, m["market_p_home"].to_numpy()), 5),
        "modelSameGames": {"accuracy": round(float(((model_p >= 0.5) == (y == 1)).mean()), 4),
                           "brier": round(_brier(y, model_p), 5), "logloss": round(_logloss(y, model_p), 5)},
        "eloSameGames": {"accuracy": round(float(((m["elo_p"] >= 0.5) == (y == 1)).mean()), 4),
                         "brier": round(_brier(y, m["elo_p"].to_numpy(float)), 5)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history-start", default="2024-01-01")
    ap.add_argument("--league-seasons", default="2020-2023")
    ap.add_argument("--market-csv", default=None, help="Optional ESPN odds export (bf_mlb_2025.csv) for the market baseline")
    ap.add_argument("--no-network", action="store_true", help="Use only the local normalized tables")
    ap.add_argument("--output", default=str(pm.PARAMS_PATH))
    args = ap.parse_args()

    params = pm.load_params()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    client = None if args.no_network else MLBStatsClient()
    games = pm.load_results(range(pm.FIRST_SEASON, today.year + 1), client=client, allow_network=not args.no_network)
    lines = pm.assemble_starter_lines(games, client=client, allow_network=not args.no_network)
    a, b = (int(x) for x in args.league_seasons.split("-"))
    params["league"] = league_constants(games, lines, range(a, b + 1))
    F = pm.build_features(games, lines, params)

    start = pd.Timestamp(args.history_start)
    pred, _ = pm.walk_forward(F, params, start=start, end=today - pd.Timedelta(days=1))
    seasons = sorted(int(s) for s in F.loc[F["official_date"] >= start, "season"].unique())
    ev = evaluate(F, pred, seasons)

    cutoff = today.to_period("M").to_timestamp()
    train = F[F["is_final"] & (F["official_date"] < cutoff)]
    win = pm.fit_win_model(train, params)
    runs = pm.fit_runs_model(pm.team_game_frame(train), params)

    stored = json.loads(Path(args.output).read_text()) if Path(args.output).exists() else {}
    baselines = stored.get("market_baselines", [])
    if args.market_csv:
        mc = market_comparison(F, pred, Path(args.market_csv))
        baselines = [x for x in baselines if x.get("season") != mc["season"]] + [mc]

    out = {
        "version": params["version"],
        "description": "Honest pregame MLB model (sports/mlb/pregame_model.py). Win: logistic regression on Elo + "
                       "decayed starter FIP/K-BB%/IP + park/weather. Runs: Poisson GLM + negative-binomial dispersion. "
                       "Refit monthly on all games before the month; evaluated walk-forward.",
        "elo": params["elo"], "decay": params["decay"], "league": params["league"],
        "win_model": params["win_model"], "runs_model": params["runs_model"],
        "fitted": {"train_cutoff": cutoff.strftime("%Y-%m-%d"), "train_games": int(len(train)),
                   "win": win.to_dict(), "runs": runs.to_dict()},
        "evaluation": {
            "method": "monthly walk-forward from " + start.strftime("%Y-%m-%d") + "; refit at each month start on every "
                      "completed game before it (2020 onward, all game types); regular_season scope is comparable "
                      "with the plan's figures",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_through": str(F.loc[F["is_final"], "official_date"].max().date()),
            "seasons": ev,
            "acceptance": acceptance(ev),
        },
        "market_baselines": baselines,
    }
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"acceptance": out["evaluation"]["acceptance"],
                      "seasons": {s: {k: v["model"] for k, v in d.items()} for s, d in ev.items()},
                      "market": baselines}, indent=1))


if __name__ == "__main__":
    main()
