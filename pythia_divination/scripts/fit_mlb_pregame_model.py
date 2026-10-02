"""Fit + evaluate the honest pregame MLB model and write sports/mlb/model_params.json.

    cd pythia_divination && python scripts/fit_mlb_pregame_model.py [--market-csv bf_mlb_2025.csv]

What it does (all from pregame information; see sports/mlb/pregame_model.py):
  1. Loads every game 2020 -> today (live Stats API, local tables as fallback) and the
     starter pitching lines (local + live top-up).
  2. Measures the league constants used as shrinkage targets on the 2020-2023 regular
     seasons.
  3. Runs the MONTHLY WALK-FORWARD from 2024: each month, both models are refit on all
     games before the month and predict every game in it -- exactly what the exporter
     serves -- and scores them against always-home, Elo and constant baselines. The run
     line / totals are also scored with a SEASON walk-forward (fit on all seasons before,
     predict the whole season), the protocol of the plan's P1-11 acceptance.
  4. Optionally compares with a 2025 closing-line file (ESPN BET, de-vigged): moneyline
     favourite rate and Brier, posted-total over/under, and a market-anchored run-line
     baseline (logistic on the de-vigged moneyline + posted total, fit Apr-Jun, scored
     Jul-Sep) next to the NB model on the same games, with paired bootstrap CIs.
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
CALIBRATION_GAP_LIMIT = 0.05


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


def _rate_line(y: np.ndarray, p: np.ndarray, base_rate: float, *, quintiles: bool = True) -> dict:
    y = np.asarray(y, float); p = np.asarray(p, float)
    gap, cal = _quintile_gap(p, y)
    out = {"n": int(len(y)), "brier": round(_brier(y, p), 5), "constant_brier": round(_brier(y, np.full(len(y), base_rate)), 5),
           "constant_rate_from_training": round(float(base_rate), 4), "mean_predicted": round(float(p.mean()), 4),
           "actual_rate": round(float(y.mean()), 4), "max_quintile_gap": gap,
           # one quintile holds ~n/5 games: a perfectly calibrated forecast still misses by ~2 SE now and then
           "quintile_se": round(float(np.sqrt(max(y.mean() * (1 - y.mean()), 1e-9) / max(len(y) / 5, 1))), 4)}
    if quintiles:
        out["quintiles"] = cal
    return out


def _paired_ci(loss_a: np.ndarray, loss_b: np.ndarray, *, draws: int = 2000, seed: int = 0) -> dict:
    """Mean of (a - b) with a 95% paired bootstrap CI (negative = a better)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(draws)]
    return {"mean": round(float(d.mean()), 5), "ci95": [round(float(np.percentile(boots, 2.5)), 5),
                                                        round(float(np.percentile(boots, 97.5)), 5)]}


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


def market_blocks(d: pd.DataFrame, train: pd.DataFrame, *, quintiles: bool = True) -> dict:
    """Run line / total / team total calibration of the walk-forward runs model on the
    completed games `d`, against constant base rates measured on `train`."""
    pr = pm.batch_probabilities(d["mu_home"], d["mu_away"], d["nb_r"], total_lines=(8.5,), team_lines=(4.5,))
    ph, pa = pr["home_by_2"], pr["away_by_2"]
    hs, as_ = d["home_score"].to_numpy(float), d["away_score"].to_numpy(float)
    margin, total = hs - as_, hs + as_
    tm = (train["home_score"] - train["away_score"]).to_numpy(float)
    tt = (train["home_score"] + train["away_score"]).to_numpy(float)
    fav_home = d["p_home"].to_numpy(float) >= 0.5
    y_h2, y_a2 = (margin >= 2).astype(float), (margin <= -2).astype(float)
    # Training rows have no walk-forward prediction; Elo names their favourite.
    train_fav_home = train["elo_p"].to_numpy(float) >= 0.5
    fav_base = float(np.where(train_fav_home, tm >= 2, tm <= -2).mean())
    dog_base = float(np.where(train_fav_home, tm <= -2, tm >= 2).mean())
    home_base, away_base = float((tm >= 2).mean()), float((tm <= -2).mean())
    rl = {
        "home_minus_1_5": _rate_line(y_h2, ph, home_base, quintiles=quintiles),
        "away_minus_1_5": _rate_line(y_a2, pa, away_base, quintiles=quintiles),
        "favorite_minus_1_5": _rate_line(np.where(fav_home, y_h2, y_a2), np.where(fav_home, ph, pa), fav_base,
                                         quintiles=quintiles),
        "underdog_minus_1_5": _rate_line(np.where(fav_home, y_a2, y_h2), np.where(fav_home, pa, ph), dog_base,
                                         quintiles=quintiles),
        # what the boards serve: both -1.5 lines on every game
        "served_both_sides": _rate_line(np.r_[y_h2, y_a2], np.r_[ph, pa], (home_base + away_base) / 2,
                                        quintiles=quintiles),
    }
    th, ta = train["home_score"].to_numpy(float), train["away_score"].to_numpy(float)
    y_over_fair = (total > pr["total_fair_line"]).astype(float)
    return {
        "run_line": rl,
        "total_over_8_5": _rate_line((total > 8.5).astype(float), pr["over_8.5"], float((tt > 8.5).mean()),
                                     quintiles=quintiles),
        # the served total: Over at each game's fair x.5 line (a coin flip by construction)
        "total_over_fair_line": {**_rate_line(y_over_fair, pr["over_fair"], 0.5, quintiles=quintiles),
                                 "mean_line": round(float(pr["total_fair_line"].mean()), 3)},
        "team_total_over_4_5": {
            "home": _rate_line((hs > 4.5).astype(float), pr["home_over_4.5"], float((th > 4.5).mean()), quintiles=quintiles),
            "away": _rate_line((as_ > 4.5).astype(float), pr["away_over_4.5"], float((ta > 4.5).mean()), quintiles=quintiles),
        },
        "team_total_over_fair_line": {
            side: _rate_line((score > pr[f"{side}_fair_line"]).astype(float), pr[f"{side}_over_fair"], 0.5,
                             quintiles=quintiles)
            for side, score in (("home", hs), ("away", as_))
        },
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
            res.update(market_blocks(d, train, quintiles=scope == "regular_season"))
            out.setdefault(str(season), {})[scope] = res
    return out


def _check_rate(block: dict) -> tuple[bool, bool]:
    return block["brier"] < block["constant_brier"], block["max_quintile_gap"] <= CALIBRATION_GAP_LIMIT


def acceptance(ev: dict, ev_season: dict) -> dict:
    """Plan P1-1 (monthly walk-forward win model) and P1-11 (run line: season
    walk-forward home -1.5, the plan's protocol; plus what the boards actually serve,
    scored monthly). Fixed home/away slices of the monthly run are reported under
    `notes` for transparency."""
    checks, notes = {}, {}
    for season in ("2024", "2025"):
        r = ev.get(season, {}).get("regular_season")
        s = ev_season.get(season, {}).get("regular_season")
        if not r:
            continue
        checks[f"{season}_monthly_brier_le_0.2430"] = r["model"]["brier"] <= 0.2430
        checks[f"{season}_monthly_brier_le_elo"] = r["model"]["brier"] <= r["elo"]["brier"]
        if s:
            below, gap = _check_rate(s["run_line"]["home_minus_1_5"])
            checks[f"{season}_season_wf_home_minus_1_5_brier_below_constant"] = below
            checks[f"{season}_season_wf_home_minus_1_5_quintile_gap_le_0.05"] = gap
        for key in ("favorite_minus_1_5", "underdog_minus_1_5", "served_both_sides"):
            below, gap = _check_rate(r["run_line"][key])
            checks[f"{season}_monthly_{key}_brier_below_constant"] = below
            checks[f"{season}_monthly_{key}_quintile_gap_le_0.05"] = gap
        for key in ("home_minus_1_5", "away_minus_1_5"):
            block = r["run_line"][key]
            notes[f"{season}_monthly_{key}"] = {"brier_below_constant": block["brier"] < block["constant_brier"],
                                                 "max_quintile_gap": block["max_quintile_gap"],
                                                 "quintile_se": block["quintile_se"]}
    return {"checks": checks, "passed": all(checks.values()), "notes": notes}


def _american_implied(ml: pd.Series) -> np.ndarray:
    ml = pd.to_numeric(ml, errors="coerce").astype(float).to_numpy()
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def market_comparison(F: pd.DataFrame, pred: pd.DataFrame, csv_path: Path) -> dict:
    """De-vigged closing lines vs the walk-forward model on the same games: moneyline,
    posted total, and a market-anchored run-line baseline."""
    b = pd.read_csv(csv_path)
    b = b[b["season_type"] == 2].copy()
    b["home_team_id"] = b["home"].map(ESPN_ABBR); b["away_team_id"] = b["away"].map(ESPN_ABBR)
    b["edt"] = pd.to_datetime(b["date"], utc=True).dt.tz_localize(None)
    D = F.merge(pred, on="game_pk")
    D = D[D["is_final"] & D["home_win"].notna() & (D["game_type"] == "R")]
    m = b.merge(D[["game_pk", "home_team_id", "away_team_id", "home_score", "away_score", "dt", "home_win", "p_home",
                   "elo_p", "mu_home", "mu_away", "nb_r"]],
                on=["home_team_id", "away_team_id"], suffixes=("_e", ""))
    m = m[(m["edt"] - m["dt"]).abs() <= pd.Timedelta(hours=6)]
    m = m.sort_values("game_pk").drop_duplicates("event_id").drop_duplicates("game_pk")
    m = m[(m["home_score_e"] == m["home_score"]) & (m["away_score_e"] == m["away_score"])].reset_index(drop=True)

    ih, ia = _american_implied(m["p0_home_ml"]), _american_implied(m["p0_away_ml"])
    m["market_p_home"] = ih / (ih + ia)
    pickem = (m["p0_home_ml"] == m["p0_away_ml"]).to_numpy()
    y = m["home_win"].to_numpy(float)
    fav_home = m["market_p_home"].to_numpy() > 0.5
    fav_won = np.where(fav_home, y == 1, y == 0)[~pickem]
    model_p = m["p_home"].to_numpy(float)
    out = {
        "season": int(m["dt"].dt.year.mode().iloc[0]),
        "source": f"{csv_path.name}: ESPN BET closing lines (ESPN public odds API), de-vigged proportionally; "
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
        "modelMinusMarketBrier": _paired_ci((model_p - y) ** 2, (m["market_p_home"].to_numpy() - y) ** 2),
    }

    # Posted total: the model's P(over | no push) at the posted line vs the de-vigged over price.
    t = m[m["p0_total"].notna() & m["p0_over_odds"].notna() & m["p0_under_odds"].notna()].copy()
    line = t["p0_total"].astype(float).to_numpy()
    io, iu = _american_implied(t["p0_over_odds"]), _american_implied(t["p0_under_odds"])
    p_over_mkt = io / (io + iu)
    p_over_model = np.empty(len(t))
    for k, (mh, ma, r, L) in enumerate(zip(t["mu_home"], t["mu_away"], t["nb_r"], line)):
        mat = pm.score_matrix(mh, ma, r)
        i, j = np.indices(mat.shape)
        over, under = mat[(i + j) > L].sum(), mat[(i + j) < L].sum()
        p_over_model[k] = over / (over + under)
    tot = (t["home_score"] + t["away_score"]).to_numpy(float)
    decided = tot != line
    yo = (tot > line).astype(float)[decided]
    out["totals"] = {
        "line": "posted closing total (p0_total) with its over/under prices",
        "games": int(decided.sum()), "pushesExcluded": int((~decided).sum()),
        "overRate": round(float(yo.mean()), 4),
        "marketBrier": round(_brier(yo, p_over_mkt[decided]), 5),
        "modelBrier": round(_brier(yo, p_over_model[decided]), 5),
        "coinFlipBrier": 0.25,
        "modelMinusMarketBrier": _paired_ci((p_over_model[decided] - yo) ** 2, (p_over_mkt[decided] - yo) ** 2),
        "lineMAE": round(float(np.mean(np.abs(tot - line))), 3),
        "modelMeanMAE": round(float(np.mean(np.abs(tot - (t["mu_home"] + t["mu_away"]).to_numpy(float)))), 3),
    }

    # Run line: no run-line prices in the file, so the market baseline is "market-anchored":
    # a logistic on the de-vigged moneyline and the posted total (and their product), fit on
    # games before July and scored on July onward, next to the NB model on the same games.
    from sklearn.linear_model import LogisticRegression

    r = m[m["p0_total"].notna()].copy()
    pr = pm.batch_probabilities(r["mu_home"], r["mu_away"], r["nb_r"])
    r["p_h2"], r["p_a2"] = pr["home_by_2"], pr["away_by_2"]
    r["margin"] = r["home_score"] - r["away_score"]
    lg = np.log(r["market_p_home"] / (1 - r["market_p_home"])).to_numpy()
    tl = r["p0_total"].astype(float).to_numpy()
    X = np.c_[lg, tl, lg * tl]
    split = pd.Timestamp(f"{out['season']}-07-01")
    tr = (r["dt"] < split).to_numpy(); te = ~tr
    fav_model = (r["p_home"].to_numpy(float) >= 0.5)[te]
    sides = {}
    for name, y_all, p_model_all in (("home_minus_1_5", (r["margin"] >= 2).to_numpy(float), r["p_h2"].to_numpy()),
                                     ("away_minus_1_5", (r["margin"] <= -2).to_numpy(float), r["p_a2"].to_numpy())):
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(X[tr], y_all[tr].astype(int))
        sides[name] = (y_all[te], lr.predict_proba(X[te])[:, 1], p_model_all[te], float(y_all[tr].mean()))
    (yh, pmh, pnh, ch), (ya, pma, pna, ca) = sides["home_minus_1_5"], sides["away_minus_1_5"]
    sides["favorite_minus_1_5"] = (np.where(fav_model, yh, ya), np.where(fav_model, pmh, pma),
                                   np.where(fav_model, pnh, pna), None)
    sides["underdog_minus_1_5"] = (np.where(fav_model, ya, yh), np.where(fav_model, pma, pmh),
                                   np.where(fav_model, pna, pnh), None)
    sides["served_both_sides"] = (np.r_[yh, ya], np.r_[pmh, pma], np.r_[pnh, pna], (ch + ca) / 2)
    run_line = {"method": "market-anchored = logistic on logit(de-vigged moneyline), posted total and their "
                          f"product, fit on games before {split.date()} (n={int(tr.sum())}), scored on games from "
                          f"{split.date()} (n={int(te.sum())}); NB = the walk-forward runs model on the same games",
                "sides": {}}
    for name, (yy, p_mkt, p_nb, const) in sides.items():
        row = {"n": int(len(yy)), "actualRate": round(float(yy.mean()), 4),
               "marketAnchoredBrier": round(_brier(yy, p_mkt), 5), "modelBrier": round(_brier(yy, p_nb), 5),
               "marketMinusModelBrier": _paired_ci((p_mkt - yy) ** 2, (p_nb - yy) ** 2)}
        if const is not None:
            row["constantBrier"] = round(_brier(yy, np.full(len(yy), const)), 5)
        run_line["sides"][name] = row
    out["runLine"] = run_line
    return out


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
    end = today - pd.Timedelta(days=1)
    pred, _ = pm.walk_forward(F, params, start=start, end=end)
    seasons = sorted(int(s) for s in F.loc[F["official_date"] >= start, "season"].unique())
    ev = evaluate(F, pred, seasons)
    pred_season, _ = pm.walk_forward(F, params, start=start, end=end, refit="season")
    ev_season = evaluate(F, pred_season, seasons)

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
            "season_walk_forward": {
                "method": "refit once per season on every completed game of earlier seasons (plan P1-11 protocol)",
                "seasons": {s: {k: {kk: vv for kk, vv in v.items() if kk in ("games", "model", "elo", "run_line",
                                                                                "total_over_8_5", "team_total_over_4_5")}
                                for k, v in d.items() if k == "regular_season"}
                            for s, d in ev_season.items()},
            },
            "acceptance": acceptance(ev, ev_season),
        },
        "market_baselines": baselines,
    }
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"acceptance": out["evaluation"]["acceptance"],
                      "seasons": {s: {k: v["model"] for k, v in d.items()} for s, d in ev.items()},
                      "market": [{k: v for k, v in x.items() if k not in ("source",)} for x in baselines]}, indent=1))


if __name__ == "__main__":
    main()
