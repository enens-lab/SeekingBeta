"""MLB bullpen leak tripwire (plan P1-1 acceptance 2).

    cd pythia_divination && python scripts/mlb_leak_tripwire.py

The served model used to read team_box["bullpen"] -- the relievers who did NOT pitch --
so four closer/leverage counts alone "predicted" 65.4% of the 2025-06-19..2026-07-26
holdout (always-home: 52.8%). This rebuilds the bullpen features the way the fixed
pipeline does (sports/mlb/roster_features.build_pregame_bullpen_roster: relievers who
pitched for the team in the 14 days BEFORE the game) and checks that a model on any
four of them, or on all of them, cannot beat always-home by more than 2 points.

Exit code 1 when the tripwire fires. Needs the normalized tables
(mlb_training_dataset_latest, reliever_game_logs_latest).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports.mlb.roster_features import (  # noqa: E402
    build_bullpen_feature_frame,
    build_pregame_bullpen_roster,
    merge_bullpen_features,
)
from sports.mlb.train_baseline import _build_estimator  # noqa: E402

NORMALIZED = DIV_ROOT / "data" / "sports" / "mlb" / "normalized"
HOLDOUT_START = pd.Timestamp("2025-06-19")
HOLDOUT_END = pd.Timestamp("2026-07-26")
TOLERANCE = 0.02
LEVERAGE_FOUR = ["home_bullpen_high_leverage_arms_count", "away_bullpen_high_leverage_arms_count",
                 "home_bullpen_closer_saves_max", "away_bullpen_closer_saves_max"]


def _accuracy(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> float:
    est = _build_estimator("hist_gradient_boosting")
    x_tr = train[columns].apply(pd.to_numeric, errors="coerce")
    x_te = test[columns].apply(pd.to_numeric, errors="coerce")
    est.fit(x_tr, train["home_win"].astype(int))
    p = est.predict_proba(x_te)[:, 1]
    return float(((p >= 0.5) == (test["home_win"].to_numpy() == 1)).mean())


def main() -> int:
    base_cols = ["game_pk", "official_date", "season", "home_win", "home_team_id", "home_team_name",
                 "away_team_id", "away_team_name"]
    full = pd.read_parquet(NORMALIZED / "mlb_training_dataset_latest.parquet")
    games = full[base_cols].dropna(subset=["home_win"]).drop_duplicates("game_pk").copy()
    games["official_date"] = pd.to_datetime(games["official_date"], errors="coerce")
    relievers = pd.read_parquet(NORMALIZED / "reliever_game_logs_latest.parquet")

    team_games = pd.concat([
        games.rename(columns={f"{side}_team_id": "team_id", f"{side}_team_name": "team_name"})
             .assign(team_side=side)[["game_pk", "official_date", "season", "team_side", "team_id", "team_name"]]
        for side in ("home", "away")
    ], ignore_index=True)
    roster = build_pregame_bullpen_roster(relievers, team_games)
    features = build_bullpen_feature_frame(roster, relievers)
    data = merge_bullpen_features(games, features)
    bullpen_columns = [c for c in data.columns if c.startswith(("home_bullpen_", "away_bullpen_"))
                       and pd.to_numeric(data[c], errors="coerce").notna().any()]

    train = data[data["official_date"] < HOLDOUT_START]
    test = data[(data["official_date"] >= HOLDOUT_START) & (data["official_date"] <= HOLDOUT_END)]
    always_home = float(test["home_win"].mean())
    rng = np.random.default_rng(0)
    results = {"holdout": f"{HOLDOUT_START.date()}..{HOLDOUT_END.date()}", "n": int(len(test)),
               "always_home": round(always_home, 4), "limit": round(always_home + TOLERANCE, 4)}
    results["pregame_leverage_four"] = round(_accuracy(train, test, LEVERAGE_FOUR), 4)
    results["pregame_all_bullpen"] = round(_accuracy(train, test, bullpen_columns), 4)
    random_fours = []
    for _ in range(10):
        cols = list(rng.choice(bullpen_columns, size=4, replace=False))
        random_fours.append(round(_accuracy(train, test, cols), 4))
    results["pregame_random_fours_max"] = max(random_fours)
    # For contrast: the same four columns as stored in the OLD dataset (built from the
    # unused-reliever list). Expected to fire (~0.65) until the dataset is rebuilt.
    old = full.drop_duplicates("game_pk").set_index("game_pk")
    if all(c in old.columns for c in LEVERAGE_FOUR):
        tr_old = train[["game_pk", "home_win"]].join(old[LEVERAGE_FOUR], on="game_pk")
        te_old = test[["game_pk", "home_win"]].join(old[LEVERAGE_FOUR], on="game_pk")
        results["stored_dataset_leverage_four"] = round(_accuracy(tr_old, te_old, LEVERAGE_FOUR), 4)
    worst = max(results["pregame_leverage_four"], results["pregame_all_bullpen"], results["pregame_random_fours_max"])
    results["tripwire"] = "PASS" if worst <= always_home + TOLERANCE else "FAIL"
    print(json.dumps(results, indent=1))
    return 0 if results["tripwire"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
