"""Experimental PyTorch home-win model for the NFL -- a research comparison, NOT served.

The boards serve the de-vigged nflverse moneyline as the headline and the ridge +
Platt model view (train_baseline.py / model_view.py) next to it. This MLP is kept so
the comparison can be re-run, with the hygiene the old version lacked:
  * explicit FEATURE_ALLOWLIST (model_view.py), no "every numeric column", no lines,
    no post-game columns (overtime), no game-week roster statuses;
  * ties excluded from the label;
  * the epoch is chosen on an INNER chronological split of the training seasons (the
    old script picked the best-AUC epoch on the very holdout it then reported, and the
    exporter used that AUC to decide which model to serve);
  * no pos_weight (it shifts the probabilities and breaks calibration);
  * metrics are a season walk-forward with the market beside every number.

    python -m sports.football.train_torch --output-dir /tmp/football_torch [-v]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn

from sports.football import model_view as mv
from sports.football.torch_model import FootballTorchModel
from sports.football.train_baseline import _load_table, binary_metrics, season_status

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental NFL torch home-win model (walk-forward).")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--games", default=None)
    parser.add_argument("--first-test-season", type=int, default=2022)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--inner-fraction", type=float, default=0.2,
                        help="Latest share of the TRAINING games used to pick the epoch.")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "football_torch"))
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x))).numpy()


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_with_inner_split(train: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    """Fit on the earlier (1 - inner_fraction) of `train`, choose the epoch by log-loss
    on the latest inner_fraction (chronological), return the best-epoch model."""
    train = train.sort_values(["official_date", "game_id"])
    cut = int(len(train) * (1.0 - args.inner_fraction))
    fit_df, inner_df = train.iloc[:cut], train.iloc[cut:]
    features = [c for c in mv.FEATURE_ALLOWLIST if mv.feature_matrix(fit_df, [c])[c].notna().any()]
    imputer = SimpleImputer(strategy="median").fit(mv.feature_matrix(fit_df, features))
    scaler = StandardScaler().fit(imputer.transform(mv.feature_matrix(fit_df, features)))

    def xform(df: pd.DataFrame) -> np.ndarray:
        return scaler.transform(imputer.transform(mv.feature_matrix(df, features))).astype(np.float32)

    x_fit, y_fit = xform(fit_df), fit_df["home_win"].to_numpy(np.float32)
    x_inner, y_inner = xform(inner_df), inner_df["home_win"].to_numpy(float)
    model = FootballTorchModel(input_dim=len(features), hidden_width=args.hidden_width, dropout=args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    history = []
    patience = args.patience
    xt, yt = torch.from_numpy(x_fit), torch.from_numpy(y_fit)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(xt))
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            if len(idx) < 2:
                continue  # BatchNorm needs >1 row
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        inner_loss = _log_loss(y_inner, _predict(model, x_inner))
        history.append({"epoch": epoch, "inner_log_loss": inner_loss})
        if inner_loss < best["loss"] - 1e-5:
            best = {"loss": inner_loss, "epoch": epoch, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best["state"])
    return {"model": model, "features": features, "imputer": imputer, "scaler": scaler, "xform": xform,
            "best_epoch": best["epoch"], "inner_log_loss": best["loss"], "history": history}


def train_torch(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    dataset = _load_table(args.dataset, "football_training_dataset_latest")
    games = _load_table(args.games, "football_games_latest")
    current, train_through = season_status(games)
    params = mv.load_params() or {}
    frame = mv.prepare_training_frame(dataset, games, params.get("rating_params"))
    labelled = frame[frame["home_win"].notna()]

    per_season: dict[str, Any] = {}
    preds = []
    for season in range(args.first_test_season, train_through + 1):
        fit = fit_with_inner_split(labelled[labelled["season"] < season], args)
        test = labelled[labelled["season"] == season]
        p = _predict(fit["model"], fit["xform"](test))
        y = test["home_win"].to_numpy(float)
        block = {"torch": binary_metrics(y, p), "best_epoch": fit["best_epoch"]}
        mkt = test["market_home_win_probability"].notna().to_numpy()
        if mkt.any():
            block["market"] = binary_metrics(y[mkt], test["market_home_win_probability"].to_numpy(float)[mkt])
        per_season[str(season)] = block
        preds.append(pd.DataFrame({"game_id": test["game_id"].values, "season": season, "home_win": y,
                                   "torch_p": p, "market_p": test["market_home_win_probability"].values}))
    allp = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    pooled: dict[str, Any] = {}
    if not allp.empty:
        pooled["torch"] = binary_metrics(allp["home_win"], allp["torch_p"])
        m = allp["market_p"].notna()
        pooled["market"] = binary_metrics(allp.loc[m, "home_win"], allp.loc[m, "market_p"])
        pooled["torch_calibration"] = mv.calibration_fit(allp["torch_p"], allp["home_win"])

    final = fit_with_inner_split(labelled[labelled["season"] <= train_through], args)
    out = Path(args.output_dir) / "nfl" / "home_win"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final["model"].state_dict(), "input_dim": len(final["features"]),
                "hidden_width": args.hidden_width, "dropout": args.dropout, "feature_columns": final["features"]},
               out / "model.pt")
    joblib.dump(final["imputer"], out / "imputer.joblib")
    joblib.dump(final["scaler"], out / "scaler.joblib")
    pd.DataFrame({"feature": final["features"]}).to_csv(out / "feature_columns.csv", index=False)
    pd.DataFrame(final["history"]).to_csv(out / "training_history.csv", index=False)
    if not allp.empty:
        allp.to_csv(out / "walkforward_predictions.csv", index=False)
    metrics = {
        "league": "nfl",
        "served": False,
        "note": "Research comparison only. Served headline = de-vigged market; served model view = ridge+Platt.",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": "season walk-forward; epoch chosen by log-loss on the latest inner_fraction of each training window",
        "feature_count": len(final["features"]),
        "feature_columns": final["features"],
        "final_best_epoch": final["best_epoch"],
        "train_through_season": int(train_through),
        "pooled": pooled,
        "per_season": per_season,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    logger.info("NFL torch walk-forward pooled: %s", json.dumps(pooled, default=float))
    return metrics


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    train_torch(args)


if __name__ == "__main__":
    main()
