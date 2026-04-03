"""Train a first basketball home-win baseline model on generated matchup data."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from sports.basketball.build_training_dataset import DEFAULT_BASKETBALL_DATA_ROOT, build_training_dataset
from sports.pga.storage import write_json

logger = logging.getLogger(__name__)

_IDENTIFIER_COLUMNS = {
    "league",
    "season_display",
    "game_id",
    "game_code",
    "game_date",
    "game_date_est",
    "game_date_time_est",
    "game_date_time_utc",
    "game_time_utc",
    "official_date",
    "status_code",
    "status_text",
    "game_label",
    "game_sub_label",
    "game_subtype",
    "week_name",
    "week_number",
    "series_text",
    "series_game_number",
    "is_neutral",
    "if_necessary",
    "arena_name",
    "arena_city",
    "arena_state",
    "away_team_id",
    "away_team_name",
    "away_team_city",
    "away_team_tricode",
    "home_team_id",
    "home_team_name",
    "home_team_city",
    "home_team_tricode",
    "away_team_key",
    "home_team_key",
    "game_id_prefix",
}
_LABEL_COLUMNS = {"home_win", "away_win"}


def _to_datetime_mixed(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basketball home-win baseline model.")
    parser.add_argument(
        "--league",
        default="nba",
        choices=["nba", "wnba", "all"],
        help="League dataset to train on.",
    )
    parser.add_argument("--dataset", default=None, help="Optional explicit basketball training dataset CSV path.")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild the dataset before training.")
    parser.add_argument(
        "--model",
        choices=["random_forest", "extra_trees", "hist_gradient_boosting"],
        default="hist_gradient_boosting",
        help="Baseline model family to train.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "artifacts" / "basketball_baseline"),
        help="Directory for trained baseline artifacts.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _default_dataset_path(league: str) -> Path:
    stem = "basketball_training_dataset_latest.csv" if league == "all" else f"basketball_{league}_training_dataset_latest.csv"
    return DEFAULT_BASKETBALL_DATA_ROOT / "normalized" / stem


def _select_feature_columns(dataset: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for column in dataset.columns:
        if column in _IDENTIFIER_COLUMNS or column in _LABEL_COLUMNS:
            continue
        if dataset[column].dtype.kind in {"i", "u", "f", "b"}:
            series = pd.to_numeric(dataset[column], errors="coerce").dropna()
            if not series.empty and series.nunique() <= 1:
                continue
            features.append(column)
    return features


def _drop_all_missing_columns(frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    retained: list[str] = []
    for column in feature_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            retained.append(column)
    return retained


def _time_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_games = dataset[["game_id", "official_date"]].drop_duplicates().sort_values(["official_date", "game_id"])
    unique_count = len(ordered_games)
    if unique_count < 20:
        raise ValueError("Need at least 20 completed games to train a basketball baseline split.")
    holdout_count = max(10, math.ceil(unique_count * 0.2))
    train_ids = set(ordered_games.iloc[:-holdout_count]["game_id"])
    val_ids = set(ordered_games.iloc[-holdout_count:]["game_id"])
    return dataset[dataset["game_id"].isin(train_ids)].copy(), dataset[dataset["game_id"].isin(val_ids)].copy()


def _build_estimator(model_name: str) -> Pipeline:
    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            min_samples_leaf=15,
            l2_regularization=0.5,
            random_state=42,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", model)])


def train_baseline(args: argparse.Namespace) -> dict[str, Any]:
    if args.rebuild_dataset:
        dataset, _ = build_training_dataset(
            argparse.Namespace(
                league=args.league,
                refresh_source_data=True,
                history_limit=None,
                max_workers=8,
                output_root=None,
                snapshot_tag=None,
                verbose=args.verbose,
            )
        )
    else:
        dataset_path = Path(args.dataset) if args.dataset else _default_dataset_path(args.league)
        dataset = pd.read_csv(dataset_path, low_memory=False)

    dataset["official_date"] = _to_datetime_mixed(dataset["official_date"])
    dataset = dataset.dropna(subset=["home_win"]).reset_index(drop=True)
    if args.league != "all":
        dataset = dataset.loc[dataset["league"] == args.league].copy()

    feature_columns = _select_feature_columns(dataset)
    train_df, val_df = _time_split(dataset)
    feature_columns = _drop_all_missing_columns(train_df, feature_columns)

    x_train = train_df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_train = pd.to_numeric(train_df["home_win"], errors="coerce").fillna(0).astype(int)
    x_val = val_df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_val = pd.to_numeric(val_df["home_win"], errors="coerce").fillna(0).astype(int)

    estimator = _build_estimator(args.model)
    estimator.fit(x_train, y_train)
    probabilities = estimator.predict_proba(x_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "league": args.league,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "feature_count": int(len(feature_columns)),
        "model": args.model,
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "log_loss": float(log_loss(y_val, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_val, probabilities)),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "validation_start_date": str(val_df["official_date"].min().date()),
        "validation_end_date": str(val_df["official_date"].max().date()),
    }

    output_dir = Path(args.output_dir) / args.league / args.model / "home_win"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, output_dir / "model.joblib")
    pd.DataFrame({"feature": feature_columns}).to_csv(output_dir / "feature_columns.csv", index=False)
    pd.DataFrame(
        {
            "game_id": val_df["game_id"].values,
            "official_date": val_df["official_date"].dt.date.astype(str).values,
            "league": val_df["league"].values,
            "away_team_name": val_df["away_team_name"].values,
            "home_team_name": val_df["home_team_name"].values,
            "home_win": y_val.values,
            "home_win_probability": probabilities,
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)

    if hasattr(estimator.named_steps["model"], "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": estimator.named_steps["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance.to_csv(output_dir / "feature_importances.csv", index=False)

    write_json(output_dir / "metrics.json", metrics)
    logger.info("Basketball baseline metrics: %s", json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train_baseline(args)


if __name__ == "__main__":
    main()
