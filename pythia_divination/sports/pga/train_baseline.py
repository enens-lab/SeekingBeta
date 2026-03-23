"""Train a first PGA baseline classifier before committing to a DNN architecture."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.pga.build_training_dataset import build_training_dataset  # noqa: E402
from sports.pga.storage import DEFAULT_PGA_DATA_ROOT, write_json  # noqa: E402

logger = logging.getLogger(__name__)

_IDENTIFIER_COLUMNS = {
    "tournament_id",
    "tournament_name",
    "display_date",
    "month",
    "event_sort_key",
    "player_id",
    "player_name",
    "country",
    "course_key",
    "champion_name",
    "champion_player_id",
    "course_name",
    "course_city",
    "course_state_code",
    "course_country",
    "standings_heading",
    "standings_value",
    "feature_snapshot_display_season",
    "feature_snapshot_mode",
}
_LABEL_COLUMNS = {"won", "top_5", "top_10", "made_cut", "withdrawn"}
_LEAKY_COLUMNS = {
    "position",
    "position_numeric",
    "position_pct_in_field",
    "total_score",
    "total_score_sort",
    "total_strokes",
    "thru",
    "round_score",
    "player_state",
    "current_round",
    "official_position",
    "official_position_sort",
    "projected_fedex_rank",
    "projected_fedex_rank_sort",
    "round_1_score",
    "round_2_score",
    "round_3_score",
    "round_4_score",
    "rounds_completed",
    "strokes_per_round",
    "score_to_par",
    "score_to_par_vs_field",
    "strokes_per_round_vs_field",
    "event_field_size",
    "field_size",
    "implied_uniform_win_probability",
    "event_cut_rate",
    "event_top10_rate",
    "event_win_rate",
    "event_avg_position",
    "event_avg_score_to_par",
    "event_avg_total_strokes",
    "event_avg_strokes_per_round",
    "event_best_score_to_par",
}


def _coerce_binary(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
        }
    )
    return mapped.fillna(0).astype(int)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a first PGA baseline model on the generated training dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_PGA_DATA_ROOT / "normalized" / "pga_training_dataset_latest.csv"),
        help="Path to the training dataset CSV.",
    )
    parser.add_argument(
        "--target",
        default="won",
        choices=["won", "top_5", "top_10", "made_cut"],
        help="Target label to train.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild the dataset before training using the configured public PGA snapshot mode.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season to use if rebuilding the dataset.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=None,
        help="Optional completed tournament limit if rebuilding the dataset.",
    )
    parser.add_argument(
        "--stat-limit",
        type=int,
        default=None,
        help="Optional stat limit if rebuilding the dataset.",
    )
    parser.add_argument(
        "--snapshot-mode",
        default="prior_season",
        choices=["prior_season", "same_season", "none"],
        help="Snapshot mode to use if rebuilding the dataset.",
    )
    parser.add_argument(
        "--model-type",
        default="random_forest",
        choices=["random_forest", "logreg"],
        help="Baseline model family to train.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "pga_baseline"),
        help="Directory for model artifacts.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _select_feature_columns(dataset: pd.DataFrame, target: str) -> list[str]:
    candidate_columns: list[str] = []
    for column in dataset.columns:
        if column in _LABEL_COLUMNS or column in _IDENTIFIER_COLUMNS or column in _LEAKY_COLUMNS:
            continue
        if column.endswith("_warning") or column.endswith("_found"):
            candidate_columns.append(column)
            continue
        if dataset[column].dtype.kind in {"i", "u", "f", "b"}:
            candidate_columns.append(column)
    filtered_columns: list[str] = []
    for column in candidate_columns:
        if column == target:
            continue
        series = pd.to_numeric(dataset[column], errors="coerce").dropna()
        if not series.empty and series.nunique() <= 1:
            continue
        filtered_columns.append(column)
    return filtered_columns


def _prepare_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def _time_ordered_group_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordering_column = "event_sort_key" if "event_sort_key" in dataset.columns else "season_event_index"
    tournaments = (
        dataset[["tournament_id", ordering_column]]
        .drop_duplicates()
        .sort_values(by=[ordering_column, "tournament_id"])
    )
    unique_count = len(tournaments)
    if unique_count < 4:
        raise ValueError("Need at least 4 tournaments to create a baseline validation split.")

    holdout_count = max(2, math.ceil(unique_count * 0.25))
    train_ids = set(tournaments.iloc[:-holdout_count]["tournament_id"])
    val_ids = set(tournaments.iloc[-holdout_count:]["tournament_id"])

    train_df = dataset[dataset["tournament_id"].isin(train_ids)].copy()
    val_df = dataset[dataset["tournament_id"].isin(val_ids)].copy()
    return train_df, val_df


def _normalize_by_tournament(frame: pd.DataFrame, probability_column: str) -> pd.Series:
    group_sums = frame.groupby("tournament_id")[probability_column].transform("sum")
    field_sizes = frame.groupby("tournament_id")["player_id"].transform("count")
    normalized = frame[probability_column] / group_sums.replace(0, np.nan)
    normalized = normalized.fillna(1.0 / field_sizes)
    return normalized


def _build_estimator(model_type: str) -> Pipeline:
    if model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        solver="liblinear",
                        C=0.25,
                    ),
                ),
            ]
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_baseline(args: argparse.Namespace) -> dict[str, Any]:
    if args.rebuild_dataset:
        build_training_dataset(
            argparse.Namespace(
                season=args.season,
                tour_code="R",
                history_limit=args.history_limit,
                stat_limit=args.stat_limit,
                snapshot_mode=args.snapshot_mode,
                output_root=None,
                snapshot_tag=None,
                verbose=args.verbose,
            )
        )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path, low_memory=False)
    if args.target not in dataset.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset")

    feature_columns = _select_feature_columns(dataset, args.target)
    if not feature_columns:
        raise ValueError("No usable baseline feature columns found")

    dataset[args.target] = _coerce_binary(dataset[args.target])
    train_df, val_df = _time_ordered_group_split(dataset)

    X_train = _prepare_feature_frame(train_df, feature_columns)
    y_train = train_df[args.target]
    X_val = _prepare_feature_frame(val_df, feature_columns)
    y_val = val_df[args.target]
    feature_columns = [column for column in feature_columns if X_train[column].notna().any()]
    X_train = X_train[feature_columns]
    X_val = X_val[feature_columns]

    pipeline = _build_estimator(args.model_type)
    pipeline.fit(X_train, y_train)

    raw_prob = pipeline.predict_proba(X_val)[:, 1]
    eval_df = val_df[["tournament_id", "player_id", "player_name", args.target]].copy()
    eval_df["raw_probability"] = raw_prob
    eval_df["normalized_win_probability"] = _normalize_by_tournament(eval_df, "raw_probability")

    auc = roc_auc_score(y_val, raw_prob) if y_val.nunique() > 1 else None
    ll = log_loss(y_val, raw_prob, labels=[0, 1])
    brier = brier_score_loss(y_val, raw_prob)

    top_pick = eval_df.sort_values(["tournament_id", "normalized_win_probability"], ascending=[True, False]).groupby("tournament_id").head(1)
    top_pick_hit_rate = float(top_pick[args.target].mean()) if not top_pick.empty else 0.0
    mean_winner_probability = float(
        eval_df.loc[eval_df[args.target] == 1, "normalized_win_probability"].mean()
    ) if (eval_df[args.target] == 1).any() else 0.0

    artifact_dir = Path(args.output_dir) / args.model_type / args.target
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / "model.joblib")
    with (artifact_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_columns, handle, indent=2)
        handle.write("\n")
    eval_df.to_csv(artifact_dir / "validation_predictions.csv", index=False)

    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(artifact_dir / "feature_importances.csv", index=False)

    metrics = {
        "model_type": args.model_type,
        "target": args.target,
        "dataset_path": str(dataset_path),
        "feature_count": len(feature_columns),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "train_tournaments": int(train_df["tournament_id"].nunique()),
        "validation_tournaments": int(val_df["tournament_id"].nunique()),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_validation": float(y_val.mean()),
        "roc_auc": auc,
        "log_loss": float(ll),
        "brier_score": float(brier),
        "top_pick_hit_rate": top_pick_hit_rate,
        "mean_winner_probability": mean_winner_probability,
        "artifact_dir": str(artifact_dir),
        "validation_prediction_path": str(artifact_dir / "validation_predictions.csv"),
        "feature_snapshot_mode": (
            str(dataset["feature_snapshot_mode"].iloc[0])
            if "feature_snapshot_mode" in dataset.columns and not dataset.empty
            else args.snapshot_mode
        ),
        "leakage_warning": bool(
            dataset["feature_leakage_warning"].astype(bool).any()
            if "feature_leakage_warning" in dataset.columns and not dataset.empty
            else args.snapshot_mode == "same_season"
        ),
        "notes": [
            "Baseline can use prior-season or same-season public PGA stat snapshots depending on dataset build mode.",
            "Evaluation remains optimistic until we collect true pre-tournament historical stat snapshots.",
            "Tree-based random forest is the default baseline because this is a sparse, tabular, tournament-ranking problem rather than a sequential price-series problem.",
        ],
    }
    write_json(artifact_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    metrics = train_baseline(args)
    print("=" * 72)
    print("PGA BASELINE TRAINING COMPLETE")
    print("=" * 72)
    print(f"Model type:             {metrics['model_type']}")
    print(f"Target:                 {metrics['target']}")
    print(f"Feature count:          {metrics['feature_count']}")
    print(f"Train tournaments:      {metrics['train_tournaments']}")
    print(f"Validation tournaments: {metrics['validation_tournaments']}")
    print(f"Validation ROC AUC:     {metrics['roc_auc']}")
    print(f"Validation log loss:    {metrics['log_loss']:.6f}")
    print(f"Validation Brier:       {metrics['brier_score']:.6f}")
    print(f"Top-pick hit rate:      {metrics['top_pick_hit_rate']:.2%}")
    print(f"Artifact dir:           {metrics['artifact_dir']}")
    print(f"Snapshot mode:          {metrics['feature_snapshot_mode']}")
    if metrics["leakage_warning"]:
        print("Leakage note:           same-season public stat snapshots still include end-of-season leakage")
    else:
        print("Leakage note:           reduced; snapshot priors come from prior seasons or are disabled")
    print("=" * 72)


if __name__ == "__main__":
    main()
