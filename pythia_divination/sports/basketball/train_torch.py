"""Train a PyTorch classifier for basketball home-win prediction."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sports.basketball.build_training_dataset import DEFAULT_BASKETBALL_DATA_ROOT, build_training_dataset
from sports.basketball.torch_model import BasketballTorchModel
from sports.basketball.train_baseline import _IDENTIFIER_COLUMNS, _LABEL_COLUMNS
from sports.pga.storage import write_json

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basketball PyTorch home-win model.")
    parser.add_argument("--league", default="nba", choices=["nba", "wnba", "all"], help="League dataset to train on.")
    parser.add_argument("--dataset", default=None, help="Optional explicit basketball training dataset path.")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild the dataset before training.")
    parser.add_argument("--epochs", type=int, default=35, help="Maximum epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--hidden-width", type=int, default=384, help="Hidden layer width.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout.")
    parser.add_argument("--learning-rate", type=float, default=7e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Optional label smoothing.")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"], help="Training device.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "basketball_torch"),
        help="Directory for torch model artifacts.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _default_dataset_path(league: str) -> Path:
    stem = "basketball_training_dataset_latest.csv" if league == "all" else f"basketball_{league}_training_dataset_latest.csv"
    return DEFAULT_BASKETBALL_DATA_ROOT / "normalized" / stem


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested MPS device, but it is not available.")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_feature_columns(dataset: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for column in dataset.columns:
        if column in _IDENTIFIER_COLUMNS or column in _LABEL_COLUMNS:
            continue
        if dataset[column].dtype.kind not in {"i", "u", "f", "b"}:
            continue
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
        raise ValueError("Need at least 20 games to train the basketball torch model.")
    holdout_count = max(10, math.ceil(unique_count * 0.2))
    train_ids = set(ordered_games.iloc[:-holdout_count]["game_id"])
    val_ids = set(ordered_games.iloc[-holdout_count:]["game_id"])
    return dataset[dataset["game_id"].isin(train_ids)].copy(), dataset[dataset["game_id"].isin(val_ids)].copy()


def _prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    x_train = train_df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_val = val_df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(x_train)
    val_imputed = imputer.transform(x_val)
    train_scaled = scaler.fit_transform(train_imputed).astype(np.float32)
    val_scaled = scaler.transform(val_imputed).astype(np.float32)
    return train_scaled, val_scaled, imputer, scaler


def _positive_weight(y: np.ndarray) -> float:
    positives = float(np.sum(y))
    negatives = float(len(y) - positives)
    if positives <= 0 or negatives <= 0:
        return 1.0
    return float(min(max(negatives / positives, 1.0), 10.0))


def _label_smooth(target: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return target
    return target * (1.0 - smoothing) + 0.5 * smoothing


@dataclass
class BasketballBatch:
    x: torch.Tensor
    y: torch.Tensor


class BasketballDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> BasketballBatch:
        return BasketballBatch(self.x[index], self.y[index])


def _collate(batch: list[BasketballBatch]) -> BasketballBatch:
    return BasketballBatch(
        x=torch.stack([item.x for item in batch]),
        y=torch.stack([item.y for item in batch]),
    )


def _train_epoch(
    model: BasketballTorchModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    label_smoothing: float,
    clip_grad_norm: float,
) -> float:
    model.train()
    running_loss = 0.0
    count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        x = batch.x.to(device)
        y = batch.y.to(device)
        logits = model(x)
        loss = criterion(logits, _label_smooth(y, label_smoothing))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        batch_size = x.shape[0]
        running_loss += float(loss.detach().cpu()) * batch_size
        count += batch_size
    return running_loss / max(count, 1)


def _predict(model: BasketballTorchModel, loader: DataLoader, *, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            outputs.append(probs)
    return np.concatenate(outputs, axis=0) if outputs else np.asarray([], dtype=np.float32)


def train_torch(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    if args.rebuild_dataset:
        build_training_dataset(
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

    dataset_path = Path(args.dataset) if args.dataset else _default_dataset_path(args.league)
    dataset = pd.read_parquet(dataset_path) if dataset_path.suffix == ".parquet" else pd.read_csv(dataset_path, low_memory=False)
    dataset["official_date"] = pd.to_datetime(dataset["official_date"], errors="coerce", format="mixed")
    dataset = dataset.dropna(subset=["home_win"]).reset_index(drop=True)
    if args.league != "all":
        dataset = dataset.loc[dataset["league"] == args.league].copy()

    feature_columns = _select_feature_columns(dataset)
    train_df, val_df = _time_split(dataset)
    feature_columns = _drop_all_missing_columns(train_df, feature_columns)

    x_train, x_val, imputer, scaler = _prepare_features(train_df, val_df, feature_columns)
    y_train = pd.to_numeric(train_df["home_win"], errors="coerce").fillna(0).astype(np.float32).to_numpy()
    y_val = pd.to_numeric(val_df["home_win"], errors="coerce").fillna(0).astype(np.float32).to_numpy()

    train_loader = DataLoader(
        BasketballDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        BasketballDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    model = BasketballTorchModel(input_dim=len(feature_columns), hidden_width=args.hidden_width, dropout=args.dropout).to(device)
    pos_weight = torch.tensor([_positive_weight(y_train)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -float("inf")
    best_epoch = -1
    history: list[dict[str, float | int | None]] = []
    patience = 6
    patience_left = patience

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            label_smoothing=args.label_smoothing,
            clip_grad_norm=args.clip_grad_norm,
        )
        probabilities = _predict(model, val_loader, device=device)
        auc = roc_auc_score(y_val.astype(int), probabilities) if len(np.unique(y_val)) > 1 else None
        ll = log_loss(y_val.astype(int), probabilities, labels=[0, 1])
        brier = brier_score_loss(y_val.astype(int), probabilities)
        scheduler.step(ll)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "roc_auc": auc,
                "log_loss": float(ll),
                "brier_score": float(brier),
            }
        )
        current_auc = auc if auc is not None else -float("inf")
        if args.verbose:
            logger.info("epoch=%s train_loss=%.5f val_auc=%s lr=%.6f", epoch, train_loss, auc, optimizer.param_groups[0]["lr"])
        if current_auc > best_auc:
            best_auc = current_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Torch training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    probabilities = _predict(model, val_loader, device=device)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "league": args.league,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "feature_count": int(len(feature_columns)),
        "device": str(device),
        "best_epoch": int(best_epoch),
        "roc_auc": float(roc_auc_score(y_val.astype(int), probabilities)),
        "log_loss": float(log_loss(y_val.astype(int), probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_val.astype(int), probabilities)),
        "accuracy": float(accuracy_score(y_val.astype(int), predictions)),
        "validation_start_date": str(val_df["official_date"].min().date()),
        "validation_end_date": str(val_df["official_date"].max().date()),
    }

    output_dir = Path(args.output_dir) / args.league / "home_win"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": len(feature_columns),
            "hidden_width": args.hidden_width,
            "dropout": args.dropout,
            "feature_columns": feature_columns,
        },
        output_dir / "model.pt",
    )
    joblib.dump(imputer, output_dir / "imputer.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    pd.DataFrame({"feature": feature_columns}).to_csv(output_dir / "feature_columns.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    val_predictions = val_df[["game_id", "official_date", "league", "away_team_name", "home_team_name", "home_win"]].copy()
    val_predictions["home_win_probability"] = probabilities
    val_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)
    logger.info("Basketball torch metrics: %s", json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    metrics = train_torch(args)
    print("=" * 72)
    print("BASKETBALL TORCH TRAINING COMPLETE")
    print("=" * 72)
    print(json.dumps(metrics, indent=2))
    print("=" * 72)


if __name__ == "__main__":
    main()
