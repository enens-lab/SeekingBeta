"""Train a tournament-aware ATP/WTA winner ranking model with PyTorch."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

logger = logging.getLogger(__name__)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TournamentGroup:
    tournament_id: str
    tournament_name: str
    player_ids: np.ndarray
    player_names: np.ndarray
    static: np.ndarray
    winner: np.ndarray


@dataclass
class TournamentBatch:
    static: torch.Tensor
    valid_mask: torch.Tensor
    winner: torch.Tensor
    tournament_id: list[str]
    tournament_name: list[str]
    player_id: list[list[str]]
    player_name: list[list[str]]


class TournamentDataset(Dataset):
    def __init__(self, groups: list[TournamentGroup]) -> None:
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> TournamentGroup:
        return self.groups[index]


def _collate_tournaments(batch: list[TournamentGroup]) -> TournamentBatch:
    max_players = max(group.static.shape[0] for group in batch)
    static_features = batch[0].static.shape[1]

    static = torch.zeros((len(batch), max_players, static_features), dtype=torch.float32)
    valid_mask = torch.zeros((len(batch), max_players), dtype=torch.bool)
    winner = torch.zeros((len(batch), max_players), dtype=torch.float32)

    tournament_ids: list[str] = []
    tournament_names: list[str] = []
    player_ids: list[list[str]] = []
    player_names: list[list[str]] = []

    for batch_index, group in enumerate(batch):
        player_count = group.static.shape[0]
        static[batch_index, :player_count] = torch.from_numpy(group.static.astype(np.float32))
        valid_mask[batch_index, :player_count] = True
        winner[batch_index, :player_count] = torch.from_numpy(group.winner.astype(np.float32))
        tournament_ids.append(group.tournament_id)
        tournament_names.append(group.tournament_name)
        player_ids.append(group.player_ids.tolist())
        player_names.append(group.player_names.tolist())

    return TournamentBatch(
        static=static,
        valid_mask=valid_mask,
        winner=winner,
        tournament_id=tournament_ids,
        tournament_name=tournament_names,
        player_id=player_ids,
        player_name=player_names,
    )


class WTATournamentRanker(nn.Module):
    def __init__(self, *, static_feature_count: int, static_width: int, dense_width: int, dropout: float) -> None:
        super().__init__()
        self.static_tower = nn.Sequential(
            nn.Linear(static_feature_count, static_width),
            nn.ReLU(),
            nn.BatchNorm1d(static_width),
            nn.Dropout(dropout),
            nn.Linear(static_width, dense_width),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.winner_head = nn.Linear(dense_width, 1)

    def forward(self, static: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, player_count, static_features = static.shape
        flat_static = static.reshape(batch_size * player_count, static_features)

        static_out = self.static_tower(flat_static)
        winner_logits = self.winner_head(static_out).reshape(batch_size, player_count)
        return {"winner": winner_logits}


def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, -1e9)
    target_index = torch.argmax(targets, dim=1)
    return nn.functional.cross_entropy(masked_logits, target_index)


def _train_epoch(
    model: WTATournamentRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        static = batch.static.to(device)
        valid_mask = batch.valid_mask.to(device)
        winner = batch.winner.to(device)
        outputs = model(static)
        loss = _masked_cross_entropy(outputs["winner"], winner, valid_mask)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.detach().cpu())
    return running_loss / max(len(loader), 1)


def _predict(model: WTATournamentRanker, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            static = batch.static.to(device)
            valid_mask = batch.valid_mask.to(device)
            outputs = model(static)
            winner_logits = outputs["winner"].masked_fill(~valid_mask, -1e9)
            winner_probs = torch.softmax(winner_logits, dim=1).detach().cpu().numpy()

            valid_mask_np = batch.valid_mask.numpy()
            winner_truth = batch.winner.numpy()
            for batch_index, tournament_id in enumerate(batch.tournament_id):
                for player_index, is_valid in enumerate(valid_mask_np[batch_index]):
                    if not is_valid:
                        continue
                    rows.append(
                        {
                            "tournament_id": tournament_id,
                            "tournament_name": batch.tournament_name[batch_index],
                            "player_id": batch.player_id[batch_index][player_index],
                            "player_name": batch.player_name[batch_index][player_index],
                            "winner_probability": float(winner_probs[batch_index, player_index]),
                            "won": int(winner_truth[batch_index, player_index]),
                        },
                    )
    return pd.DataFrame(rows)


def train_wta_ranker() -> None:
    _set_seed(42)
    device = _resolve_device("auto")

    dataset_path = PACKAGE_ROOT / "data" / "sports" / "wta" / "normalized" / "wta_training_dataset_latest.csv"
    df = pd.read_csv(dataset_path)

    static_columns = [
        "elo",
        "surf_elo",
        "player_rolling_win_rate_5",
        "player_elo_diff_5",
        "serve_won",
        "return_won",
        "age",
        "height",
        "elo_field_percentile",
    ]

    df = df.sort_values("date")
    tournament_ids = df["tournament_id"].unique()
    split_index = int(len(tournament_ids) * 0.8)
    train_ids = tournament_ids[:split_index]
    val_ids = tournament_ids[split_index:]
    train_df = df[df["tournament_id"].isin(train_ids)].copy()
    val_df = df[df["tournament_id"].isin(val_ids)].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train = scaler.fit_transform(imputer.fit_transform(train_df[static_columns]))
    x_val = scaler.transform(imputer.transform(val_df[static_columns]))

    def make_groups(frame: pd.DataFrame, processed_static: np.ndarray) -> list[TournamentGroup]:
        groups: list[TournamentGroup] = []
        frame = frame.reset_index(drop=True)
        for tournament_id, group in frame.groupby("tournament_id", sort=False):
            indices = group.index.tolist()
            winner = group["won_tournament"].to_numpy()
            if winner.sum() != 1:
                continue
            groups.append(
                TournamentGroup(
                    tournament_id=tournament_id,
                    tournament_name=group["tournament_name"].iloc[0],
                    player_ids=group.get("player_key", group["player_id"]).to_numpy(),
                    player_names=group["player_name"].to_numpy(),
                    static=processed_static[indices],
                    winner=winner,
                ),
            )
        return groups

    train_groups = make_groups(train_df, x_train)
    val_groups = make_groups(val_df, x_val)
    if not train_groups or not val_groups:
        raise RuntimeError("Tennis training split produced empty tournament groups. Rebuild the dataset first.")

    train_loader = DataLoader(TournamentDataset(train_groups), batch_size=16, shuffle=True, collate_fn=_collate_tournaments)
    val_loader = DataLoader(TournamentDataset(val_groups), batch_size=16, shuffle=False, collate_fn=_collate_tournaments)

    model = WTATournamentRanker(
        static_feature_count=len(static_columns),
        static_width=256,
        dense_width=128,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    logger.info("Starting training on %s with %s features...", device, len(static_columns))
    for epoch in range(1, 31):
        loss = _train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if epoch % 5 == 0:
            predictions = _predict(model, val_loader, device)
            top_pick_hit = float(
                predictions.sort_values(["tournament_id", "winner_probability"], ascending=[True, False])
                .groupby("tournament_id")
                .head(1)["won"]
                .mean(),
            )
            logger.info("Epoch %s: loss=%.4f, val_top_pick_hit=%.2f%%", epoch, loss, top_pick_hit * 100)

    artifact_dir = PACKAGE_ROOT / "artifacts" / "wta_tournament_ranker_torch"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    joblib.dump(imputer, artifact_dir / "imputer.joblib")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")

    with open(artifact_dir / "feature_columns.json", "w") as handle:
        json.dump(static_columns, handle)

    logger.info("Training complete. Artifacts saved to %s", artifact_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    train_wta_ranker()
