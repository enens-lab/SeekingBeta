"""Train a tournament-aware WTA winner ranking model with PyTorch."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    # Handle cases where multiple winners might exist (though rare in tennis, happens in labels sometimes)
    # Cross entropy expects a target index. We use the argmax of targets.
    target_index = torch.argmax(targets, dim=1)
    return nn.functional.cross_entropy(masked_logits, target_index)

def _train_epoch(model: WTATournamentRanker, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
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
            for b_idx, t_id in enumerate(batch.tournament_id):
                for p_idx, is_valid in enumerate(valid_mask_np[b_idx]):
                    if not is_valid: continue
                    rows.append({
                        "tournament_id": t_id,
                        "tournament_name": batch.tournament_name[b_idx],
                        "player_id": batch.player_id[b_idx][p_idx],
                        "player_name": batch.player_name[b_idx][p_idx],
                        "winner_probability": float(winner_probs[b_idx, p_idx]),
                        "won": int(winner_truth[b_idx, p_idx]),
                    })
    return pd.DataFrame(rows)

def train_wta_ranker():
    _set_seed(42)
    device = _resolve_device("auto")
    
    dataset_path = ROOT / "data" / "sports" / "wta" / "normalized" / "wta_training_dataset_latest.csv"
    df = pd.read_csv(dataset_path)
    
    # Feature columns
    static_columns = ["elo", "surf_elo", "player_rolling_win_rate_5", "player_elo_diff_5"]
    
    # Split by time
    tourneys = df["tournament_id"].unique()
    split_idx = int(len(tourneys) * 0.8)
    train_ids = tourneys[:split_idx]
    val_ids = tourneys[split_idx:]
    
    train_df = df[df["tournament_id"].isin(train_ids)].copy()
    val_df = df[df["tournament_id"].isin(val_ids)].copy()
    
    # Preprocessing
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_train_raw = train_df[static_columns]
    X_val_raw = val_df[static_columns]
    
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_train = scaler.fit_transform(X_train_imputed)
    X_val = scaler.transform(imputer.transform(X_val_raw))
    
    # Build groups
    def make_groups(frame, processed_static):
        groups = []
        frame = frame.reset_index(drop=True)
        for t_id, group in frame.groupby("tournament_id", sort=False):
            indices = group.index.tolist()
            winner = group["won_tournament"].to_numpy()
            if winner.sum() != 1: continue # Only keep tournaments with a single winner recorded
            groups.append(TournamentGroup(
                tournament_id=t_id,
                tournament_name=group["tournament_name"].iloc[0],
                player_ids=group["player_id"].to_numpy(),
                player_names=group["player_name"].to_numpy(),
                static=processed_static[indices],
                winner=winner
            ))
        return groups

    train_groups = make_groups(train_df, X_train)
    val_groups = make_groups(val_df, X_val)
    
    train_loader = DataLoader(TournamentDataset(train_groups), batch_size=16, shuffle=True, collate_fn=_collate_tournaments)
    val_loader = DataLoader(TournamentDataset(val_groups), batch_size=16, shuffle=False, collate_fn=_collate_tournaments)
    
    model = WTATournamentRanker(
        static_feature_count=len(static_columns),
        static_width=128,
        dense_width=64,
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    logger.info("Starting training on %s...", device)
    for epoch in range(1, 21):
        loss = _train_epoch(model, train_loader, optimizer, device)
        if epoch % 5 == 0:
            preds = _predict(model, val_loader, device)
            top_pick_hit = float(preds.sort_values(["tournament_id", "winner_probability"], ascending=[True, False]).groupby("tournament_id").head(1)["won"].mean())
            logger.info("Epoch %s: loss=%.4f, val_top_pick_hit=%.2f%%", epoch, loss, top_pick_hit * 100)

    # Save artifacts
    artifact_dir = ROOT / "artifacts" / "wta_tournament_ranker_torch"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    joblib.dump(imputer, artifact_dir / "imputer.joblib")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    logger.info("Training complete. Artifacts saved to %s", artifact_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    train_wta_ranker()
