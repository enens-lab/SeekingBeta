"""Quick Apple-Silicon / CPU benchmark for the PGA PyTorch training environment."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


class BenchmarkModel(nn.Module):
    def __init__(
        self,
        sequence_features: int,
        static_features: int,
        hidden: int,
        static_width: int,
        dense_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=sequence_features,
            hidden_size=hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.seq_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.static_tower = nn.Sequential(
            nn.Linear(static_features, static_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_width, max(64, static_width // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        merged_width = hidden + max(64, static_width // 2)
        self.shared = nn.Sequential(
            nn.Linear(merged_width, dense_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_width, max(64, dense_width // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {
                'won': nn.Linear(max(64, dense_width // 2), 1),
                'top_10': nn.Linear(max(64, dense_width // 2), 1),
                'made_cut': nn.Linear(max(64, dense_width // 2), 1),
            }
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_out, _ = self.lstm(sequence)
        seq_last = seq_out[:, -1, :]
        seq_last = self.seq_head(seq_last)
        static_out = self.static_tower(static)
        shared = self.shared(torch.cat([seq_last, static_out], dim=1))
        return {name: head(shared).squeeze(-1) for name, head in self.heads.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark the PGA PyTorch MPS/CPU environment.')
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--sequence-length', type=int, default=12)
    parser.add_argument('--sequence-features', type=int, default=12)
    parser.add_argument('--static-features', type=int, default=265)
    parser.add_argument('--hidden', type=int, default=192)
    parser.add_argument('--static-width', type=int, default=384)
    parser.add_argument('--dense-width', type=int, default=384)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--learning-rate', type=float, default=4e-4)
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    if device_name == 'cpu':
        return torch.device('cpu')
    if device_name == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        raise RuntimeError('Requested MPS device, but torch.backends.mps.is_available() is false.')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _sync(device: torch.device) -> None:
    if device.type == 'mps':
        torch.mps.synchronize()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='[%(levelname)s] %(message)s')
    device = _resolve_device(args.device)

    model = BenchmarkModel(
        sequence_features=args.sequence_features,
        static_features=args.static_features,
        hidden=args.hidden,
        static_width=args.static_width,
        dense_width=args.dense_width,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    sequence = torch.randn(args.batch_size, args.sequence_length, args.sequence_features, device=device)
    static = torch.randn(args.batch_size, args.static_features, device=device)
    labels = {
        'won': torch.randint(0, 2, (args.batch_size,), device=device, dtype=torch.float32),
        'top_10': torch.randint(0, 2, (args.batch_size,), device=device, dtype=torch.float32),
        'made_cut': torch.randint(0, 2, (args.batch_size,), device=device, dtype=torch.float32),
    }

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(sequence, static)
        loss = sum(criterion(outputs[name], labels[name]) for name in labels)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    for _ in range(args.warmup_steps):
        step()
    _sync(device)

    start = time.perf_counter()
    last_loss = 0.0
    for _ in range(args.steps):
        last_loss = step()
    _sync(device)
    elapsed = time.perf_counter() - start

    rows_per_second = (args.batch_size * args.steps) / elapsed
    summary = {
        'device': str(device),
        'torch_version': torch.__version__,
        'mps_built': torch.backends.mps.is_built(),
        'mps_available': torch.backends.mps.is_available(),
        'batch_size': args.batch_size,
        'steps': args.steps,
        'elapsed_seconds': elapsed,
        'rows_per_second': rows_per_second,
        'last_loss': last_loss,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
