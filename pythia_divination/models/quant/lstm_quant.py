"""Torch LSTMBaseline model + `lstm_quant` predictor.

The model architecture is ported verbatim from TradeBot's
`lstm/models_torch/lstm_baseline.py` so the saved state_dict loads exactly. The
predictor builds the 48-feature / 60-step window (live options reconstruction),
scales with the saved StandardScaler, runs the net, and returns the same
prediction shape pythia's other LSTM endpoints use (P(>2% in 5d)).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DIV_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = DIV_ROOT / "artifacts" / "lstm_quant"
SEQ_LEN = 60

_STATE: dict[str, Any] = {}  # lazy singleton cache: model, scaler, feature_cols, cfg, device


# ── Architecture (verbatim port of TradeBot lstm/models_torch/lstm_baseline.py) ──

@dataclass
class LSTMBaselineConfig:
    seq_len: int = 60
    num_features: int = 48
    conv_channels: int = 64
    lstm1: int = 256
    lstm2: int = 128
    attn_heads: int = 4
    dropout: float = 0.3


def _build_net(cfg: LSTMBaselineConfig):
    import torch.nn as nn

    class LSTMBaseline(nn.Module):
        def __init__(self, cfg: LSTMBaselineConfig):
            super().__init__()
            self.cfg = cfg
            self.conv = nn.Conv1d(cfg.num_features, cfg.conv_channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.lstm1 = nn.LSTM(cfg.conv_channels, cfg.lstm1, batch_first=True)
            self.attn = nn.MultiheadAttention(cfg.lstm1, cfg.attn_heads, batch_first=True)
            self.norm = nn.LayerNorm(cfg.lstm1)
            self.lstm2 = nn.LSTM(cfg.lstm1, cfg.lstm2, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(cfg.lstm2, 32), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(32, 1),
            )

        def forward(self, x):
            h = x.transpose(1, 2)
            h = self.relu(self.conv(h))
            h = h.transpose(1, 2)
            h, _ = self.lstm1(h)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            h = self.norm(h + attn_out)
            h, _ = self.lstm2(h)
            last = h[:, -1, :]
            return self.head(last).squeeze(-1)

    return LSTMBaseline(cfg)


def load_quant_model() -> dict[str, Any]:
    """Lazy-load the torch model + scaler + feature columns (cached)."""
    if _STATE:
        return _STATE
    import warnings
    import torch
    import joblib

    ckpt = torch.load(ARTIFACT_DIR / "model.pt", map_location="cpu", weights_only=False)
    cfg = LSTMBaselineConfig(**{k: v for k, v in (ckpt.get("config") or {}).items()
                                if k in LSTMBaselineConfig.__dataclass_fields__})
    net = _build_net(cfg)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    # Unpickle the scaler only to read mean_/scale_ below; scaling is applied by hand,
    # so a sklearn version mismatch is harmless here — silence the benign warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    with open(ARTIFACT_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)

    # Extract StandardScaler params as plain arrays so scaling is independent of the
    # sklearn version that pickled the scaler (EC2 may differ from the trainer's).
    # transform() == (X - mean_) / scale_; replicate it by hand to avoid version warnings/skew.
    scaler_mean = np.asarray(getattr(scaler, "mean_", 0.0), dtype=np.float64)
    scaler_scale = np.asarray(getattr(scaler, "scale_", 1.0), dtype=np.float64)
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)  # guard zero-variance cols

    _STATE.update({
        "model": net,
        "scaler": scaler,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "feature_cols": feature_cols,
        "cfg": cfg,
        "horizon": ckpt.get("pred_horizon", 5),
        "target_return": ckpt.get("target_return", 0.02),
    })
    logger.info("Loaded lstm_quant: %d features, seq_len=%d, %d params",
                cfg.num_features, cfg.seq_len, ckpt.get("n_params", 0))
    return _STATE


def _signal(prob: float) -> str:
    if prob >= 0.60:
        return "strong_buy"
    if prob >= 0.55:
        return "buy"
    if prob >= 0.45:
        return "hold"
    return "avoid"


def predict_quant(ohlcv_df, symbol: str, *, options_source: str = "auto",
                  schwab_chain_json: dict | None = None) -> dict[str, Any]:
    """Run the options-enriched quant LSTM for one ticker. Returns a dict with
    probability/signal/horizon (P(>2% in 5d)). Raises ValueError if < 60 bars."""
    import torch
    from .feature_engine import engineer_quant_features
    from .options_features import FEATURE_COLUMNS as OPTION_COLS

    state = load_quant_model()
    feat = engineer_quant_features(
        ohlcv_df, state["feature_cols"], symbol=symbol,
        options_source=options_source, schwab_chain_json=schwab_chain_json,
    )
    if len(feat) < SEQ_LEN:
        raise ValueError(f"need >= {SEQ_LEN} bars for lstm_quant, got {len(feat)}")

    # Did live options reconstruction actually populate the 10 options features,
    # or did it fail and zero-fill? (last row carries today's reconstructed values.)
    opt_cols = [c for c in OPTION_COLS if c in feat.columns]
    options_live = bool(np.abs(feat.iloc[-1][opt_cols].to_numpy(dtype=np.float64)).sum() > 0) if opt_cols else False

    recent = feat.iloc[-SEQ_LEN:].to_numpy(dtype=np.float64)
    X = ((recent - state["scaler_mean"]) / state["scaler_scale"]).astype(np.float32)  # version-proof StandardScaler
    X = X.reshape(1, SEQ_LEN, len(state["feature_cols"]))

    with torch.no_grad():
        logit = state["model"](torch.from_numpy(X)).item()
    prob = float(1.0 / (1.0 + np.exp(-logit)))  # logit -> probability

    return {
        "probability": round(prob, 4),
        "signal": _signal(prob),
        "horizon_days": int(state["horizon"]),
        "target_return_pct": round(float(state["target_return"]) * 100, 1),
        "options_source": options_source,
        "options_live": options_live,
    }
