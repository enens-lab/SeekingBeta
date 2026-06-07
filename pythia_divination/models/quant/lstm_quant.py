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
ARTIFACTS_ROOT = DIV_ROOT / "artifacts"
SEQ_LEN = 60

# Torch options models -> artifact dir holding model.pt / scaler.joblib / feature_columns.json.
# Keras models live under artifacts/<name>/classifier/; the torch options models live under
# artifacts/<name>/torch/ (so they don't clobber the dormant keras files). "lstm_quant" is kept
# as an alias of lstm_5d (the interim parallel name, byte-identical model) so the older
# endpoint keeps working during/after the keras->torch migration.
_MODEL_DIRS = {
    "lstm_5d": ARTIFACTS_ROOT / "lstm_5d" / "torch",
    "lstm_jackpot": ARTIFACTS_ROOT / "lstm_jackpot" / "torch",
    "lstm_quant": ARTIFACTS_ROOT / "lstm_5d" / "torch",
}
DEFAULT_MODEL = "lstm_5d"

_STATES: dict[str, dict[str, Any]] = {}  # per-model lazy cache


def is_torch_model(model_name: str) -> bool:
    """True if this model name is served by the torch options path (vs the keras path)."""
    return model_name in _MODEL_DIRS


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


def load_quant_model(model_name: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Lazy-load a torch options model (+ scaler + feature columns) by name, cached per model."""
    if model_name in _STATES:
        return _STATES[model_name]
    art_dir = _MODEL_DIRS.get(model_name)
    if art_dir is None:
        raise ValueError(f"unknown torch options model '{model_name}' (known: {sorted(_MODEL_DIRS)})")
    import warnings
    import torch
    import joblib

    ckpt = torch.load(art_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = LSTMBaselineConfig(**{k: v for k, v in (ckpt.get("config") or {}).items()
                                if k in LSTMBaselineConfig.__dataclass_fields__})
    net = _build_net(cfg)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    # Unpickle the scaler only to read mean_/scale_ below; scaling is applied by hand,
    # so a sklearn version mismatch is harmless here — silence the benign warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = joblib.load(art_dir / "scaler.joblib")
    with open(art_dir / "feature_columns.json") as f:
        feature_cols = json.load(f)

    # Extract StandardScaler params as plain arrays so scaling is independent of the
    # sklearn version that pickled the scaler (EC2 may differ from the trainer's).
    # transform() == (X - mean_) / scale_; replicate it by hand to avoid version warnings/skew.
    scaler_mean = np.asarray(getattr(scaler, "mean_", 0.0), dtype=np.float64)
    scaler_scale = np.asarray(getattr(scaler, "scale_", 1.0), dtype=np.float64)
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)  # guard zero-variance cols

    state = {
        "model": net,
        "scaler": scaler,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "feature_cols": feature_cols,
        "cfg": cfg,
        "horizon": ckpt.get("pred_horizon", 5),
        "target_return": ckpt.get("target_return", 0.02),
        "model_name": model_name,
    }
    _STATES[model_name] = state
    logger.info("Loaded torch options model %s: %d features, seq_len=%d, %d params, horizon=%sd target=%s",
                model_name, cfg.num_features, cfg.seq_len, ckpt.get("n_params", 0),
                state["horizon"], state["target_return"])
    return state


def _signal(prob: float) -> str:
    if prob >= 0.60:
        return "strong_buy"
    if prob >= 0.55:
        return "buy"
    if prob >= 0.45:
        return "hold"
    return "avoid"


def predict_quant(ohlcv_df, symbol: str, *, model_name: str = DEFAULT_MODEL,
                  options_source: str = "auto",
                  schwab_chain_json: dict | None = None) -> dict[str, Any]:
    """Run an options-enriched torch LSTM for one ticker. `model_name` selects the
    model (lstm_5d / lstm_jackpot / lstm_quant). Returns probability/signal and the
    model's horizon + target_return (read from the checkpoint). Raises ValueError if < 60 bars."""
    import torch
    from .feature_engine import engineer_quant_features
    from .options_features import FEATURE_COLUMNS as OPTION_COLS

    state = load_quant_model(model_name)
    feat = engineer_quant_features(
        ohlcv_df, state["feature_cols"], symbol=symbol,
        options_source=options_source, schwab_chain_json=schwab_chain_json,
    )
    if len(feat) < SEQ_LEN:
        raise ValueError(f"need >= {SEQ_LEN} bars for {model_name}, got {len(feat)}")

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
        "model_name": model_name,
        "probability": round(prob, 4),
        "signal": _signal(prob),
        "horizon_days": int(state["horizon"]),
        "target_return_pct": round(float(state["target_return"]) * 100, 1),
        "options_source": options_source,
        "options_live": options_live,
    }
