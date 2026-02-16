"""Base model class and shared types for Pythia ML models."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ModelTask(str, Enum):
    CLASSIFICATION = "classifier"
    REGRESSION = "regressor"


@dataclass
class ModelConfig:
    model_type: str
    task: ModelTask
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    sequence_length: int = 10


@dataclass
class PredictionResult:
    prob_up: Optional[float] = None
    signal: Optional[str] = None
    predicted_return: Optional[float] = None
    probabilities: Optional[np.ndarray] = None


class BaseModel:
    """Abstract base for all Pythia ML models."""

    requires_sequences: bool = False

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.metrics: Dict[str, Any] = {}

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> PredictionResult:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Save model, scaler, and metrics to disk."""
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            joblib.dump(self.model, path / "model.joblib")
        if self.scaler is not None:
            joblib.dump(self.scaler, path / "scaler.joblib")
        if self.metrics:
            with open(path / "metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2, default=str)

        logger.info("Saved artifacts to %s", path)

    def load(self, path: Path) -> None:
        """Load model, scaler, and metrics from disk."""
        model_path = path / "model.joblib"
        if model_path.exists():
            self.model = joblib.load(model_path)

        scaler_path = path / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                self.metrics = json.load(f)

        logger.info("Loaded artifacts from %s", path)

    def get_scaler(self) -> Optional[StandardScaler]:
        """Return the scaler for this model.

        If the scaler is already loaded in-memory return it. Otherwise attempt
        to locate and load `scaler.joblib` from the repository `artifacts/`
        directory using the model config (works for containerized runtime).
        """
        if self.scaler is not None:
            return self.scaler

        try:
            repo_root = Path(__file__).resolve().parents[1]
            artifacts_dir = repo_root / "artifacts" / self.config.model_type / self.config.task.value
            scaler_path = artifacts_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler from %s", scaler_path)
                return self.scaler
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.debug("get_scaler failed: %s", exc)

        return None

    def _classify_signal(self, prob_up: float, threshold: float = 0.55) -> str:
        if prob_up >= threshold:
            return "buy"
        elif prob_up <= 1 - threshold:
            return "sell"
        return "hold"
