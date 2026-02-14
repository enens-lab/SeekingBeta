"""Utility functions for loading model artifacts."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from models.base import ModelConfig, ModelTask
from models.registry import get_artifacts_path, get_model, model_exists as _model_exists

logger = logging.getLogger(__name__)


def model_exists(model_name: str, task: str) -> bool:
    """Check if model artifacts exist."""
    return _model_exists(model_name, task)


def load_artifacts(
    model_name: str = "gradient_boosting",
    task: str = "classifier",
) -> Tuple[Any, Any, List[str], Dict[str, Any]]:
    """
    Load a trained model's artifacts (model, scaler, feature names, metrics).

    Returns:
        (model, scaler, feature_names, metrics)
    """
    from features.technical import get_feature_columns

    task_enum = ModelTask(task)
    path = get_artifacts_path(model_name, task_enum)

    if not path.exists():
        raise FileNotFoundError(f"No artifacts at {path}")

    config = ModelConfig(model_type=model_name, task=task_enum)
    model_obj = get_model(model_name, config)
    model_obj.load(path)

    feature_names = get_feature_columns()

    return model_obj.model, model_obj.scaler, feature_names, model_obj.metrics


def load_metrics(
    model_name: str = "gradient_boosting",
    task: str = "classifier",
) -> Optional[Dict[str, Any]]:
    """Load just the metrics for a trained model."""
    task_enum = ModelTask(task)
    path = get_artifacts_path(model_name, task_enum)
    metrics_path = path / "metrics.json"

    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        return json.load(f)
