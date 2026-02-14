"""Model registry — maps model names to classes and artifact paths."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type

from models.base import BaseModel, ModelConfig, ModelTask

logger = logging.getLogger(__name__)

ARTIFACTS_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


def _lazy_imports() -> Dict[str, Type[BaseModel]]:
    """Import model classes lazily to avoid heavy imports at startup."""
    from models.gradient_boosting import GradientBoostingModel
    from models.linear_regression import LinearRegressionModel
    from models.random_forest import RandomForestModel
    from models.lstm import LSTMModel

    return {
        "gradient_boosting": GradientBoostingModel,
        "linear_regression": LinearRegressionModel,
        "random_forest": RandomForestModel,
        "lstm": LSTMModel,
    }


# Lightweight registry — just names; actual classes loaded on demand
MODEL_REGISTRY: Dict[str, str] = {
    "gradient_boosting": "models.gradient_boosting.GradientBoostingModel",
    "linear_regression": "models.linear_regression.LinearRegressionModel",
    "random_forest": "models.random_forest.RandomForestModel",
    "lstm": "models.lstm.LSTMModel",
}


def get_model(name: str, config: Optional[ModelConfig] = None) -> BaseModel:
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    classes = _lazy_imports()
    cls = classes[name]

    if config is None:
        config = ModelConfig(model_type=name, task=ModelTask.CLASSIFICATION)

    return cls(config)


def get_artifacts_path(model_name: str, task: ModelTask) -> Path:
    """Return the artifacts directory for a given model + task."""
    return ARTIFACTS_ROOT / model_name / task.value


def model_exists(model_name: str, task: str) -> bool:
    """Check whether trained artifacts exist for a model/task pair."""
    task_enum = ModelTask(task)
    path = get_artifacts_path(model_name, task_enum)
    # Check for either joblib (sklearn) or pt (pytorch) model files
    return (path / "model.joblib").exists() or (path / "model.pt").exists()


def get_trained_models() -> List[Dict[str, str]]:
    """List all models that have saved artifacts."""
    trained = []
    for name in MODEL_REGISTRY:
        for task in ModelTask:
            if model_exists(name, task.value):
                path = get_artifacts_path(name, task)
                metrics_file = path / "metrics.json"
                trained.append({
                    "model": name,
                    "task": task.value,
                    "path": str(path),
                    "has_metrics": metrics_file.exists(),
                })
    return trained
