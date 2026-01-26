"""Analytics API endpoints for model evaluation and comparison."""

from fastapi import APIRouter, HTTPException
from typing import Optional

from models.registry import MODEL_REGISTRY, get_artifacts_path, get_trained_models
from models.base import ModelTask, ModelConfig
from models.utils import load_metrics, model_exists
from analytics.comparison import compare_models, get_best_model, generate_comparison_report

from .schemas import ModelComparisonResponse, FeatureImportanceResponse

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/metrics/{model_type}/{task}")
def get_model_metrics(model_type: str, task: str):
    """Get training metrics for a specific model."""
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model not found: {model_type}")

    if task not in ("classifier", "regressor"):
        raise HTTPException(400, f"Invalid task: {task}")

    task_enum = ModelTask.CLASSIFICATION if task == "classifier" else ModelTask.REGRESSION
    metrics = load_metrics(model_type, task_enum)

    if not metrics:
        raise HTTPException(404, f"No metrics found for {model_type}/{task}")

    return metrics


@router.get("/compare", response_model=ModelComparisonResponse)
def compare_all_models(
    task: str = "classifier",
    metric: Optional[str] = None
):
    """
    Compare all trained models on a specific metric.

    Args:
        task: "classifier" or "regressor"
        metric: Metric to compare (default: auc_roc for classifier, r2_score for regressor)
    """
    if task not in ("classifier", "regressor"):
        raise HTTPException(400, f"Invalid task: {task}")

    result = compare_models(task, metric)

    return ModelComparisonResponse(
        task=task,
        metric=result["metric"],
        models={k: {kk: vv for kk, vv in v.items() if kk != "all_metrics"}
                for k, v in result["models"].items()},
        ranking=result["ranking"]
    )


@router.get("/compare/report")
def comparison_report(task: str = "classifier"):
    """Generate a text comparison report for all models."""
    if task not in ("classifier", "regressor"):
        raise HTTPException(400, f"Invalid task: {task}")

    report = generate_comparison_report(task)
    return {"task": task, "report": report}


@router.get("/best/{task}")
def get_best(task: str, metric: Optional[str] = None):
    """Get the best performing model for a task."""
    if task not in ("classifier", "regressor"):
        raise HTTPException(400, f"Invalid task: {task}")

    best = get_best_model(task, metric)

    if not best:
        raise HTTPException(404, f"No trained {task} models found")

    return {"task": task, "best_model": best, "metric": metric}


@router.get("/feature-importance/{model_type}/{task}", response_model=FeatureImportanceResponse)
def get_feature_importance(model_type: str, task: str):
    """Get feature importance for a model (if available)."""
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model not found: {model_type}")

    if task not in ("classifier", "regressor"):
        raise HTTPException(400, f"Invalid task: {task}")

    task_enum = ModelTask.CLASSIFICATION if task == "classifier" else ModelTask.REGRESSION

    if not model_exists(model_type, task_enum):
        raise HTTPException(404, f"Model not trained: {model_type}/{task}")

    # Load model and get feature importance
    from models.registry import get_model

    config = ModelConfig(model_type=model_type, task=task_enum, hyperparams={})
    model = get_model(model_type, config)
    artifacts_path = get_artifacts_path(model_type, task_enum)
    model.load(artifacts_path)

    importance = model.get_feature_importance()

    if not importance:
        raise HTTPException(404, f"Feature importance not available for {model_type}")

    # Sort by importance
    sorted_importance = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))

    return FeatureImportanceResponse(
        model=model_type,
        task=task,
        feature_importance=sorted_importance
    )


@router.get("/models")
def list_all_models():
    """List all available models and their training status."""
    return get_trained_models()


@router.get("/models/available")
def list_available_model_types():
    """List all available model types."""
    return {
        "model_types": list(MODEL_REGISTRY.keys()),
        "tasks": ["classifier", "regressor"]
    }
