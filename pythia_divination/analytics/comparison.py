"""Model comparison utilities."""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from models.registry import MODEL_REGISTRY, get_artifacts_path
from models.base import ModelTask
from models.utils import load_metrics


def compare_models(
    task: str = "classifier",
    metric: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare all trained models on a specific metric.

    Args:
        task: "classifier" or "regressor"
        metric: Metric to compare (default: auc_roc for classifier, r2_score for regressor)

    Returns:
        Dictionary with comparison results
    """
    task_enum = ModelTask.CLASSIFICATION if task == "classifier" else ModelTask.REGRESSION

    if metric is None:
        metric = "auc_roc" if task == "classifier" else "r2_score"

    results = {}

    for model_type in MODEL_REGISTRY.keys():
        metrics_data = load_metrics(model_type, task_enum)

        if metrics_data:
            results[model_type] = {
                "value": metrics_data.get(metric),
                "cv_mean": metrics_data.get("cv_mean"),
                "cv_std": metrics_data.get("cv_std"),
                "training_time": metrics_data.get("training_time_seconds"),
                "all_metrics": metrics_data,
            }

    return {
        "task": task,
        "metric": metric,
        "models": results,
        "ranking": get_model_rankings(results, metric, task),
    }


def get_model_rankings(
    results: Dict[str, Dict],
    metric: str,
    task: str
) -> Dict[str, int]:
    """
    Rank models based on a specific metric.

    Args:
        results: Model results dictionary
        metric: Metric to rank by
        task: Task type for determining sort order

    Returns:
        Dictionary mapping model names to their rank (1 is best)
    """
    # Determine if higher is better
    lower_is_better = {"mae", "rmse", "mape", "max_error", "max_drawdown"}
    reverse = metric not in lower_is_better

    # Filter models with valid metric values
    valid_models = [
        (name, data["value"])
        for name, data in results.items()
        if data.get("value") is not None
    ]

    if not valid_models:
        return {}

    # Sort by metric value
    sorted_models = sorted(valid_models, key=lambda x: x[1], reverse=reverse)

    return {name: rank + 1 for rank, (name, _) in enumerate(sorted_models)}


def get_all_model_metrics(task: str = "classifier") -> Dict[str, Dict]:
    """
    Get all metrics for all trained models.

    Args:
        task: "classifier" or "regressor"

    Returns:
        Dictionary mapping model names to their full metrics
    """
    task_enum = ModelTask.CLASSIFICATION if task == "classifier" else ModelTask.REGRESSION

    results = {}
    for model_type in MODEL_REGISTRY.keys():
        metrics_data = load_metrics(model_type, task_enum)
        if metrics_data:
            results[model_type] = metrics_data

    return results


def get_best_model(task: str = "classifier", metric: Optional[str] = None) -> Optional[str]:
    """
    Get the best performing model for a task.

    Args:
        task: "classifier" or "regressor"
        metric: Metric to use for comparison

    Returns:
        Name of the best model, or None if no models trained
    """
    comparison = compare_models(task, metric)
    ranking = comparison.get("ranking", {})

    if not ranking:
        return None

    # Find model with rank 1
    for model_name, rank in ranking.items():
        if rank == 1:
            return model_name

    return None


def generate_comparison_report(task: str = "classifier") -> str:
    """
    Generate a text report comparing all models.

    Args:
        task: "classifier" or "regressor"

    Returns:
        Formatted comparison report
    """
    comparison = compare_models(task)
    models = comparison.get("models", {})
    ranking = comparison.get("ranking", {})

    if not models:
        return f"No trained {task} models found."

    lines = [
        f"Model Comparison Report ({task})",
        "=" * 50,
        "",
    ]

    # Sort by ranking
    sorted_models = sorted(
        [(name, data, ranking.get(name, 99)) for name, data in models.items()],
        key=lambda x: x[2]
    )

    if task == "classifier":
        lines.append(f"{'Rank':<6} {'Model':<20} {'AUC':<10} {'Accuracy':<10} {'CV Mean':<12}")
        lines.append("-" * 60)

        for name, data, rank in sorted_models:
            all_metrics = data.get("all_metrics", {})
            auc = all_metrics.get("auc_roc", "N/A")
            acc = all_metrics.get("accuracy", "N/A")
            cv_mean = data.get("cv_mean", "N/A")

            auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else acc
            cv_str = f"{cv_mean:.4f}" if isinstance(cv_mean, float) else cv_mean

            lines.append(f"#{rank:<5} {name:<20} {auc_str:<10} {acc_str:<10} {cv_str:<12}")
    else:
        lines.append(f"{'Rank':<6} {'Model':<20} {'R2':<10} {'MAE':<12} {'RMSE':<12}")
        lines.append("-" * 62)

        for name, data, rank in sorted_models:
            all_metrics = data.get("all_metrics", {})
            r2 = all_metrics.get("r2_score", "N/A")
            mae = all_metrics.get("mae", "N/A")
            rmse = all_metrics.get("rmse", "N/A")

            r2_str = f"{r2:.4f}" if isinstance(r2, float) else r2
            mae_str = f"{mae:.6f}" if isinstance(mae, float) else mae
            rmse_str = f"{rmse:.6f}" if isinstance(rmse, float) else rmse

            lines.append(f"#{rank:<5} {name:<20} {r2_str:<10} {mae_str:<12} {rmse_str:<12}")

    lines.append("")
    lines.append(f"Best model: {get_best_model(task)}")

    return "\n".join(lines)
