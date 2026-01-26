"""Metric computation utilities for model evaluation."""

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Probability predictions for positive class (optional)

    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            # AUC undefined if only one class present
            metrics["auc_roc"] = None

    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

        # Specificity (true negative rate)
        if tn + fp > 0:
            metrics["specificity"] = float(tn / (tn + fp))

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_score": float(r2_score(y_true, y_pred)),
    }

    # MAPE (avoid division by zero)
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        metrics["mape"] = float(mape)

    # Mean error (bias)
    metrics["mean_error"] = float(np.mean(y_pred - y_true))

    # Max absolute error
    metrics["max_error"] = float(np.max(np.abs(y_true - y_pred)))

    return metrics


def compute_trading_metrics(
    signals: np.ndarray,
    returns: np.ndarray,
    threshold: float = 0.55
) -> Dict[str, float]:
    """
    Compute trading-specific metrics.

    Args:
        signals: Model probability outputs
        returns: Actual next-day returns
        threshold: Probability threshold for buy signals

    Returns:
        Dictionary of trading metrics
    """
    # Convert signals to positions
    positions = np.where(signals >= threshold, 1, np.where(signals <= 1 - threshold, -1, 0))

    # Strategy returns (position * actual return)
    strategy_returns = positions * returns

    metrics = {}

    # Win rate
    if (positions != 0).sum() > 0:
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (positions != 0).sum()
        metrics["win_rate"] = float(winning_trades / total_trades)
        metrics["num_trades"] = int(total_trades)
    else:
        metrics["win_rate"] = 0.0
        metrics["num_trades"] = 0

    # Total strategy return
    metrics["total_return"] = float(strategy_returns.sum())

    # Sharpe ratio (annualized, assuming daily returns)
    if len(strategy_returns) > 0 and strategy_returns.std() > 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        metrics["sharpe_ratio"] = float(sharpe)

    # Maximum drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    metrics["max_drawdown"] = float(drawdown.min())

    return metrics


def get_metric_comparison_key(task: str) -> str:
    """Get the primary metric for model comparison based on task."""
    if task == "classifier":
        return "auc_roc"
    else:
        return "r2_score"


def is_metric_higher_better(metric: str) -> bool:
    """Determine if higher values are better for a given metric."""
    lower_is_better = {"mae", "rmse", "mape", "max_error", "max_drawdown"}
    return metric not in lower_is_better
