"""
Training script for Pythia ML models.

Usage:
    python -m models.train --all
    python -m models.train --model gradient_boosting
    python -m models.train --model lstm --task classifier
    python -m models.train --list
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fetch import fetch_ohlcv
from features.technical import make_features, get_feature_columns
from features.sequences import create_sequences_per_ticker
from models.base import ModelConfig, ModelTask
from models.registry import MODEL_REGISTRY, get_model, get_artifacts_path

logger = logging.getLogger(__name__)

_TRAIN_TICKERS_FALLBACK = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
_TRAIN_PERIOD = "5y"
_SEQUENCE_LENGTH = 10


def _get_train_tickers() -> List[str]:
    try:
        from config.settings import settings
        return list(settings.universe)
    except Exception:
        return _TRAIN_TICKERS_FALLBACK


def _fetch_training_data(tickers: List[str], include_regression: bool = False) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        try:
            raw = fetch_ohlcv(ticker, period=_TRAIN_PERIOD, data_source="auto")
            feat = make_features(raw, include_regression_target=include_regression)
            feat["ticker"] = ticker
            frames.append(feat)
            logger.info("Fetched %d rows for %s", len(feat), ticker)
        except Exception as e:
            logger.warning("Skipping %s: %s", ticker, e)

    if not frames:
        raise RuntimeError("Could not fetch data for any ticker.")

    combined = pd.concat(frames)
    logger.info("Total training samples: %d from %d tickers", len(combined), len(frames))
    return combined


def train_model(model_name: str, task: ModelTask, tickers: List[str] = None) -> dict:
    tickers = tickers or _get_train_tickers()
    include_regression = task == ModelTask.REGRESSION

    logger.info("Training %s / %s on %d tickers ...", model_name, task.value, len(tickers))

    data = _fetch_training_data(tickers, include_regression=include_regression)
    feat_cols = get_feature_columns()

    target_col = "y_return" if task == ModelTask.REGRESSION else "y_class"
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    config = ModelConfig(model_type=model_name, task=task, sequence_length=_SEQUENCE_LENGTH)
    model = get_model(model_name, config)

    if model.requires_sequences:
        X_seq, y_seq = create_sequences_per_ticker(
            data, feat_cols, target_col, ticker_col="ticker",
            sequence_length=_SEQUENCE_LENGTH,
        )
        metrics = model.train(X_seq, y_seq)
    else:
        X = data[feat_cols].values
        y = data[target_col].values
        metrics = model.train(X, y)

    save_path = get_artifacts_path(model_name, task)
    model.save(save_path)

    logger.info("Saved %s / %s to %s", model_name, task.value, save_path)
    return metrics


def train_all(models=None, tasks=None, tickers=None):
    models = models or list(MODEL_REGISTRY.keys())
    tasks = tasks or list(ModelTask)

    results = []
    for model_name in models:
        for task in tasks:
            t0 = time.time()
            try:
                metrics = train_model(model_name, task, tickers)
                elapsed = time.time() - t0
                logger.info("Finished %s / %s in %.1fs", model_name, task.value, elapsed)
                results.append((model_name, task.value, metrics))
            except Exception as e:
                logger.error("Failed %s / %s: %s", model_name, task.value, e)
                results.append((model_name, task.value, {"error": str(e)}))

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Pythia ML models")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--model", type=str, help="Specific model to train")
    parser.add_argument("--task", type=str, choices=["classifier", "regressor"])
    parser.add_argument("--tickers", nargs="+", help="Override training tickers")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list:
        print("Available models:")
        for name in MODEL_REGISTRY:
            print(f"  - {name}")
        print("\nTasks: classifier, regressor")
        return

    if args.all:
        results = train_all(tickers=args.tickers)
    elif args.model:
        if args.model not in MODEL_REGISTRY:
            print(f"Unknown model: {args.model}. Available: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)
        if args.task:
            task = ModelTask(args.task)
            metrics = train_model(args.model, task, tickers=args.tickers)
            results = [(args.model, task.value, metrics)]
        else:
            results = train_all(models=[args.model], tickers=args.tickers)
    else:
        parser.print_help()
        return

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for model_name, task_val, metrics in results:
        status = "ERROR" if "error" in metrics else "OK"
        print(f"  {model_name:20s} / {task_val:10s} — {status}")
        if "error" not in metrics:
            for k, v in metrics.items():
                if k not in ("task", "model"):
                    print(f"    {k}: {v}")
        else:
            print(f"    {metrics['error']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
