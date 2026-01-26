"""Sequence data preparation for LSTM and other sequence-based models."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flat feature array to sequences for LSTM.

    Args:
        X: Shape (n_samples, n_features)
        y: Shape (n_samples,)
        sequence_length: Number of timesteps per sequence

    Returns:
        X_seq: Shape (n_samples - sequence_length + 1, sequence_length, n_features)
        y_seq: Shape (n_samples - sequence_length + 1,)
    """
    n_samples, n_features = X.shape

    if n_samples < sequence_length:
        raise ValueError(
            f"Not enough samples ({n_samples}) for sequence length ({sequence_length})"
        )

    n_sequences = n_samples - sequence_length + 1

    X_seq = np.zeros((n_sequences, sequence_length, n_features))
    y_seq = np.zeros(n_sequences)

    for i in range(n_sequences):
        X_seq[i] = X[i:i + sequence_length]
        y_seq[i] = y[i + sequence_length - 1]  # Target at end of sequence

    return X_seq, y_seq


def create_sequences_per_ticker(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    ticker_col: str = "ticker",
    sequence_length: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences respecting ticker boundaries (no cross-ticker sequences).

    This prevents data leakage by ensuring sequences don't span multiple tickers.

    Args:
        data: DataFrame with features, target, and ticker column
        feature_cols: List of feature column names
        target_col: Name of target column
        ticker_col: Name of ticker column
        sequence_length: Number of timesteps per sequence

    Returns:
        X_seq: Combined sequences from all tickers
        y_seq: Combined targets from all tickers
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for ticker in data[ticker_col].unique():
        ticker_data = data[data[ticker_col] == ticker].sort_index()

        if len(ticker_data) < sequence_length:
            continue

        X = ticker_data[feature_cols].values
        y = ticker_data[target_col].values

        X_seq, y_seq = create_sequences(X, y, sequence_length)
        all_X.append(X_seq)
        all_y.append(y_seq)

    if not all_X:
        raise ValueError(
            f"No ticker had enough data for sequence_length={sequence_length}"
        )

    return np.concatenate(all_X), np.concatenate(all_y)


def create_inference_sequence(
    X: np.ndarray,
    sequence_length: int = 10
) -> np.ndarray:
    """
    Create a single sequence for inference from the most recent data.

    Args:
        X: Shape (n_samples, n_features) - historical feature data
        sequence_length: Number of timesteps

    Returns:
        X_seq: Shape (1, sequence_length, n_features) - single sequence for prediction
    """
    if len(X) < sequence_length:
        raise ValueError(
            f"Not enough samples ({len(X)}) for sequence length ({sequence_length})"
        )

    # Take the most recent sequence_length samples
    X_recent = X[-sequence_length:]
    return X_recent.reshape(1, sequence_length, -1)


def split_sequences_train_val(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    val_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequence data into train/validation sets.

    Uses time-based split (last val_ratio of data for validation)
    to respect temporal ordering.

    Args:
        X_seq: Sequence features
        y_seq: Sequence targets
        val_ratio: Fraction of data for validation

    Returns:
        X_train, y_train, X_val, y_val
    """
    n_samples = len(X_seq)
    split_idx = int(n_samples * (1 - val_ratio))

    X_train = X_seq[:split_idx]
    y_train = y_seq[:split_idx]
    X_val = X_seq[split_idx:]
    y_val = y_seq[split_idx:]

    return X_train, y_train, X_val, y_val
