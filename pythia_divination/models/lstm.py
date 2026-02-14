"""LSTM neural network model for classification and regression."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

from models.base import BaseModel, ModelConfig, ModelTask, PredictionResult

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "epochs": 50,
    "batch_size": 32,
}


def _get_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _LSTMNet:
    @staticmethod
    def build(input_size, hidden_size, num_layers, dropout, output_size, task):
        torch, nn = _get_torch()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, output_size)
                self.task = task

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last = lstm_out[:, -1, :]
                out = self.fc(last)
                if self.task == ModelTask.CLASSIFICATION:
                    out = torch.sigmoid(out)
                return out.squeeze(-1)

        return Net()


class LSTMModel(BaseModel):
    requires_sequences = True

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.net = None
        self._device = None

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        torch, nn = _get_torch()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params = {**_DEFAULT_PARAMS, **self.config.hyperparams}
        val_ratio = kwargs.get("val_ratio", 0.2)
        split = int(len(X) * (1 - val_ratio))

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        n_samples, seq_len, n_features = X_train.shape
        self.scaler = StandardScaler()
        self.scaler.fit(X_train.reshape(-1, n_features))

        X_train_s = self._scale_sequences(X_train)
        X_val_s = self._scale_sequences(X_val)

        self.net = _LSTMNet.build(
            input_size=n_features,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            output_size=1,
            task=self.config.task,
        ).to(self._device)

        criterion = nn.BCELoss() if self.config.task == ModelTask.CLASSIFICATION else nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=params["learning_rate"])

        epochs = params["epochs"]
        batch_size = params["batch_size"]
        X_t = torch.FloatTensor(X_train_s).to(self._device)
        y_t = torch.FloatTensor(y_train).to(self._device)

        self.net.train()
        for epoch in range(epochs):
            indices = np.arange(len(X_t))
            np.random.shuffle(indices)
            for start in range(0, len(X_t), batch_size):
                batch_idx = indices[start:start + batch_size]
                optimizer.zero_grad()
                output = self.net(X_t[batch_idx])
                loss = criterion(output, y_t[batch_idx])
                loss.backward()
                optimizer.step()

        self.net.eval()
        with torch.no_grad():
            val_output = self.net(torch.FloatTensor(X_val_s).to(self._device)).cpu().numpy()

        if self.config.task == ModelTask.CLASSIFICATION:
            val_pred = (val_output >= 0.5).astype(int)
            self.metrics = {
                "accuracy": float(accuracy_score(y_val, val_pred)),
                "auc": float(roc_auc_score(y_val, val_output)) if len(np.unique(y_val)) > 1 else 0.0,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "task": self.config.task.value,
                "model": "lstm",
            }
        else:
            self.metrics = {
                "mse": float(mean_squared_error(y_val, val_output)),
                "rmse": float(np.sqrt(mean_squared_error(y_val, val_output))),
                "r2": float(r2_score(y_val, val_output)),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "task": self.config.task.value,
                "model": "lstm",
            }

        self.model = self.net
        logger.info("LSTM (%s) trained — metrics: %s", self.config.task.value, self.metrics)
        return self.metrics

    def predict(self, X: np.ndarray) -> PredictionResult:
        if self.net is None:
            raise RuntimeError("LSTM not trained / loaded.")
        torch, _ = _get_torch()
        device = self._device or torch.device("cpu")

        self.net.eval()
        with torch.no_grad():
            output = self.net(torch.FloatTensor(X).to(device)).cpu().numpy()

        if self.config.task == ModelTask.CLASSIFICATION:
            prob_up = float(output[-1])
            signal = self._classify_signal(prob_up)
            return PredictionResult(prob_up=prob_up, signal=signal)
        else:
            return PredictionResult(predicted_return=float(output[-1]))

    def save(self, path: Path) -> None:
        torch, _ = _get_torch()
        path.mkdir(parents=True, exist_ok=True)

        if self.net is not None:
            torch.save(self.net.state_dict(), path / "model.pt")
            arch = {
                "input_size": self.net.lstm.input_size,
                "hidden_size": self.net.lstm.hidden_size,
                "num_layers": self.net.lstm.num_layers,
                "dropout": self.net.lstm.dropout,
                "task": self.config.task.value,
            }
            with open(path / "architecture.json", "w") as f:
                json.dump(arch, f, indent=2)

        if self.scaler is not None:
            joblib.dump(self.scaler, path / "scaler.joblib")
        if self.metrics:
            with open(path / "metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2, default=str)

    def load(self, path: Path) -> None:
        torch, _ = _get_torch()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(path / "architecture.json") as f:
            arch = json.load(f)

        task = ModelTask(arch["task"])
        self.config = ModelConfig(model_type="lstm", task=task, sequence_length=self.config.sequence_length)

        self.net = _LSTMNet.build(
            input_size=arch["input_size"],
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            dropout=arch.get("dropout", 0.0),
            output_size=1,
            task=task,
        ).to(self._device)

        state = torch.load(path / "model.pt", map_location=self._device, weights_only=True)
        self.net.load_state_dict(state)
        self.model = self.net

        scaler_path = path / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                self.metrics = json.load(f)

    def _scale_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        flat = X.reshape(-1, n_features)
        scaled = self.scaler.transform(flat)
        return scaled.reshape(n_samples, seq_len, n_features)
