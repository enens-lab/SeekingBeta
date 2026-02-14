"""Gradient Boosting model for classification and regression."""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from models.base import BaseModel, ModelConfig, ModelTask, PredictionResult

logger = logging.getLogger(__name__)

_DEFAULT_CLF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
}

_DEFAULT_REG_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
}


class GradientBoostingModel(BaseModel):
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        val_ratio = kwargs.get("val_ratio", 0.2)
        split = int(len(X) * (1 - val_ratio))

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        if self.config.task == ModelTask.CLASSIFICATION:
            params = {**_DEFAULT_CLF_PARAMS, **self.config.hyperparams}
            self.model = GradientBoostingClassifier(**params)
            self.model.fit(X_train_s, y_train)

            val_proba = self.model.predict_proba(X_val_s)[:, 1]
            val_pred = self.model.predict(X_val_s)

            self.metrics = {
                "accuracy": float(accuracy_score(y_val, val_pred)),
                "auc": float(roc_auc_score(y_val, val_proba)),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "task": self.config.task.value,
                "model": "gradient_boosting",
            }
        else:
            params = {**_DEFAULT_REG_PARAMS, **self.config.hyperparams}
            self.model = GradientBoostingRegressor(**params)
            self.model.fit(X_train_s, y_train)

            val_pred = self.model.predict(X_val_s)
            self.metrics = {
                "mse": float(mean_squared_error(y_val, val_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
                "r2": float(r2_score(y_val, val_pred)),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "task": self.config.task.value,
                "model": "gradient_boosting",
            }

        logger.info("Gradient Boosting (%s) trained — metrics: %s", self.config.task.value, self.metrics)
        return self.metrics

    def predict(self, X: np.ndarray) -> PredictionResult:
        if self.model is None:
            raise RuntimeError("Model not trained / loaded.")

        if self.config.task == ModelTask.CLASSIFICATION:
            proba = self.model.predict_proba(X)[:, 1]
            prob_up = float(proba[-1])
            signal = self._classify_signal(prob_up)
            return PredictionResult(prob_up=prob_up, signal=signal, probabilities=proba)
        else:
            preds = self.model.predict(X)
            return PredictionResult(predicted_return=float(preds[-1]))
