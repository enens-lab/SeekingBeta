"""LSTM neural network model for classification and regression."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

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
        self.tf_model = None
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
        # TensorFlow/Keras branch
        if getattr(self, "tf_model", None) is not None:
            try:
                import tensorflow as tf
                self.tf_model.trainable = False
                preds = self.tf_model(tf.convert_to_tensor(X, dtype=tf.float32))
                output = preds.numpy()
            except Exception as exc:  # pragma: no cover - runtime safety
                raise RuntimeError(f"TF LSTM prediction failed: {exc}")

            if self.config.task == ModelTask.CLASSIFICATION:
                prob_up = float(output[-1])
                signal = self._classify_signal(prob_up)
                return PredictionResult(prob_up=prob_up, signal=signal)
            else:
                return PredictionResult(predicted_return=float(output[-1]))

        # PyTorch branch (legacy)
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
        # Prefer TensorFlow/Keras artifact if present
        keras_path = path / "model.keras"
        if keras_path.exists():
            try:
                import tensorflow as tf
                from tensorflow import keras
                import warnings
                warnings.filterwarnings('ignore')

                # Monkey-patch Dense layer to accept quantization_config
                original_dense_init = keras.layers.Dense.__init__

                def patched_dense_init(self, *args, **kwargs):
                    # Remove quantization_config if present
                    kwargs.pop('quantization_config', None)
                    original_dense_init(self, *args, **kwargs)

                keras.layers.Dense.__init__ = patched_dense_init

                # Also patch other layers that might have quantization_config
                for layer_class in [keras.layers.Conv1D, keras.layers.LSTM, keras.layers.BatchNormalization,
                                   keras.layers.Dropout, keras.layers.LayerNormalization]:
                    if hasattr(layer_class, '__init__'):
                        original_init = layer_class.__init__

                        def make_patched_init(orig):
                            def patched_init(self, *args, **kwargs):
                                kwargs.pop('quantization_config', None)
                                orig(self, *args, **kwargs)
                            return patched_init

                        layer_class.__init__ = make_patched_init(original_init)

                # Load model
                self.tf_model = tf.keras.models.load_model(str(keras_path), compile=False)
                self.model = self.tf_model
                logger.info("Successfully loaded Keras model from %s", keras_path)

            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning("Failed to load Keras model (%s), will try PyTorch fallback", exc)

        # If no TF model loaded, fall back to PyTorch
        if getattr(self, "tf_model", None) is None:
            torch, _ = _get_torch()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            arch_path = path / "architecture.json"
            if arch_path.exists():
                with open(arch_path) as f:
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

        # Scaler / metrics / feature columns are shared across frameworks
        scaler_path = path / "scaler.joblib"
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception:
                self.scaler = None

        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = {}

        # Load feature column ordering if present (exported from LSTM training repo)
        feat_path = path / "feature_columns.json"
        if feat_path.exists():
            try:
                with open(feat_path) as f:
                    self.feature_columns = json.load(f)
            except Exception:
                self.feature_columns = None

    def _scale_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        flat = X.reshape(-1, n_features)
        scaled = self.scaler.transform(flat)
        return scaled.reshape(n_samples, seq_len, n_features)

    def compute_features(self, raw: "pd.DataFrame", ticker: Optional[str] = None) -> "pd.DataFrame":
        """Compute feature DataFrame expected by the LSTM model.

        This computes features for both lstm_5d (38 features) and lstm_jackpot (30 features).
        Missing external data (sentiment, insider, fundamentals) are filled with sensible defaults.

        Args:
            raw: OHLCV DataFrame with columns Open, High, Low, Close, Volume

        Returns:
            DataFrame with columns matching `self.feature_columns` if available,
            otherwise falls back to a small set of technical features.
        """
        import pandas as pd
        import numpy as np

        df = raw.copy()
        df.index = pd.to_datetime(df.index)

        cols = getattr(self, "feature_columns", None)
        sentiment_score = 0.0
        sentiment_articles = 0
        sentiment_source = "default"
        if ticker and cols and any(c in cols for c in ("sentiment", "sentiment_change", "num_articles")):
            try:
                from features.sentiment import get_sentiment_snapshot

                snapshot = get_sentiment_snapshot(ticker)
                sentiment_score = float(snapshot.score)
                sentiment_articles = int(snapshot.num_articles)
                sentiment_source = snapshot.source
            except Exception as exc:
                logger.debug("Sentiment feature fetch failed for %s: %s", ticker, exc)

        if not cols:
            # Fallback: compute basic features using existing helpers
            feat = pd.DataFrame(index=df.index)
            feat["ret_1d"] = df["Close"].pct_change(1)
            feat["rsi_14"] = (df["Close"].diff().rolling(14).apply(lambda x: (x[x>0].mean() if len(x[x>0])>0 else 0)/(abs(x[x<0]).mean() if len(x[x<0])>0 else 1)))
            return feat.dropna()

        out = pd.DataFrame(index=df.index)

        # Helper short-hands
        close = df["Close"]
        openp = df["Open"]
        high = df["High"]
        low = df["Low"]
        vol = df.get("Volume", pd.Series(0, index=df.index))

        # Detect which model we're computing for
        is_jackpot = any(c.startswith("BB") and "_" in c for c in cols)

        if is_jackpot:
            # Jackpot model features (uses pandas_ta)
            try:
                import pandas_ta as ta
            except ImportError:
                logger.warning("pandas_ta not installed, using fallback Bollinger Bands calculation")
                ta = None

            # Bollinger Bands using pandas_ta (BBB, BBL, BBM, BBP, BBU)
            if ta is not None:
                bb = ta.bbands(close, length=20, std=2.0)
                if bb is not None:
                    out["BBL_20_2.0_2.0"] = bb[f"BBL_20_2.0"]
                    out["BBM_20_2.0_2.0"] = bb[f"BBM_20_2.0"]
                    out["BBU_20_2.0_2.0"] = bb[f"BBU_20_2.0"]
                    out["BBB_20_2.0_2.0"] = bb[f"BBB_20_2.0"]
                    out["BBP_20_2.0_2.0"] = bb[f"BBP_20_2.0"]
            else:
                # Fallback manual calculation
                ma20 = close.rolling(20).mean()
                sd20 = close.rolling(20).std()
                out["BBL_20_2.0_2.0"] = ma20 - 2 * sd20
                out["BBM_20_2.0_2.0"] = ma20
                out["BBU_20_2.0_2.0"] = ma20 + 2 * sd20
                out["BBB_20_2.0_2.0"] = 4 * sd20 / ma20  # width
                out["BBP_20_2.0_2.0"] = (close - (ma20 - 2 * sd20)) / (4 * sd20)  # percent

            # MACD using pandas_ta
            if ta is not None:
                macd_data = ta.macd(close, fast=12, slow=26, signal=9)
                if macd_data is not None:
                    out["MACD_12_26_9"] = macd_data["MACD_12_26_9"]
                    out["MACDh_12_26_9"] = macd_data["MACDh_12_26_9"]
                    out["MACDs_12_26_9"] = macd_data["MACDs_12_26_9"]
            else:
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                out["MACD_12_26_9"] = macd
                out["MACDs_12_26_9"] = macd.ewm(span=9, adjust=False).mean()
                out["MACDh_12_26_9"] = macd - out["MACDs_12_26_9"]

            # Log returns
            out["YesterdayCloseLogR"] = np.log(close.shift(1) / close.shift(2)).replace([np.inf, -np.inf], 0).fillna(0)
            out["YesterdayVolumeLogR"] = np.log(vol.shift(1) / vol.shift(2)).replace([np.inf, -np.inf], 0).fillna(0)

            # ATR (Average True Range)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            out["atr"] = tr.rolling(14).mean()

            # Bollinger Band width
            out["bb_width"] = (out["BBU_20_2.0_2.0"] - out["BBL_20_2.0_2.0"]) / out["BBM_20_2.0_2.0"]

            # Distance to 52-week high
            out["high_52"] = high.rolling(252, min_periods=1).max()
            out["dist_to_high"] = (out["high_52"] - close) / out["high_52"]

            # Momentum
            out["momentum_5d"] = close.pct_change(5)

            # RSI
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            roll_up = up.rolling(14).mean()
            roll_down = down.rolling(14).mean().replace(0, 1e-9)
            out["rsi"] = 100 - (100 / (1 + roll_up / roll_down))

            # Rate of Change
            out["roc_10"] = close.pct_change(10)

            # Relative Volume
            vol_sma = vol.rolling(20).mean()
            out["vol_sma_20"] = vol_sma
            out["rvol"] = vol / (vol_sma + 1e-9)

            # Volatility
            out["volatility_20d"] = close.pct_change().rolling(20).std()

            # External data placeholders
            out["insider_shares"] = 0.0
            out["insider_amount"] = 0.0
            out["insider_buy_flag"] = 0
            out["insider_own"] = 0.0
            out["inst_own"] = 0.0
            out["market_cap"] = 0.0
            out["num_articles"] = sentiment_articles
            out["pe_ratio"] = 0.0
            out["sentiment"] = sentiment_score
            out["short_float"] = 0.0

        else:
            # lstm_5d (production) model features
            out["YesterdayClose"] = close.shift(1)
            out["YesterdayOpenLogR"] = np.log(openp.shift(1) / close.shift(2)).replace([np.inf, -np.inf], 0).fillna(0)
            out["YesterdayHighLogR"] = np.log(high.shift(1) / close.shift(2)).replace([np.inf, -np.inf], 0).fillna(0)
            out["YesterdayLowLogR"] = np.log(low.shift(1) / close.shift(2)).replace([np.inf, -np.inf], 0).fillna(0)
            out["YesterdayVolumeLogR"] = np.log(vol.shift(1) / vol.shift(2)).replace([np.inf, -np.inf], 0).fillna(0)
            out["YesterdayCloseLogR"] = np.log(close.shift(1) / close.shift(2)).replace([np.inf, -np.inf], 0).fillna(0)

            out["MA10"] = close.rolling(10).mean()
            out["MA20"] = close.rolling(20).mean()
            out["MA30"] = close.rolling(30).mean()
            out["DayOfWeek"] = df.index.dayofweek
            out["DayOfMonth"] = df.index.day
            out["MonthNumber"] = df.index.month
            out["EMA10"] = close.ewm(span=10, adjust=False).mean()
            out["EMA30"] = close.ewm(span=30, adjust=False).mean()

            # Technical indicators
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            roll_up = up.rolling(14).mean()
            roll_down = down.rolling(14).mean().replace(0, 1e-9)
            out["RSI"] = 100 - (100 / (1 + roll_up / roll_down))

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            out["MACD"] = macd
            out["MACD_Signal"] = macd.ewm(span=9, adjust=False).mean()

            # Bollinger bands
            ma20 = close.rolling(20).mean()
            sd20 = close.rolling(20).std()
            out["BollingerUpper"] = ma20 + 2 * sd20
            out["BollingerLower"] = ma20 - 2 * sd20

            out["Volatility_10"] = close.pct_change().rolling(10).std()
            out["Volatility_20"] = close.pct_change().rolling(20).std()
            out["Volatility_30"] = close.pct_change().rolling(30).std()
            out["volatility_5d"] = close.pct_change().rolling(5).std()
            out["volatility_20d"] = out["Volatility_20"]

            # OBV
            obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
            out["OBV"] = obv

            # ZScore of returns
            out["ZScore"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)

            out["overnight_gap"] = (openp - close.shift(1)) / close.shift(1)
            out["abnormal_vol"] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)
            out["momentum_5d"] = close.pct_change(5)
            out["momentum_20d"] = close.pct_change(20)
            out["skew_5d"] = close.pct_change().rolling(5).skew()
            out["intraday_range"] = (high - low) / (close + 1e-9)

            # External data placeholders
            out["insider_shares"] = 0.0
            out["insider_amount"] = 0.0
            out["insider_buy_flag"] = 0
            out["sentiment"] = sentiment_score
            out["num_articles"] = sentiment_articles
            out["sentiment_change"] = 0.0

        # Ensure all expected columns present and ordered
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0

        out = out[cols]
        out.attrs["sentiment"] = {
            "score": sentiment_score,
            "num_articles": sentiment_articles,
            "source": sentiment_source,
        }
        return out.dropna()
