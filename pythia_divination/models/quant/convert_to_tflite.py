"""Convert the torch options-enriched LSTM (lstm_5d / lstm_jackpot) to TFLite for
on-device (iOS) inference, with numerical-parity verification.

The model (Conv1d -> LSTM -> MultiheadAttention -> LayerNorm -> LSTM -> head) can't
be hand-reconstructed in Keras safely, so we go torch -> ONNX -> TFLite. Because
torch and a working tf/onnx2tf stack are awkward to install in one env on Apple
Silicon, the pipeline runs in two stages in two interpreters:

  # stage 1 — system python (has torch): torch -> ONNX (+ sigmoid head) + golden vectors
  python models/quant/convert_to_tflite.py export

  # stage 2 — a torch-free tf/onnx2tf venv: ONNX -> TFLite, verify vs golden, dump ops
  /tmp/sb_conv/bin/python models/quant/convert_to_tflite.py convert

Artifacts land in OUT_DIR (default /tmp/sb_tflite): <model>.onnx, golden_<model>.npz,
<model>.tflite. The exported graph applies sigmoid, so the TFLite output is the
positive-class probability in [0,1] (matches the metadata `outputKind`).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

DIV_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = DIV_ROOT / "artifacts"
MODELS = ["lstm_5d", "lstm_jackpot"]
OUT_DIR = Path(os.environ.get("SB_TFLITE_OUT", "/tmp/sb_tflite"))
SEQ_LEN = 60
OPSET = 17
N_GOLDEN = 8
SEED = 7


def _art(model: str) -> Path:
    return ARTIFACTS_ROOT / model / "torch"


# ───────────────────────── stage 1: torch -> ONNX ─────────────────────────

def export() -> None:
    sys.path.insert(0, str(DIV_ROOT))
    import torch
    import torch.nn as nn
    from models.quant.lstm_quant import LSTMBaselineConfig, _build_net

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    class WithSigmoid(nn.Module):
        """Wrap so the exported graph emits P(class=1) in [0,1] directly."""

        def __init__(self, net: nn.Module):
            super().__init__()
            self.net = net

        def forward(self, x):
            return torch.sigmoid(self.net(x))

    for model in MODELS:
        ckpt = torch.load(_art(model) / "model.pt", map_location="cpu", weights_only=False)
        cfg = LSTMBaselineConfig(**{k: v for k, v in (ckpt.get("config") or {}).items()
                                    if k in LSTMBaselineConfig.__dataclass_fields__})
        net = _build_net(cfg)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        wrapped = WithSigmoid(net).eval()

        dummy = torch.zeros(1, cfg.seq_len, cfg.num_features, dtype=torch.float32)
        onnx_path = OUT_DIR / f"{model}.onnx"
        torch.onnx.export(
            wrapped, dummy, str(onnx_path),
            input_names=["input"], output_names=["prob"],
            opset_version=OPSET, do_constant_folding=True,
        )

        # Golden vectors: random standardized inputs (post-scaler distribution) + torch refs.
        rng = np.random.default_rng(SEED)
        xs = rng.standard_normal((N_GOLDEN, cfg.seq_len, cfg.num_features)).astype(np.float32)
        with torch.no_grad():
            probs = wrapped(torch.from_numpy(xs)).numpy().reshape(-1)
            logits = net(torch.from_numpy(xs)).numpy().reshape(-1)
        np.savez(OUT_DIR / f"golden_{model}.npz", xs=xs, probs=probs, logits=logits)
        print(f"[export] {model}: ONNX={onnx_path.name} params={ckpt.get('n_params')} "
              f"prob[min={probs.min():.4f} max={probs.max():.4f}] -> golden_{model}.npz")


# ───────────────────────── stage 2: ONNX -> TFLite ─────────────────────────

def _make_interpreter(model_path: Path):
    """Use the LiteRT runtime (== the on-device TFLite runtime). A successful
    allocate/invoke here means builtin-only ops; a model needing Flex ops would
    raise, so this doubles as the Flex check."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except Exception:
        from tensorflow.lite import Interpreter  # type: ignore
    return Interpreter(model_path=str(model_path))


def _tflite_op_names(model_path: Path) -> list[str]:
    interp = _make_interpreter(model_path)
    interp.allocate_tensors()
    try:
        return sorted({d["op_name"] for d in interp._get_ops_details()})  # type: ignore[attr-defined]
    except Exception:
        return ["<op-introspection-unavailable>"]


def _run_tflite(model_path: Path, xs: np.ndarray) -> np.ndarray:
    interp = _make_interpreter(model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    want = list(inp["shape"])
    print(f"           tflite input shape={want} output shape={list(out['shape'])}")
    preds = []
    for i in range(xs.shape[0]):
        x = xs[i:i + 1].astype(np.float32)  # (1, 60, 48)
        if len(want) == 3 and want[1] == x.shape[2] and want[2] == x.shape[1]:
            x = np.transpose(x, (0, 2, 1))  # adapt if converter kept channel-first
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        preds.append(float(np.asarray(interp.get_tensor(out["index"])).reshape(-1)[0]))
    return np.asarray(preds, dtype=np.float64)


def convert() -> None:
    import onnxruntime as ort
    import onnx2tf

    summary = {}
    for model in MODELS:
        onnx_path = OUT_DIR / f"{model}.onnx"
        golden = np.load(OUT_DIR / f"golden_{model}.npz")
        xs, ref_probs = golden["xs"], golden["probs"].astype(np.float64)

        # (a) ONNX export fidelity: onnxruntime vs torch golden
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_probs = np.asarray(
            [sess.run(None, {"input": xs[i:i + 1]})[0].reshape(-1)[0] for i in range(xs.shape[0])],
            dtype=np.float64,
        )
        onnx_max_err = float(np.max(np.abs(ort_probs - ref_probs)))

        # (b) Simplify first: constant-fold the static Expand/Reshape that
        # torch's MultiheadAttention export emits (onnx2tf's Expand handler
        # trips on it). With fixed batch=1/seq_len=60 every shape is static.
        conv_input = onnx_path
        try:
            import onnxslim
            slim_path = OUT_DIR / f"{model}_slim.onnx"
            onnxslim.slim(str(onnx_path), str(slim_path))
            conv_input = slim_path
            print(f"[convert] {model}: onnxslim simplified -> {slim_path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[convert] {model}: onnxslim unavailable/failed ({exc}); using raw onnx")

        # (c) ONNX -> TFLite
        out_folder = OUT_DIR / f"{model}_tf"
        onnx2tf.convert(
            input_onnx_file_path=str(conv_input),
            output_folder_path=str(out_folder),
            copy_onnx_input_output_names_to_tflite=True,
            keep_shape_absolutely_input_names=["input"],  # keep tflite input (1,60,48)
            non_verbose=True,
        )
        # onnx2tf emits several variants; the plain float32 model is what we ship.
        cand = sorted(out_folder.glob("*float32.tflite")) or sorted(out_folder.glob("*.tflite"))
        if not cand:
            raise RuntimeError(f"{model}: no tflite produced in {out_folder}")
        tflite_src = cand[0]
        tflite_dst = OUT_DIR / f"{model}.tflite"
        tflite_dst.write_bytes(tflite_src.read_bytes())

        # (c) TFLite fidelity vs torch golden + op-set inspection (Flex => not iOS-runnable)
        tfl_probs = _run_tflite(tflite_dst, xs)
        tflite_max_err = float(np.max(np.abs(tfl_probs - ref_probs)))
        ops = _tflite_op_names(tflite_dst)
        flex = [o for o in ops if o.lower().startswith("flex")]

        summary[model] = {
            "onnx_vs_torch_max_err": onnx_max_err,
            "tflite_vs_torch_max_err": tflite_max_err,
            "tflite_bytes": tflite_dst.stat().st_size,
            "ops": ops,
            "flex_ops": flex,
        }
        print(f"[convert] {model}: onnx_err={onnx_max_err:.2e} tflite_err={tflite_max_err:.2e} "
              f"bytes={tflite_dst.stat().st_size} flex={flex or 'NONE'}")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    worst = max(s["tflite_vs_torch_max_err"] for s in summary.values())
    any_flex = any(s["flex_ops"] for s in summary.values())
    print(f"\nworst tflite-vs-torch err: {worst:.2e}  | any Flex ops: {any_flex}")
    if any_flex:
        print("!! Flex/SELECT_TF_OPS required — NOT runnable by the app's builtin-only TFLite runtime.")
    if worst > 1e-3:
        print("!! parity error exceeds 1e-3 — investigate before shipping.")


# ───────────────────────── stage 3: emit iOS metadata ─────────────────────────

# Curated display metadata (description / horizon / cutoffs / backtest) — source of
# truth for the on-device model cards. featureColumns + scaler come from the artifacts.
_DISPLAY = {
    "lstm_5d": {
        "description": "5-Day Options-Flow Model (options-enriched LSTM)",
        "horizon": "5d", "target_return": ">2%",
        "signal_cutoffs": {"high": 0.6, "medium": 0.4},
        "backtest": {"total_return_pct": 90.8, "sharpe": 0.92, "alpha": 0.082,
                     "benchmark_return_pct": 60.7, "window": "2024-01 to 2026-05"},
    },
    "lstm_jackpot": {
        "description": "20-Day Jackpot Model (options-enriched, high-beta)",
        "horizon": "20d", "target_return": ">20%",
        "signal_cutoffs": {"high": 0.55, "medium": 0.45},
        "backtest": {"total_return_pct": 81.2, "sharpe": 0.77, "alpha": 0.045,
                     "benchmark_return_pct": 60.7, "window": "2024-01 to 2026-05"},
    },
}


def emit_metadata() -> None:
    """Write the iOS-side 48-feature metadata JSON (featureColumns + 48-dim scaler +
    curated display block) into OUT_DIR for both models. Run with a python that has
    joblib + sklearn (the .venv)."""
    import warnings
    import joblib

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        art = _art(model)
        with open(art / "feature_columns.json") as f:
            feature_cols = json.load(f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = joblib.load(art / "scaler.joblib")
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        var = np.asarray(getattr(scaler, "var_", scale ** 2), dtype=np.float64)
        assert len(feature_cols) == len(mean) == 48, f"{model}: expected 48 features"

        disp = _DISPLAY[model]
        meta = {
            "schemaVersion": 1,
            "model": model,
            "sequenceLength": SEQ_LEN,
            "featureColumns": feature_cols,
            "inputShape": [1, SEQ_LEN, len(feature_cols)],
            "outputKind": "positive_class_probability",
            "defaults": {"sentimentScore": 0.0, "sentimentArticles": 0, "sentimentSource": "default"},
            "metadata": {
                "description": disp["description"],
                "horizon": disp["horizon"],
                "target_return": disp["target_return"],
                "signal_cutoffs": disp["signal_cutoffs"],
                "backtest": disp["backtest"],
            },
            "scaler": {"mean": mean.tolist(), "scale": scale.tolist(), "variance": var.tolist()},
        }
        out = OUT_DIR / f"{model}_metadata.json"
        out.write_text(json.dumps(meta, indent=2))
        print(f"[metadata] {model}: {len(feature_cols)} features, scaler[{len(mean)}] -> {out.name}")


# ───────────────────────── stage 4: parity golden fixture ─────────────────────────

def golden() -> None:
    """Emit a deterministic fixture for the Swift parity test: synthetic OHLCV + a synthetic
    option chain, the server's expected 48-feature last row (options zero-filled via symbol=None),
    and the expected 10 options features (BS greeks + extract_from_frames). Run with the .venv
    (needs pandas + the quant modules). Epochs are midnight-UTC so the Swift UTC day-count matches."""
    sys.path.insert(0, str(DIV_ROOT))
    import pandas as pd
    from datetime import timedelta
    from models.quant.feature_engine import engineer_quant_features
    from models.quant.options_live import bs_delta_gamma
    from models.quant.options_features import extract_from_frames, FEATURE_COLUMNS as OPT_COLS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 140
    start = pd.Timestamp("2025-01-02", tz="UTC")
    dates = [start + timedelta(days=i) for i in range(n)]
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.015, n))
    openp = close * (1.0 + rng.normal(0, 0.004, n))
    high = np.maximum(openp, close) * (1.0 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(openp, close) * (1.0 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    ohlcv = pd.DataFrame({"date": [d.tz_localize(None) for d in dates],
                          "open": openp, "high": high, "low": low, "close": close, "volume": vol})

    feat_cols = json.load(open(_art("lstm_5d") / "feature_columns.json"))
    feats = engineer_quant_features(ohlcv.copy(), feat_cols, symbol=None)  # symbol=None -> options 0
    expected_last = [float(x) for x in feats.iloc[-1].to_numpy(dtype=float)]

    asof = pd.Timestamp("2025-05-21", tz="UTC")
    asof_naive = asof.tz_localize(None)
    spot = 100.0
    rows = []
    for dte in (10, 14, 21, 38, 60, 120, 200):
        exp = (asof + timedelta(days=dte)).strftime("%Y-%m-%d")
        for strike in range(80, 121, 2):
            for right in ("CALL", "PUT"):
                iv = 0.25 + 0.0015 * abs(strike - 100) + (0.012 if right == "PUT" else 0.0)
                rows.append({"expiration": exp, "strike": float(strike), "right": right,
                             "implied_vol": iv, "open_interest": float(100 + (strike % 7) * 11),
                             "volume": float(40 + (strike % 5) * 9), "bid": 1.25, "underlying_price": spot})
    cdf = pd.DataFrame(rows)
    dte_arr = (pd.to_datetime(cdf["expiration"]) - asof_naive).dt.days.to_numpy()
    delta, gamma = bs_delta_gamma(spot, cdf["strike"].to_numpy(float), dte_arr,
                                  cdf["implied_vol"].to_numpy(float), cdf["right"].to_numpy())
    cdf["delta"], cdf["gamma"], cdf["iv_error"], cdf["symbol"] = delta, gamma, 0.0, "TEST"
    greeks = cdf[["symbol", "expiration", "strike", "right", "implied_vol", "delta", "gamma",
                  "volume", "bid", "underlying_price", "iv_error"]]
    oi = cdf[["expiration", "strike", "right", "open_interest"]]
    opt = extract_from_frames(greeks, oi, asof_naive.strftime("%Y-%m-%d"))

    def epoch_utc(ts) -> int:
        return int(pd.Timestamp(ts, tz="UTC").timestamp())

    payload = {
        "feature_columns": feat_cols,
        "expected_features_last": expected_last,
        "bars": [{"epoch": int(d.timestamp()), "open": float(o), "high": float(h),
                  "low": float(l), "close": float(c), "volume": float(v)}
                 for d, o, h, l, c, v in zip(dates, openp, high, low, close, vol)],
        "spot": spot,
        "asof_epoch": int(asof.timestamp()),
        "chain": [{"strike": float(r.strike), "iv": float(r.implied_vol), "bid": float(r.bid),
                   "volume": float(r.volume), "oi": float(r.open_interest),
                   "isCall": (r.right == "CALL"), "exp_epoch": epoch_utc(r.expiration)}
                  for r in cdf.itertuples()],
        "expected_options": {k: float(opt[k]) for k in OPT_COLS},
    }
    (OUT_DIR / "golden_parity.json").write_text(json.dumps(payload))
    print(f"[golden] {len(payload['bars'])} bars, {len(payload['chain'])} contracts -> golden_parity.json")
    print("  expected options:", {k: round(v, 5) for k, v in payload["expected_options"].items()})


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "export"
    if stage == "export":
        export()
    elif stage == "convert":
        convert()
    elif stage == "metadata":
        emit_metadata()
    elif stage == "golden":
        golden()
    else:
        sys.exit(f"unknown stage '{stage}' (use: export | convert | metadata | golden)")
