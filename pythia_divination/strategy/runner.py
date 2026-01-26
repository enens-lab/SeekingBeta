from typing import Dict, List, Tuple
import pandas as pd
from data.fetch import fetch_ohlcv
from features.technical import make_features
from models.utils import load_artifacts
from broker.alpaca import place_order, get_positions
from storage.postgres import insert_trade

def _sizer(budget: float, picks: List[Tuple[str,float,float]]) -> List[Tuple[str,int]]:
    if not picks: return []
    each = budget / len(picks)
    out = []
    for sym, _, px in picks:
        qty = int(each // px)
        if qty > 0: out.append((sym, qty))
    return out

def run_once(universe: List[str], threshold: float, budget: float, data_source: str="alpaca", persist: bool = True) -> Dict:
    MODEL, SCALER, FEATS, _ = load_artifacts()
    preds = []
    for t in universe:
        df = fetch_ohlcv(t, period="120d", data_source=data_source, timeframe="1Day")
        feat = make_features(df)
        Xs = SCALER.transform(feat[FEATS].values)
        prob = float(MODEL.predict_proba(Xs)[-1,1])
        last_close = float(df["Close"].iloc[-1])
        preds.append((t, prob, last_close))
    preds.sort(key=lambda x: x[1], reverse=True)
    chosen = [x for x in preds if x[1] >= threshold]

    held = {p["symbol"]: int(float(p["qty"])) for p in get_positions()}
    plan = _sizer(budget, chosen)

    orders = []
    chosen_syms = {s for s,_,_ in chosen}
    # sells
    for sym, qty in held.items():
        if sym not in chosen_syms and qty>0:
            orders.append({"symbol": sym, "side":"sell", "qty": qty, "reason": "rebalance"})
    # buys
    for sym, target_qty in plan:
        cur = held.get(sym, 0)
        add = max(0, target_qty - cur)
        if add>0:
            orders.append({"symbol": sym, "side":"buy", "qty": add, "reason": "entered_on_signal"})

    results = []
    for o in orders:
        try:
            res = place_order(o["symbol"], o["side"], o["qty"])
            results.append({"request": o, "response": res})
            if persist:
                insert_trade(
                    ts=None,
                    symbol=o["symbol"],
                    side=o["side"],
                    qty=o["qty"],
                    price=None,
                    status=res.get("status"),
                    provider_order_id=res.get("id"),
                    reason=o.get("reason"),
                    raw_json=res
                )
        except Exception as e:
            results.append({"request": o, "error": str(e)})
            if persist:
                insert_trade(
                    ts=None,
                    symbol=o["symbol"],
                    side=o["side"],
                    qty=o["qty"],
                    price=None,
                    status="error",
                    provider_order_id=None,
                    reason=o.get("reason"),
                    raw_json={"error": str(e)}
                )

    return {"preds": preds, "chosen": chosen, "orders": results}
