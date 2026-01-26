import os, requests
from typing import Optional, Dict, Any, List
from config.settings import settings

PAPER_BASE = "https://paper-api.alpaca.markets"
HTTP_TIMEOUT = 20

def _hdrs():
    key = settings.alpaca.key_id
    sec = settings.alpaca.secret_key
    if not key or not sec:
        raise RuntimeError("Alpaca keys are not configured (check .env)")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": sec,
        "Accept": "application/json",
        "User-Agent": "pythia-api-mod/1.0"
    }

def account() -> Dict[str, Any]:
    r = requests.get(f"{PAPER_BASE}/v2/account", headers=_hdrs(), timeout=HTTP_TIMEOUT)
    r.raise_for_status(); return r.json()

def get_positions() -> List[Dict[str, Any]]:
    r = requests.get(f"{PAPER_BASE}/v2/positions", headers=_hdrs(), timeout=HTTP_TIMEOUT)
    r.raise_for_status(); return r.json()

def get_open_orders() -> List[Dict[str, Any]]:
    r = requests.get(f"{PAPER_BASE}/v2/orders", params={"status":"open","limit":500},
                     headers=_hdrs(), timeout=HTTP_TIMEOUT)
    r.raise_for_status(); return r.json()

def cancel_order(order_id: str):
    r = requests.delete(f"{PAPER_BASE}/v2/orders/{order_id}", headers=_hdrs(), timeout=HTTP_TIMEOUT)
    r.raise_for_status(); return {"status":"ok"}

def place_order(symbol: str, side: str, qty: int, type="market", tif="day") -> Dict[str, Any]:
    payload = {"symbol": symbol, "side": side, "qty": qty, "type": type, "time_in_force": tif}
    r = requests.post(f"{PAPER_BASE}/v2/orders", json=payload, headers=_hdrs(), timeout=HTTP_TIMEOUT)
    r.raise_for_status(); return r.json()
