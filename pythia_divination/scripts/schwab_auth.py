#!/usr/bin/env python3
"""Headless Schwab OAuth (re-)authorization — for the EC2 box.

Schwab's refresh token expires every 7 days, so this must be re-run roughly
weekly to keep QUANT_OPTIONS_SOURCE=schwab serving real-greeks option chains.
(When the token is missing/expired, lstm_quant silently falls back to yfinance —
predictions keep working, just without Schwab greeks.)

Unlike a desktop flow, this uses schwab-py's MANUAL flow: it needs NO browser and
NO open port on the server. It prints an authorize URL; you open that URL in your
OWN browser, log in + approve, get redirected to the callback (which won't load —
that's fine), then paste the FULL redirected URL back into this prompt.

Run it inside the divination container so it sees the same creds + token path the
API uses, and writes the token to the mounted artifacts volume (persists across
rebuilds):

    ssh -i Pythia.pem ec2-user@<box>
    cd /home/ec2-user/seekingbeta
    docker compose exec divination-api python scripts/schwab_auth.py
    #            ^ no -T: this needs an interactive TTY to paste the redirect URL

Env (from docker-compose):
    SCHWAB_APP_KEY, SCHWAB_APP_SECRET  — Schwab developer app credentials (required)
    SCHWAB_CALLBACK_URL                — must match the app's registered callback
                                         (default https://127.0.0.1:8182)
    SCHWAB_TOKEN_PATH                  — where to write the token
                                         (default /app/artifacts/schwab_token.json)
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # harmless in-container (no .env); picks up a local .env if run elsewhere
except Exception:
    pass

APP_KEY = os.environ.get("SCHWAB_APP_KEY", "")
APP_SECRET = os.environ.get("SCHWAB_APP_SECRET", "")
CALLBACK_URL = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
TOKEN_PATH = os.environ.get("SCHWAB_TOKEN_PATH", "/app/artifacts/schwab_token.json")

if not APP_KEY or not APP_SECRET:
    sys.exit("ERROR: SCHWAB_APP_KEY and SCHWAB_APP_SECRET must be set in the environment (.env on the box).")

try:
    import schwab
except ImportError:
    sys.exit("ERROR: schwab-py is not installed. Rebuild the image after adding schwab-py to requirements.txt.")

Path(TOKEN_PATH).parent.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Schwab manual OAuth flow")
print(f"  callback : {CALLBACK_URL}   (must match the app's registered callback)")
print(f"  token    : {TOKEN_PATH}")
print("=" * 70)
print("Steps:")
print("  1. Open the URL printed below in YOUR browser.")
print("  2. Log in to Schwab and approve the app.")
print("  3. You'll be redirected to the callback URL — the page will fail to load.")
print("     That is EXPECTED. Copy the FULL URL from the address bar.")
print("  4. Paste that full redirected URL back here when prompted.")
print()

# Manual flow: prints the auth URL, reads the redirected URL from stdin. No
# local browser, no listening socket — works fine over SSH.
client = schwab.auth.client_from_manual_flow(
    api_key=APP_KEY,
    app_secret=APP_SECRET,
    callback_url=CALLBACK_URL,
    token_path=TOKEN_PATH,
)

print()
print(f"Token written to: {TOKEN_PATH}")

# Verify the token actually works by hitting a lightweight authed endpoint.
try:
    resp = client.get_account_numbers()
    if getattr(resp, "status_code", None) == 200:
        accts = resp.json()
        print(f"Verified — Schwab returned {len(accts)} account(s). Auth is good for ~7 days.")
    else:
        print(f"WARNING: token saved but verification call returned HTTP {getattr(resp, 'status_code', '?')}.")
except Exception as exc:
    print(f"WARNING: token saved but verification call failed: {exc}")

print()
print("Next: set QUANT_OPTIONS_SOURCE=schwab in the box .env and restart divination-api,")
print("then check /predict/lstm_quant/AAPL?force_refresh=true shows options_source=schwab.")
