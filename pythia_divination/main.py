"""
Single-file launcher for the FastAPI app.

Usage:
  python main.py                       # 127.0.0.1:8000
  python main.py --host 0.0.0.0 --port 8080
  python main.py --reload              # dev hot-reload (needs source path string)
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path (so imports like api.service work when run from anywhere)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env before bringing up anything else
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ROOT / ".env")
except Exception:
    pass

import uvicorn

# Import centralized settings & app pieces
from config.settings import settings
from storage.postgres import init_db  # async
# For non-reload mode we import the app object directly
from api.service import app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Pythia API web server")
    p.add_argument("--host", default=os.getenv("UVICORN_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("UVICORN_PORT", "8000")))
    p.add_argument("--log-level", default=os.getenv("UVICORN_LOG_LEVEL", "info"),
                   choices=["critical", "error", "warning", "info", "debug", "trace"])
    p.add_argument("--reload", action="store_true",
                   help="Enable dev autoreload (uses 'api.service:app' import path).")
    return p.parse_args()


def main():
    args = parse_args()

    # Fast fail if DB is unreachable (migrations are handled separately)
    # When using reload, uvicorn spawns a new process, so skip this check to avoid double init.
    if not args.reload:
        try:
            asyncio.run(init_db())
        except Exception as e:
            logging.getLogger("startup").error("Database connectivity check failed: %s", e)
            sys.exit(2)

    # When --reload is set, uvicorn requires an import string target, not the in-memory app object.
    if args.reload:
        uvicorn.run(
            "api.service:app",
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            reload=True,
            reload_dirs=[str(ROOT)],  # watch the repo root
        )
    else:
        # Non-reload: pass the app object directly
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    # On Windows + Python 3.8–3.11, selecting the proactor loop can help with async I/O.
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            pass
    main()
