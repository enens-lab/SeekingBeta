"""
Pythia Claude - Main entry point
"""
import argparse
import sys
from pathlib import Path

# Ensure pythia_claude's own modules take precedence
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Add sibling pythia to path for ML predictions (AFTER current dir)
PYTHIA_PATH = Path(__file__).parent.parent / "pythia"
if PYTHIA_PATH.exists() and str(PYTHIA_PATH) not in sys.path:
    sys.path.append(str(PYTHIA_PATH))

import uvicorn

from api.logging_config import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Pythia Claude API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    # Initialize logging
    setup_logging(args.log_level)
    logger = get_logger("main")
    logger.info(f"Starting Pythia API server on {args.host}:{args.port}")
    logger.info(f"Log level: {args.log_level.upper()}")

    uvicorn.run(
        "api.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
