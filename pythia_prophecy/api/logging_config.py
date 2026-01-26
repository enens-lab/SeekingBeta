"""
Logging configuration for Pythia Prophecy backend.
Configures structured logging to both console and file.
"""
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Log directory - same level as data directory
LOG_DIR = Path(__file__).parent.parent / "logs"

# Log level from environment (default: INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Log file settings
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5  # Keep 5 rotated files


def setup_logging(log_level: str = None) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Root logger for the application
    """
    level = getattr(logging, (log_level or LOG_LEVEL).upper(), logging.INFO)

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create log file path with date
    log_file = LOG_DIR / f"pythia_{datetime.now().strftime('%Y-%m-%d')}.log"

    # Create formatters
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Configure root logger
    root_logger = logging.getLogger("pythia")
    root_logger.setLevel(level)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Also create a separate access log for HTTP requests
    access_log_file = LOG_DIR / f"access_{datetime.now().strftime('%Y-%m-%d')}.log"
    access_logger = logging.getLogger("pythia.access")
    access_logger.setLevel(logging.INFO)

    access_file_handler = RotatingFileHandler(
        access_log_file,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    access_formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    access_file_handler.setFormatter(access_formatter)
    access_logger.addHandler(access_file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (will be prefixed with 'pythia.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"pythia.{name}")
