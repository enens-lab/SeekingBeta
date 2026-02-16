"""
Logging configuration for Pythia Prophecy backend.
Configures structured JSON logging to both console and file.
"""
import logging
import os
import sys
import json
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Log directory - same level as data directory
LOG_DIR = Path(__file__).parent.parent / "logs"

# Log level from environment (default: INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Log format from environment (json or text, default: json)
LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower()

# Log file settings
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5  # Keep 5 rotated files


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def __init__(self):
        super().__init__()
        self.hostname = os.getenv("HOSTNAME", "unknown")
        self.environment = os.getenv("ENVIRONMENT", "development")

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": self.environment,
            "hostname": self.hostname,
        }

        # Add trace ID if available (can be set via context)
        trace_id = getattr(record, "trace_id", None)
        if trace_id:
            log_data["trace_id"] = trace_id

        # Add exception info if present
        if record.exc_info:
            log_data["error"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
            }

        # Add extra fields (user_id, request_id, etc)
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code

        return json.dumps(log_data, default=str)


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
    if LOG_FORMAT == "json":
        file_formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
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

    if LOG_FORMAT == "json":
        access_formatter = JSONFormatter()
    else:
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
