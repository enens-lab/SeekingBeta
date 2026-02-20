"""
Unified settings loader:
- Loads secrets from .env
- Loads project defaults from config.yaml
- Environment variables override YAML where applicable
- Exposes typed properties and provider blocks (e.g., Alpaca)

Usage:
  from config.settings import settings
  print(settings.database_url, settings.alpaca.base_url, settings.data_source)
"""

from __future__ import annotations
import os, typing as t
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env once, early
load_dotenv()

# Paths
ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"

def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

@dataclass(frozen=True)
class AlpacaSettings:
    base_url: str
    feed: str
    key_id: str
    secret_key: str

@dataclass(frozen=True)
class Settings:
    # Core / infra
    database_url: str

    # Strategy / data
    data_source: str
    universe: t.List[str]
    start: str
    threshold: float
    k_top: int
    lookback_download: str

    # Providers
    alpaca: AlpacaSettings

    # ---- Factory ----
    @staticmethod
    def load() -> "Settings":
        yml = _read_yaml(CFG_PATH)

        # helpers to resolve with precedence: ENV > YAML > default
        def env(name: str, default: t.Optional[str] = None) -> t.Optional[str]:
            v = os.getenv(name)
            return v if v is not None and v != "" else default

        def yget(path: str, default: t.Any = None) -> t.Any:
            # simple "a.b.c" path getter
            cur = yml
            for part in path.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur

        # Database
        database_url = env("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is not set (define it in .env)")

        # Data source + core knobs
        data_source = (env("DATA_SOURCE", None) or yget("data_source", "yahoo")).lower()
        universe = yget("universe", ["AAPL","MSFT","GOOGL","AMZN","META"])
        start = yget("start", "2016-01-01")
        threshold = float(yget("threshold", 0.55))
        k_top = int(yget("k_top", 10))
        lookback_download = yget("lookback_download", "120d")

        # Alpaca block (ENV overrides YAML)
        alp_base = env("ALPACA_BASE_URL", None) or yget("alpaca.base_url", "https://data.alpaca.markets")
        alp_feed = env("ALPACA_FEED", None) or yget("alpaca.feed", "iex")
        alp_key = env("ALPACA_KEY_ID", "")
        alp_secret = env("ALPACA_SECRET_KEY", "")
        if data_source == "alpaca" and (not alp_key or not alp_secret):
            raise RuntimeError("Alpaca selected but ALPACA_KEY_ID / ALPACA_SECRET_KEY not set in .env")

        alpaca = AlpacaSettings(
            base_url=alp_base.rstrip("/"),
            feed=alp_feed,
            key_id=alp_key,
            secret_key=alp_secret,
        )

        return Settings(
            database_url=database_url,
            data_source=data_source,
            universe=universe,
            start=start,
            threshold=threshold,
            k_top=k_top,
            lookback_download=lookback_download,
            alpaca=alpaca,
        )

# Singleton settings object
settings = Settings.load()

def reload_settings() -> Settings:
    """Hot-reload settings if you change .env or config.yaml at runtime."""
    global settings
    object.__setattr__(settings, "__dict__", Settings.load().__dict__)  # mypy: ignore
    return settings
