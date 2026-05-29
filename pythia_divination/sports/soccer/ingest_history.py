"""Pre-warm the soccer match-history cache.

Downloads football-data.co.uk season CSVs for each predictable league so the
export / training steps run offline. Safe to re-run (CSVs are cached on disk).

    python -m sports.soccer.ingest_history --seasons 5
"""

from __future__ import annotations

import argparse
import logging

from . import client
from .constants import PREDICTABLE_LEAGUE_KEYS, tour_for_league

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache soccer match history")
    parser.add_argument("--seasons", type=int, default=5, help="number of seasons back to fetch")
    parser.add_argument("--leagues", nargs="*", default=list(PREDICTABLE_LEAGUE_KEYS))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for league_key in args.leagues:
        try:
            frame = client.load_history(league_key, seasons=args.seasons)
            logger.info("%s (%s): %d matches cached", tour_for_league(league_key), league_key, len(frame))
        except Exception as exc:  # pragma: no cover
            logger.warning("failed to ingest %s: %s", league_key, exc)


if __name__ == "__main__":
    main()
