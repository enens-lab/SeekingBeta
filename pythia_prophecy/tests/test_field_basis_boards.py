"""Checks for tennis/golf boards built from last season's field.

Boards labelled fieldBasis="previous_edition" must never be graded into this
season's record by the runtime graders, the BFF's own re-dated tennis fallback
must carry the label and drop last season's winner flags, and the new optional
board fields must survive the response models.

Run:  python tests/test_field_basis_boards.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JWT_SECRET_KEY", "test")

from api import service as s  # noqa: E402
from api.models import SportsHistoricalBoard, SportsUpcomingBoard  # noqa: E402


def _today_minus(days: int) -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d"))


def test_runtime_tennis_grading_skips_previous_edition():
    year = datetime.now(timezone.utc).year
    finished = _today_minus(1)
    s._fetch_live_atp_completed_tournaments = lambda _year: [
        {"tournament": "Chengdu", "canonical": s._canonical_tennis_event_name("Chengdu"), "latestDate": finished,
         "actualWinner": "Alex Player", "surface": "Hard"}
    ]

    def board(basis, field=28):
        predictions = [{"rank": 1, "playerName": "Alex Player", "winProbability": 30.0}] + [
            {"rank": i + 2, "playerName": f"Other {i}", "winProbability": 1.0} for i in range(field - 1)
        ]
        item = {"tour": "ATP", "name": f"{year} Chengdu", "scheduledDate": finished, "latestDate": finished,
                "predictions": predictions}
        if basis:
            item["fieldBasis"] = basis
        return item

    assert s._build_runtime_tennis_backtests([board("previous_edition")], []) == []
    graded = s._build_runtime_tennis_backtests([board("current_draw")], [])
    assert len(graded) == 1 and graded[0]["hitStatus"] == "Top Pick"
    assert len(s._build_runtime_tennis_backtests([board(None)], [])) == 1  # unlabelled boards unchanged
    assert s._build_runtime_tennis_backtests([board("current_draw", field=8)], []) == []  # finals-size field


def test_runtime_golf_grading_skips_previous_edition():
    year = datetime.now(timezone.utc).year
    finished = datetime.now(timezone.utc) - timedelta(days=3)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / f"schedule_{year}_latest.csv").write_text(
            "tournament_name,champion_name,display_date\n"
            f"The RSM Classic,Sam Golfer,{finished:%b} {finished.day} - {finished:%b} {finished.day}\n"
        )
        original = s.PGA_NORMALIZED_DIR
        s.PGA_NORMALIZED_DIR = directory
        try:
            def board(basis):
                item = {"tour": "PGA", "name": f"{year} The RSM Classic", "original_name": "The RSM Classic",
                        "scheduledDate": int(finished.strftime("%Y%m%d")),
                        "predictions": [{"rank": 1, "playerName": "Sam Golfer", "winProbability": 6.0}]}
                if basis:
                    item["fieldBasis"] = basis
                return item

            assert s._build_runtime_golf_backtests([board("previous_edition")], []) == []
            assert len(s._build_runtime_golf_backtests([board(None)], [])) == 1
        finally:
            s.PGA_NORMALIZED_DIR = original


def test_dynamic_tennis_fallback_is_labelled():
    today = datetime.now(timezone.utc)
    soon = today + timedelta(days=5)
    last_year = int(f"{today.year - 1}{soon:%m%d}")
    backtests = [{"tour": "ATP", "tournament": "Basel", "latestDate": last_year, "surface": "Hard",
                  "predictedWinner": "A", "fullField": [{"rank": 1, "playerName": "A", "winProbability": 20.0, "actualWinner": True}]}]
    upcoming = s._build_dynamic_tennis_upcoming(backtests)
    assert upcoming and upcoming[0]["fieldBasis"] == "previous_edition"
    assert all("actualWinner" not in entry for entry in upcoming[0]["predictions"])


def test_models_keep_new_fields():
    board = SportsUpcomingBoard(id="x", name="2026 Basel", tour="ATP", course="Hard", fieldBasis="previous_edition",
                                fieldNote="Field is last year's entry list.", ratingsAsOf=20260929)
    dumped = board.model_dump()
    assert dumped["fieldBasis"] == "previous_edition" and dumped["ratingsAsOf"] == 20260929 and dumped["fieldNote"]
    history = SportsHistoricalBoard(year=2026, tournament="Basel", tour="ATP", hitStatus="Miss", fieldSize=32,
                                    rankingFavorite="B", rankingFavoriteWon=True)
    assert history.model_dump()["rankingFavorite"] == "B" and history.fieldSize == 32
    assert SportsUpcomingBoard(id="y", name="n", tour="PGA", course="c").fieldBasis is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("test_field_basis_boards: all checks passed")
