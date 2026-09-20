"""Checks for the parsed-backtests cache (api/service.py::_load_sports_backtests).

The cache must be invisible to callers: same boards, same order, same season
summary as parsing the full file every time -- just parsed once per file version.

Run:  python tests/test_sports_backtests_cache.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JWT_SECRET_KEY", "test")

from api import service as s  # noqa: E402

CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg)


def _board(i: int, year: int, hit: str, tour: str = "PGA") -> dict:
    return {
        "year": year, "tournament": f"Event {i}", "tour": tour, "hitStatus": hit,
        "scheduledDate": int(f"{year}01{(i % 28) + 1:02d}"), "latestDate": int(f"{year}01{(i % 28) + 1:02d}"),
        "fullField": [],
        "notes": "x" * 2000,  # bulk per board, like the real files' lineups
    }


def main():
    with tempfile.TemporaryDirectory() as d:
        data_dir = Path(d)
        s.SPORTS_DATA_SOURCE_DIR = data_dir  # _sports_data_path prefers this dir
        s.SPORTS_FRONTEND_SOURCE_DIR = data_dir
        s._SPORTS_BACKTESTS_CACHE.clear()
        year = time.gmtime().tm_year
        # 900 boards across two seasons, mixed hit statuses, deliberately unsorted.
        boards = []
        for i in range(900):
            y = year if i % 3 else year - 1
            hit = ["Top Pick", "Top 3", "Top 5", "Miss"][i % 4]
            boards.append(_board(i, y, hit))
        path = data_dir / "golf_historical_backtests.json"
        path.write_text(json.dumps(boards))

        # 1. cached load == full parse, sorted newest-first, trimmed to the headroom
        kept, rows = s._load_sports_backtests("golf_historical_backtests.json")
        full = s._load_sports_json("golf_historical_backtests.json")
        check(len(rows) == len(full) == 900, "summary rows cover the full history")
        headroom = s._backtests_cache_headroom()
        check(len(kept) == min(headroom, 900), f"kept {len(kept)} boards, expected {min(headroom, 900)}")
        expected_sorted = s._sort_sports_backtests(full)
        check([b["tournament"] for b in kept] == [b["tournament"] for b in expected_sorted[:len(kept)]], "kept boards are the newest, in sorted order")

        # 2. the served collection is identical to the uncached path
        cached_coll = s._build_sports_board_collection(upcoming=[], backtests=kept, updated_at=s.datetime.now(s.timezone.utc),
                                                       source="t", summary_rows=rows)
        direct_coll = s._build_sports_board_collection(upcoming=[], backtests=full, updated_at=s.datetime.now(s.timezone.utc),
                                                       source="t")
        check([b.tournament for b in cached_coll.backtests] == [b.tournament for b in direct_coll.backtests], "same capped backtests")
        check(cached_coll.seasonSummary == direct_coll.seasonSummary, f"same season summary: {cached_coll.seasonSummary} vs {direct_coll.seasonSummary}")
        check(cached_coll.seasonSummary.sampleSize == sum(1 for b in full if b["year"] == year), "summary counts the FULL season, not the capped list")

        # 3. second call is a cache hit (no re-parse) and returns copies, not the cached lists
        t0 = time.perf_counter(); kept2, rows2 = s._load_sports_backtests("golf_historical_backtests.json"); dt = time.perf_counter() - t0
        check(dt < 0.05, f"cache hit took {dt*1000:.0f} ms")
        kept2.append({"tournament": "mutation"})
        check(len(s._load_sports_backtests("golf_historical_backtests.json")[0]) == len(kept), "callers get a copy; cache unaffected")

        # 4. runtime merge keeps the summary honest (replace-by-key + new boards)
        runtime = [dict(kept[0], hitStatus="Miss"), _board(5000, year, "Top Pick")]
        merged_rows = s._merge_summary_rows(rows, runtime)
        check(len(merged_rows) == 901, "one replaced, one added")
        merged_boards = s._merge_runtime_backtests(kept, runtime)
        coll = s._build_sports_board_collection(upcoming=[], backtests=merged_boards, updated_at=s.datetime.now(s.timezone.utc), source="t", summary_rows=merged_rows)
        direct = s._build_sports_board_collection(upcoming=[], backtests=s._merge_runtime_backtests(full, runtime), updated_at=s.datetime.now(s.timezone.utc), source="t")
        check(coll.seasonSummary == direct.seasonSummary, "merged summary matches the uncached merged path")
        check([b.tournament for b in coll.backtests] == [b.tournament for b in direct.backtests], "merged capped boards match")

        # 5. a rewritten file (new mtime/size) invalidates the entry
        boards.append(_board(9000, year, "Top Pick"))
        path.write_text(json.dumps(boards)); os.utime(path, (time.time() + 5, time.time() + 5))
        kept3, rows3 = s._load_sports_backtests("golf_historical_backtests.json")
        check(len(rows3) == 901, "file change picked up")

        # 6. missing file -> empty, no crash
        check(s._load_sports_backtests("nope_historical_backtests.json") == ([], []), "missing file handled")

    print(f"test_sports_backtests_cache: {CHECKS} checks passed")


if __name__ == "__main__":
    main()
