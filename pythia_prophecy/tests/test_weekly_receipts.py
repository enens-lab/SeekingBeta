"""Unit tests for the weekly Receipts digest (Wave 1.3). Offline fixtures only.

    cd pythia_prophecy && .venv/bin/python tests/test_weekly_receipts.py

Fixtures mirror real production shapes observed on 2026-07-26: tennis grading is
current, MLB publishes daily boards but its graded history stops a season back,
and golf carries undated backtests. The honesty rules (placed != win, stale
pipeline never reads as a clean sheet) are what these tests exist to protect.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import weekly_receipts as wr  # noqa: E402

_passed = []


def check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    _passed.append(name)


NOW = datetime(2026, 7, 26, 8, 0, tzinfo=timezone.utc)
START, END, START_ISO, END_ISO = wr.week_window(NOW)
check("window is the 7 days ending yesterday",
      (START, END, START_ISO, END_ISO) == (20260719, 20260725, "2026-07-19", "2026-07-25"))

BOARDS = {
    # Grading current: two in-window results, one outright hit and one placed.
    "tennis": {
        "upcoming": [
            {
                "name": "2026 Washington",
                "tournament": "2026 Washington",
                "scheduledDate": 20260726,
                "predictions": [{"rank": 1, "playerName": "Taylor Fritz", "winProbability": 0.17}],
            }
        ],
        "backtests": [
            {"tournament": "2026 Newport", "scheduledDate": 20260719,
             "hitStatus": "Top Pick", "predictedWinner": "A. Player", "actualWinner": "A. Player"},
            {"tournament": "2026 Bastad", "scheduledDate": 20260722,
             "hitStatus": "Top 3", "predictedWinner": "B. Player", "actualWinner": "C. Other"},
            # Out of window: must not be counted, but should set graded_through.
            {"tournament": "2026 Old Event", "scheduledDate": 20260710,
             "hitStatus": "Top Pick", "predictedWinner": "D. Player", "actualWinner": "D. Player"},
        ],
        "seasonSummary": {"year": 2026, "sampleSize": 95, "topPickHits": 21,
                          "topPickAccuracy": 0.221, "top3Accuracy": 0.526, "top5Accuracy": 0.695},
    },
    # Publishes boards daily, but graded history stopped last season.
    "mlb": {
        "upcoming": [
            {"name": "A @ B", "awayTeam": "Athletics", "homeTeam": "Twins",
             "scheduledDate": 20260726, "predictedWinner": "Athletics",
             "awayWinProbability": 0.93, "homeWinProbability": 0.07, "predictions": []},
            {"name": "C @ D", "awayTeam": "Braves", "homeTeam": "Orioles",
             "scheduledDate": 20260726, "predictedWinner": "Braves",
             "awayWinProbability": 0.59, "homeWinProbability": 0.41, "predictions": []},
        ],
        "backtests": [
            {"tournament": "Rangers at Guardians", "scheduledDate": 20250928,
             "hitStatus": "Top Pick", "predictedWinner": "Guardians", "actualWinner": "Guardians"},
        ],
        "seasonSummary": {"year": 2026, "sampleSize": 0, "topPickHits": 0, "topPickAccuracy": None},
    },
    # Undated backtests can never fall in a window.
    "golf": {
        "upcoming": [],
        "backtests": [{"tournament": "Some Open", "hitStatus": "Top Pick", "predictedWinner": "X"}],
        "seasonSummary": {"year": 2026, "sampleSize": 0, "topPickHits": 0, "topPickAccuracy": None},
    },
}

TRACK = {"available": True, "summary": {"hit_rate": 0.53, "sample_size": 1100,
                                        "as_of": "2026-05-11", "total_return_net": 0.9,
                                        "benchmark_return": 0.6}}

r = wr.build_receipts(BOARDS, TRACK, now=NOW)
by_sport = {s["sport"]: s for s in r["sports"]}

# --- windowing + honest scoring ---
check("only sports with data appear", set(by_sport) == {"tennis"})
check("mlb omitted (no in-window grades, no season sample)", "mlb" not in by_sport)
check("golf omitted (undated backtests cannot be windowed)", "golf" not in by_sport)
check("in-window boards counted, out-of-window excluded", by_sport["tennis"]["week"]["count"] == 2)
check("outright hit is the only win", by_sport["tennis"]["week"]["wins"] == 1)
check("placed finish counts as a loss", by_sport["tennis"]["week"]["losses"] == 1)
check("placed finish disclosed separately", by_sport["tennis"]["week"]["placed"] == 1)
check("week totals aggregate", r["week_record"] == {"wins": 1, "losses": 1, "graded": 2})
check("graded_through reflects newest grade, even out of window",
      by_sport["tennis"]["graded_through"] == 20260722)
check("graded details sorted oldest first",
      [d["date_key"] for d in by_sport["tennis"]["week"]["details"]] == [20260719, 20260722])

# --- season blocks require a real sample ---
check("season block carries sample size", by_sport["tennis"]["season"]["sample_size"] == 95)
check("zero-sample season is dropped", wr._season_block({"sampleSize": 0}) is None)
check("missing season is dropped", wr._season_block(None) is None)

# --- teaser + remaining count ---
teaser = r["teaser"]
check("teaser is the highest-confidence upcoming pick", teaser["pick"] == "Athletics")
check("teaser carries its probability", abs(teaser["probability"] - 0.93) < 1e-9)
check("remaining count excludes the revealed teaser", r["remaining_boards"] == 2)

# --- stocks are cumulative, never weekly ---
check("stock record carried with sample", r["stocks"]["sample_size"] == 1100)
check("stock record keeps its own as-of date", r["stocks"]["as_of"] == "2026-05-11")
check("unavailable track record -> None", wr.summarize_stock_record({"available": False}) is None)
check("track record without sample -> None",
      wr.summarize_stock_record({"available": True, "summary": {"hit_rate": 0.5, "sample_size": 0}}) is None)

# --- the critical guard: a stale pipeline must not read as a perfect week ---
STALE_ONLY = {"mlb": BOARDS["mlb"], "golf": BOARDS["golf"]}
stale = wr.build_receipts(STALE_ONLY, None, now=NOW)
check("all-stale week reports zero graded", stale["week_record"]["graded"] == 0)
check("all-stale week is not sent", wr.has_content(stale) is False)
stale_text = wr.render_receipts_text(stale, "https://x")
check("all-stale copy says nothing graded, not 0-0",
      "no boards finished grading" in stale_text and "0-0" not in stale_text)

# --- has_content ---
check("week with grades has content", wr.has_content(r) is True)
season_only = wr.build_receipts({"tennis": dict(BOARDS["tennis"], backtests=[])}, None, now=NOW)
check("season-only week still worth sending", wr.has_content(season_only) is True)
check("empty everything has no content", wr.has_content(wr.build_receipts({}, None, now=NOW)) is False)

# --- rendering ---
subject = wr.receipts_subject(r)
check("subject states the real record", subject == "Your receipts: we went 1-1 on 2 graded boards")
check("subject falls back without weekly grades",
      wr.receipts_subject(stale) == "Your receipts: season records, losses included")

text = wr.render_receipts_text(r, "https://seekingbeta.ai")
check("text: weekly headline", "LAST WEEK: 1-1 across 2 graded boards." in text)
check("text: placed finishes disclosed as losses",
      "(1 placed top 3/5, counted as losses)" in text)
check("text: season line carries sample + graded-through",
      "top pick 21/95 (22%)" in text and "graded through Jul 22" in text)
check("text: losing pick shown with actual result",
      "2026 Bastad — B. Player -> Top 3 (actual: C. Other)" in text)
check("text: stock record labeled cumulative",
      "cumulative record, not a weekly number" in text.lower())
check("text: teaser revealed in full", "model likes Athletics (93%)" in text)
check("text: honesty footer", "We publish losses the same way we publish wins." in text)

html = wr.render_receipts_html_body(r, "https://seekingbeta.ai")
check("html: weekly headline", "<strong>Last week: 1-1</strong>" in html)
check("html: season line present", "top pick 21/95 (22%)" in html)
check("html: teaser present", "model likes <strong>Athletics</strong>" in html)
check("html: honesty footer", "losses the same way we publish wins" in html)

# --- truncation is disclosed, never silent ---
many = dict(BOARDS["tennis"])
many["backtests"] = [
    {"tournament": f"Event {i}", "scheduledDate": 20260720, "hitStatus": "Miss",
     "predictedWinner": "P", "actualWinner": "Q"}
    for i in range(8)
]
big = wr.build_receipts({"tennis": many}, None, now=NOW)
big_text = wr.render_receipts_text(big, "https://x")
check("long lists disclose what was trimmed", "and 3 more, all in the public record" in big_text)

print(f"\nAll {len(_passed)} checks passed:")
for name in _passed:
    print("  PASS", name)
