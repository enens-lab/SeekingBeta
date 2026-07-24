"""Unit tests for the Daily Brief (Wave 1.2). Offline — fixture dicts only.

    cd pythia_prophecy && .venv/bin/python tests/test_daily_brief.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import daily_brief  # noqa: E402

_passed = []


def check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    _passed.append(name)


NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
TODAY, YESTERDAY, ISO = daily_brief.utc_date_keys(NOW)
check("date keys computed", TODAY == 20260707 and YESTERDAY == 20260706 and ISO == "2026-07-07")

BOARDS = {
    "golf": {
        "upcoming": [
            {
                "name": "The Open Championship",
                "tournament": "The Open Championship",
                "scheduledDate": TODAY,
                "eventState": "upcoming",
                "predictions": [{"rank": 1, "playerName": "S. Scheffler", "winProbability": 0.18}],
            }
        ],
        "backtests": [
            {
                "tournament": "John Deere Classic",
                "scheduledDate": YESTERDAY,
                "hitStatus": "Top Pick",
                "predictedWinner": "A. Player",
                "actualWinner": "A. Player",
            },
            {
                "tournament": "Euro Open",
                "scheduledDate": YESTERDAY,
                "hitStatus": "Top 5",
                "predictedWinner": "B. Player",
                "actualWinner": "C. Other",
            },
        ],
        "seasonSummary": {"topPickAccuracy": 0.21},
    },
    "mlb": {
        "upcoming": [
            {
                "name": "Yankees @ Red Sox",
                "awayTeam": "Yankees",
                "homeTeam": "Red Sox",
                "scheduledDate": TODAY,
                "predictedWinner": "Yankees",
                "homeWinProbability": 0.44,
                "awayWinProbability": 0.56,
                "predictions": [],
            }
        ],
        "backtests": [
            {
                "tournament": "Mets @ Braves",
                "awayTeam": "Mets",
                "homeTeam": "Braves",
                "scheduledDate": YESTERDAY,
                "hitStatus": "Miss",
                "predictedWinner": "Mets",
                "actualWinner": "Braves",
            }
        ],
    },
    "tennis": {"upcoming": [], "backtests": []},
}

HOMEPAGE = {
    "available": True,
    "rows": [
        {
            "model": "lstm_5d",
            "predictions": [
                {"ticker": "NVDA", "signal": "buy", "prob_up": 0.71},
                {"ticker": "AAPL", "signal": "hold", "prob_up": 0.52},
                {"ticker": "TSLA", "signal": "sell", "prob_up": 0.31},
            ],
        },
        {"model": "lstm_jackpot", "predictions": [{"ticker": "NVDA", "signal": "buy", "prob_up": 0.66}]},
    ],
}

TRACK = {"available": True, "summary": {"hit_rate": 0.58, "sample_size": 412}}

brief = daily_brief.build_brief(BOARDS, HOMEPAGE, TRACK, now=NOW)

# --- sports assembly ---
sports = {s["sport"]: s for s in brief["sports"]}
check("golf + mlb present, empty tennis omitted", set(sports) == {"golf", "mlb"})
check("golf today board picked up", sports["golf"]["today"][0]["pick"] == "S. Scheffler")
check("golf top-pick win counted", sports["golf"]["yesterday"]["wins"] == 1)
check("golf Top 5 counted as loss (honest scoring)",
      sports["golf"]["yesterday"]["losses"] == 1 and sports["golf"]["yesterday"]["placed"] == 1)
check("mlb matchup headline", sports["mlb"]["today"][0]["event"] == "Yankees @ Red Sox")
check("mlb away-side probability chosen", abs(sports["mlb"]["today"][0]["probability"] - 0.56) < 1e-9)
check("mlb miss graded", sports["mlb"]["yesterday"]["losses"] == 1)
check("total record aggregated", brief["yesterday_record"] == {"wins": 1, "losses": 2})

# --- stocks assembly ---
stocks = brief["stocks"]
check("only lstm_5d row counted", stocks["counts"] == {"buy": 1, "hold": 1, "sell": 1})
check("strongest conviction first", stocks["top_signals"][0]["ticker"] == "NVDA")
check("track record carried", stocks["track_record"]["sample_size"] == 412)

# --- degraded inputs ---
empty = daily_brief.build_brief(None, None, None, now=NOW)
check("all-sources-down brief is empty but valid",
      empty["stocks"] is None and empty["sports"] == [] and empty["date"] == ISO)
no_stock = daily_brief.build_brief(BOARDS, {"available": False}, {"available": False}, now=NOW)
check("unavailable stock sources -> stocks None", no_stock["stocks"] is None)

# --- rendering ---
subject = daily_brief.brief_subject(brief)
check("subject counts actionable signals", "2 stock signals" in subject)
check("subject shows record", "yesterday 1-2" in subject)

text = daily_brief.render_brief_text(brief, "https://seekingbeta.ai")
check("text: fixed stocks section", "STOCKS (lstm_5d): 1 buy · 1 hold · 1 sell" in text)
check("text: golf graded line shows result",
      "John Deere Classic — A. Player -> Top Pick (actual: A. Player)" in text)
check("text: honesty footer", "Losses are published like wins." in text)

html = daily_brief.render_brief_html_body(brief, "https://seekingbeta.ai")
check("html: sports header rendered", "<strong>Golf</strong>" in html)
check("html: escapes applied", "Yankees @ Red Sox" in html)
check("html: honesty footer", "Losses are published like wins." in html)

# Empty brief renders a subject that still makes sense.
check("empty brief subject fallback", "your boards are ready" in daily_brief.brief_subject(empty))

# --- real-world regression: engine emits 'avoid' + null prob_up ---
AVOID_HOMEPAGE = {
    "available": True,
    "rows": [
        {
            "model": "lstm_5d",
            "predictions": [
                {"ticker": "AAPL", "signal": "avoid", "prob_up": None},
                {"ticker": "MSFT", "signal": "hold", "prob_up": 0.51},
            ],
        }
    ],
}
avoid_brief = daily_brief.build_brief(None, AVOID_HOMEPAGE, None, now=NOW)
check("unknown signal kind counted", avoid_brief["stocks"]["counts"]["avoid"] == 1)
avoid_text = daily_brief.render_brief_text(avoid_brief, "https://x")
check("counts line includes avoid", "1 avoid" in avoid_text)
check("null prob_up renders without empty parens",
      "AAPL: AVOID\n" in avoid_text + "\n" and "( up)" not in avoid_text)
check("avoid counts as actionable in subject", "1 stock signal" in daily_brief.brief_subject(avoid_brief))

print(f"\nAll {len(_passed)} checks passed:")
for name in _passed:
    print("  PASS", name)
