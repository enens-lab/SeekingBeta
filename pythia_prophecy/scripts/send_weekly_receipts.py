"""Send the weekly "Receipts" email (MARKET_ROADMAP Wave 1.3).

Runs inside the prophecy-api container:

    # manual test to one address (ignores the opt-in list):
    docker compose exec -T prophecy-api python scripts/send_weekly_receipts.py --to you@example.com

    # dry run (render + count recipients, send nothing):
    docker compose exec -T prophecy-api python scripts/send_weekly_receipts.py --dry-run

    # the real thing (host cron, Monday mornings):
    docker compose exec -T prophecy-api python scripts/send_weekly_receipts.py

Recipients are ONLY users with newsletter_enabled = TRUE (explicit marketing
opt-in). This is a marketing send, so it never goes to users who merely have an
account: the research flagged emailing lapsed or non-consenting users as a
GDPR/CAN-SPAM problem, not a growth tactic. Unsubscribe uses the existing
marketing scope, so one click stops these without touching the Daily Brief.

Assembly is deliberately script-side rather than a public endpoint: it needs the
full backtests payload, which is far too heavy to expose to anonymous traffic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import weekly_receipts as receipts_lib  # noqa: E402
from api.compliance_store import list_newsletter_recipients  # noqa: E402
from api.email_service import (  # noqa: E402
    FRONTEND_URL,
    _render_email_html,
    _send_email,
    build_unsubscribe_url,
)

BOARDS_TIMEOUT_SECONDS = 240.0


def fetch_sources(base_url: str) -> tuple[dict, dict]:
    base = base_url.rstrip("/")
    with httpx.Client(timeout=BOARDS_TIMEOUT_SECONDS) as client:
        boards = client.get(f"{base}/api/sports/boards", params={"lean_backtests": "true"})
        boards.raise_for_status()
        track = client.get(f"{base}/api/performance/track-record", params={"model": "lstm_5d"})
        track.raise_for_status()
    return boards.json(), track.json()


def render(receipts: dict) -> tuple[str, str, str]:
    frontend = FRONTEND_URL.rstrip("/")
    subject = receipts_lib.receipts_subject(receipts)
    text = receipts_lib.render_receipts_text(receipts, frontend)
    html = _render_email_html(
        title=f"Receipts — {receipts['week_start']} to {receipts['week_end']}",
        body_html=receipts_lib.render_receipts_html_body(receipts, frontend),
        cta_label="See the Full Record",
        cta_url=f"{frontend}/track-record",
    )
    return subject, text, html


def main() -> int:
    parser = argparse.ArgumentParser(description="Send the SeekingBeta weekly Receipts email")
    parser.add_argument("--base-url", default="http://localhost:8001", help="prophecy API base URL")
    parser.add_argument("--to", default=None, help="send only to this address (test mode)")
    parser.add_argument("--dry-run", action="store_true", help="render + count recipients, send nothing")
    args = parser.parse_args()

    boards, track_record = fetch_sources(args.base_url)
    receipts = receipts_lib.build_receipts(boards, track_record)
    subject, text, html = render(receipts)

    if not receipts_lib.has_content(receipts):
        print("No graded results and no season records to report; not sending.")
        return 0

    recipients = (
        [{"user_id": "manual-test", "email": args.to}]
        if args.to
        else list_newsletter_recipients()
    )

    print(f"Subject: {subject}")
    print(f"Window: {receipts['week_start']} to {receipts['week_end']}")
    print(f"Recipients: {len(recipients)}")
    if args.dry_run:
        print("--- text body ---")
        print(text)
        for r in recipients:
            print(f"would send -> {r['email']}")
        return 0

    sent = failed = 0
    for r in recipients:
        unsubscribe = build_unsubscribe_url(r["user_id"], r["email"])
        per_user_html = html.replace(
            "</body>",
            '<div style="text-align:center;color:#64748b;font-size:12px;padding:8px 0;">'
            f'<a href="{unsubscribe}" style="color:#64748b;">Unsubscribe from weekly Receipts</a></div></body>',
        )
        ok = _send_email(
            r["email"],
            subject,
            text + f"\n\nUnsubscribe from weekly Receipts: {unsubscribe}",
            per_user_html,
            category="weekly_receipts",
            unsubscribe_url=unsubscribe,
        )
        sent += ok
        failed += not ok
    print(f"Sent: {sent}, failed/suppressed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
