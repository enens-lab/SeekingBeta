"""Send the Daily Brief email to opted-in users (MARKET_ROADMAP Wave 1.2).

Runs inside the prophecy-api container (all env/SMTP config present):

    # manual test to one address (ignores opt-in list):
    docker compose exec -T prophecy-api python scripts/send_daily_brief.py --to you@example.com

    # dry run (render + list recipients, send nothing):
    docker compose exec -T prophecy-api python scripts/send_daily_brief.py --dry-run

    # the real thing (host cron, weekday mornings):
    docker compose exec -T prophecy-api python scripts/send_daily_brief.py

Recipients = user_compliance_preferences.daily_digest_enabled = TRUE (the pref
already exposed in Profile → preferences). Suppression list is enforced by
email_service._send_email. One-click unsubscribe flips ONLY the digest pref
(scope=digest), never the account or newsletter.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import daily_brief as brief_lib  # noqa: E402
from api.compliance_store import list_daily_digest_recipients  # noqa: E402
from api.email_service import (  # noqa: E402
    FRONTEND_URL,
    _render_email_html,
    _send_email,
    build_unsubscribe_url,
)


def fetch_brief(base_url: str) -> dict:
    response = httpx.get(f"{base_url.rstrip('/')}/api/brief/today", timeout=60.0)
    response.raise_for_status()
    return response.json()


def render(brief: dict) -> tuple[str, str, str]:
    frontend = FRONTEND_URL.rstrip("/")
    subject = brief_lib.brief_subject(brief)
    text = brief_lib.render_brief_text(brief, frontend)
    body_html = brief_lib.render_brief_html_body(brief, frontend)
    html = _render_email_html(
        title=f"Daily Brief — {brief['date']}",
        body_html=body_html,
        cta_label="Open Today's Boards",
        cta_url=f"{frontend}/sports",
    )
    return subject, text, html


def main() -> int:
    parser = argparse.ArgumentParser(description="Send the SeekingBeta Daily Brief")
    parser.add_argument("--base-url", default="http://localhost:8001", help="prophecy API base URL")
    parser.add_argument("--to", default=None, help="send only to this address (test mode)")
    parser.add_argument("--dry-run", action="store_true", help="render + list recipients, send nothing")
    args = parser.parse_args()

    brief = fetch_brief(args.base_url)
    subject, text, html = render(brief)

    has_content = bool(brief.get("stocks")) or bool(brief.get("sports"))
    if not has_content:
        print("Brief is empty (no signals, boards, or graded results) — not sending.")
        return 0

    if args.to:
        recipients = [{"user_id": "manual-test", "email": args.to}]
    else:
        recipients = list_daily_digest_recipients()

    print(f"Subject: {subject}")
    print(f"Recipients: {len(recipients)}")
    if args.dry_run:
        print("--- text body ---")
        print(text)
        for r in recipients:
            print(f"would send -> {r['email']}")
        return 0

    sent = failed = 0
    for r in recipients:
        unsubscribe = build_unsubscribe_url(r["user_id"], r["email"]) + "&scope=digest"
        per_user_html = html.replace(
            "</body>",
            f'<div style="text-align:center;color:#64748b;font-size:12px;padding:8px 0;">'
            f'<a href="{unsubscribe}" style="color:#64748b;">Unsubscribe from the Daily Brief</a></div></body>',
        )
        ok = _send_email(
            r["email"],
            subject,
            text + f"\n\nUnsubscribe from the Daily Brief: {unsubscribe}",
            per_user_html,
            category="daily_digest",
            unsubscribe_url=unsubscribe,
        )
        sent += ok
        failed += not ok
    print(f"Sent: {sent}, failed/suppressed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
