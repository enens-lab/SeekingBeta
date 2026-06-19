#!/usr/bin/env python3
"""
Provision a dedicated app-store *reviewer* account (pre-verified, Pro by default).

Intended to be run by YOU against the PRODUCTION prophecy database. The password is
supplied via the REVIEWER_PW environment variable so it is never hardcoded, committed,
or echoed by this script. Idempotent: if the account already exists, its password and
tier are updated and it is (re)marked email-verified.

Run it WITHOUT rebuilding the image by piping it into the running container (the api.*
modules are already in the image; the user DB is the mounted prophecy_data volume):

    # on the EC2 box, from /home/ec2-user/seekingbeta
    read -s -p "Reviewer password (min 8 chars): " REVIEWER_PW; echo
    docker compose exec -T \
      -e REVIEWER_PW="$REVIEWER_PW" \
      -e REVIEWER_EMAIL="playreview@seekingbeta.ai" \
      prophecy-api python - < pythia_prophecy/scripts/create_review_user.py

Env vars:
    REVIEWER_PW     (required) the login password the reviewer will use
    REVIEWER_EMAIL  (default playreview@seekingbeta.ai)
    REVIEWER_TIER   (default pro; one of free|basic|pro)
    REVIEWER_FIRST  (default Play)
    REVIEWER_LAST   (default Reviewer)
"""
import os
import sys
import uuid
from datetime import datetime

# Work both as a file (local run) and piped via stdin into the container (`python -`,
# where __file__ is undefined). In the container the app root is /app; locally it is
# the pythia_prophecy directory (this file's grandparent).
if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:  # piped via stdin
    sys.path.insert(0, "/app")

from api.database import (
    create_user,
    get_user_by_email,
    init_database,
    update_user_password,
    update_user_tier,
    verify_user_email,
)
from api.auth import hash_password
from api.models import UserInDB, SubscriptionTier


def main() -> int:
    email = os.environ.get("REVIEWER_EMAIL", "playreview@seekingbeta.ai").strip().lower()
    password = os.environ.get("REVIEWER_PW", "")
    tier_name = os.environ.get("REVIEWER_TIER", "pro").strip().lower()
    first = os.environ.get("REVIEWER_FIRST", "Play").strip() or "Play"
    last = os.environ.get("REVIEWER_LAST", "Reviewer").strip() or "Reviewer"

    if not password:
        print("ERROR: set REVIEWER_PW (the reviewer login password). Nothing changed.")
        return 2
    if len(password) < 8:
        print("ERROR: REVIEWER_PW must be at least 8 characters. Nothing changed.")
        return 2
    try:
        tier = SubscriptionTier(tier_name)
    except ValueError:
        print(f"ERROR: REVIEWER_TIER must be one of free|basic|pro (got {tier_name!r}). Nothing changed.")
        return 2

    init_database()
    existing = get_user_by_email(email)
    if existing:
        update_user_password(existing.id, hash_password(password))
        update_user_tier(existing.id, tier)
        if not existing.email_verified:
            verify_user_email(existing.id)
        print(f"[UPDATED] {email} -> tier={tier.value}, email_verified=True (password reset)")
    else:
        now = datetime.utcnow()
        user = UserInDB(
            id=str(uuid.uuid4()),
            email=email,
            first_name=first,
            last_name=last,
            hashed_password=hash_password(password),
            tier=tier,
            email_verified=True,
            verification_token=None,
            created_at=now,
            updated_at=now,
        )
        create_user(user)
        print(f"[CREATED] {email} -> tier={tier.value}, email_verified=True")

    print("Done. Enter this email + the password you chose in Play Console -> Policy -> App access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
