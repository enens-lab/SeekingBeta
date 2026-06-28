"""Unit tests for social-login Phase 1 (backend).

Self-contained (no pytest dependency) so it runs anywhere:

    cd pythia_prophecy && .venv/bin/python tests/test_social_login.py

Exercises the find-or-create / link matrix in database.resolve_or_create_social_user
(the security-critical core) plus the verifier dispatch, against an ISOLATED temp
SQLite DB — never the dev/prod pythia.db.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # pythia_prophecy/
sys.path.insert(0, str(ROOT))

from api import database, auth, social_auth  # noqa: E402

# --- Isolate the DB: point at a throwaway file, then build the schema there. ---
_tmp = tempfile.mkdtemp(prefix="sblogin_test_")
database.DATABASE_PATH = Path(_tmp) / "test.db"
database.init_database()

_passed = []


def check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    _passed.append(name)


# 1. New Google user with a verified email -> created, normalized, verified.
u1, created1 = database.resolve_or_create_social_user(
    provider="google", subject="g-123", email="Alice@Example.com",
    email_verified=True, first_name="Alice", last_name="Smith",
)
check("new google user created", created1 is True)
check("email normalized to lowercase", u1.email == "alice@example.com")
check("email_verified carried from provider", u1.email_verified is True)
check("password sentinel stored", u1.hashed_password == auth.SOCIAL_ONLY_PASSWORD_SENTINEL)
check("password login impossible for social account",
      auth.verify_password("any-guess", u1.hashed_password) is False)
_ids = database.get_identities_by_user_id(u1.id)
check("one identity linked", len(_ids) == 1 and _ids[0]["provider"] == "google")
check("identity subject stored", _ids[0]["subject"] == "g-123")

# 2. Same (provider, subject) again -> same user, NOT created.
u2, created2 = database.resolve_or_create_social_user(
    provider="google", subject="g-123", email="alice@example.com", email_verified=True,
)
check("repeat identity -> not created", created2 is False)
check("repeat identity -> same user", u2.id == u1.id)

# 3. A DIFFERENT provider with the SAME verified email -> links (no duplicate).
u3, created3 = database.resolve_or_create_social_user(
    provider="apple", subject="a-999", email="alice@example.com", email_verified=True,
)
check("apple links to existing account by verified email", created3 is False and u3.id == u1.id)
check("two identities now linked to the user",
      len(database.get_identities_by_user_id(u1.id)) == 2)

# 4. Unverified email matching an existing account -> REFUSE (no link, no dup).
try:
    database.resolve_or_create_social_user(
        provider="facebook", subject="f-1", email="alice@example.com", email_verified=False,
    )
    check("unverified email on existing account raises conflict", False)
except database.SocialEmailConflict:
    check("unverified email on existing account raises conflict", True)
# ...and it must NOT have created a stray facebook identity.
check("no identity created on conflict",
      database.get_identity_by_provider_subject("facebook", "f-1") is None)

# 5. Facebook with no email -> standalone account on a synthetic address.
u5, created5 = database.resolve_or_create_social_user(
    provider="facebook", subject="f-noemail", email=None, email_verified=False,
    first_name="Bob", last_name=None,
)
check("fb no-email creates standalone account", created5 is True and u5.id != u1.id)
check("synthetic email assigned", u5.email.endswith("@noreply.seekingbeta.ai"))
check("synthetic account is not email-verified", u5.email_verified is False)
check("missing last name -> non-empty placeholder (passes model validation)",
      u5.last_name == "Member")

# 6. Brand-new verified email -> fresh account.
u6, created6 = database.resolve_or_create_social_user(
    provider="google", subject="g-new", email="carol@example.com", email_verified=True,
)
check("brand-new verified email -> new account", created6 is True and u6.id not in (u1.id, u5.id))

# 7. Deleting the account clears its linked identities.
database.delete_user_account(u1.id)
check("delete_user_account clears identities", database.get_identities_by_user_id(u1.id) == [])
check("deleted user's apple identity gone",
      database.get_identity_by_provider_subject("apple", "a-999") is None)

# 8. Verifier dispatch guards (no network — config absent).
try:
    social_auth.verify_social_credential("myspace", "tok")
    check("unsupported provider raises", False)
except social_auth.SocialAuthError:
    check("unsupported provider raises", True)
try:
    social_auth.verify_social_credential("google", "   ")
    check("empty credential raises", False)
except social_auth.SocialAuthError:
    check("empty credential raises", True)
try:
    social_auth.verify_social_credential("google", "fake.token.value")
    check("unconfigured provider raises (never grants)", False)
except social_auth.SocialAuthError:
    check("unconfigured provider raises (never grants)", True)

print(f"\nAll {len(_passed)} checks passed:")
for _name in _passed:
    print("  PASS", _name)
