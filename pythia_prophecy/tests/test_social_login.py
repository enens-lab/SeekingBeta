"""Unit tests for social-login Phase 1 (backend).

Self-contained (no pytest dependency) so it runs anywhere:

    cd pythia_prophecy && .venv/bin/python tests/test_social_login.py

Exercises the find-or-create / link matrix in database.resolve_or_create_social_user
(the security-critical core) plus the verifier dispatch, against an ISOLATED temp
SQLite DB — never the dev/prod pythia.db.
"""
import base64
import hashlib
import hmac
import json
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

# 9. Facebook signed_request verification (stdlib only — runs everywhere).
_fb_secret = "test-app-secret"
social_auth.FACEBOOK_APP_SECRET = _fb_secret


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


_fb_payload = {"algorithm": "HMAC-SHA256", "user_id": "fb-123", "issued_at": 1700000000}
_payload_b64 = _b64(json.dumps(_fb_payload).encode("utf-8"))
_good_sig = hmac.new(_fb_secret.encode("utf-8"), _payload_b64.encode("utf-8"), hashlib.sha256).digest()
_good_sr = f"{_b64(_good_sig)}.{_payload_b64}"
check("fb signed_request: valid signature parses",
      social_auth.parse_facebook_signed_request(_good_sr).get("user_id") == "fb-123")
try:
    social_auth.parse_facebook_signed_request(f"{_b64(b'not-the-sig')}.{_payload_b64}")
    check("fb signed_request: tampered signature rejected", False)
except social_auth.SocialAuthError:
    check("fb signed_request: tampered signature rejected", True)
try:
    social_auth.parse_facebook_signed_request("garbage-no-dot")
    check("fb signed_request: malformed rejected", False)
except social_auth.SocialAuthError:
    check("fb signed_request: malformed rejected", True)

# 10. Apple refresh-token encryption round-trips (Fernet from the JWT secret).
_enc = social_auth.encrypt_token("apple-refresh-xyz")
check("token encrypts to ciphertext", bool(_enc) and _enc != "apple-refresh-xyz")
check("token decrypts back", social_auth.decrypt_token(_enc) == "apple-refresh-xyz")
check("decrypt(None) -> None", social_auth.decrypt_token(None) is None)

# 11. Apple ES256 client_secret build (guarded: needs jose + cryptography).
try:
    from jose import jwt as _jose_jwt
    from cryptography.hazmat.primitives.asymmetric import ec as _ec
    from cryptography.hazmat.primitives import serialization as _ser
    _have_jose = True
except Exception:
    _have_jose = False

if _have_jose:
    _key = _ec.generate_private_key(_ec.SECP256R1())
    _pem = _key.private_bytes(_ser.Encoding.PEM, _ser.PrivateFormat.PKCS8, _ser.NoEncryption()).decode("utf-8")
    _kf = Path(_tmp) / "apple_key.p8"
    _kf.write_text(_pem)
    social_auth.APPLE_OAUTH_PRIVATE_KEY_FILE = str(_kf)
    social_auth.APPLE_OAUTH_TEAM_ID = "TEAM123456"
    social_auth.APPLE_OAUTH_KEY_ID = "KEY1234567"
    _cs = social_auth._build_apple_client_secret("ai.seekingbeta.app")
    _hdr = _jose_jwt.get_unverified_header(_cs)
    _claims = _jose_jwt.get_unverified_claims(_cs)
    check("apple client_secret: ES256 + kid header",
          _hdr.get("alg") == "ES256" and _hdr.get("kid") == "KEY1234567")
    check("apple client_secret: iss/sub/aud claims",
          _claims.get("iss") == "TEAM123456"
          and _claims.get("sub") == "ai.seekingbeta.app"
          and _claims.get("aud") == "https://appleid.apple.com")
else:
    print("  (skipped apple client_secret test — jose/cryptography not in this venv)")

# 12. Facebook Limited Login JWT path (guarded: needs jose + cryptography).
if _have_jose:
    from cryptography.hazmat.primitives.asymmetric import rsa as _rsa
    from jose import jwk as _jose_jwk
    import time as _time

    social_auth.FACEBOOK_APP_ID = "1027261346420299"
    social_auth.FACEBOOK_APP_SECRET = _fb_secret

    check("opaque token not mistaken for JWT", social_auth._looks_like_jwt("EAAOhZBZC...opaque") is False)
    check("fb signed_request not mistaken for JWT", social_auth._looks_like_jwt(_good_sr) is False)

    _rsa_key = _rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _rsa_pem = _rsa_key.private_bytes(_ser.Encoding.PEM, _ser.PrivateFormat.PKCS8, _ser.NoEncryption()).decode()
    _pub_jwk = _jose_jwk.construct(_rsa_key.public_key(), algorithm="RS256").to_dict()
    _pub_jwk["kid"] = "fbkid1"
    social_auth._fb_keys_cache["keys"] = [_pub_jwk]
    social_auth._fb_keys_cache["fetched_at"] = _time.time()

    _now = int(_time.time())
    _fb_claims = {
        "iss": "https://www.facebook.com", "aud": "1027261346420299", "sub": "fb-limited-777",
        "iat": _now, "exp": _now + 3600, "nonce": "n-1",
        "given_name": "Lim", "family_name": "Ited", "email": "lim@example.com",
    }
    _fb_jwt = _jose_jwt.encode(_fb_claims, _rsa_pem, algorithm="RS256", headers={"kid": "fbkid1"})
    check("limited-login credential detected as JWT", social_auth._looks_like_jwt(_fb_jwt) is True)

    _ident = social_auth.verify_social_credential("facebook", _fb_jwt)
    check("limited JWT verifies -> subject from sub", _ident.subject == "fb-limited-777")
    check("limited JWT email present but NOT verified (no auto-link)",
          _ident.email == "lim@example.com" and _ident.email_verified is False)
    check("limited JWT name claims mapped", _ident.first_name == "Lim" and _ident.last_name == "Ited")

    try:
        social_auth.verify_social_credential("facebook", _fb_jwt, nonce="wrong-nonce")
        check("limited JWT nonce mismatch rejected", False)
    except social_auth.SocialAuthError:
        check("limited JWT nonce mismatch rejected", True)

    _bad_aud = dict(_fb_claims, aud="999999")
    _bad_jwt = _jose_jwt.encode(_bad_aud, _rsa_pem, algorithm="RS256", headers={"kid": "fbkid1"})
    try:
        social_auth.verify_social_credential("facebook", _bad_jwt)
        check("limited JWT wrong audience rejected", False)
    except social_auth.SocialAuthError:
        check("limited JWT wrong audience rejected", True)
else:
    print("  (skipped facebook limited-login tests — jose/cryptography not in this venv)")

print(f"\nAll {len(_passed)} checks passed:")
for _name in _passed:
    print("  PASS", _name)
