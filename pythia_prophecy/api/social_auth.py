"""Social login: server-side verification of third-party identity tokens.

Provider-agnostic by design — adding a provider is a new ``_verify_*`` function
plus an audience allow-list entry, NOT a new endpoint or schema. Mirrors the
billing token-verify posture (``/api/billing/{apple,google}/verify``): the
server is the only authority, and a forged / foreign / expired token RAISES
``SocialAuthError`` and never grants.

Heavy provider SDKs (``google-auth``, ``python-jose``, ``httpx``) are imported
lazily inside each verifier so this module stays importable — and unit-testable
— without them installed locally. They are present in the container image
(see requirements.txt).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from .auth import SECRET_KEY
from .logging_config import get_logger

logger = get_logger("social_auth")

SUPPORTED_PROVIDERS = ("google", "apple", "facebook")


def _csv_env(name: str) -> list[str]:
    return [v.strip() for v in os.getenv(name, "").split(",") if v.strip()]


# --- Provider config (env; values come from the provider consoles, plan §A) ---
GOOGLE_OAUTH_CLIENT_IDS = _csv_env("GOOGLE_OAUTH_CLIENT_IDS")
APPLE_OAUTH_BUNDLE_ID = os.getenv("APPLE_OAUTH_BUNDLE_ID", "").strip()      # iOS native client_id
APPLE_OAUTH_SERVICES_ID = os.getenv("APPLE_OAUTH_SERVICES_ID", "").strip()  # web / Android client_id
APPLE_OAUTH_AUDIENCES = [a for a in (APPLE_OAUTH_BUNDLE_ID, APPLE_OAUTH_SERVICES_ID) if a]
# Needed to build the ES256 client_secret for the /auth/token + /auth/revoke calls.
APPLE_OAUTH_TEAM_ID = os.getenv("APPLE_OAUTH_TEAM_ID", "").strip()
APPLE_OAUTH_KEY_ID = os.getenv("APPLE_OAUTH_KEY_ID", "").strip()
APPLE_OAUTH_PRIVATE_KEY_FILE = os.getenv("APPLE_OAUTH_PRIVATE_KEY_FILE", "").strip()
# Web/Services-ID auth-code exchange requires the redirect_uri registered on the
# Services ID; native-app exchange omits it.
APPLE_OAUTH_REDIRECT_URI = os.getenv("APPLE_OAUTH_REDIRECT_URI", "").strip()
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_REVOKE_URL = "https://appleid.apple.com/auth/revoke"

FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID", "").strip()
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET", "").strip()
FACEBOOK_GRAPH_VERSION = os.getenv("FACEBOOK_GRAPH_VERSION", "v19.0").strip()

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")

_HTTP_TIMEOUT = int(os.getenv("SOCIAL_HTTP_TIMEOUT_SECONDS", "10"))


class SocialAuthError(Exception):
    """Raised on ANY verification failure — bad signature, wrong audience,
    expired, revoked, provider/app mismatch, or network error. The endpoint
    maps this to HTTP 401 and NEVER grants on it."""


@dataclass
class SocialIdentity:
    provider: str
    subject: str                      # provider 'sub' / 'id' — the durable join key
    email: Optional[str] = None       # Facebook may omit it
    email_verified: bool = False      # the provider's assertion (Facebook: always False here)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    raw: dict = field(default_factory=dict)


def verify_social_credential(
    provider: str, credential: str, nonce: Optional[str] = None
) -> SocialIdentity:
    """Dispatch to the provider verifier. Raises SocialAuthError on failure."""
    provider = (provider or "").lower().strip()
    if not credential or not credential.strip():
        raise SocialAuthError("Missing credential")
    if provider == "google":
        return _verify_google(credential, nonce)
    if provider == "apple":
        return _verify_apple(credential, nonce)
    if provider == "facebook":
        return _verify_facebook(credential)
    raise SocialAuthError(f"Unsupported provider: {provider}")


def configured_providers() -> list[str]:
    """Providers that currently have the env config required to verify."""
    out = []
    if GOOGLE_OAUTH_CLIENT_IDS:
        out.append("google")
    if APPLE_OAUTH_AUDIENCES:
        out.append("apple")
    if FACEBOOK_APP_ID and FACEBOOK_APP_SECRET:
        out.append("facebook")
    return out


# --------------------------------------------------------------------------- #
# Google — OpenID ID token (RS256), verified against Google's JWKS by google-auth
# --------------------------------------------------------------------------- #
def _verify_google(credential: str, nonce: Optional[str]) -> SocialIdentity:
    if not GOOGLE_OAUTH_CLIENT_IDS:
        raise SocialAuthError("Google OAuth is not configured")
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests
    except Exception as exc:  # pragma: no cover - dep present in container
        raise SocialAuthError(f"Google verify dependency missing: {exc}")

    try:
        # audience=None: skip the library's single-audience check (we allow-list
        # multiple platform client IDs below). Signature, exp, and issuer are
        # still verified by the library.
        idinfo = google_id_token.verify_oauth2_token(credential, google_requests.Request())
    except Exception as exc:
        raise SocialAuthError(f"Google token invalid: {exc}")

    if idinfo.get("iss") not in GOOGLE_ISSUERS:
        raise SocialAuthError("Google token issuer mismatch")
    if idinfo.get("aud") not in GOOGLE_OAUTH_CLIENT_IDS:
        raise SocialAuthError("Google token audience not allow-listed")
    sub = idinfo.get("sub")
    if not sub:
        raise SocialAuthError("Google token missing subject")

    return SocialIdentity(
        provider="google",
        subject=str(sub),
        email=(idinfo.get("email") or None),
        email_verified=bool(idinfo.get("email_verified")),
        first_name=idinfo.get("given_name"),
        last_name=idinfo.get("family_name"),
        raw={k: idinfo[k] for k in ("name", "picture", "locale") if idinfo.get(k)},
    )


# --------------------------------------------------------------------------- #
# Apple — identityToken (RS256), verified against Apple's JWKS via python-jose
# --------------------------------------------------------------------------- #
_apple_keys_cache: dict = {"keys": None, "fetched_at": 0.0}
_APPLE_KEYS_TTL = 3600


def _fetch_apple_keys(force: bool = False) -> list[dict]:
    now = time.time()
    if (not force) and _apple_keys_cache["keys"] is not None and (
        now - _apple_keys_cache["fetched_at"] < _APPLE_KEYS_TTL
    ):
        return _apple_keys_cache["keys"]
    try:
        req = urllib.request.Request(APPLE_KEYS_URL, headers={"User-Agent": "SeekingBetaAI/1.0"})
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        keys = data.get("keys") or []
    except Exception as exc:
        raise SocialAuthError(f"Could not fetch Apple signing keys: {exc}")
    _apple_keys_cache["keys"] = keys
    _apple_keys_cache["fetched_at"] = now
    return keys


def _apple_key_for_kid(kid: Optional[str]) -> Optional[dict]:
    if not kid:
        return None
    for k in _fetch_apple_keys():
        if k.get("kid") == kid:
            return k
    # Unknown kid — keys may have rotated; refetch once.
    for k in _fetch_apple_keys(force=True):
        if k.get("kid") == kid:
            return k
    return None


def _verify_apple(credential: str, nonce: Optional[str]) -> SocialIdentity:
    if not APPLE_OAUTH_AUDIENCES:
        raise SocialAuthError("Apple OAuth is not configured")
    try:
        from jose import jwt as jose_jwt
    except Exception as exc:  # pragma: no cover - dep present in container
        raise SocialAuthError(f"Apple verify dependency missing: {exc}")

    try:
        header = jose_jwt.get_unverified_header(credential)
    except Exception as exc:
        raise SocialAuthError(f"Apple token malformed: {exc}")
    jwk = _apple_key_for_kid(header.get("kid"))
    if not jwk:
        raise SocialAuthError("Apple signing key not found")

    try:
        # Audience allow-list (Bundle ID + Services ID) is checked manually below,
        # so disable the library's single-audience check.
        claims = jose_jwt.decode(
            credential,
            jwk,
            algorithms=["RS256"],
            issuer=APPLE_ISSUER,
            options={"verify_aud": False},
        )
    except Exception as exc:
        raise SocialAuthError(f"Apple token invalid: {exc}")

    if claims.get("aud") not in APPLE_OAUTH_AUDIENCES:
        raise SocialAuthError("Apple token audience not allow-listed")
    if nonce is not None:
        token_nonce = claims.get("nonce")
        if token_nonce and token_nonce != nonce:
            raise SocialAuthError("Apple nonce mismatch")
    sub = claims.get("sub")
    if not sub:
        raise SocialAuthError("Apple token missing subject")

    # Apple may send email_verified as the string "true".
    ev = claims.get("email_verified")
    email_verified = (ev is True) or (str(ev).lower() == "true")
    email = claims.get("email") or None

    return SocialIdentity(
        provider="apple",
        subject=str(sub),
        email=email,
        email_verified=bool(email_verified and email),
        raw={
            "is_private_email": str(claims.get("is_private_email")).lower() == "true",
            # The audience == the client_id the client used (Bundle ID for native,
            # Services ID for web). The auth-code exchange + revoke must reuse it.
            "aud": claims.get("aud"),
        },
    )


# --------------------------------------------------------------------------- #
# Facebook — access token, verified server-to-server via the Graph API
# --------------------------------------------------------------------------- #
def _verify_facebook(credential: str) -> SocialIdentity:
    if not (FACEBOOK_APP_ID and FACEBOOK_APP_SECRET):
        raise SocialAuthError("Facebook OAuth is not configured")
    try:
        import httpx
    except Exception as exc:  # pragma: no cover - dep present in container
        raise SocialAuthError(f"Facebook verify dependency missing: {exc}")

    base = f"https://graph.facebook.com/{FACEBOOK_GRAPH_VERSION}"
    app_token = f"{FACEBOOK_APP_ID}|{FACEBOOK_APP_SECRET}"
    # appsecret_proof hardens the /me call against access-token theft.
    proof = hmac.new(
        FACEBOOK_APP_SECRET.encode("utf-8"), credential.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            dbg = client.get(
                f"{base}/debug_token",
                params={"input_token": credential, "access_token": app_token},
            )
            dbg.raise_for_status()
            data = (dbg.json() or {}).get("data") or {}
            if not data.get("is_valid"):
                raise SocialAuthError("Facebook token is not valid")
            if str(data.get("app_id")) != FACEBOOK_APP_ID:
                raise SocialAuthError("Facebook token app mismatch")
            fb_user_id = data.get("user_id")
            if not fb_user_id:
                raise SocialAuthError("Facebook token missing user id")

            me = client.get(
                f"{base}/me",
                params={
                    "fields": "id,first_name,last_name,email",
                    "access_token": credential,
                    "appsecret_proof": proof,
                },
            )
            me.raise_for_status()
            profile = me.json() or {}
    except SocialAuthError:
        raise
    except Exception as exc:
        raise SocialAuthError(f"Facebook verify failed: {exc}")

    subject = str(profile.get("id") or fb_user_id)
    # Facebook gives no clean per-user verified-email boolean, so treat email as
    # UNVERIFIED — the link-by-email path will never auto-link on it (plan §D).
    return SocialIdentity(
        provider="facebook",
        subject=subject,
        email=(profile.get("email") or None),
        email_verified=False,
        first_name=profile.get("first_name"),
        last_name=profile.get("last_name"),
        raw={},
    )


# --------------------------------------------------------------------------- #
# Token encryption at rest (Apple refresh tokens). Symmetric key derived from
# the app's JWT secret — no extra secret to manage; rotating JWT_SECRET_KEY
# invalidates stored tokens (acceptable: a re-login re-stores them).
# --------------------------------------------------------------------------- #
def _token_cipher():
    from cryptography.fernet import Fernet
    key = base64.urlsafe_b64encode(hashlib.sha256(SECRET_KEY.encode("utf-8")).digest())
    return Fernet(key)


def encrypt_token(plaintext: Optional[str]) -> Optional[str]:
    if not plaintext:
        return None
    try:
        return _token_cipher().encrypt(plaintext.encode("utf-8")).decode("utf-8")
    except Exception as exc:  # pragma: no cover
        logger.warning("Token encryption failed: %s", exc)
        return None


def decrypt_token(ciphertext: Optional[str]) -> Optional[str]:
    if not ciphertext:
        return None
    try:
        return _token_cipher().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except Exception as exc:  # pragma: no cover
        logger.warning("Token decryption failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Apple: client_secret (ES256), authorization-code exchange, token revocation
# --------------------------------------------------------------------------- #
def _load_apple_private_key() -> str:
    if not APPLE_OAUTH_PRIVATE_KEY_FILE:
        raise SocialAuthError("Apple private key file not configured")
    try:
        with open(APPLE_OAUTH_PRIVATE_KEY_FILE, "r") as fh:
            return fh.read()
    except Exception as exc:
        raise SocialAuthError(f"Could not read Apple private key: {exc}")


def _build_apple_client_secret(client_id: str) -> str:
    """ES256 JWT signed by the .p8 — the client_secret for /auth/token + /auth/revoke.
    Built fresh and short-lived per call (Apple caps exp at 6 months)."""
    if not (APPLE_OAUTH_TEAM_ID and APPLE_OAUTH_KEY_ID and client_id):
        raise SocialAuthError("Apple client-secret config incomplete (team/key/client id)")
    try:
        from jose import jwt as jose_jwt
    except Exception as exc:  # pragma: no cover - dep present in container
        raise SocialAuthError(f"Apple verify dependency missing: {exc}")
    now = int(time.time())
    claims = {
        "iss": APPLE_OAUTH_TEAM_ID,
        "iat": now,
        "exp": now + 3600,
        "aud": APPLE_ISSUER,
        "sub": client_id,
    }
    try:
        return jose_jwt.encode(
            claims, _load_apple_private_key(), algorithm="ES256",
            headers={"kid": APPLE_OAUTH_KEY_ID},
        )
    except SocialAuthError:
        raise
    except Exception as exc:
        raise SocialAuthError(f"Apple client-secret build failed: {exc}")


def apple_client_id_for_aud(aud: Optional[str]) -> Optional[str]:
    """The client_id to use for exchange/revoke == the audience of the token the
    client presented (Bundle ID for native, Services ID for web/Android)."""
    if aud and aud in APPLE_OAUTH_AUDIENCES:
        return aud
    return APPLE_OAUTH_BUNDLE_ID or APPLE_OAUTH_SERVICES_ID or None


def exchange_apple_auth_code(
    authorization_code: str, client_id: str, redirect_uri: Optional[str] = None
) -> dict:
    """Exchange the single-use authorization code for tokens. Returns the JSON
    (refresh_token, access_token, id_token). Raises SocialAuthError on failure.
    redirect_uri is required for the web/Services-ID flow and omitted for native."""
    if not authorization_code:
        raise SocialAuthError("Missing Apple authorization code")
    client_secret = _build_apple_client_secret(client_id)
    data = {
        "grant_type": "authorization_code",
        "code": authorization_code,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    if redirect_uri:
        data["redirect_uri"] = redirect_uri
    try:
        import httpx
    except Exception as exc:  # pragma: no cover
        raise SocialAuthError(f"Apple token exchange dependency missing: {exc}")
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.post(
                APPLE_TOKEN_URL, data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if resp.status_code != 200:
            raise SocialAuthError(
                f"Apple token exchange failed: {resp.status_code} {resp.text[:200]}"
            )
        return resp.json() or {}
    except SocialAuthError:
        raise
    except Exception as exc:
        raise SocialAuthError(f"Apple token exchange error: {exc}")


def revoke_apple_identity(refresh_token: Optional[str], client_id: Optional[str] = None) -> bool:
    """Revoke an Apple refresh token on account deletion (App Store requirement).
    Best-effort: returns True on success, False on any failure (never raises) so
    deletion always proceeds. No-ops when no token/config is available — e.g.
    users who signed in before authorization-code capture shipped."""
    if not refresh_token:
        return False
    cid = client_id or APPLE_OAUTH_BUNDLE_ID or APPLE_OAUTH_SERVICES_ID
    if not cid:
        logger.warning("Apple revoke skipped: no client_id configured")
        return False
    try:
        client_secret = _build_apple_client_secret(cid)
        import httpx
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.post(
                APPLE_REVOKE_URL,
                data={
                    "client_id": cid,
                    "client_secret": client_secret,
                    "token": refresh_token,
                    "token_type_hint": "refresh_token",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if resp.status_code == 200:
            return True
        logger.warning("Apple revoke non-200: %s %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        logger.warning("Apple revoke failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Facebook: verify a Meta signed_request (data-deletion callback)
# --------------------------------------------------------------------------- #
def parse_facebook_signed_request(signed_request: str) -> dict:
    """Verify + parse a Meta signed_request ('<sig>.<payload>', base64url). The
    HMAC-SHA256 is keyed by the APP SECRET over the RAW (still-encoded) payload
    string. Raises SocialAuthError on any tampering. Returns the payload dict."""
    if not FACEBOOK_APP_SECRET:
        raise SocialAuthError("Facebook app secret not configured")
    if not signed_request or "." not in signed_request:
        raise SocialAuthError("Malformed Facebook signed_request")
    sig_b64, payload_b64 = signed_request.split(".", 1)

    def _b64url(data: str) -> bytes:
        return base64.urlsafe_b64decode(data + ("=" * (-len(data) % 4)))

    try:
        sig = _b64url(sig_b64)
        payload = json.loads(_b64url(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise SocialAuthError(f"Facebook signed_request decode failed: {exc}")

    expected = hmac.new(
        FACEBOOK_APP_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(expected, sig):
        raise SocialAuthError("Facebook signed_request signature mismatch")
    algo = str(payload.get("algorithm", "")).upper()
    if algo and algo != "HMAC-SHA256":
        raise SocialAuthError("Facebook signed_request unexpected algorithm")
    return payload
