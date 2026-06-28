"""
Authentication utilities - password hashing and JWT tokens
"""
import hashlib
import hmac
import secrets
import base64
import json
import re
from datetime import datetime, timedelta
from typing import Optional
import os

from .logging_config import get_logger

logger = get_logger("auth")

# Secret key for JWT - in production, use environment variable
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    logger.warning("JWT_SECRET_KEY not set - using default dev key. SET THIS IN PRODUCTION!")
    SECRET_KEY = "pythia-dev-secret-key-change-in-production"

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
REFRESH_TOKEN_EXPIRE_DAYS = 7
PASSWORD_RESET_TOKEN_EXPIRE_HOURS = 1
UNSUBSCRIBE_TOKEN_EXPIRE_DAYS = 365

# Stored in users.hashed_password for social-login-only accounts (no password
# set). It deliberately lacks the "salt$key" separator that verify_password()
# requires, so password login can never succeed against it (verify_password
# returns False on the split). This avoids a NOT NULL schema migration.
SOCIAL_ONLY_PASSWORD_SENTINEL = "!social-login-no-password!"


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    )
    return f"{salt}${key.hex()}"


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password strength. Returns (is_valid, message)."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*()_\-+=\[\]{};:'\",.<>?/\\|`~]", password):
        return False, "Password must contain at least one special character (!@#$%^&*, etc.)"
    return True, "Password is valid"


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    try:
        salt, key_hex = hashed.split("$")
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,
        )
        return hmac.compare_digest(key.hex(), key_hex)
    except (ValueError, AttributeError):
        return False


def create_access_token(user_id: str, email: str, token_type: str = "access") -> str:
    """Create a JWT token (access, refresh, password_reset, or unsubscribe)."""
    now = datetime.utcnow()

    if token_type == "access":
        expires = now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    elif token_type == "refresh":
        expires = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    elif token_type == "password_reset":
        expires = now + timedelta(hours=PASSWORD_RESET_TOKEN_EXPIRE_HOURS)
    elif token_type == "unsubscribe":
        expires = now + timedelta(days=UNSUBSCRIBE_TOKEN_EXPIRE_DAYS)
    else:
        raise ValueError(f"Unknown token type: {token_type}")

    payload = {
        "sub": user_id,
        "email": email,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }

    logger.debug(f"Created {token_type} token for user: {email}")
    return _encode_jwt(payload)


def decode_access_token(token: str, expected_type: str = "access") -> Optional[dict]:
    """Decode and validate a JWT token, optionally checking token type."""
    try:
        payload = _decode_jwt(token)
        if not payload:
            return None

        # Check expiration
        if payload.get("exp", 0) <= datetime.utcnow().timestamp():
            logger.debug(f"Token expired for user: {payload.get('email', 'unknown')}")
            return None

        # Check token type if specified
        if expected_type and payload.get("type") != expected_type:
            logger.debug(f"Invalid token type: expected {expected_type}, got {payload.get('type')}")
            return None

        return payload
    except Exception as e:
        logger.debug(f"Token decode failed: {e}")
    return None


def generate_verification_token() -> str:
    """Generate a 6-digit verification code."""
    code = secrets.randbelow(1000000)
    return f"{code:06d}"


def _encode_jwt(payload: dict) -> str:
    """Simple JWT encoding (HS256)."""
    header = {"alg": ALGORITHM, "typ": "JWT"}

    def b64_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    header_b64 = b64_encode(json.dumps(header).encode("utf-8"))
    payload_b64 = b64_encode(json.dumps(payload).encode("utf-8"))

    message = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        SECRET_KEY.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature_b64 = b64_encode(signature)

    return f"{message}.{signature_b64}"


def _decode_jwt(token: str) -> Optional[dict]:
    """Simple JWT decoding (HS256)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        message = f"{header_b64}.{payload_b64}"
        expected_signature = hmac.new(
            SECRET_KEY.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        def b64_decode(data: str) -> bytes:
            padding = 4 - len(data) % 4
            if padding != 4:
                data += "=" * padding
            return base64.urlsafe_b64decode(data)

        actual_signature = b64_decode(signature_b64)
        if not hmac.compare_digest(expected_signature, actual_signature):
            return None

        payload = json.loads(b64_decode(payload_b64))
        return payload
    except Exception:
        return None
