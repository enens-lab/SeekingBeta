"""
Authentication utilities - password hashing and JWT tokens
"""
import hashlib
import hmac
import secrets
import base64
import json
from datetime import datetime, timedelta
from typing import Optional
import os

from .logging_config import get_logger

logger = get_logger("auth")

# Secret key for JWT - in production, use environment variable
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "pythia-dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24


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


def create_access_token(user_id: str, email: str) -> str:
    """Create a JWT access token."""
    now = datetime.utcnow()
    expires = now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }

    logger.debug(f"Created access token for user: {email}")
    return _encode_jwt(payload)


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT access token."""
    try:
        payload = _decode_jwt(token)
        if payload and payload.get("exp", 0) > datetime.utcnow().timestamp():
            return payload
        if payload:
            logger.debug(f"Token expired for user: {payload.get('email', 'unknown')}")
    except Exception as e:
        logger.debug(f"Token decode failed: {e}")
    return None


def generate_verification_token() -> str:
    """Generate a random verification token."""
    return secrets.token_urlsafe(32)


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
