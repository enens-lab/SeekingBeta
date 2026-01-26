"""JWT authentication for Pythia API."""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from storage.postgres import get_user_by_username, get_rate_limit, increment_rate_limit
from api.tiers import get_tier_config, UserTier

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "pythia-dev-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


# Test accounts to seed
TEST_ACCOUNTS = [
    {"username": "free_user", "password": "free123", "tier": "free"},
    {"username": "pro_user", "password": "pro123", "tier": "pro"},
    {"username": "enterprise_user", "password": "enterprise123", "tier": "enterprise"},
]


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: int, username: str, tier: str) -> str:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "tier": tier,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


class User:
    """Represents an authenticated user."""
    def __init__(self, id: int, username: str, tier: str):
        self.id = id
        self.username = username
        self.tier = tier
        self.tier_config = get_tier_config(tier)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """FastAPI dependency to get the current authenticated user."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)
    user_id = int(payload.get("sub", 0))
    username = payload.get("username", "")
    tier = payload.get("tier", "free")

    if not user_id or not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return User(id=user_id, username=username, tier=tier)


async def get_optional_user(token: str = Depends(oauth2_scheme)) -> Optional[User]:
    """FastAPI dependency to get user if authenticated, None otherwise."""
    if not token:
        return None
    try:
        return await get_current_user(token)
    except HTTPException:
        return None


async def check_rate_limit(user: User) -> None:
    """Check if user is within their rate limit. Raises 429 if exceeded."""
    tier_config = user.tier_config

    # Unlimited for enterprise
    if tier_config.daily_rate_limit is None:
        return

    current_count = await get_rate_limit(user.id)

    if current_count >= tier_config.daily_rate_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. {tier_config.daily_rate_limit} requests per day for {user.tier} tier.",
        )


async def increment_user_rate_limit(user: User) -> int:
    """Increment user's rate limit counter and return new count."""
    return await increment_rate_limit(user.id)


async def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate a user and return their info if valid."""
    user = await get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user
