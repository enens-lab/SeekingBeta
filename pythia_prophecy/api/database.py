"""
Database layer for user management
Uses SQLite for simplicity - can be swapped for PostgreSQL
"""
from __future__ import annotations

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

from .models import UserInDB, SubscriptionTier, UserOracle
from .auth import SOCIAL_ONLY_PASSWORD_SENTINEL
from .logging_config import get_logger

logger = get_logger("database")

DATABASE_PATH = Path(__file__).parent.parent / "data" / "pythia.db"


def ensure_db_dir():
    """Ensure the data directory exists."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db():
    """Context manager for database connections."""
    ensure_db_dir()
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize database tables."""
    logger.info(f"Initializing database at {DATABASE_PATH}")
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                hashed_password TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'free',
                email_verified INTEGER NOT NULL DEFAULT 0,
                verification_token TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_verification_token ON users(verification_token)
        """)
        # User Oracle (watchlist) table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_oracle (
                user_id TEXT PRIMARY KEY,
                watchlist TEXT NOT NULL DEFAULT '[]',
                timeframes TEXT NOT NULL DEFAULT '["5d"]',
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        # Companies anchor table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                asset_type TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # Company info (fundamentals) table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS company_info (
                ticker TEXT PRIMARY KEY,
                sector TEXT,
                industry TEXT,
                description TEXT,
                market_cap INTEGER,
                enterprise_value INTEGER,
                employees INTEGER,
                website TEXT,
                country TEXT,
                state TEXT,
                city TEXT,
                exchange TEXT,
                currency TEXT,
                dividend_yield REAL,
                beta REAL,
                pe_ratio REAL,
                forward_pe REAL,
                price_to_book REAL,
                fifty_two_week_high REAL,
                fifty_two_week_low REAL,
                avg_volume INTEGER,
                raw_info TEXT,
                fetched_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (ticker) REFERENCES companies(ticker)
            )
        """)
        # Company news table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS company_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                article_id TEXT,
                title TEXT,
                publisher TEXT,
                link TEXT,
                published_at TEXT,
                article_type TEXT,
                thumbnail_url TEXT,
                related_tickers TEXT,
                fetched_at TEXT NOT NULL,
                FOREIGN KEY (ticker) REFERENCES companies(ticker),
                UNIQUE(ticker, article_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_company_news_ticker ON company_news(ticker)
        """)
        # Social login: provider identities linked to a user (Apple/Google/Facebook/...).
        # One provider identity (provider, subject) maps to at most one user; a user
        # may link several providers. `subject` (the provider 'sub'/'id') is the durable
        # join key — emails change, subjects don't.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_identities (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                subject TEXT NOT NULL,
                email_at_link TEXT,
                raw_profile TEXT,
                linked_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(provider, subject),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_auth_identities_user_id ON auth_identities(user_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_auth_identities_email ON auth_identities(email_at_link)
        """)
        conn.commit()


def create_user(user: UserInDB) -> UserInDB:
    """Create a new user in the database."""
    logger.debug(f"Creating user: {user.email}")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO users (id, email, first_name, last_name, hashed_password,
                             tier, email_verified, verification_token, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user.id,
                user.email,
                user.first_name,
                user.last_name,
                user.hashed_password,
                user.tier.value,
                1 if user.email_verified else 0,
                user.verification_token,
                user.created_at.isoformat(),
                user.updated_at.isoformat(),
            ),
        )
        conn.commit()
    logger.info(f"User created in database: {user.email} (id={user.id})")
    return user


def get_user_by_email(email: str) -> Optional[UserInDB]:
    """Get user by email address."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower(),)
        ).fetchone()
        if row:
            return _row_to_user(row)
    return None


def get_user_by_id(user_id: str) -> Optional[UserInDB]:
    """Get user by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row:
            return _row_to_user(row)
    return None


def get_user_by_verification_token(token: str) -> Optional[UserInDB]:
    """Get user by verification token."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE verification_token = ?", (token,)
        ).fetchone()
        if row:
            return _row_to_user(row)
    return None


def verify_user_email(user_id: str) -> bool:
    """Mark user's email as verified."""
    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE users
            SET email_verified = 1, verification_token = NULL, updated_at = ?
            WHERE id = ?
            """,
            (datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Email verified for user_id={user_id}")
            return True
        logger.warning(f"Failed to verify email for user_id={user_id} - user not found")
        return False


def update_user_verification_token(user_id: str, token: str) -> bool:
    """Update user's verification token."""
    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE users
            SET verification_token = ?, updated_at = ?
            WHERE id = ?
            """,
            (token, datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def update_user_tier(user_id: str, tier: SubscriptionTier) -> bool:
    """Update user's subscription tier."""
    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE users
            SET tier = ?, updated_at = ?
            WHERE id = ?
            """,
            (tier.value, datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Tier updated for user_id={user_id} to {tier.value}")
            return True
        logger.warning(f"Failed to update tier for user_id={user_id} - user not found")
        return False


def update_user_password(user_id: str, hashed_password: str) -> bool:
    """Update user's hashed password."""
    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE users
            SET hashed_password = ?, updated_at = ?
            WHERE id = ?
            """,
            (hashed_password, datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Password updated for user_id={user_id}")
            return True
        logger.warning(f"Failed to update password for user_id={user_id} - user not found")
        return False


def delete_user_account(user_id: str) -> bool:
    """Delete user account and user-owned sqlite records."""
    with get_db() as conn:
        # Explicitly delete dependent rows since foreign-key cascades are not guaranteed.
        conn.execute("DELETE FROM user_oracle WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM auth_identities WHERE user_id = ?", (user_id,))
        cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Deleted user account user_id={user_id}")
            return True
        logger.warning(f"Failed to delete user account user_id={user_id} - user not found")
        return False


def list_users_by_tiers(tiers: list[SubscriptionTier]) -> list[UserInDB]:
    """List users whose tier is in the provided set."""
    if not tiers:
        return []

    tier_values = [tier.value for tier in tiers]
    placeholders = ",".join("?" for _ in tier_values)
    query = f"SELECT * FROM users WHERE tier IN ({placeholders})"

    with get_db() as conn:
        rows = conn.execute(query, tuple(tier_values)).fetchall()
    return [_row_to_user(row) for row in rows]


def _row_to_user(row: sqlite3.Row) -> UserInDB:
    """Convert database row to UserInDB model."""
    return UserInDB(
        id=row["id"],
        email=row["email"],
        first_name=row["first_name"],
        last_name=row["last_name"],
        hashed_password=row["hashed_password"],
        tier=SubscriptionTier(row["tier"]),
        email_verified=bool(row["email_verified"]),
        verification_token=row["verification_token"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ============================================================
# Social Login (Auth Identities) Functions
# ============================================================

class SocialEmailConflict(Exception):
    """Raised when a social login presents an email that already belongs to an
    existing account but cannot be safely linked because the provider did NOT
    assert the email is verified. Linking on an unverified email is an
    account-takeover vector, so we refuse rather than link or silently duplicate.
    The endpoint maps this to HTTP 409."""


def upsert_auth_identity(
    user_id: str,
    provider: str,
    subject: str,
    email_at_link: Optional[str] = None,
    raw_profile: Optional[dict] = None,
) -> None:
    """Idempotently link a provider identity to a user. Keyed on
    (provider, subject); re-linking the same identity refreshes the audit fields."""
    now = datetime.utcnow().isoformat()
    profile_json = json.dumps(raw_profile) if raw_profile else None
    email_norm = (email_at_link or "").strip().lower() or None
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO auth_identities
                (id, user_id, provider, subject, email_at_link, raw_profile, linked_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, subject) DO UPDATE SET
                email_at_link = excluded.email_at_link,
                raw_profile = COALESCE(excluded.raw_profile, auth_identities.raw_profile),
                updated_at = excluded.updated_at
            """,
            (str(uuid.uuid4()), user_id, provider, subject, email_norm, profile_json, now, now),
        )
        conn.commit()


def get_identity_by_provider_subject(provider: str, subject: str) -> Optional[dict]:
    """Look up a linked identity by its durable (provider, subject) key."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM auth_identities WHERE provider = ? AND subject = ?",
            (provider, subject),
        ).fetchone()
        return dict(row) if row else None


def get_identities_by_user_id(user_id: str) -> list[dict]:
    """All provider identities linked to a user (for account deletion / display)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM auth_identities WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_identities_for_user(user_id: str) -> int:
    """Remove all linked identities for a user. Returns the number removed."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM auth_identities WHERE user_id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount


def _delete_identity_row(identity_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM auth_identities WHERE id = ?", (identity_id,))
        conn.commit()


def _synth_social_email(provider: str, subject: str) -> str:
    """A non-deliverable, unique placeholder email for providers that omit one
    (e.g. Facebook when the email permission is denied). Keeps the NOT NULL /
    UNIQUE email invariant; such accounts are standalone and cannot be linked
    to others by email."""
    safe = "".join(ch for ch in subject if ch.isalnum() or ch in ".-_") or "user"
    return f"{provider}.{safe}@noreply.seekingbeta.ai"


def resolve_or_create_social_user(
    provider: str,
    subject: str,
    email: Optional[str],
    email_verified: bool,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    raw_profile: Optional[dict] = None,
) -> tuple[UserInDB, bool]:
    """Find-or-create the user behind a verified social identity.

    Resolution order (security rationale in SOCIAL_LOGIN_PLAN.md section D):
      1. (provider, subject) already linked -> that user. Email irrelevant.
      2. else, IF the provider asserts a VERIFIED email matching an existing
         user -> link the identity to that user (no duplicate account).
      3. else -> create a new password-less user + link the identity.

    Refuses (SocialEmailConflict) to link onto an existing account when the
    provider has NOT verified the email — an unverified-email token must never
    be handed someone else's account.

    Returns (user, created).
    """
    norm_email = (email or "").strip().lower() or None

    # 1. Existing identity wins — subject is the durable join key.
    existing = get_identity_by_provider_subject(provider, subject)
    if existing:
        user = get_user_by_id(existing["user_id"])
        if user:
            return user, False
        # Orphaned identity (user was deleted) — drop it and fall through.
        _delete_identity_row(existing["id"])

    # 2. Link by VERIFIED email only.
    if norm_email:
        matching = get_user_by_email(norm_email)
        if matching:
            if not email_verified:
                raise SocialEmailConflict(
                    "An account with this email already exists. Sign in with your "
                    "password, or use a provider that verifies your email."
                )
            upsert_auth_identity(matching.id, provider, subject, norm_email, raw_profile)
            if not matching.email_verified:
                verify_user_email(matching.id)
                matching = get_user_by_id(matching.id) or matching
            return matching, False

    # 3. Create a new password-less user.
    account_email = norm_email or _synth_social_email(provider, subject)
    now = datetime.utcnow()
    new_user = UserInDB(
        id=str(uuid.uuid4()),
        email=account_email,
        # first/last are NOT NULL with min_length=1 on the model; fall back to a
        # non-empty placeholder when the provider supplies nothing.
        first_name=(first_name or "").strip() or "Member",
        last_name=(last_name or "").strip() or "Member",
        hashed_password=SOCIAL_ONLY_PASSWORD_SENTINEL,
        tier=SubscriptionTier.FREE,
        email_verified=bool(email_verified and norm_email),
        verification_token=None,
        created_at=now,
        updated_at=now,
    )
    try:
        create_user(new_user)
    except sqlite3.IntegrityError:
        # Race: a concurrent request created the same identity or email first.
        existing = get_identity_by_provider_subject(provider, subject)
        if existing:
            raced = get_user_by_id(existing["user_id"])
            if raced:
                return raced, False
        if norm_email and email_verified:
            raced = get_user_by_email(norm_email)
            if raced:
                upsert_auth_identity(raced.id, provider, subject, norm_email, raw_profile)
                return raced, False
        raise
    upsert_auth_identity(new_user.id, provider, subject, norm_email, raw_profile)
    return new_user, True


# ============================================================
# User Oracle (Watchlist) Functions
# ============================================================

def get_user_oracle(user_id: str) -> Optional[UserOracle]:
    """Get user's oracle (watchlist and timeframes)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM user_oracle WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            return UserOracle(
                user_id=row["user_id"],
                watchlist=json.loads(row["watchlist"]),
                timeframes=json.loads(row["timeframes"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
    return None


def create_or_update_user_oracle(
    user_id: str,
    watchlist: list[str],
    timeframes: list[str],
) -> UserOracle:
    """Create or update user's oracle settings."""
    now = datetime.utcnow()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO user_oracle (user_id, watchlist, timeframes, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                watchlist = excluded.watchlist,
                timeframes = excluded.timeframes,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                json.dumps(watchlist),
                json.dumps(timeframes),
                now.isoformat(),
            ),
        )
        conn.commit()
    return UserOracle(
        user_id=user_id,
        watchlist=watchlist,
        timeframes=timeframes,
        updated_at=now,
    )


def add_to_watchlist(user_id: str, ticker: str) -> UserOracle:
    """Add a ticker to user's watchlist."""
    oracle = get_user_oracle(user_id)
    if oracle:
        watchlist = oracle.watchlist
        if ticker.upper() not in [t.upper() for t in watchlist]:
            watchlist.append(ticker.upper())
        timeframes = oracle.timeframes
    else:
        watchlist = [ticker.upper()]
        timeframes = ["5d"]
    return create_or_update_user_oracle(user_id, watchlist, timeframes)


def remove_from_watchlist(user_id: str, ticker: str) -> UserOracle:
    """Remove a ticker from user's watchlist."""
    oracle = get_user_oracle(user_id)
    if oracle:
        watchlist = [t for t in oracle.watchlist if t.upper() != ticker.upper()]
        timeframes = oracle.timeframes
    else:
        watchlist = []
        timeframes = ["5d"]
    return create_or_update_user_oracle(user_id, watchlist, timeframes)


def update_oracle_timeframes(user_id: str, timeframes: list[str]) -> UserOracle:
    """Update user's preferred timeframes."""
    oracle = get_user_oracle(user_id)
    if oracle:
        watchlist = oracle.watchlist
    else:
        watchlist = []
    return create_or_update_user_oracle(user_id, watchlist, timeframes)


# ============================================================
# Company Functions
# ============================================================

def upsert_company(ticker: str, name: Optional[str] = None, asset_type: Optional[str] = None) -> None:
    """Insert or update a company record."""
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies (ticker, name, asset_type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                name = COALESCE(excluded.name, companies.name),
                asset_type = COALESCE(excluded.asset_type, companies.asset_type),
                updated_at = excluded.updated_at
            """,
            (ticker.upper(), name, asset_type, now, now),
        )
        conn.commit()


def upsert_company_info(ticker: str, info: dict) -> None:
    """Insert or update company info (fundamentals)."""
    now = datetime.utcnow().isoformat()
    raw_info_json = json.dumps(info.get("raw_info")) if info.get("raw_info") else None
    related_fields = (
        ticker.upper(),
        info.get("sector"),
        info.get("industry"),
        info.get("description"),
        info.get("market_cap"),
        info.get("enterprise_value"),
        info.get("employees"),
        info.get("website"),
        info.get("country"),
        info.get("state"),
        info.get("city"),
        info.get("exchange"),
        info.get("currency"),
        info.get("dividend_yield"),
        info.get("beta"),
        info.get("pe_ratio"),
        info.get("forward_pe"),
        info.get("price_to_book"),
        info.get("fifty_two_week_high"),
        info.get("fifty_two_week_low"),
        info.get("avg_volume"),
        raw_info_json,
        now,
        now,
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO company_info (
                ticker, sector, industry, description,
                market_cap, enterprise_value, employees,
                website, country, state, city,
                exchange, currency,
                dividend_yield, beta, pe_ratio, forward_pe, price_to_book,
                fifty_two_week_high, fifty_two_week_low, avg_volume,
                raw_info, fetched_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                sector = excluded.sector,
                industry = excluded.industry,
                description = excluded.description,
                market_cap = excluded.market_cap,
                enterprise_value = excluded.enterprise_value,
                employees = excluded.employees,
                website = excluded.website,
                country = excluded.country,
                state = excluded.state,
                city = excluded.city,
                exchange = excluded.exchange,
                currency = excluded.currency,
                dividend_yield = excluded.dividend_yield,
                beta = excluded.beta,
                pe_ratio = excluded.pe_ratio,
                forward_pe = excluded.forward_pe,
                price_to_book = excluded.price_to_book,
                fifty_two_week_high = excluded.fifty_two_week_high,
                fifty_two_week_low = excluded.fifty_two_week_low,
                avg_volume = excluded.avg_volume,
                raw_info = excluded.raw_info,
                fetched_at = excluded.fetched_at,
                updated_at = excluded.updated_at
            """,
            related_fields,
        )
        conn.commit()


def add_company_news(ticker: str, articles: list[dict]) -> int:
    """Insert news articles, ignoring duplicates. Returns count of new articles."""
    now = datetime.utcnow().isoformat()
    inserted = 0
    with get_db() as conn:
        for article in articles:
            related_tickers_json = json.dumps(article.get("related_tickers")) if article.get("related_tickers") else None
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO company_news (
                        ticker, article_id, title, publisher, link,
                        published_at, article_type, thumbnail_url,
                        related_tickers, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker.upper(),
                        article.get("article_id"),
                        article.get("title"),
                        article.get("publisher"),
                        article.get("link"),
                        article.get("published_at"),
                        article.get("article_type"),
                        article.get("thumbnail_url"),
                        related_tickers_json,
                        now,
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0] > 0:
                    inserted += 1
            except Exception as e:
                logger.debug(f"Failed to insert news article for {ticker}: {e}")
                continue
        conn.commit()
    if inserted > 0:
        logger.debug(f"Inserted {inserted} news articles for {ticker}")
    return inserted


def get_company(ticker: str) -> Optional[dict]:
    """Get company by ticker."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM companies WHERE ticker = ?", (ticker.upper(),)
        ).fetchone()
        if row:
            return dict(row)
    return None


def get_company_info(ticker: str) -> Optional[dict]:
    """Get company info (fundamentals) by ticker."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM company_info WHERE ticker = ?", (ticker.upper(),)
        ).fetchone()
        if row:
            result = dict(row)
            if result.get("raw_info"):
                result["raw_info"] = json.loads(result["raw_info"])
            return result
    return None


def get_company_news(ticker: str, limit: int = 50) -> list[dict]:
    """Get company news articles, most recent first."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM company_news
            WHERE ticker = ?
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (ticker.upper(), limit),
        ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            if item.get("related_tickers"):
                item["related_tickers"] = json.loads(item["related_tickers"])
            results.append(item)
        return results


def get_stale_companies(hours: int = 24) -> list[str]:
    """Get tickers of companies whose info is older than `hours` hours."""
    from datetime import timedelta
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with get_db() as conn:
        # Companies that have info but it's stale
        rows = conn.execute(
            """
            SELECT c.ticker FROM companies c
            LEFT JOIN company_info ci ON c.ticker = ci.ticker
            WHERE ci.ticker IS NULL OR ci.updated_at < ?
            """,
            (cutoff,),
        ).fetchall()
        return [row["ticker"] for row in rows]


# Initialize database on module load
init_database()
