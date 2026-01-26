"""
Database layer for user management
Uses SQLite for simplicity - can be swapped for PostgreSQL
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

from .models import UserInDB, SubscriptionTier, UserOracle
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
                timeframes TEXT NOT NULL DEFAULT '["1d"]',
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
        timeframes = ["1d"]
    return create_or_update_user_oracle(user_id, watchlist, timeframes)


def remove_from_watchlist(user_id: str, ticker: str) -> UserOracle:
    """Remove a ticker from user's watchlist."""
    oracle = get_user_oracle(user_id)
    if oracle:
        watchlist = [t for t in oracle.watchlist if t.upper() != ticker.upper()]
        timeframes = oracle.timeframes
    else:
        watchlist = []
        timeframes = ["1d"]
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
