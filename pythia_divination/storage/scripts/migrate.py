"""
Migration runner using asyncpg.
Usage:
  python -m storage.scripts.migrate
  python -m storage.scripts.migrate status
"""

import sys, glob, asyncio, datetime as dt
from pathlib import Path

# Repo root + settings
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.settings import settings  # centralized config

import asyncpg

DATABASE_URL = settings.database_url
MIGRATIONS_DIR = REPO_ROOT / "storage" / "migrations"

async def ensure_migrations_table(conn):
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version TEXT PRIMARY KEY,
          applied_at TIMESTAMPTZ NOT NULL
        )
    """)

async def applied_versions(conn):
    rows = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
    return {r["version"] for r in rows}

async def apply_migration(conn, path, version):
    sql = Path(path).read_text(encoding="utf-8")
    async with conn.transaction():
        await conn.execute(sql)
        await conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES($1,$2)",
            version, dt.datetime.utcnow()
        )

async def main(show_status=False):
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await ensure_migrations_table(conn)
        done = await applied_versions(conn)

        files = sorted(glob.glob(str(MIGRATIONS_DIR / '*.sql')))
        if show_status:
            print("Applied migrations:")
            for v in sorted(done):
                print(f"  {v}")
            print("\nPending migrations:")
            for f in files:
                v = Path(f).name
                if v not in done:
                    print(f"  {v}")
            return

        pending = [f for f in files if Path(f).name not in done]
        if not pending:
            print("No pending migrations.")
            return

        print("Applying migrations:")
        for f in pending:
            v = Path(f).name
            print(f"  -> {v} ...", end="", flush=True)
            try:
                await apply_migration(conn, f, v)
                print(" OK")
            except Exception as e:
                print(" FAILED")
                print(f"\nError applying {v}:\n{e}", file=sys.stderr)
                sys.exit(2)

        print("All pending migrations applied.")
    finally:
        await conn.close()

if __name__ == "__main__":
    import sys
    asyncio.run(main(show_status=(len(sys.argv) > 1 and sys.argv[1] == "status")))
