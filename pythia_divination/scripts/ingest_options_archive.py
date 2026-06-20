"""Ingest a day's options-archive parquet from S3 into Postgres.

Companion to the RunPod options worker (runpod/sports/handler.py, type=options_archive),
which snapshots the daily options features and writes the parquet to
s3://$S3_BUCKET/$OPTIONS_ARCHIVE_S3_PREFIX/dt=<date>/options_features_<date>.parquet
(it can't reach the private Postgres). This runs ON the box (inside the
divination container, which has DATABASE_URL + AWS creds) to upsert that parquet
into options_features_daily — the same table the on-box archive wrote to.

Usage (inside the divination container):
    python scripts/ingest_options_archive.py [YYYY-MM-DD]   # default: today (UTC)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

import pandas as pd

from storage.market_data_store import (
    OPTIONS_FEATURE_COLUMNS,
    is_enabled,
    upsert_options_features_daily,
)


def main() -> None:
    trade_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bucket = os.getenv("S3_BUCKET", "pythia-ml-artifacts")
    region = os.getenv("AWS_REGION", "us-east-1")
    prefix = os.getenv("OPTIONS_ARCHIVE_S3_PREFIX", "options_archive").strip("/")
    s3_uri = f"s3://{bucket}/{prefix}/dt={trade_date}/options_features_{trade_date}.parquet"

    if not is_enabled():
        print("DATABASE_URL not set; cannot ingest options archive", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / f"options_features_{trade_date}.parquet"
        subprocess.run(
            ["aws", "s3", "cp", s3_uri, str(local), "--region", region, "--only-show-errors"],
            check=True,
        )
        df = pd.read_parquet(local)

    stored = 0
    for record in df.to_dict(orient="records"):
        feats = {c: record.get(c) for c in OPTIONS_FEATURE_COLUMNS}
        td = pd.to_datetime(record["trade_date"]).date()
        upsert_options_features_daily(
            symbol=str(record["symbol"]),
            trade_date=td,
            source=str(record.get("source") or "unknown"),
            features=feats,
            underlying_price=record.get("underlying_price"),
            options_live=bool(record.get("options_live")),
        )
        stored += 1

    print(f"Ingested {stored} options rows for {trade_date} from {s3_uri}")


if __name__ == "__main__":
    main()
