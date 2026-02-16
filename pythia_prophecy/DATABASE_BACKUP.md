# Database Backup & Recovery Strategy

This document outlines the backup and disaster recovery procedures for the Pythia Prophecy SQLite database.

## Database Overview

- **Type**: SQLite3
- **Location**: `pythia_prophecy/data/pythia.db`
- **Backend**: FastAPI with async SQLite access
- **Data**: User accounts, authentication tokens, preferences, watchlists, oracle data

## Backup Architecture

### Backup Strategy: 3-2-1 Rule

Follow the industry-standard 3-2-1 backup strategy:
- **3 copies** of data (original + 2 backups)
- **2 different storage types** (local + remote)
- **1 off-site backup** (cloud or external location)

### Backup Schedule

```
Daily:   Automated daily backup at 2 AM UTC
Weekly:  Full backup every Sunday at 1 AM UTC
Monthly: Archive backup on the 1st of each month
On-Demand: Manual backups before major deployments
```

## Local Backup Procedures

### 1. File-Based Backup (Easiest for Small DBs)

Since SQLite stores the entire database in a single file:

```bash
# Create backup directory
mkdir -p backups/daily backups/weekly backups/monthly

# Daily backup
cp pythia_prophecy/data/pythia.db backups/daily/pythia_$(date +%Y-%m-%d).db

# Compress to save space
gzip backups/daily/pythia_$(date +%Y-%m-%d).db

# Verify backup integrity
sqlite3 backups/daily/pythia_$(date +%Y-%m-%d).db.gz "PRAGMA integrity_check;" 2>&1 | head -1
```

### 2. SQL Dump Backup

For a text-based backup that's portable:

```bash
# Export entire database to SQL dump
sqlite3 pythia_prophecy/data/pythia.db .dump > backups/daily/pythia_$(date +%Y-%m-%d).sql

# Compress the dump
gzip backups/daily/pythia_$(date +%Y-%m-%d).sql

# Verify dump is readable
gunzip -c backups/daily/pythia_$(date +%Y-%m-%d).sql.gz | head -20
```

### 3. Automated Backup Script

Create `scripts/backup.sh`:

```bash
#!/bin/bash
set -e

BACKUP_DIR="backups"
DB_FILE="pythia_prophecy/data/pythia.db"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_FILE="${BACKUP_DIR}/daily/pythia_${TIMESTAMP}.db"

# Ensure backup directories exist
mkdir -p ${BACKUP_DIR}/{daily,weekly,monthly}

# Create backup (SQLite WAL-aware copy)
cp "${DB_FILE}" "${BACKUP_FILE}"
echo "Database backup created: ${BACKUP_FILE}"

# Verify backup
if sqlite3 "${BACKUP_FILE}" "PRAGMA integrity_check;" | grep -q "ok"; then
    echo "✓ Backup integrity verified"
    gzip "${BACKUP_FILE}"
    echo "✓ Backup compressed"
else
    echo "✗ Backup verification failed! Removing corrupted backup."
    rm "${BACKUP_FILE}"
    exit 1
fi

# Keep only last 7 daily backups (cleanup old files)
find ${BACKUP_DIR}/daily -name "*.db.gz" -mtime +7 -delete
echo "✓ Old backups cleaned up"

# Create weekly backup (every Sunday)
if [ "$(date +%A)" = "Sunday" ]; then
    cp "${BACKUP_FILE}.gz" "${BACKUP_DIR}/weekly/pythia_week_$(date +%Y-W%V).db.gz"
    echo "✓ Weekly backup created"
fi

# Create monthly backup (1st of month)
if [ "$(date +%d)" = "01" ]; then
    cp "${BACKUP_FILE}.gz" "${BACKUP_DIR}/monthly/pythia_$(date +%Y-%m).db.gz"
    echo "✓ Monthly backup created"
fi

echo "Backup process completed at $(date)"
```

Run with cron:
```bash
# Add to crontab
0 2 * * * cd /path/to/pythia_prophecy && bash scripts/backup.sh >> logs/backup.log 2>&1
```

## Remote Backup Procedures

### AWS S3 Backup

For production, automate backups to AWS S3:

```bash
#!/bin/bash
BACKUP_FILE="backups/daily/pythia_$(date +%Y-%m-%d).db.gz"

# Upload to S3
aws s3 cp "${BACKUP_FILE}" \
  "s3://pythia-backups/$(date +%Y/%m)/${BACKUP_FILE##*/}" \
  --sse AES256 \
  --storage-class STANDARD_IA

# Enable versioning on S3 bucket for additional protection
aws s3api put-bucket-versioning \
  --bucket pythia-backups \
  --versioning-configuration Status=Enabled
```

### Google Cloud Storage Backup

Alternative cloud backup option:

```bash
#!/bin/bash
BACKUP_FILE="backups/daily/pythia_$(date +%Y-%m-%d).db.gz"

# Upload to GCS
gsutil cp "${BACKUP_FILE}" \
  "gs://pythia-backups/$(date +%Y/%m)/${BACKUP_FILE##*/}"

# Set lifecycle policy to delete old backups after 90 days
gsutil lifecycle set - "gs://pythia-backups" << 'EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90}
      }
    ]
  }
}
EOF
```

## Recovery Procedures

### Full Database Recovery

#### From Latest Daily Backup:

```bash
# 1. Stop the application
docker-compose down

# 2. Locate latest backup
LATEST_BACKUP=$(ls -t backups/daily/*.db.gz | head -1)

# 3. Decompress backup
gunzip -c "${LATEST_BACKUP}" > pythia_prophecy/data/pythia_recovered.db

# 4. Verify recovery
sqlite3 pythia_prophecy/data/pythia_recovered.db "PRAGMA integrity_check;"

# 5. Restore if verified
if [ $? -eq 0 ]; then
    mv pythia_prophecy/data/pythia.db pythia_prophecy/data/pythia.db.backup
    mv pythia_prophecy/data/pythia_recovered.db pythia_prophecy/data/pythia.db
    echo "Database recovered from: ${LATEST_BACKUP}"
else
    echo "Recovery failed - backup corrupted"
    exit 1
fi

# 6. Restart application
docker-compose up -d
```

#### From SQL Dump:

```bash
# 1. Stop application
docker-compose down

# 2. Get dump file
DUMP_FILE=$(ls -t backups/daily/*.sql.gz | head -1)

# 3. Create new database from dump
rm pythia_prophecy/data/pythia.db
gunzip -c "${DUMP_FILE}" | sqlite3 pythia_prophecy/data/pythia.db

# 4. Verify
sqlite3 pythia_prophecy/data/pythia.db "SELECT COUNT(*) FROM sqlite_master WHERE type='table';"

# 5. Restart
docker-compose up -d
```

### Point-in-Time Recovery

For recovering to a specific point before corruption:

```bash
# List available backups with timestamps
ls -lh backups/daily/*.db.gz | awk '{print $6, $7, $8, $9}'

# Restore from specific timestamp
RECOVERY_BACKUP="backups/daily/pythia_2025-02-14_02-00-00.db.gz"
gunzip -c "${RECOVERY_BACKUP}" > pythia_prophecy/data/pythia.db
```

## Disaster Recovery Plan

### Scenario 1: Accidental Data Deletion

1. Identify when deletion occurred
2. Restore from backup taken BEFORE deletion
3. Verify data integrity
4. Notify affected users if necessary
5. Document incident

### Scenario 2: Database Corruption

1. Stop application immediately (prevent further writes)
2. Check backup integrity
3. Restore from backup
4. Verify with `PRAGMA integrity_check;`
5. Restart application

### Scenario 3: Ransomware/Security Breach

1. Isolate affected systems
2. Restore from known-good off-site backup
3. Change all credentials (JWT secret, DB passwords)
4. Audit application security
5. Restore and test in isolated environment first

## Backup Best Practices

### DO's
- ✅ Test recovery procedures regularly (monthly)
- ✅ Store backups in multiple locations
- ✅ Verify backup integrity after creation
- ✅ Use compression to save storage space
- ✅ Maintain backup logs and audit trails
- ✅ Document recovery procedures
- ✅ Practice disaster recovery drills

### DON'Ts
- ❌ Store all backups in the same location as the database
- ❌ Skip integrity checks
- ❌ Keep backups in unencrypted cloud storage
- ❌ Forget about backup maintenance and cleanup
- ❌ Assume backups work without testing
- ❌ Store database backups on the same disk

## Monitoring & Alerts

### Backup Health Checks

```python
# Add to your monitoring system
import os
from datetime import datetime, timedelta

def check_backup_health():
    """Verify recent backup exists and is healthy."""
    backup_dir = "backups/daily"

    # Check for recent backup (created in last 25 hours)
    now = datetime.now()
    recent_backups = []

    for backup in os.listdir(backup_dir):
        filepath = os.path.join(backup_dir, backup)
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

        if now - mtime < timedelta(hours=25):
            recent_backups.append(filepath)

    if not recent_backups:
        alert("NO_RECENT_BACKUP", "No backup created in last 25 hours")
        return False

    # Verify latest backup
    latest = max(recent_backups, key=os.path.getctime)

    try:
        import sqlite3
        conn = sqlite3.connect(latest)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()[0]
        conn.close()

        if result == "ok":
            return True
        else:
            alert("BACKUP_CORRUPTION", f"Backup integrity check failed: {result}")
            return False
    except Exception as e:
        alert("BACKUP_ERROR", f"Error verifying backup: {e}")
        return False
```

## Migration to Production Database

When moving to production with larger datasets:

1. **Move to PostgreSQL** (recommended for production)
   - Better concurrency handling
   - More mature backup tools
   - Easier scaling
   - Better security features

2. **Migration Steps**:
   - Export SQLite data to SQL
   - Create PostgreSQL database
   - Import data with pg_restore
   - Test thoroughly
   - Update connection strings
   - Deploy and verify

## Backup Storage Costs

Estimated costs for 12-month backup retention:

| Storage Type | Daily Copies | Cost/Month | Annual Cost |
|--------------|-------------|-----------|------------|
| Local SSD    | 7 days      | $0        | $0         |
| S3 Standard  | 90 days     | $1-2      | $15-24     |
| S3 Glacier   | 365 days    | $0.40     | $5-10      |
| GCS Archive  | 365 days    | $0.04/GB  | $0.50-1    |

*Costs assume ~10MB database size*

## Checklist for Production Deployment

- [ ] Automated backup script configured
- [ ] Remote backup (S3/GCS) configured
- [ ] Backup integrity checks pass
- [ ] Recovery procedure tested and documented
- [ ] Backup monitoring and alerts active
- [ ] Backup retention policy defined
- [ ] Off-site backup routine verified
- [ ] Disaster recovery plan approved
- [ ] Team trained on recovery procedures
- [ ] Compliance requirements met (GDPR, etc.)

## References

- SQLite Documentation: https://www.sqlite.org/backup.html
- AWS S3 Backup Best Practices: https://docs.aws.amazon.com/AmazonS3/latest/userguide/
- Database Backup & Recovery: https://en.wikipedia.org/wiki/Backup_and_recovery
