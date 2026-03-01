# Reliability Hardening Runbook

This runbook covers:

- Postgres backups
- Restore testing
- Runtime alert checks (5xx spikes, webhook failures, container restarts)
- Auth + webhook rate limit controls
- Pre-deploy safety checks (required backtests + TLS wiring)

## 1) Backup and restore scripts

Scripts:

- `/Users/huyngo/Downloads/pythia/scripts/ops/postgres_backup.sh`
- `/Users/huyngo/Downloads/pythia/scripts/ops/postgres_restore_test.sh`
- `/Users/huyngo/Downloads/pythia/scripts/ops/sqlite_restore_test.sh`

Manual backup:

```bash
cd /home/ec2-user/seekingbeta
bash scripts/ops/postgres_backup.sh
```

Manual restore test (uses latest backup by default):

```bash
cd /home/ec2-user/seekingbeta
bash scripts/ops/postgres_restore_test.sh
bash scripts/ops/sqlite_restore_test.sh
```

Restore test against a specific backup:

```bash
cd /home/ec2-user/seekingbeta
bash scripts/ops/postgres_restore_test.sh backups/postgres/pythia_20260301T010000Z.dump.gz
```

## 2) Runtime alert checks

Script:

- `/Users/huyngo/Downloads/pythia/scripts/ops/runtime_alerts.sh`

Manual run:

```bash
cd /home/ec2-user/seekingbeta
bash scripts/ops/runtime_alerts.sh
```

Optional webhook alert destination:

```bash
cd /home/ec2-user/seekingbeta
ALERT_WEBHOOK_URL="https://hooks.slack.com/services/..." \
WINDOW_MINUTES=10 HTTP_5XX_THRESHOLD=20 WEBHOOK_FAILURE_THRESHOLD=1 \
bash scripts/ops/runtime_alerts.sh
```

Exit codes:

- `0`: healthy
- `2`: alert condition found
- `1`: script/config error

## 3) Suggested cron schedule on EC2

Open crontab:

```bash
crontab -e
```

Add:

```cron
# Ensure log directory exists first:
# mkdir -p /home/ec2-user/seekingbeta/logs

# Hourly backup
0 * * * * cd /home/ec2-user/seekingbeta && bash scripts/ops/postgres_backup.sh >> logs/postgres_backup.log 2>&1

# Daily restore drill at 03:15 UTC
15 3 * * * cd /home/ec2-user/seekingbeta && bash scripts/ops/postgres_restore_test.sh >> logs/postgres_restore_test.log 2>&1
20 3 * * * cd /home/ec2-user/seekingbeta && bash scripts/ops/sqlite_restore_test.sh >> logs/sqlite_restore_test.log 2>&1

# Runtime alerts every 5 minutes
*/5 * * * * cd /home/ec2-user/seekingbeta && bash scripts/ops/runtime_alerts.sh >> logs/runtime_alerts.log 2>&1
```

## 4) Rate-limit env controls (prophecy-api)

These are now configurable in `docker-compose.yml`:

- Auth:
  - `AUTH_SIGNUP_RATE_LIMIT`
  - `AUTH_SIGNUP_RATE_WINDOW_SECONDS`
  - `AUTH_SIGNUP_IP_RATE_LIMIT`
  - `AUTH_SIGNUP_IP_RATE_WINDOW_SECONDS`
  - `AUTH_LOGIN_RATE_LIMIT`
  - `AUTH_LOGIN_RATE_WINDOW_SECONDS`
  - `AUTH_LOGIN_IP_RATE_LIMIT`
  - `AUTH_LOGIN_IP_RATE_WINDOW_SECONDS`
  - `AUTH_VERIFY_RATE_LIMIT`
  - `AUTH_VERIFY_RATE_WINDOW_SECONDS`
  - `AUTH_RESEND_RATE_LIMIT`
  - `AUTH_RESEND_RATE_WINDOW_SECONDS`
  - `AUTH_PASSWORD_RESET_RATE_LIMIT`
  - `AUTH_PASSWORD_RESET_RATE_WINDOW_SECONDS`
  - `AUTH_PASSWORD_RESET_CONFIRM_RATE_LIMIT`
  - `AUTH_PASSWORD_RESET_CONFIRM_RATE_WINDOW_SECONDS`
  - `AUTH_REFRESH_RATE_LIMIT`
  - `AUTH_REFRESH_RATE_WINDOW_SECONDS`
  - `AUTH_CHANGE_PASSWORD_RATE_LIMIT`
  - `AUTH_CHANGE_PASSWORD_RATE_WINDOW_SECONDS`
  - `AUTH_DELETE_ACCOUNT_RATE_LIMIT`
  - `AUTH_DELETE_ACCOUNT_RATE_WINDOW_SECONDS`
- Webhooks:
  - `SES_WEBHOOK_RATE_LIMIT`
  - `SES_WEBHOOK_RATE_WINDOW_SECONDS`
  - `STRIPE_WEBHOOK_RATE_LIMIT`
  - `STRIPE_WEBHOOK_RATE_WINDOW_SECONDS`

After changing limits:

```bash
cd /home/ec2-user/seekingbeta
bash scripts/ops/pre_deploy_check.sh
docker compose up -d --build --force-recreate prophecy-api
```

## 5) Pre-deploy safety checks

Script:

- `/Users/huyngo/Downloads/pythia/scripts/ops/pre_deploy_check.sh`

Manual run:

```bash
cd /home/ec2-user/seekingbeta
bash scripts/ops/pre_deploy_check.sh
```

What it validates:

- Required backtest artifacts exist:
  - `production_trade_log.csv`
  - `production_metrics.json`
  - `jackpot_trade_log.csv`
  - `jackpot_metrics.json`
- `docker-compose.yml` includes the backtest mount:
  - `./pythia_prophecy/data/backtests:/app/data/backtests:ro`
- If frontend nginx is TLS-enabled (`listen 443 ssl;`), compose must include:
  - frontend port mapping `443:443`
  - letsencrypt mount `/etc/letsencrypt:/etc/letsencrypt:ro`

Optional cert file check (recommended on EC2 deploys):

```bash
cd /home/ec2-user/seekingbeta
CHECK_LOCAL_CERT_FILES=true bash scripts/ops/pre_deploy_check.sh
```

## 6) Daily backtest artifact refresh

Script:

- `/Users/huyngo/Downloads/pythia/scripts/ops/sync_backtest_artifacts.sh`
- `/Users/huyngo/Downloads/pythia/scripts/ops/refresh_backtests_daily.sh`

Expected source files in `SOURCE_DIR` (default: `./backtest_results`):

- `production_trade_log.csv`
- `production_metrics.json`
- `jackpot_trade_log.csv`
- `jackpot_metrics.json`

Manual sync:

```bash
cd /home/ec2-user/seekingbeta
SOURCE_DIR=/home/ec2-user/Stock_Prediction_Model/backtest_results \
bash scripts/ops/sync_backtest_artifacts.sh
```

The script:

- Validates all required files exist and are non-empty
- Copies files into `pythia_prophecy/data/backtests`
- Recreates `prophecy-api` by default to clear in-memory performance cache

Daily cron example (after your backtest job finishes):

```cron
# Daily backtest artifact sync at 02:10 UTC
10 2 * * * cd /home/ec2-user/seekingbeta && SOURCE_DIR=/home/ec2-user/Stock_Prediction_Model/backtest_results bash scripts/ops/sync_backtest_artifacts.sh >> logs/backtest_sync.log 2>&1
```

End-to-end daily run + sync (recommended):

```cron
# Run both backtests (choice=3) then sync + restart prophecy-api
10 2 * * * cd /home/ec2-user/seekingbeta && BACKTEST_SCRIPT_PATH=/home/ec2-user/Stock_Prediction_Model/run_backtest_v4_1.py bash scripts/ops/refresh_backtests_daily.sh >> logs/backtest_refresh.log 2>&1
```
