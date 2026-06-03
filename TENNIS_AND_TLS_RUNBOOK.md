# Tennis Live Feed + TLS Renewal Runbook

Covers two operational concerns surfaced during the sports expansion work:
1. how tennis boards stay current (live ESPN schedule + periodic data refresh)
2. how the site's HTTPS certificate is renewed (and why the site once went dark)

---

## 1. Tennis data: how it stays current

Tennis boards come from **two layers**:

### a) Curated board (model predictions)
`pythia_prophecy/frontend/src/data/wta_upcoming_tournaments.json` + `wta_historical_backtests.json`
are produced by `pythia_divination/scripts/export_wta_frontend_data.py` (runs the
WTA/ATP tournament-ranker model). This is the source of **which tournaments show
and their predicted fields**.

### b) Live ESPN schedule overlay (dates only)
The BFF (`pythia_prophecy/api/service.py`, `_apply_live_tennis_schedule`) fetches
the live ATP+WTA schedule from ESPN
(`site.api.espn.com/apis/site/v2/sports/tennis/{atp,wta}/scoreboard`) and
**overrides the start/end dates** on curated tournaments so in-progress majors
(e.g. Roland Garros) track their real window instead of drifting. Cached 15 min
(`TENNIS_LIVE_SCHEDULE_CACHE_TTL_SECONDS`). ESPN gives schedule, NOT the draw, so
it only corrects dates — it does not add new tournaments or fields.

> Design note: we deliberately do **not** append ESPN's full ~50-event calendar
> (those entries have no model field and ESPN naming duplicates curated events).
> A `TENNIS_UPCOMING_GRACE_DAYS` window in `_filter_upcoming_tennis` is the
> fallback that keeps an in-progress event visible if ESPN is unreachable.

### Periodic refresh (to pick up NEW tournaments + fresh predictions)
Run the full ingest -> dataset -> export chain. Weekly is plenty.

On EC2 the divination Python deps live ONLY inside the divination image, and
the export must write to the host's prophecy data dir. So the cron calls a
containerized wrapper that runs the chain in a throwaway divination container
with the host repo bind-mounted, then bounces prophecy:

```bash
# manual run on EC2:
/home/ec2-user/seekingbeta/ops/refresh_tennis_cron.sh
```

Installed cron (Mondays 06:10 UTC):
```cron
10 6 * * 1 /home/ec2-user/seekingbeta/ops/refresh_tennis_cron.sh >> /var/log/refresh_tennis.log 2>&1
```

> Local dev (Mac, with the divination venv) can still run the script directly:
> `cd pythia_divination && scripts/refresh_tennis_data.sh`.

> The refresh reuses the existing ranker artifacts (no retrain). Add a
> `python -m sports.wta.train_tournament_ranker_torch` step if you want fresh
> model weights.

> Gotcha: `export_wta_frontend_data.py` needs `from __future__ import annotations`
> to run on the Python 3.9 runtime (it uses `str | None` hints). Already added.

---

## 2. TLS certificate renewal

**Incident (2026-05-27):** the Let's Encrypt cert for `seekingbeta.ai` expired,
taking the **entire HTTPS site down** — including `/sports` and every
`/api/...` call (browser/app TLS handshake fails). Two root causes:
- no auto-renew timer/cron existed (it was being renewed by hand), and
- the renewal `authenticator = standalone` needs port 80, but the **frontend
  container already binds :80**, so a bare `certbot renew` fails.

### Cert facts
- Issuer: Let's Encrypt; domains: `seekingbeta.ai`, `www.seekingbeta.ai`
- Lives on the host at `/etc/letsencrypt/...`, mounted read-only into the
  frontend nginx container (`docker-compose.yml`: `/etc/letsencrypt:/etc/letsencrypt:ro`)
- nginx reads `…/live/seekingbeta.ai/{fullchain,privkey}.pem`

### Manual renewal (if ever needed)
The renewal config now has **pre/post hooks** that stop/start the frontend so
standalone can bind :80, so this is all you need:
```bash
sudo certbot renew            # add --force-renewal to renew before the 30-day window
sudo certbot certificates     # confirm new Expiry Date
```
The frontend re-reads the mounted cert on restart (handled by the post-hook).

### Auto-renewal (installed)
A systemd timer runs `certbot renew` twice daily; certbot only acts within 30
days of expiry. Unit files are version-controlled at `ops/certbot-renew.{service,timer}`.
```bash
# install / re-install:
sudo cp ops/certbot-renew.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now certbot-renew.timer
systemctl list-timers certbot-renew.timer    # check next run
sudo certbot renew --dry-run                  # prove the path end-to-end
```

### Monitoring
Quick external check (alert if < ~10 days):
```bash
echo | openssl s_client -connect seekingbeta.ai:443 -servername seekingbeta.ai 2>/dev/null \
  | openssl x509 -noout -enddate
```

---

## 3. ML model artifacts (sports + stock)

**Why this matters:** the sports boards (tennis ranker, pga, mlb/basketball/
football models) load trained artifacts from `pythia_divination/artifacts/`.
If those files are missing, the export step fails and boards go empty/stale.

### How artifacts reach the server
- **Source of truth on EC2:** the host dir `pythia_divination/artifacts/`,
  bind-mounted into the container (`docker-compose.yml`:
  `./pythia_divination/artifacts:/app/artifacts`). This **survives container
  rebuilds/recreates** — it does NOT survive a host/EBS loss.
- **Intended durable store:** S3 bucket `pythia-ml-artifacts`, auto-pulled on
  container start by `scripts/download_artifacts.sh` when
  `AUTO_DOWNLOAD_ARTIFACTS=true`.
- **Interim backup:** a tarball on the host at
  `/home/ec2-user/artifact-backups/divination-artifacts-<date>.tgz`.
  Restore: `tar xzf <tarball> -C /home/ec2-user/seekingbeta/pythia_divination`.

### ⚠️ Known gap (action required)
As of 2026-06-03 the S3 bucket holds **only the 6 stock models** — no sports
artifacts — and the `pythia-app` IAM user is **read-only** (`s3:PutObject`
denied), so the bucket can't be populated from the app credentials. Until that's
fixed, the host bind-mount + tarball are the only copies of the sports
artifacts. **The artifacts themselves are gitignored (382MB) and are NOT in the
repo.**

The image side is already wired and waiting:
- `Dockerfile` installs `awscli`.
- `download_artifacts.sh` now requires a sports artifact (not just a stock
  model) before skipping the download.

**To finish self-healing once an IAM user with `s3:PutObject` exists:**
```bash
# from EC2 (host has aws cli), with write-capable creds in the env:
cd /home/ec2-user/seekingbeta
aws s3 sync pythia_divination/artifacts/ s3://pythia-ml-artifacts/artifacts/artifacts/ --region us-east-1
# then enable auto-download and recreate divination:
echo 'AUTO_DOWNLOAD_ARTIFACTS=true' >> .env
docker compose up -d --force-recreate divination-api
```
After that, a fresh box restores all artifacts automatically on start.

### Adding/refreshing an artifact
Train locally, then get it onto EC2 (until S3 write works): stream it over ssh
```bash
tar czf - artifacts/<new_model> | ssh -i Pythia.pem ec2-user@<host> \
  'cd /home/ec2-user/seekingbeta/pythia_divination && tar xzf -'
```
and re-tar the host backup.
