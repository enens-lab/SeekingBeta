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

```bash
# on EC2 (or locally), from the divination dir with its venv:
cd /home/ec2-user/seekingbeta/pythia_divination
scripts/refresh_tennis_data.sh            # defaults: 2020..current year
# then serve the refreshed JSON:
cd /home/ec2-user/seekingbeta && docker compose up -d --build prophecy-api
```

Suggested cron (Mondays 06:10 UTC):
```cron
10 6 * * 1 cd /home/ec2-user/seekingbeta/pythia_divination && PYTHON=.venv/bin/python scripts/refresh_tennis_data.sh >> /var/log/refresh_tennis.log 2>&1
```

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
