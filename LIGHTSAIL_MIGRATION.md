# Moving seekingbeta.ai from EC2 (m7i-flex.large, ~$74/mo) to Lightsail ($12/mo)

Status: **ready to execute**. The web box no longer needs a model server: since
2026-09-19 stock predictions are a daily RunPod batch (`runpod/sports/handler.py`
`stock_predictions`) that prophecy-api serves from a file, the live sports slates
come from the RunPod exports, and `divination-api` is behind a compose profile.
What runs on the box is nginx + prophecy-api (~0.6 GB) + postgres (~0.1 GB).

| | EC2 today | Lightsail target |
|---|---|---|
| Plan | m7i-flex.large, 2 vCPU / 8 GB, 50 GB gp3 | **$12/mo: 2 vCPU / 2 GB / 60 GB SSD / 3 TB transfer** |
| Cost | ~$70 compute + ~$4 EBS + EIP | $12 flat (static IP free while attached) |
| RAM used | ~1.5 GB (was ~4.5 GB with divination) | same workload, 2 GB + 2 GB swap |
| Daily builds | prophecy rebuilt on every board refresh | none (boards + sweep are bind mounts) |

If 2 GB proves tight in practice (watch `free -m` after a week), the $24 plan
(4 GB) is the next step; nothing else changes.

## 0. Before the cutover day (you)

1. **Lightsail console → Create instance**: region **Ohio (us-east-2)** (same as
   the EC2 box and the backups bucket), platform Linux, blueprint **OS only →
   Amazon Linux 2023**, plan **$12**. Name it `seekingbeta-web`.
   Under "SSH key pair" pick/download a key (or upload the public half of
   `Pythia.pem`). Default user is `ec2-user`, same as EC2.
2. **Networking tab** of the instance: add a **static IP** and attach it. Firewall:
   keep SSH (22) but restrict it to your IP; add **HTTP 80** and **HTTPS 443**.
3. **Namecheap DNS** (the zone lives at `dns1/dns2.registrar-servers.com`): lower
   the TTL of the `@` and `www` A records to **60 seconds** now, so the switch
   propagates in a minute instead of an hour. Do not change the IPs yet.
4. Tell me the static IP and which key opens it. I do the rest over SSH.

## 1. Prepare the new box (me, ~10 min)

```bash
ssh -i <key> ec2-user@<static-ip>
curl -fsSL https://raw.githubusercontent.com/enens-lab/SeekingBeta/feature/sports-prediction-market/ops/lightsail_bootstrap.sh | bash
# log out / in so the docker group applies
```

`ops/lightsail_bootstrap.sh` installs docker + compose, git, aws cli, certbot,
adds 2 GB swap and docker log rotation, and clones the repo to
`/home/ec2-user/seekingbeta` (identical path, so the crontab and every ops script
transfer unchanged).

## 2. Move the state (me, ~10 min, ~2 min of write downtime)

On the **old** box:
```bash
cd /home/ec2-user/seekingbeta && bash ops/export_state_bundle.sh --consistent
```
`--consistent` stops prophecy-api and postgres for the copy (nginx keeps serving
the static site; API calls fail for ~1 minute) so Postgres and the SQLite users DB
are exact. The bundle holds `.env`, `pythia_prophecy/secrets/`, the two docker
volumes, `/etc/letsencrypt`, the boards, today's stock sweep, the artifacts, the
crontab and the small home dirs. It contains live secrets: it goes box-to-box
over SSH only and is deleted afterwards.

```bash
# from your Mac (or straight between boxes if you add the key):
scp -i Pythia.pem ec2-user@18.119.71.168:/home/ec2-user/seekingbeta-state-*.tgz /tmp/
scp -i <newkey> /tmp/seekingbeta-state-*.tgz ec2-user@<static-ip>:/home/ec2-user/
```

On the **new** box:
```bash
cd /home/ec2-user/seekingbeta && bash ops/import_state_bundle.sh /home/ec2-user/seekingbeta-state-*.tgz
```
This restores everything, builds prophecy-api + frontend, starts postgres,
prophecy-api and frontend, installs the crontab and enables the certbot renewal
timer (renewal uses `--standalone` on :80 with the pre/post hooks that stop and
start nginx; that config travels inside `/etc/letsencrypt`).

Check before switching DNS (the cert is for seekingbeta.ai, so `-k` locally):
```bash
curl -sk https://localhost/healthz
curl -sk "https://localhost/api/sports/boards?sports=golf&include_backtests=false" | head -c 300
curl -sk https://localhost/predict/homepage | head -c 300
curl -sk -H "Host: seekingbeta.ai" https://localhost/ | head -c 200
free -m; docker stats --no-stream
```

## 3. Cutover (you: 2 DNS edits; me: verification)

1. Namecheap: point `@` and `www` A records at the Lightsail static IP.
2. Within ~2 minutes: `dig +short seekingbeta.ai` shows the new IP; I verify
   sign-in, sports boards, the homepage cards, `/api/market/history/AAPL` (mobile),
   Stripe webhook delivery (Stripe dashboard → Webhooks → resend one test), and
   that the first hourly backup lands in S3.
3. Old box: leave it **stopped** (not terminated) for one week as the rollback
   (stopped instances only bill EBS, ~$4/mo). If nothing regresses, terminate it,
   delete the volume and **release the Elastic IP** (an unattached EIP bills hourly).
4. Restore the A-record TTL to 30 min once stable.

## What changes for the crons (already in the bundle's crontab)

| Cron | Note |
|---|---|
| hourly backup + S3 upload | unchanged (uses `.env` AWS keys) |
| `*/5` runtime alerts | no divination checks; alerts if the sweep is >54h old |
| `10 6 * * 1` weekly + `10 7 * * *` daily sports refresh | no prophecy rebuild any more (bind mount) |
| `30 21 * * 1-5` options archive | unchanged (RunPod) |
| **`15 22 * * 1-5` stock sweep** | new: `ops/refresh_stock_predictions_runpod.sh` |
| `0 12 * * *` Daily Brief, `0 13 * * 1` Receipts | unchanged |

Things that are **not** on the box any more and where they went:
- LSTM inference → RunPod `stock_predictions` (~$0.05-0.10 per run at ~4 min on
  the 16 GB CPU endpoint, weekdays only).
- `/api/market/history` for the mobile apps → prophecy fetches Yahoo directly.
- Attribution (integrated gradients) → 501, as it already was for the torch models.
- Options archive Postgres ingest (`ops/refresh_options_runpod.sh` runs a one-off
  `docker compose run divination-api`) → still works: `run` enables the `ml`
  profile implicitly, but it needs the divination image built once on the new box
  (`docker compose --profile ml build divination-api`, ~4 GB of disk, ~10 min).
  Skip it unless the archive ingest is actually wanted on the web box.

## Rollback

DNS back to `18.119.71.168` and `docker compose up -d` on the old box (its data is
at most one hour behind the new box's via the volumes; anything users changed
after cutover on the new box is lost, so decide within the day).
