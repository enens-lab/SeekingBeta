#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Prepare a fresh Lightsail instance (Ubuntu 24.04 or Amazon Linux 2023) to run
# SeekingBeta.
# ==============================================================================
# Run ONCE on the new box as its default user (ubuntu on Ubuntu, ec2-user on AL2023),
# right after first SSH:
#   scp -i <key> ops/lightsail_bootstrap.sh <user>@<ip>:/tmp/ && ssh -i <key> <user>@<ip> bash /tmp/lightsail_bootstrap.sh
#
# Idempotent. What it does and why:
#   * docker + compose plugin, git, aws cli v2, cron, certbot: the same host
#     tools the EC2 box used. python3 (3.9 on AL2023, 3.12 on Ubuntu) runs the
#     ops validators.
#   * a 2 GB swapfile: the $12 plan has 2 GB RAM; steady state is ~1.2 GB
#     (prophecy ~0.6, postgres ~0.1, nginx, docker, OS). Images are NOT built
#     here (they come prebuilt in the state bundle), but swap keeps a memory
#     spike from OOM-killing the app.
#   * docker log rotation: unbounded json logs filled the old box's disk twice.
#   * /home/ec2-user -> the real home, when the user is not ec2-user. Every ops
#     script defaults to /home/ec2-user/seekingbeta, the crontab is copied
#     verbatim and the certbot renewal hooks cd into that path; one symlink
#     keeps all of it working without editing anything.
#   * clones the repo into <home>/seekingbeta.
# After this: log out/in (docker group), then ops/import_state_bundle.sh <bundle.tgz>.
# ==============================================================================
REPO_URL="${REPO_URL:-https://github.com/enens-lab/SeekingBeta.git}"
BRANCH="${BRANCH:-feature/sports-prediction-market}"
ME="$(id -un)"; HOME_DIR="$(getent passwd "$ME" | cut -d: -f6)"
DEPLOY_DIR="${DEPLOY_DIR:-$HOME_DIR/seekingbeta}"
SWAP_GB="${SWAP_GB:-2}"

log() { echo "[bootstrap] $*"; }
ARCH=$(uname -m); [ "$ARCH" = "aarch64" ] && CARCH=aarch64 || CARCH=x86_64

if command -v apt-get >/dev/null 2>&1; then
  log "packages (apt)"
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  # Ubuntu's docker.io + docker-compose-v2 give `docker compose` (v2 plugin) with
  # no third-party repo; certbot from apt ships certbot.timer for renewals.
  sudo apt-get install -y -qq docker.io docker-compose-v2 git cron python3 certbot unzip tar gzip curl >/dev/null
elif command -v dnf >/dev/null 2>&1; then
  log "packages (dnf)"
  sudo dnf -y -q install docker git cronie python3 certbot unzip tar gzip >/dev/null
  if ! docker compose version >/dev/null 2>&1; then
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${CARCH}" -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  fi
else
  log "unsupported distro (need apt or dnf)"; exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  log "aws cli v2"
  (cd /tmp && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${CARCH}.zip" -o awscliv2.zip && unzip -q -o awscliv2.zip && sudo ./aws/install >/dev/null && rm -rf aws awscliv2.zip)
fi

log "docker daemon (log rotation) + group"
sudo mkdir -p /etc/docker
if [ ! -f /etc/docker/daemon.json ]; then
  printf '{\n  "log-driver": "json-file",\n  "log-opts": {"max-size": "20m", "max-file": "3"}\n}\n' | sudo tee /etc/docker/daemon.json >/dev/null
  sudo systemctl restart docker 2>/dev/null || true
fi
sudo systemctl enable --now docker >/dev/null
sudo systemctl enable --now cron >/dev/null 2>&1 || sudo systemctl enable --now crond >/dev/null
sudo usermod -aG docker "$ME"

log "swap (${SWAP_GB}G)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l "${SWAP_GB}G" /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB*1024)) status=none
  sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-swap.conf >/dev/null && sudo sysctl -q -p /etc/sysctl.d/99-swap.conf
fi

if [ "$ME" != "ec2-user" ] && [ ! -e /home/ec2-user ]; then
  log "symlink /home/ec2-user -> $HOME_DIR (hard-coded paths in crontab, ops scripts, certbot hooks)"
  sudo ln -s "$HOME_DIR" /home/ec2-user
fi

log "repo -> $DEPLOY_DIR ($BRANCH)"
if [ ! -d "$DEPLOY_DIR/.git" ]; then
  git clone -q --branch "$BRANCH" "$REPO_URL" "$DEPLOY_DIR"
else
  (cd "$DEPLOY_DIR" && git fetch -q origin && git checkout -q "$BRANCH" && git pull -q --ff-only origin "$BRANCH")
fi
mkdir -p "$DEPLOY_DIR/logs" "$DEPLOY_DIR/backups/postgres" "$DEPLOY_DIR/pythia_prophecy/secrets" \
         "$DEPLOY_DIR/pythia_prophecy/data/predictions" "$DEPLOY_DIR/pythia_prophecy/data/backtests" \
         "$HOME_DIR/logs" "$HOME_DIR/archive"

log "done. Next: log out/in (docker group), then ops/import_state_bundle.sh <bundle.tgz>"
log "versions: $(docker --version | cut -d, -f1); $(docker compose version | head -1); $(aws --version | cut -d' ' -f1); $(python3 -V); user=$ME home=$HOME_DIR"
