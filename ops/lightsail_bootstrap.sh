#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Prepare a fresh Lightsail (Amazon Linux 2023) instance to run SeekingBeta.
# ==============================================================================
# Run ONCE on the new box as ec2-user, right after first SSH:
#   curl -fsSL https://raw.githubusercontent.com/enens-lab/SeekingBeta/feature/sports-prediction-market/ops/lightsail_bootstrap.sh | bash
#   (or scp this file over and `bash ops/lightsail_bootstrap.sh`)
#
# Idempotent. What it does and why:
#   * docker + compose plugin, git, aws cli v2, cronie, certbot: the same host
#     tools the EC2 box used (AL2023 ships python3.9 already -- the ops
#     validators run on it).
#   * a 2 GB swapfile: the $12 plan has 2 GB RAM; steady state is ~1.2 GB
#     (prophecy ~0.6, postgres ~0.1, nginx, docker, OS) but `docker compose
#     build` peaks above that. Swap is the difference between a slow build and
#     an OOM-killed one.
#   * docker log rotation: unbounded json logs filled the old box's disk twice.
#   * clones the repo into /home/ec2-user/seekingbeta (same path as EC2, so the
#     crontab and every ops script transfer unchanged).
# After this: ops/import_state_bundle.sh <bundle.tgz> (see LIGHTSAIL_MIGRATION.md).
# ==============================================================================
REPO_URL="${REPO_URL:-https://github.com/enens-lab/SeekingBeta.git}"
BRANCH="${BRANCH:-feature/sports-prediction-market}"
DEPLOY_DIR="${DEPLOY_DIR:-/home/ec2-user/seekingbeta}"
SWAP_GB="${SWAP_GB:-2}"

log() { echo "[bootstrap] $*"; }

if [ "$(id -un)" != "ec2-user" ]; then
  log "run as ec2-user (the crontab and paths assume it)"; exit 1
fi

log "packages"
sudo dnf -y -q install docker git cronie python3 certbot unzip tar gzip >/dev/null
if ! docker compose version >/dev/null 2>&1; then
  # AL2023's docker package has no compose plugin; install the official binary.
  sudo mkdir -p /usr/local/lib/docker/cli-plugins
  ARCH=$(uname -m); [ "$ARCH" = "aarch64" ] && CARCH=aarch64 || CARCH=x86_64
  sudo curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${CARCH}" -o /usr/local/lib/docker/cli-plugins/docker-compose
  sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi
if ! command -v aws >/dev/null 2>&1; then
  ARCH=$(uname -m); [ "$ARCH" = "aarch64" ] && CARCH=aarch64 || CARCH=x86_64
  (cd /tmp && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${CARCH}.zip" -o awscliv2.zip && unzip -q -o awscliv2.zip && sudo ./aws/install >/dev/null && rm -rf aws awscliv2.zip)
fi

log "docker daemon (log rotation) + group"
sudo mkdir -p /etc/docker
if [ ! -f /etc/docker/daemon.json ]; then
  printf '{\n  "log-driver": "json-file",\n  "log-opts": {"max-size": "20m", "max-file": "3"}\n}\n' | sudo tee /etc/docker/daemon.json >/dev/null
fi
sudo systemctl enable --now docker >/dev/null
sudo systemctl enable --now crond >/dev/null
sudo usermod -aG docker ec2-user

log "swap (${SWAP_GB}G)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l "${SWAP_GB}G" /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB*1024)) status=none
  sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-swap.conf >/dev/null && sudo sysctl -q -p /etc/sysctl.d/99-swap.conf
fi

log "repo -> $DEPLOY_DIR ($BRANCH)"
if [ ! -d "$DEPLOY_DIR/.git" ]; then
  git clone -q --branch "$BRANCH" "$REPO_URL" "$DEPLOY_DIR"
else
  (cd "$DEPLOY_DIR" && git fetch -q origin && git checkout -q "$BRANCH" && git pull -q --ff-only origin "$BRANCH")
fi
mkdir -p "$DEPLOY_DIR/logs" "$DEPLOY_DIR/backups/postgres" "$DEPLOY_DIR/pythia_prophecy/secrets" \
         "$DEPLOY_DIR/pythia_prophecy/data/predictions" "$DEPLOY_DIR/pythia_prophecy/data/backtests" \
         /home/ec2-user/logs /home/ec2-user/archive
sudo chown -R ec2-user:ec2-user "$DEPLOY_DIR" /home/ec2-user/logs /home/ec2-user/archive

log "done. Next: log out/in (docker group), then ops/import_state_bundle.sh <bundle.tgz>"
log "versions: $(docker --version | cut -d, -f1); $(docker compose version | head -1); $(aws --version | cut -d' ' -f1); $(python3 -V)"
