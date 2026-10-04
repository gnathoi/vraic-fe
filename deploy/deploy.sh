#!/usr/bin/env bash
# Podman quadlets route: builds the image, installs the quadlets, (re)starts services. Usage: deploy/deploy.sh [--no-build]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "create .env from .env.example first"; exit 1; }
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
if [ "${1:-}" != "--no-build" ]; then
  podman build --build-arg GIT_SHA="$SHA" -t localhost/jfe-app:"$SHA" -t localhost/jfe-app:latest -f Containerfile .
fi
mkdir -p ~/.config/containers/systemd
for f in deploy/quadlets/*; do sed "s|%h/vraic-fe/.env|$PWD/.env|" "$f" > ~/.config/containers/systemd/"$(basename "$f")"; done
systemctl --user daemon-reload
systemctl --user start jfe-db.service jfe-llm.service
systemctl --user restart jfe-api.service jfe-worker.service jfe-reports.service
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8090/healthz >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:8090/healthz && echo " api up ($SHA): http://localhost:8090"
