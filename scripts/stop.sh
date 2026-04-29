#!/bin/bash
set -e
cd /home/ubuntu/crypto-bot
echo "[stop] Bringing down containers..."
docker compose down || true
echo "[stop] Done."
