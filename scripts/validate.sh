#!/bin/bash
set -e
cd /home/ubuntu/crypto-bot

echo "[validate] Checking container is up..."
sleep 5

STATUS=$(docker compose ps --format json | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line:
        d = json.loads(line)
        print(d.get('State', d.get('Status', '')))
" 2>/dev/null | head -1)

if echo "$STATUS" | grep -qi "running\|up"; then
  echo "[validate] Container is running. Deploy successful."
  exit 0
else
  echo "[validate] Container not running (State=$STATUS). Deploy failed."
  docker compose logs --tail=20
  exit 1
fi
