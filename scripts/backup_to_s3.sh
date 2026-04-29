#!/bin/bash
set -e
BUCKET="spiderbot-backups-639381489406"
DATE=$(date +%Y-%m-%d)
DATA_DIR="/home/ubuntu/crypto-bot/data"

aws s3 cp "$DATA_DIR/trades.csv"     "s3://$BUCKET/trades/$DATE/trades.csv"     2>/dev/null || true
aws s3 cp "$DATA_DIR/bot_state.json" "s3://$BUCKET/state/$DATE/bot_state.json"  2>/dev/null || true
aws s3 cp "$DATA_DIR/trader.log"     "s3://$BUCKET/logs/$DATE/trader.log"       2>/dev/null || true

echo "[backup] Done: $DATE"
