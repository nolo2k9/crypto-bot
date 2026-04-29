#!/bin/bash
# Pulls all /spiderbot/* secrets from SSM and writes .env
set -e
REGION="eu-west-1"
DEST="/home/ubuntu/crypto-bot/.env"

echo "[secrets] Fetching from SSM Parameter Store..."

fetch() {
  aws ssm get-parameter --region "$REGION" --name "/spiderbot/$1" \
    --with-decryption --query "Parameter.Value" --output text 2>/dev/null || echo ""
}

cat > "$DEST" <<EOF
BINANCE_KEY=$(fetch BINANCE_KEY)
BINANCE_SECRET=$(fetch BINANCE_SECRET)
TELEGRAM_BOT_TOKEN=$(fetch TELEGRAM_BOT_TOKEN)
TELEGRAM_CHAT_ID=$(fetch TELEGRAM_CHAT_ID)
MODE=paper
USE_TESTNET=false
EXCHANGE=binance
MARKET_TYPE=futures
LEVERAGE=10
RISK_PER_TRADE=0.01
FEE_RATE=0.0005
DAILY_LOSS_LIMIT=-0.05
MAX_HOURS=24
FORCE_DEFAULT=true
EOF

chmod 600 "$DEST"
echo "[secrets] .env written."
