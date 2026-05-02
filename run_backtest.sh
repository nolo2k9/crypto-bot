#!/usr/bin/env bash
# Backtest SpiderBot strategy on BTCUSDT (365 days, 4h bars)
# Usage: ./run_backtest.sh [--days N] [--symbol SYM] [extra python flags...]
set -euo pipefail

DAYS=${DAYS:-365}
SYMBOL=${SYMBOL:-BTCUSDT}
INTERVAL=${INTERVAL:-4h}

# allow override via positional flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --days) DAYS="$2"; shift 2;;
    --symbol) SYMBOL="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    *) break;;
  esac
done

echo "Running backtest: symbol=$SYMBOL interval=$INTERVAL days=$DAYS"
python spider_bot.py \
  --mode backtest \
  --symbols "$SYMBOL" \
  --interval "$INTERVAL" \
  --days "$DAYS" \
  --no-ml \
  --adx-threshold 15 \
  --market-type futures \
  --leverage 10 \
  --tp-mult 3.0 \
  --sl-mult 2.0 \
  "$@"
