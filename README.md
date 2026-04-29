# SpiderBot — Binance USDT-M Futures Trading Bot

An automated trading bot for Binance USDT-M perpetual futures. Supports paper trading, live trading, and backtesting from a single CLI entry point.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Configuration Reference](#configuration-reference)
6. [Running the Bot](#running-the-bot)
7. [Signal Logic](#signal-logic)
8. [Risk Management](#risk-management)
9. [Alerts](#alerts)
10. [Monitoring (Prometheus)](#monitoring-prometheus)
11. [Paper Trading Checklist](#paper-trading-checklist)
12. [Live Trading Checklist](#live-trading-checklist)
13. [Backtesting](#backtesting)
14. [Project Structure](#project-structure)
15. [Troubleshooting](#troubleshooting)

---

## Features

### Trading
- **Three modes**: `paper` (simulated fills with slippage), `live` (real orders), `backtest` (historical replay)
- **USDT-M futures** on Binance mainnet or testnet
- **Exchange rule enforcement**: LOT_SIZE, NOTIONAL/MIN_NOTIONAL, PRICE_FILTER all respected before every order
- **Real venue brackets**: SL placed as `STOP_MARKET`, TP placed as `TAKE_PROFIT_MARKET` — both use `closePosition=True` and `workingType=MARK_PRICE`
- **ATR trailing stop**: moves stop to breakeven at 1× ATR profit, trails at 1× ATR above/below live price once profit reaches 2× ATR
- **Partial TP**: scale out 50% of position at a configurable ATR multiple, then move stop to breakeven
- **Multi-symbol**: run up to `--max-active-symbols` symbols in parallel, each with independent position state
- **Auto symbol selection**: auto-pick top gainer, top loser, or both by 24h volume (with dynamic rotation)

### Signal
- **Indicators**: ATR, RSI, ADX, MACD, Bollinger Bands, VWAP, Stochastic RSI, Volume Oscillator, EMA (21/50/200)
- **Regime gate**: EMA-200 filter blocks longs in a downtrend and shorts in an uptrend
- **Higher-timeframe (HTF) trend gate**: EMA-21 vs EMA-50 on a higher interval (auto-derived or configured)
- **ML overlay**: optional XGBoost classifier trained on a rolling window — can veto but not override the indicator signal
- **Fear & Greed index**: daily fetch scales risk up/down (0.40× in extreme fear → 1.25× in extreme greed)

### Risk Controls
- **ATR-based position sizing**: risk exactly `--risk` fraction of free balance per trade
- **Daily loss limit**: halt if day's realized P&L drops below the configured fraction of starting balance
- **Portfolio VaR limit**: block new entries (keep managing open positions) when annualised portfolio VaR exceeds the limit
- **Adaptive risk scaling**: linearly reduces risk as drawdown approaches the daily loss limit
- **Max open positions**: cap simultaneous open positions across all symbols
- **Correlation gate**: skip entry if return correlation with any open position exceeds threshold
- **Funding rate gate**: skip entry if funding rate is extreme against the intended direction
- **Volatility shock circuit breaker**: pause all new entries when any symbol's ATR spikes to `shock_threshold`× its 50-bar median
- **Cooldown bars**: wait N bars after a losing exit before re-entering the same symbol

### Infrastructure
- **WebSocket feed** for real-time klines with automatic reconnect and exponential back-off
- **REST fallback** if WebSocket is silent for configurable seconds
- **State persistence**: bot state serialised to `bot_state.json` every `--autosave-sec` seconds and on every exit
- **Position reconciliation on startup**: discovers and attaches any existing venue positions
- **Prometheus metrics**: equity, per-symbol realised P&L, drawdown
- **Telegram and/or email alerts** on every entry, exit, and risk event
- **Structured JSON trade log** to CSV

---

## Architecture

```
spider_bot.py  (entry point)
│
├── core/cli/cli.py              CLI argument parsing
├── core/loop/loop.py            Main trading loop (live + paper)
├── core/backtest/backtest.py    Historical backtesting engine
│
├── client/binance_client.py     Thin python-binance wrapper (spot + futures)
├── data_feed/ws_feed.py         Binance WS kline feed (REST seed + fallback)
│
├── indicators/indicators.py     All technical indicators + entry signal
├── ml_model/ml_model.py         Rolling XGBoost overlay
├── sentiment_tuner/             Fear & Greed + news sentiment
│
├── order_manager/
│   ├── order_manager.py         Market orders, IOC limit, close_position_fast
│   └── brackets.py              STOP_MARKET + TAKE_PROFIT_MARKET bracket placement
│
├── risk_management/
│   └── risk_management.py       Position sizing, P&L, VaR, adaptive risk scaling
│
├── utils/
│   ├── balance.py               Free USDT balance fetch
│   ├── health.py                Exchange heartbeat check
│   ├── sizing.py                snap_cap_and_format — caps qty by wallet/leverage
│   ├── filters.py               normalize_filters across spot/futures schemas
│   ├── select_symbols.py        Auto-select top gainer/loser
│   └── utils.py                 CSV trade log, klines fetch
│
├── core/
│   ├── state/state.py           Runtime state serialisation
│   ├── gauges/gauges.py         Prometheus gauges
│   ├── helpers/                 Numeric, time-index, futures-position helpers
│   ├── klines/                  REST kline refresh helper
│   └── sync/                    Startup position sync
│
└── alerts/alerts.py             Telegram + email alert dispatch
```

**Data flow per bar:**
```
WS closes a candle
  → data_feed.data[sym] updated
  → indicators computed (ATR, RSI, ADX, MACD, BB, VWAP, EMA, Stoch RSI)
  → regime / HTF / corr / funding checks
  → simple_signal() → ±1 / 0
  → ML veto (optional)
  → position_size_from_atr()
  → enforce_exchange_filters() → snap_cap_and_format()
  → place_market_order()
  → verify fill via get_net_position_qty()
  → place_bracket_orders() (SL + TP on venue)
  → state updated, alert sent, CSV logged
```

---

## Requirements

- Python 3.11+
- A Binance account with futures enabled (or testnet keys)
- API keys with `Enable Futures` permission (do **not** enable withdrawal)

### Python packages

```
pip install -r requirements.txt
```

Key dependencies:
| Package | Version | Purpose |
|---------|---------|---------|
| python-binance | 1.0.19 | Binance REST + WS |
| websocket-client | 1.8.0 | WebSocket feed |
| pandas | 2.2.2 | OHLCV data handling |
| numpy | 1.26.4 | Numerics |
| python-dotenv | 1.0.1 | `.env` file loading |
| prometheus-client | 0.20.0 | Metrics export |
| xgboost | 2.0.3 | ML overlay (optional) |
| scikit-learn | 1.5.1 | ML pre-processing |
| tweepy | 4.14.0 | Twitter sentiment (optional) |
| textblob | 0.18.0 | Text sentiment (optional) |

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd Crypto_Bot/Crypto_Bot
pip install -r requirements.txt
```

### 2. Create `.env`

Copy the template below and fill in your values. The bot loads this from the same directory as `spider_bot.py`.

```dotenv
# ── Binance API ────────────────────────────────────────────────────────────────
BINANCE_KEY=your_api_key_here
BINANCE_SECRET=your_api_secret_here

# Set to true to use Binance testnet instead of mainnet
USE_TESTNET=false

# ── Bot defaults (all overridable via CLI flags) ────────────────────────────────
MODE=paper                   # paper | live | backtest
MARKET_TYPE=futures          # futures | spot
LEVERAGE=10
RISK_PER_TRADE=0.01          # 1% of free balance per trade
DAILY_LOSS_LIMIT=-0.05       # Stop if daily P&L drops below -5% of starting balance
PORTFOLIO_VAR_LIMIT=-0.05    # Block new entries if annualised VaR exceeds 5%
MAX_TRADE_HOURS=24           # Close position after this many hours regardless
TP_MULT=3.0                  # Take-profit at 3× ATR from entry
SL_MULT=2.0                  # Stop-loss at 2× ATR from entry
FEE_RATE=0.001               # Taker fee (0.1%)
LOG_FILE=trader.log
TRADE_LOG_CSV=trades.csv

# ── Alerts ──────────────────────────────────────────────────────────────────────
# Telegram (recommended)
TELEGRAM_BOT_TOKEN=123456789:ABCdef...
TELEGRAM_CHAT_ID=-100123456789

# Email (optional, in addition to Telegram)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=you@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_FROM=you@gmail.com
ALERT_TO=you@gmail.com

# ── Sentiment (optional) ────────────────────────────────────────────────────────
# Santiment API key for social sentiment data
# SANTIMENT_API_KEY=your_key

# Twitter API keys for sentiment overlay
# TWITTER_API_KEY=...
# TWITTER_API_SECRET=...
# TWITTER_ACCESS_TOKEN=...
# TWITTER_ACCESS_SECRET=...
```

### 3. Create a Telegram bot (recommended)

1. Message `@BotFather` on Telegram → `/newbot` → copy the token → set `TELEGRAM_BOT_TOKEN`
2. Add the bot to a private channel or group → get the chat ID → set `TELEGRAM_CHAT_ID`
3. All trade entries, exits, and risk events will be sent as formatted messages

### 4. API key security

- Create a **separate restricted API key** for the bot
- Enable: `Read info`, `Enable Spot & Margin`, `Enable Futures`
- **Disable**: `Enable Withdrawals` — the bot never needs this
- Whitelist the IP address of the server running the bot

---

## Configuration Reference

All parameters can be set via CLI flags. If the `.env` file sets a default, the CLI flag overrides it.

### Core

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `paper` | `paper`, `live`, or `backtest` |
| `--symbols` | _(required unless `--auto-select`)_ | One or more symbols, e.g. `BTCUSDT ETHUSDT` |
| `--auto-select` | — | `top_gainer`, `top_loser`, or `both` (by 24h USDT volume) |
| `--min-volume` | `50000000` | Min 24h volume (USDT) for auto-select |
| `--interval` | `5m` | Kline interval: `1m`, `3m`, `5m`, `15m`, `1h`, `4h`, etc. |
| `--market-type` | `futures` | `futures` or `spot` |
| `--leverage` | `10` | Futures leverage (set on exchange at startup) |
| `--exchange` | `binance` | Exchange backend (`binance` only currently) |
| `--use-testnet` | `false` | Use Binance testnet |

### Sizing & Fees

| Flag | Default | Description |
|------|---------|-------------|
| `--risk` | `0.01` | Fraction of free balance to risk per trade (1%) |
| `--fee` | `0.001` | Taker fee rate (0.1%) — used for P&L calculation |
| `--tp-mult` | `3.0` | Take-profit distance as ATR multiple |
| `--sl-mult` | `2.0` | Stop-loss distance as ATR multiple |
| `--partial-tp-mult` | `1.5` | Scale out 50% at this ATR multiple profit; `0` to disable |

### Risk Limits

| Flag | Default | Description |
|------|---------|-------------|
| `--daily-loss-limit` | `-0.05` | Halt if daily P&L / starting balance falls below this (e.g. `-0.05` = -5%) |
| `--var-limit` | `-0.05` | Block new entries when portfolio annualised VaR exceeds this fraction |
| `--max-hours` | `24` | Time-based exit: close position after N hours regardless of P&L |
| `--max-open-positions` | `5` | Maximum simultaneous open positions across all symbols |
| `--max-active-symbols` | _(none)_ | Cap on symbols tracked concurrently |
| `--adx-threshold` | `20.0` | Minimum ADX for entry (trend strength gate) |
| `--min-atr-bps` | `50.0` | Minimum ATR in basis points of price (0.5%); skips ranging symbols |
| `--min-bbw-bps` | `100.0` | Minimum Bollinger Band width in basis points (1.0%) |
| `--corr-threshold` | `0.70` | Block entry if return correlation with open position ≥ this; `0` to disable |
| `--max-funding-bps` | `10.0` | Skip entry if funding rate > this in bps (10 bps = 0.1%) |

### Volatility & Cooldown

| Flag | Default | Description |
|------|---------|-------------|
| `--shock-threshold` | `3.0` | Pause entries when ATR/medianATR(50) ≥ this multiple |
| `--shock-pause-mult` | `3.0` | Pause for `shock_pause_mult × interval_seconds` after a shock |
| `--cooldown-bars` | `3` | Bars to wait before re-entering after a losing exit |

### Symbol Selection & Rotation

| Flag | Default | Description |
|------|---------|-------------|
| `--dynamic-select` | `false` | Re-evaluate symbol universe every `--reeval-interval-minutes` |
| `--reeval-interval-minutes` | `60` | How often to refresh symbol list in dynamic mode |
| `--stale-rotate-bars` | `30` | After N consecutive filtered bars with no position, swap out the symbol |

### Session & Timing

| Flag | Default | Description |
|------|---------|-------------|
| `--trade-hours` | `6,22` | UTC entry window `start,end` (e.g. `6,22` for 06:00–22:00 UTC); `none` for 24h |
| `--htf-interval` | _(auto)_ | Higher-TF interval for trend gate (e.g. `4h`); `none` to disable |
| `--bar-close-only` | `true` | Only evaluate entries/exits on closed candles |

### ML

| Flag | Default | Description |
|------|---------|-------------|
| `--no-ml` | — | Disable XGBoost overlay entirely |

### State & Logging

| Flag | Default | Description |
|------|---------|-------------|
| `--state-file` | `bot_state.json` | Path to persist/restore runtime state |
| `--autosave-sec` | `60` | How often to autosave state (seconds) |
| `--log-file` | `trader.log` | Log file path (in addition to stdout) |
| `--trade-log-csv` | — | Append every trade to this CSV file |

### Startup Behaviour

| Flag | Default | Description |
|------|---------|-------------|
| `--flatten-on-start` | `false` | Close all existing exchange positions at startup |
| `--force-default` | `true` | Fall back to `BTCUSDT` if all supplied symbols fail validation |

### Residual Cleanup (Futures)

| Flag | Default | Description |
|------|---------|-------------|
| `--residual-retries` | `3` | Attempts to clean up sub-threshold residual qty after close |
| `--residual-sleep-sec` | `0.5` | Delay between residual cleanup attempts |
| `--residual-qty-threshold` | `1e-6` | Qty below which a futures position is considered flat |

### Debug / Override

| Flag | Default | Description |
|------|---------|-------------|
| `--force-side` | — | Force a `BUY` or `SELL` on the next bar, ignoring signals |
| `--force-size-usd` | `10.0` | Notional (USDT) for forced entries |
| `--force-once` | `true` | Consume the force flag once per symbol |
| `--ignore-filters` | `false` | Skip exchange filter enforcement on qty |
| `--min-notional-override` | — | Override exchange minNotional (USDT) |
| `--step-size-override` | — | Override LOT_SIZE stepSize |
| `--min-qty-override` | — | Override LOT_SIZE minQty |

### Sentiment Autotune

```bash
python spider_bot.py --autotune
```

Fetches the Fear & Greed index, Santiment sentiment, trending coins, and prints a ready-to-paste command with recommended parameters. Does **not** start the bot.

---

## Running the Bot

### Paper trading (recommended first step)

```bash
python spider_bot.py \
  --mode paper \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --interval 5m \
  --leverage 10 \
  --risk 0.01 \
  --tp-mult 3.0 \
  --sl-mult 2.0 \
  --trade-log-csv trades.csv \
  --log-file trader.log
```

### Auto-select symbols (paper)

```bash
python spider_bot.py \
  --mode paper \
  --auto-select both \
  --min-volume 100000000 \
  --dynamic-select \
  --interval 5m \
  --max-active-symbols 5 \
  --max-open-positions 3
```

### Live trading

```bash
python spider_bot.py \
  --mode live \
  --symbols BTCUSDT ETHUSDT \
  --interval 5m \
  --leverage 5 \
  --risk 0.005 \
  --daily-loss-limit -0.03 \
  --max-open-positions 2 \
  --trade-log-csv trades.csv
```

### Testnet live

```bash
python spider_bot.py \
  --mode live \
  --use-testnet \
  --symbols BTCUSDT \
  --interval 5m
```

### Sentiment autotune then run

```bash
# Step 1: get recommendations
python spider_bot.py --autotune --mode paper --symbols BTCUSDT

# Step 2: copy and run the printed command
```

---

## Signal Logic

Entry signals come from `indicators/indicators.py:simple_signal()`. A trade is only entered when **all conditions** pass:

### Long (BUY) conditions
1. **Regime gate**: `close > EMA_200 × 0.995` (within 0.5% of EMA-200)
2. **Trend alignment**: `EMA_21 > EMA_50` (fast above slow)
3. **HTF trend**: EMA_21 > EMA_50 on the higher timeframe
4. **Trend strength**: `ADX ≥ adx_threshold` (default 20)
5. **Momentum**: `MACD_HIST > 0` and `MACD > MACD_SIGNAL`
6. **RSI range**: `40 < RSI < 72` (not overbought, trending up)
7. **Stochastic**: `Stoch_K > Stoch_D` and `Stoch_K < 85`
8. **VWAP**: `close > VWAP × 0.997`
9. **BB position**: `BB_PCT > 0.3` (upper half of band)
10. **Volume**: `Vol_Osc > -10` or volume above 80% of 20-bar MA
11. **Regime filters**: `ATR ≥ min_atr_bps` and `BBW ≥ min_bbw_bps` and `ADX ≥ adx_threshold`

### Short (SELL) conditions
Mirror of long conditions (trend down, RSI 28–60, Stoch_K < Stoch_D, close below VWAP, BB_PCT < 0.7).

### ML veto
If the XGBoost model is trained and its prediction disagrees with the indicator signal, the signal is zeroed out (no trade). The ML layer can **veto** but never **originate** a trade.

### Exit logic
Exits happen via two paths, both monitored every 0.25 seconds:
1. **Bracket fill**: Binance executes the venue STOP_MARKET or TAKE_PROFIT_MARKET order — detected by reconciling `positionAmt` going to zero
2. **Software check**: bar-close price checked against local stop/take levels (catches cases where the bracket wasn't placed)
3. **Time exit**: position open longer than `--max-hours`

---

## Risk Management

### Position sizing

Size is calculated so that if the stop-loss hits, the loss equals exactly `risk × free_balance`:

```
qty = (balance × risk) / (atr × sl_mult)
```

The resulting notional is then:
- Capped to `wallet_free × leverage × 0.90` (never exceed 90% of available margin)
- Snapped to the exchange's LOT_SIZE step
- Checked against MIN_NOTIONAL

### ATR trailing stop

Once in a position:
- At `1× ATR` profit → stop moved to breakeven
- At `2× ATR` profit → stop begins trailing 1× ATR behind live price
- Updates happen every 0.25 seconds in the fast-poll loop

### Partial TP

At `partial_tp_mult × ATR` profit (default 1.5×):
1. Close 50% of position at market
2. Move stop to breakeven
3. Cancel and replace brackets for remaining qty

### Fear & Greed scaling

The position size is multiplied by a sentiment factor fetched once per day:

| F&G Index | Multiplier |
|-----------|------------|
| < 20 (Extreme Fear) | 0.40× |
| 20–34 (Fear) | 0.65× |
| 35–65 (Neutral) | 1.00× |
| 66–80 (Greed) | 1.15× |
| > 80 (Extreme Greed) | 1.25× |

### Adaptive risk scaling

Risk is linearly reduced as the portfolio drawdown approaches the daily loss limit:

```python
risk_scale = 1.0 - (drawdown - max_drawdown) / max_drawdown  # if drawdown > max_drawdown
```

The `max_drawdown` threshold is derived from `abs(daily_loss_limit)`.

### Portfolio VaR

Annualised portfolio VaR is computed using 252-day historical covariance. When it exceeds `--var-limit`, new entries are blocked but existing positions continue to be managed.

---

## Alerts

Alerts are sent on every significant event. Both channels can be active simultaneously.

### Events that trigger alerts
- Trade entry (symbol, side, qty, entry price, SL, TP)
- Trade exit (symbol, reason, gross P&L, fees, net P&L)
- Daily loss limit hit
- Max drawdown (50%) hit
- Portfolio VaR limit crossed in either direction
- Daily Fear & Greed sentiment update

### Telegram setup

1. Create a bot via `@BotFather` → `/newbot`
2. Get token → set `TELEGRAM_BOT_TOKEN` in `.env`
3. Create a private channel, add your bot as admin
4. Get the channel ID (use `@userinfobot` or the API) → set `TELEGRAM_CHAT_ID`

### Email setup

Uses SMTP with STARTTLS. For Gmail:
1. Enable 2FA on your Google account
2. Create an App Password at myaccount.google.com/apppasswords
3. Set `SMTP_PASSWORD` to the app password (not your Gmail password)

---

## Monitoring (Prometheus)

The bot exports three Prometheus gauges on the default port (8000) when `prometheus-client` is installed:

| Metric | Labels | Description |
|--------|--------|-------------|
| `bot_equity_usdt` | — | Current portfolio equity |
| `bot_realized_pnl_usdt` | `symbol` | Cumulative realised P&L per symbol |
| `bot_max_drawdown_pct` | — | Current drawdown percentage |

Scrape with Prometheus and display in Grafana, or add a simple exporter:

```bash
# Grafana dashboard query examples
bot_equity_usdt
bot_realized_pnl_usdt{symbol="BTCUSDT"}
bot_max_drawdown_pct
```

---

## Paper Trading Checklist

Before running with real money, validate every part of the system in paper mode. This should run for **at least 5–7 days** across varying market conditions.

**Setup:**
- [ ] `.env` has valid Binance API keys (needed to fetch prices even in paper mode)
- [ ] `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` set — verify alerts arrive
- [ ] Start with `--symbols BTCUSDT ETHUSDT` — these are the most liquid and reliable

**Run:**
```bash
python spider_bot.py \
  --mode paper \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --interval 5m \
  --trade-log-csv trades.csv \
  --log-file trader.log
```

**What to check in the logs:**
- [ ] No `[ENTRY] No fill for ...` spam — this means the symbol doesn't exist on Binance
- [ ] `[ORDER-RESP]` lines appear for entries (confirm fills are being simulated)
- [ ] `[BRACKETS]` lines confirm SL and TP are being tracked
- [ ] `[FAST-EXIT][BEFORE]` and `[FAST-EXIT][AFTER]` confirm exits are being processed
- [ ] `[BOOTSTRAP]` line on startup if any position existed previously
- [ ] `DataFeed ready with recent data for all symbols` — feed is working
- [ ] Prometheus metrics responding at `localhost:8000/metrics`

**What to check in `trades.csv`:**
- [ ] Entries have matching exits (no orphaned positions)
- [ ] P&L values are non-trivial (not all zero)
- [ ] Both BUY and SELL trades appear over time
- [ ] Quantities look proportional to your balance (not 2000 units when balance is $50)

**What to check in Telegram:**
- [ ] "Trade Entry" alerts include symbol, side, quantity, prices
- [ ] "Trade Exit" alerts include close reason (`tp`, `sl`, `time`, `bracket_fill`)
- [ ] "Sentiment Update" alert fires daily with F&G index

**After 5+ days:**
- [ ] Review win rate and P&L in `trades.csv`
- [ ] Check no errors in `trader.log` that repeat continuously
- [ ] Confirm bot restarts cleanly (Ctrl-C then restart) and picks up existing state

---

## Live Trading Checklist

Only proceed when the paper trading checklist is complete and you are satisfied with bot behaviour.

**Before first live run:**
- [ ] Binance futures account funded (start small — enough for a few minimum notional trades)
- [ ] API key has `Enable Futures` but **NOT** `Enable Withdrawals`
- [ ] IP whitelist set on the API key
- [ ] `--use-testnet` removed / `USE_TESTNET=false` in `.env`
- [ ] Start with low leverage (`--leverage 3` or `--leverage 5`)
- [ ] Start with low risk (`--risk 0.005` = 0.5% per trade)
- [ ] Set a conservative daily loss limit (`--daily-loss-limit -0.02` = -2%)
- [ ] Set `--max-open-positions 2` for the first week
- [ ] Run with `--flatten-on-start` the very first time to ensure clean state

**First live run:**
```bash
python spider_bot.py \
  --mode live \
  --symbols BTCUSDT \
  --interval 5m \
  --leverage 3 \
  --risk 0.005 \
  --daily-loss-limit -0.02 \
  --max-open-positions 1 \
  --trade-log-csv trades.csv \
  --flatten-on-start
```

**Monitoring:**
- [ ] Telegram alerts arrive within seconds of trade events
- [ ] Verify first trade on Binance web UI matches the bot log
- [ ] Check `bot_state.json` is being updated regularly

**Scaling up:**
- Increase `--risk` and `--max-open-positions` only after a week of stable operation
- Add more symbols gradually (`--symbols BTCUSDT ETHUSDT`)
- Consider `--auto-select both --dynamic-select` only after you trust the bot fully

---

## Backtesting

Run a single backtest:

```bash
python spider_bot.py \
  --mode backtest \
  --symbols BTCUSDT \
  --interval 1h \
  --days 90 \
  --risk 0.01 \
  --tp-mult 3.0 \
  --sl-mult 2.0 \
  --leverage 10
```

Run a grid search over TP/SL multiples:

```bash
python spider_bot.py \
  --mode backtest \
  --symbols BTCUSDT \
  --interval 1h \
  --days 90 \
  --grid-search
```

The grid search tests combinations of TP and SL multiples and prints a sorted results table. Use it to pick good defaults before paper trading.

**Backtest output includes:**
- Total return, win rate, profit factor
- Max drawdown
- Sharpe ratio (annualised)
- Trade count, average hold time
- Per-trade log

**Backtest limitations:**
- Uses close-price fills (no spread or slippage beyond the configured `--fee`)
- Slippage and spread are configurable via `slippage_bps_entry`, `slippage_bps_exit`, `spread_bps` parameters in `backtest.py`
- Does not model funding rate payments (configurable via `funding_bps_per_day`)

---

## Project Structure

```
Crypto_Bot/
├── spider_bot.py                  Entry point
├── requirements.txt
├── .env                           (create from template — not committed)
├── bot_state.json                 Runtime state (auto-generated)
├── trades.csv                     Trade log (auto-generated)
├── trader.log                     Log file (auto-generated)
│
├── alerts/
│   └── alerts.py                  Telegram + email dispatch
├── client/
│   └── binance_client.py          python-binance wrapper
├── core/
│   ├── backtest/backtest.py       Backtesting engine
│   ├── cli/cli.py                 CLI argument parser
│   ├── gauges/gauges.py           Prometheus metrics
│   ├── helpers/                   Numeric, time-index, position helpers
│   ├── klines/coinflare_klines.py REST kline refresh utility
│   ├── logging/logging.py         Log setup
│   ├── loop/loop.py               Main trading loop
│   ├── state/state.py             State serialisation
│   └── sync/                      Startup position sync
├── data_feed/
│   └── ws_feed.py                 Binance WS feed with REST fallback
├── indicators/
│   └── indicators.py              All indicators + simple_signal()
├── ml_model/
│   └── ml_model.py                Rolling XGBoost overlay
├── order_manager/
│   ├── brackets.py                SL/TP bracket placement
│   └── order_manager.py           Order placement, sizing, filters
├── risk_management/
│   └── risk_management.py         Sizing, P&L, VaR, adaptive risk
├── sentiment_tuner/
│   └── daily_sentiment_tuner.py   Fear & Greed + sentiment fetch
└── utils/
    ├── balance.py                 Free USDT balance
    ├── filters.py                 Filter normalisation
    ├── health.py                  Exchange heartbeat
    ├── precision.py               Quantity precision formatting
    ├── select_symbols.py          Auto-select top gainers/losers
    ├── sizing.py                  snap_cap_and_format
    └── utils.py                   CSV log, klines fetch
```

---

## Troubleshooting

### `[ENTRY] No fill for XYZUSDT` every bar
The symbol doesn't exist on Binance Futures. Use standard Binance USDT-M perpetual symbols like `BTCUSDT`, `ETHUSDT`, `SOLUSDT`. Check available symbols at https://www.binance.com/en/futures/

### `[GLOBAL-STALE] All symbols stale` repeating
The WebSocket feed isn't receiving data. The bot will automatically retry via REST. If it persists:
- Check your network/firewall
- Binance WS requires connecting to `fstream.binance.com` on port 443
- Try restarting the bot

### `[CAP-BLOCK] order blocked: notional X > cap Y`
Your account balance × leverage is too small for the computed position size. Either:
- Deposit more USDT to the futures account
- Reduce `--risk` (e.g. to `0.005`)
- Reduce `--leverage` (counterintuitively, lower leverage means smaller notional cap with the same balance)

### Bot exits immediately with "Daily loss limit exceeded"
`--daily-loss-limit` is a negative fraction. The default is `-0.05` (stop if down 5%). If you pass a positive number like `0.05`, the check triggers immediately. Always use a negative value.

### `ImportError: No module named 'xgboost'`
XGBoost is optional. Either install it (`pip install xgboost`) or disable ML with `--no-ml`.

### `[BRACKETS] TP failed` or `[BRACKETS] SL failed`
Binance rejected the bracket order. Common causes:
- The stop price is too close to the current mark price (Binance requires a minimum distance)
- Insufficient margin to hold the bracket
- The symbol doesn't support `TAKE_PROFIT_MARKET` (very rare)
The position is still open — the bot's software stop/take logic will handle exits.

### State file issues after restart
If `bot_state.json` is corrupted or from a different symbol set, delete it and restart. The bot will discover existing positions from the exchange directly.

### Clock skew warning
`[HEARTBEAT] Clock skew X ms exceeds 2500 ms` — your system clock is not synced. Run `sudo ntpdate pool.ntp.org` or enable automatic time sync.
