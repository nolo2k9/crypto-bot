import argparse
import os

# -------- CLI --------
def parse_args():
    testnet_default = os.getenv("USE_TESTNET", "false").lower() in ("1", "true", "yes")
    p = argparse.ArgumentParser(description="SpiderBot — Binance USDT-M Futures")

    mx = p.add_mutually_exclusive_group()
    mx.add_argument("--use-testnet", dest="use_testnet", action="store_true", help="Use testnet APIs")
    mx.add_argument("--no-testnet", dest="use_testnet", action="store_false", help="Use mainnet APIs")
    p.set_defaults(use_testnet=testnet_default)

    p.add_argument("--symbols", nargs="+", default=[], help="Symbols to trade")
    p.add_argument("--auto-select", choices=["top_gainer", "top_loser", "both"], help="Auto-select trading symbols")
    p.add_argument("--min-volume", type=float, default=5e7, help="Min 24h USDT volume for auto-select")
    p.add_argument("--interval", default="5m", help="Kline interval (e.g., 1m,5m,1h)")
    p.add_argument("--mode", choices=["paper", "live", "backtest"], default=os.getenv("MODE", "paper"))
    p.add_argument("--days", type=int, default=60, help="Days of history for backtest")
    p.add_argument("--risk", type=float, default=float(os.getenv("RISK_PER_TRADE", "0.01")), help="Risk per trade fraction")
    p.add_argument("--daily-loss-limit", type=float, default=float(os.getenv("DAILY_LOSS_LIMIT", "-0.05")),
                   help="Daily loss limit as fraction")
    p.add_argument("--var-limit", type=float, default=float(os.getenv("PORTFOLIO_VAR_LIMIT", "-0.05")),
                   help="Portfolio VaR limit as fraction")
    p.add_argument("--max-hours", type=int, default=int(os.getenv("MAX_TRADE_HOURS", "24")),
                   help="Max hours to keep position open")
    p.add_argument("--log-file", default=os.getenv("LOG_FILE", "trader.log"), help="Optional file log path")
    p.add_argument("--trade-log-csv", default=os.getenv("TRADE_LOG_CSV", ""), help="Optional CSV trade log path")

    p.add_argument("--force-default", action="store_true",
                   default=os.getenv("FORCE_DEFAULT", "true").lower() in ("1", "true", "yes"))
    p.add_argument("--no-ml", dest="use_ml", action="store_false", help="Disable ML blending")
    p.add_argument("--fee", type=float, default=float(os.getenv("FEE_RATE", "0.001")), help="Fee rate")
    p.add_argument("--tp-mult", type=float, default=float(os.getenv("TP_MULT", "3.0")), help="TP multiple (ATR)")
    p.add_argument("--sl-mult", type=float, default=float(os.getenv("SL_MULT", "2.0")), help="SL multiple (ATR)")
    p.add_argument("--market-type", choices=["spot", "futures"], default=os.getenv("MARKET_TYPE", "futures"),
                   help="Market type")
    p.add_argument("--leverage", type=int, default=int(os.getenv("LEVERAGE", "10")), help="Futures leverage")
    p.add_argument("--dynamic-select", action="store_true", help="Re-evaluate symbols at runtime")
    p.add_argument("--autotune", action="store_true", help="Print sentiment-based parameter recommendations and exit")
    p.add_argument("--rsi-period", type=int, default=14, help="RSI lookback period")
    p.add_argument("--adx-threshold", type=float, default=20.0, help="ADX threshold for signal filtering")
    p.add_argument("--volume-filter", type=lambda x: x.lower() == 'true', default=False,
                   help="Enable/disable volume filter; true/false")
    p.add_argument("--flatten-on-start", action="store_true",
                   help="Close any existing exchange positions in tracked symbols at startup (futures only).")
    p.add_argument("--cooldown-bars", type=int, default=3,
                   help="Bars to wait before new entries on a symbol after a losing exit.")
    p.add_argument("--shock-threshold", type=float, default=3.0,
                   help="Pause new entries if ATR/medianATR(50) >= threshold on any tracked symbol.")
    p.add_argument("--shock-pause-mult", type=float, default=3.0,
                   help="How many intervals to pause entries when a vol shock is detected.")
    p.add_argument("--residual-retries", type=int, default=3, help="Retries to flatten tiny residual futures qty.")
    p.add_argument("--residual-sleep-sec", type=float, default=0.5, help="Sleep between residual clean-up retries.")
    p.add_argument("--residual-qty-threshold", type=float, default=1e-6,
                   help="Qty below which a futures position is considered flat.")
    p.add_argument("--state-file", type=str, default=os.getenv("STATE_FILE", "bot_state.json"),
                   help="Path to persist/restore runtime state.")
    p.add_argument("--autosave-sec", type=int, default=60, help="Autosave interval for runtime state file.")
    p.add_argument("--exchange", choices=["binance"], default="binance",
                   help="Exchange backend (Binance only)")
    p.add_argument("--min-atr-bps", type=float, default=50.0,
                   help="Min ATR in basis points of price (e.g., 50 = 0.50%)")
    p.add_argument("--min-bbw-bps", type=float, default=100.0,
                   help="Min Bollinger Band width in bps (e.g., 100 = 1.00%)")

    # NEW: force-entry & filter override flags
    p.add_argument("--force-side", choices=["BUY", "SELL"], default=None,
                   help="Force a market entry on next processed bar, ignoring regime/signals.")
    p.add_argument("--force-size-usd", type=float, default=10.0,
                   help="Target notional (USDT) for forced entries.")
    p.add_argument("--force-once", type=lambda x: x.lower() not in ("0", "false", "no"),
                   default=True, help="If true, consume the force after one entry per symbol.")
    p.add_argument("--min-notional-override", type=float, default=None,
                   help="Override min notional (USDT). If set, bypasses exchange minNotional.")
    p.add_argument("--step-size-override", type=float, default=None,
                   help="Override LOT_SIZE stepSize (base-asset units).")
    p.add_argument("--min-qty-override", type=float, default=None,
                   help="Override LOT_SIZE minQty (base-asset units).")
    p.add_argument("--ignore-filters", action="store_true",
                   help="Do not enforce exchange filters on qty (send as-is).")
    p.add_argument("--reeval-interval-minutes", type=int, default=60)
    p.add_argument("--stale-rotate-bars", type=int, default=30)
    p.add_argument("--max-active-symbols", type=int, default=None)
    p.add_argument("--max-open-positions", type=int, default=5,
                   help="Max simultaneous open positions across all symbols (default 5)")
    p.add_argument("--max-funding-bps", type=float, default=10.0,
                   help="Skip entry if funding rate exceeds this in bps (default 10 = 0.1%%)")
    p.add_argument("--htf-interval", default=None,
                   help="Higher-TF trend gate interval (e.g. 4h). None=auto-derive, 'none'=disable.")
    p.add_argument("--trade-hours", default="6,22",
                   help="UTC entry window 'start,end' (default '6,22'). 'none' for 24h.")
    p.add_argument("--partial-tp-mult", type=float, default=1.5,
                   help="Scale out 50%% at this × ATR profit (default 1.5). 0 to disable.")
    p.add_argument("--grid-search", action="store_true",
                   help="Run parameter grid search instead of single backtest (backtest mode only)")
    p.add_argument("--corr-threshold", type=float, default=0.70,
                   help="Block new entry if return correlation with any open position >= this (0=off, default 0.70)")
    try:
        from argparse import BooleanOptionalAction
        p.add_argument("--bar-close-only", action=BooleanOptionalAction, default=True)
    except Exception:
        p.add_argument("--bar-close-only", dest="bar_close_only", action="store_true")
        p.add_argument("--no-bar-close-only", dest="bar_close_only", action="store_false")
        p.set_defaults(bar_close_only=True)

    p.set_defaults(use_ml=True)
    return p.parse_args()