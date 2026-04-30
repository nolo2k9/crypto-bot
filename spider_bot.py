#!/usr/bin/env python3
"""
SpiderBot — Binance USDT-M futures trading bot

Key features:
- Paper/live/backtest modes
- Env-based API keys, optional testnet
- Exchange rule enforcement (LOT_SIZE, NOTIONAL/MIN_NOTIONAL, PRICE_FILTER)
- Dynamic ATR/leverage-aware position sizing
- RSI, ADX, MACD, Bollinger Bands, VWAP, Stoch RSI, Volume Oscillator
- Optional lightweight ML overlay (XGBoost with periodic retrain)
- ATR trailing stop (breakeven at 1×ATR, trails at 2×ATR)
- Structured logging, optional CSV trade log
- Real SL/TP exits via Binance TAKE_PROFIT_MARKET + STOP_MARKET brackets
- Auto-select symbols (top gainer/loser/both)
- Binance WS feed with REST fallback
- Prometheus metrics, email alerts, runtime state persistence
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(name)s|%(message)s')
import os
from dotenv import load_dotenv
from core.loop.loop import run_live_or_paper
from core.backtest.backtest import backtest, grid_search_backtest
from core.cli.cli import parse_args
# ------------- Env & Metrics -------------
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))

# -------- Entry point --------
def main():
    args = parse_args()
    trade_log_csv = args.trade_log_csv if args.trade_log_csv else None

    # Autotune quick-run
    if getattr(args, 'autotune', False):
        try:
            from sentiment_tuner.daily_sentiment_tuner import SentimentTuner
            tuner = SentimentTuner()
            recs = tuner.daily_recommendations()
        except Exception as e:
            logging.warning(f"Autotune: failed to fetch sentiment data: {e}")
            recs = {}

        fg_index = recs.get('fear_greed_index', -1)
        santiment_sentiment = recs.get('santiment_sentiment', 0.0)
        if fg_index >= 0:
            suggested = recs.get('recommended_params', {})
            print(f"Today's Fear & Greed Index: {fg_index}")
            print("Recommended Parameters Based on Sentiment:")
            print(f"  Risk per trade: {suggested.get('risk', args.risk):.4f}")
            print(f"  Stop loss mult (ATR): {suggested.get('sl_mult', args.sl_mult):.2f}")
            print(f"  Take profit mult (ATR): {suggested.get('tp_mult', args.tp_mult):.2f}")
        else:
            print("Failed to fetch Fear & Greed Index; using default parameters.")
            suggested = {'risk': args.risk, 'sl_mult': args.sl_mult, 'tp_mult': args.tp_mult}
        print(f"Santiment sentiment (7d avg): {santiment_sentiment:.4f}")

        top_news = recs.get('top_news', [])
        if top_news:
            print("\nTop Crypto News Headlines:")
            for n in top_news[:10]:
                print(f"  - {n}")

        twitter_sentiment = recs.get('twitter_sentiment', 0.0)
        print(f"\nTwitter Sentiment Score: {twitter_sentiment:.3f}")

        trending = recs.get('trending_symbols', [])
        if trending:
            print("\nRecommended Trending Coins:")
            print(" ", " ".join(trending))

        cmd = f"python {os.path.basename(__file__)} --mode {args.mode}"
        if args.market_type:
            cmd += f" --market-type {args.market_type}"
        if args.exchange:
            cmd += f" --exchange {args.exchange}"
        if trending:
            cmd += f" --symbols {' '.join(trending)}"
        elif args.symbols:
            cmd += " --symbols " + " ".join(args.symbols)
        cmd += f" --risk {suggested['risk']:.4f} --sl-mult {suggested['sl_mult']:.2f} --tp-mult {suggested['tp_mult']:.2f}"
        cmd += f" --max-hours {args.max_hours} --leverage {args.leverage}"
        if trade_log_csv:
            cmd += f" --trade-log-csv {trade_log_csv}"
        if args.log_file:
            cmd += f" --log-file {args.log_file}"
        if args.use_testnet:
            cmd += " --use-testnet"
        if not args.use_ml:
            cmd += " --no-ml"

        print("\nRun the trading bot with these parameters:")
        print(cmd)
        print("\nNote: This run only outputs recommendations; the bot did not start.\n")
        return

    # Parse trade_hours from "h0,h1" string
    trade_hours = None
    raw_th = (args.trade_hours or "").strip().lower()
    if raw_th and raw_th != "none":
        try:
            parts = [int(x.strip()) for x in raw_th.split(",")]
            trade_hours = (parts[0], parts[1])
        except Exception:
            logging.warning("Invalid --trade-hours '%s'; using 24h (no filter).", raw_th)

    # Normal operation
    if args.mode == "backtest":
        symbols = args.symbols if args.symbols else ["BTCUSDT"]
        if len(symbols) > 1:
            logging.info("Backtest supports a single symbol; using the first.")

        raw_strategies = args.strategy if hasattr(args, "strategy") else ["trend"]
        strategies = ["trend", "mean_reversion", "breakout", "momentum"] if "all" in raw_strategies else raw_strategies

        bt_kwargs = dict(
            symbol=symbols[0], interval=args.interval, days=args.days,
            risk_per_trade=args.risk, fee_rate=args.fee,
            tp_mult=args.tp_mult, sl_mult=args.sl_mult, use_ml=args.use_ml,
            market_type=args.market_type, leverage=args.leverage, rsi_period=args.rsi_period,
            htf_interval=args.htf_interval, trade_hours=trade_hours,
            partial_tp_mult=args.partial_tp_mult, adx_threshold=args.adx_threshold,
            max_hold_hours=args.max_hours, strategies=strategies,
        )
        if getattr(args, "grid_search", False):
            grid_search_backtest(**{k: v for k, v in bt_kwargs.items()
                                   if k not in ("tp_mult", "sl_mult", "adx_threshold")})
        else:
            backtest(**bt_kwargs)
        return

    run_live_or_paper(
        symbols=args.symbols, interval=args.interval, mode=args.mode, risk_per_trade=args.risk,
        daily_loss_limit=args.daily_loss_limit, var_limit=args.var_limit, max_hours=args.max_hours,
        use_testnet=args.use_testnet, log_file=args.log_file, use_ml=args.use_ml,
        auto_select=args.auto_select, min_volume=args.min_volume, fee_rate=args.fee,
        tp_mult=args.tp_mult, sl_mult=args.sl_mult, trade_log_csv=trade_log_csv,
        force_default=args.force_default, market_type=args.market_type, leverage=args.leverage,
        dynamic_select=args.dynamic_select, rsi_period=args.rsi_period, adx_threshold=args.adx_threshold,
        volume_filter=args.volume_filter, flatten_on_start=args.flatten_on_start,
        cooldown_bars=args.cooldown_bars, shock_threshold=args.shock_threshold,
        shock_pause_mult=args.shock_pause_mult, residual_retries=args.residual_retries,
        residual_sleep_sec=args.residual_sleep_sec, residual_qty_threshold=args.residual_qty_threshold,
        state_file=args.state_file, autosave_sec=args.autosave_sec, exchange=args.exchange,
        min_atr_bps=args.min_atr_bps, min_bbw_bps=args.min_bbw_bps,
        force_side=args.force_side,
        force_size_usd=args.force_size_usd,
        force_once=args.force_once,
        min_notional_override=args.min_notional_override,
        step_size_override=args.step_size_override,
        min_qty_override=args.min_qty_override,
        ignore_filters=args.ignore_filters,
        reeval_interval_minutes=args.reeval_interval_minutes,
        stale_rotate_bars=args.stale_rotate_bars,
        max_active_symbols=args.max_active_symbols,
        bar_close_only=args.bar_close_only,
        max_open_positions=args.max_open_positions,
        max_funding_bps=args.max_funding_bps,
        htf_interval=args.htf_interval,
        trade_hours=trade_hours,
        partial_tp_mult=args.partial_tp_mult,
        corr_threshold=args.corr_threshold,
        strategies=strategies if 'strategies' in dir() else ["trend"],
    )


if __name__ == "__main__":
    main()
