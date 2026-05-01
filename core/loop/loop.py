#!/usr/bin/env python3
# core/live/run_live_or_paper.py

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd

from utils.precision import format_quantity
from indicators.indicators import indicators, simple_signal, get_signal
from ml_model.ml_model import RollingML, ml_available
from data_feed.ws_feed import BinanceWsFeed as CoinflareRestFeed
from risk_management.risk_management import (
    position_size_from_atr, trade_pnl, order_notional,
    portfolio_var, adaptive_risk_scaling
)
from order_manager.order_manager import (
    enforce_exchange_filters,
    place_market_order,
    close_position_fast,
    get_symbol_filters,
    place_limit_ioc_order,
    _best_bid_ask
)

# Brackets + sizing/filter helpers
from order_manager.brackets import place_bracket_orders, cancel_bracket_orders
from utils.sizing import snap_cap_and_format
from utils.filters import normalize_filters

from alerts.alerts import send_alert
from utils.utils import append_trade_csv

try:
    from sentiment_tuner.daily_sentiment_tuner import SentimentTuner as _SentimentTuner
    _sentiment_tuner = _SentimentTuner()
except Exception:
    _sentiment_tuner = None


def _fg_risk_mult(fg: int) -> float:
    """Map Fear & Greed index to a risk multiplier. -1 (fetch failed) → 1.0."""
    if fg < 0:
        return 1.0
    if fg < 20:    # extreme fear
        return 0.40
    if fg < 35:    # fear
        return 0.65
    if fg <= 65:   # neutral
        return 1.0
    if fg <= 80:   # greed
        return 1.15
    return 1.25    # extreme greed — capped

from client.binance_client import UnifiedBinanceClient

from core.state.state import save_runtime_state
from core.logging.logging import setup_logging
from utils.health import exchange_healthy
from utils.balance import get_free_usdt
from core.klines.coinflare_klines import _refresh_coinflare_now
from core.helpers.numeric_helpers.numeric_helpers import (
    _quantize,
    fmt_price_for)

from core.helpers.time_index_helpers.time_index_helpers import (
    parse_interval_seconds, _sanitize_df
)
from core.helpers.futures_position_helpers.futures_Position_helpers import clean_up_residual_futures_position
from utils.select_symbols import auto_select_symbols
from core.gauges.gauges import equity_gauge, pnl_gauge, drawdown_gauge


# -------------------- tiny helpers --------------------

def _correlation_ok(
    sym: str,
    state: dict,
    feed_data: dict,
    threshold: float = 0.70,
    window: int = 50,
) -> bool:
    """Return False when sym's returns correlate >= threshold with any open position."""
    if threshold <= 0:
        return True
    open_syms = [s for s, v in state.items() if v.get("position", 0) != 0 and s != sym]
    if not open_syms:
        return True
    df_sym = feed_data.get(sym)
    if df_sym is None or len(df_sym) < max(20, window // 2):
        return True
    ret_sym = df_sym["Close"].pct_change().dropna().tail(window)
    for osym in open_syms:
        df_o = feed_data.get(osym)
        if df_o is None or len(df_o) < 20:
            continue
        ret_o = df_o["Close"].pct_change().dropna().tail(window)
        common = ret_sym.index.intersection(ret_o.index)
        if len(common) < 20:
            continue
        try:
            corr = float(ret_sym.loc[common].corr(ret_o.loc[common]))
            if abs(corr) >= threshold:
                logging.debug("[CORR] Block %s: corr=%.2f with open %s", sym, corr, osym)
                return False
        except Exception:
            pass
    return True


def _funding_rate_ok(client, symbol: str, side: str, max_funding_bps: float = 10.0) -> bool:
    """Return False when funding rate is extreme against our intended direction."""
    try:
        rows = client.futures_funding_rate(symbol=symbol, limit=1)
        if not rows:
            return True
        rate_bps = float(rows[-1].get("fundingRate", 0.0)) * 10_000.0
        if side == "BUY" and rate_bps > max_funding_bps:
            logging.info("[FUNDING] Skipping BUY on %s: funding=%.2f bps > %.1f bps", symbol, rate_bps, max_funding_bps)
            return False
        if side == "SELL" and rate_bps < -max_funding_bps:
            logging.info("[FUNDING] Skipping SELL on %s: funding=%.2f bps < -%.1f bps", symbol, rate_bps, max_funding_bps)
            return False
        return True
    except Exception:
        return True  # fail open


def _pos_dir_from_side(order_side: str) -> str:
    """BUY => LONG, SELL => SHORT."""
    return "LONG" if str(order_side).upper() == "BUY" else "SHORT"


def _pos_dir_from_sign(sign: int) -> str:
    """+1 => LONG, -1 => SHORT."""
    return "LONG" if int(sign) == 1 else "SHORT"


_HTF_MAP: Dict[str, str] = {
    "1m": "15m", "3m": "15m", "5m": "1h", "15m": "4h",
    "30m": "4h", "1h": "1d", "2h": "1d", "4h": "1d",
    "6h": "1d", "8h": "1d", "12h": "1d", "1d": "1w",
}


def _klines_to_df(raw) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    cols = ["open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"]
    df = pd.DataFrame(raw, columns=cols[:len(raw[0])])
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def get_net_position_qty(client, symbol: str) -> float:
    """Signed net qty from Binance futures for symbol (LONG positive / SHORT negative)."""
    try:
        rows = client.futures_position_information(symbol=symbol) or []
        net = 0.0
        for r in rows:
            if str(r.get("symbol", "")).upper() == symbol.upper():
                try:
                    net += float(r.get("positionAmt", 0.0))
                except Exception:
                    pass
        return net
    except Exception as e:
        logging.debug("[LOOP] get_net_position_qty failed for %s: %s", symbol, e)
        return 0.0


# -------------------- runner --------------------

def run_live_or_paper(
    symbols: List[str],
    interval: str,
    mode: str,
    risk_per_trade: float,
    daily_loss_limit: float,
    var_limit: float,
    max_hours: int,
    use_testnet: bool,
    log_file: Optional[str] = None,
    use_ml: bool = True,
    auto_select: Optional[str] = None,
    min_volume: float = 5e7,
    fee_rate: float = 0.001,
    tp_mult: float = 3.0,
    sl_mult: float = 2.0,
    trade_log_csv: Optional[str] = None,
    force_default: bool = True,
    market_type: str = "futures",
    leverage: int = 10,
    dynamic_select: bool = False,
    reeval_interval_minutes: int = 60,
    rsi_period: int = 14,
    adx_threshold: float = 20.0,
    volume_filter: bool = False,
    bar_close_only: bool = True,
    flatten_on_start: bool = False,
    cooldown_bars: int = 3,
    shock_threshold: float = 3.0,
    shock_pause_mult: float = 3.0,
    residual_retries: int = 3,
    residual_sleep_sec: float = 0.5,
    residual_qty_threshold: float = 1e-6,
    state_file: str = "bot_state.json",
    autosave_sec: int = 60,
    exchange: str = "binance",
    min_atr_bps: float = 50.0,
    min_bbw_bps: float = 100.0,

    # Force-entry & filter knobs
    force_side: Optional[str] = None,          # "BUY" / "SELL" / None
    force_size_usd: float = 10.0,
    force_once: bool = True,                   # consume once per symbol
    min_notional_override: Optional[float] = None,
    step_size_override: Optional[float] = None,
    min_qty_override: Optional[float] = None,
    ignore_filters: bool = False,

    # New behavior knobs
    stale_rotate_bars: int = 30,               # after N unfit bars, rotate the symbol (if no position)
    close_on_reselect: bool = False,          # if True, will close positions on symbol removal; else keep until flat
    max_active_symbols: Optional[int] = None, # cap active coverage (None = no cap)
    max_open_positions: int = 5,              # max simultaneous open positions across all symbols
    max_funding_bps: float = 10.0,           # skip entry if funding rate exceeds this (bps, 10 = 0.1%)
    htf_interval: Optional[str] = None,      # higher-TF for trend gate; None=auto, "none"=off
    trade_hours: Optional[tuple] = (6, 22),  # UTC hour range for entries; None=24h
    partial_tp_mult: float = 1.5,            # scale out 50% at this × ATR profit; 0=off
    corr_threshold: float = 0.70,            # skip entry if return corr with open pos >= this; 0=off
    strategies: Optional[List[str]] = None,  # active strategies; None = ["trend"]
) -> None:
    """
    Live/Paper loop (Binance USDT-M):
    - Discovers & manages any existing positions on startup (sets SL/TP & brackets).
    - Never double-enters a symbol that already has a venue position.
    - Periodically rotates out perma-unfit symbols (if flat) to new candidates.
    - Entries: enforce filters → cap by wallet/leverage → format → send
    - Exits:   use close_position_fast() for full closes; reduceOnly market fallback
    - Brackets: venue-visible SL/TP via place_bracket_orders + reconciliation
    - ATR trailing stop: moves stop to breakeven at 1× ATR profit, trails at 2× ATR
    """
    setup_logging(log_file)

    # -------- Local helpers --------
    def _fmt_qty(sym: str, qty: float) -> float:
        f = get_symbol_filters(client, sym, exchange)
        step = f.get("stepSize")
        qp = f.get("quantityPrecision")
        return _quantize(qty, step, qp)

    def _fmt_qty_str(sym: str, qty: float) -> str:
        f = get_symbol_filters(client, sym, exchange)
        return format_quantity(_quantize(qty, f.get("stepSize"), f.get("quantityPrecision")),
                               f.get("stepSize"), f.get("quantityPrecision")) or "0"

    def effective_filters(sym: str) -> Tuple[float, Optional[float], float]:
        """
        Return (minNotional, stepSize, minQty) with CLI overrides applied,
        using normalized filter keys for consistency.
        """
        raw = get_symbol_filters(client, sym, exchange)
        nf = normalize_filters(raw)
        eff_min_notional = float(min_notional_override if min_notional_override is not None
                                 else (nf.get("min_notional") or 10.0))
        eff_step = float(step_size_override) if step_size_override is not None else nf.get("step_size")
        eff_min_qty = float(min_qty_override if min_qty_override is not None else (nf.get("min_qty") or 0.0))
        logging.info(f"[FILTERS] {sym} eff_min_notional={eff_min_notional}, eff_step={eff_step}, eff_min_qty={eff_min_qty}")
        return eff_min_notional, eff_step, eff_min_qty

    def _ensure_brackets_for_state(sym: str, s: dict, price: float, atr: float):
        """
        If local state has a position but no SL/TP, set default stop/take and place venue brackets.
        """
        if s.get("position", 0) == 0 or s.get("qty", 0.0) <= 0:
            return
        long = s["position"] == 1
        if s.get("stop") is None or s.get("take") is None:
            ep = float(s["entry_price"] if s.get("entry_price") is not None else price)
            stop_px = ep - sl_mult * atr if long else ep + sl_mult * atr
            take_px = ep + tp_mult * atr if long else ep - tp_mult * atr
            s["stop"] = stop_px
            s["take"] = take_px
            try:
                sl_q = fmt_price_for(client, sym, exchange, float(stop_px))
                tp_q = fmt_price_for(client, sym, exchange, float(take_px))
                qty_str = _fmt_qty_str(sym, float(s["qty"]))
                br = place_bracket_orders(
                    client=client,
                    symbol=sym,
                    side_in=("BUY" if long else "SELL"),
                    qty_str=qty_str,
                    entry_price=fmt_price_for(client, sym, exchange, ep),
                    sl_px=sl_q,
                    tp_px=tp_q,
                    market_type=market_type,
                    leverage_hint=leverage,
                )
                s["brackets"] = br
                logging.info(f"[BOOTSTRAP] Placed brackets for existing position {sym}: stop={sl_q} take={tp_q}")
            except Exception as e:
                logging.warning(f"[BOOTSTRAP] Failed to place brackets for {sym}: {e}")

    # -------- Client / feed bootstrap --------
    client = UnifiedBinanceClient(market_type=market_type, testnet=use_testnet)

    if market_type == "futures":
        try:
            (getattr(client, "set_position_mode")(one_way=True))
            pmode = getattr(client, "get_position_mode", lambda: "ONE_WAY")()
            logging.info(f"Futures position mode set to: {pmode}")
        except Exception as e:
            logging.warning(f"Could not enforce ONE_WAY position mode: {e}")
        for sym in symbols:
            try:
                client.set_margin_type(sym, 'ISOLATED')
            except Exception as e:
                logging.warning(f"Failed to set ISOLATED margin for {sym}: {e}")
            try:
                client.set_leverage(sym, leverage)
            except Exception as e:
                logging.warning(f"Failed to set leverage for {sym}: {e}")

    if auto_select:
        try:
            sel = auto_select_symbols(client, auto_select, min_volume)
            if sel:
                symbols = sel if (max_active_symbols is None) else sel[:max_active_symbols]
                logging.info(f"Auto-selected symbols: {symbols}")
        except Exception as e:
            logging.warning(f"Auto-select failed: {e}")

    # Validate symbols
    confirmed = []
    for sym in symbols:
        try:
            client.get_symbol_info(sym)
            client.get_klines(sym, interval, limit=1)
            confirmed.append(sym)
        except Exception as e:
            logging.warning(f"Symbol {sym} validation error: {e}")
    if not confirmed and force_default:
        confirmed = ["BTCUSDT"]
    symbols = confirmed
    if not symbols:
        logging.error("No valid symbols to run.")
        return

    data_feed = CoinflareRestFeed(UnifiedBinanceClient(market_type=market_type, testnet=use_testnet),
                                  symbols, interval, limit=500, poll_sec=2.0, testnet=use_testnet)
    data_feed.start()

    # -------- State --------
    state = {
        sym: {"position": 0, "qty": 0.0, "entry_price": None, "entry_time": None,
              "stop": None, "take": None, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
              "small_qty_count": 0, "disabled": False, "peak_equity": 0.0,
              "consecutive_losses": 0, "stale_count": 0, "last_bar_ts": None,
              "cooldown_bars_left": 0, "brackets": None, "scaled_out": False}
        for sym in symbols
    }
    starting_balance = max(1.0, get_free_usdt(client, market_type))
    logging.info(f"Free USDT balance (contracts): {starting_balance:.2f}")
    for s in state.values():
        s["peak_equity"] = starting_balance

    day_start = datetime.now(timezone.utc)
    day_realized_pnl = 0.0
    peak_equity = starting_balance

    # Feed warmup
    start_time = time.time()
    while True:
        if all(sym in data_feed.data and not data_feed.data[sym].empty for sym in symbols):
            logging.info("DataFeed ready with recent data for all symbols")
            break
        _refresh_coinflare_now(client, data_feed, symbols, interval, bars=500)
        if time.time() - start_time > 120:
            logging.error("Timeout waiting for DataFeed to receive data")
            return
        logging.info("Waiting for DataFeed to receive data...")
        time.sleep(5)

    # ---- Discover/attach existing venue positions & set brackets (robust on restarts) ----
    def _bootstrap_existing_positions():
        nonlocal symbols, data_feed
        discovered = []
        try:
            rows = client.futures_position_information() or []
        except Exception as e:
            logging.warning(f"[BOOTSTRAP] Could not fetch positions: {e}")
            rows = []
        for r in rows:
            sym = str(r.get("symbol", "")).upper()
            if not sym:
                continue
            try:
                signed = float(r.get("positionAmt", 0.0))
            except Exception:
                signed = 0.0
            if abs(signed) > 1e-12:
                entry_price = None
                for k in ("entryPrice", "avgEntryPrice"):
                    v = r.get(k)
                    if v not in (None, "", "0", 0):
                        try:
                            entry_price = float(v)
                            break
                        except Exception:
                            pass
                discovered.append((sym, signed, abs(signed), entry_price))
        if not discovered:
            return

        # ensure we track all discovered symbols
        add_syms = [sym for (sym, _, _, _) in discovered if sym not in symbols]
        if add_syms:
            logging.info(f"[BOOTSTRAP] Adding existing-position symbols to coverage: {add_syms}")
            for sym in add_syms:
                state[sym] = {"position": 0, "qty": 0.0, "entry_price": None, "entry_time": None,
                              "stop": None, "take": None, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
                              "small_qty_count": 0, "disabled": False, "peak_equity": starting_balance,
                              "consecutive_losses": 0, "stale_count": 0, "last_bar_ts": None,
                              "cooldown_bars_left": 0, "brackets": None, "scaled_out": False}
            new_symbols = symbols + add_syms
            # respect max_active_symbols cap if provided
            if max_active_symbols is not None:
                new_symbols = new_symbols[:max_active_symbols]
            # restart feed with superset
            try:
                data_feed.stop()
            except Exception:
                pass
            nonlocal_symbols = list(dict.fromkeys(new_symbols))  # stable-unique
            logging.info(f"[BOOTSTRAP] Restarting feed to include discovered symbols: {nonlocal_symbols}")
            # rebind outer symbols
            symbols = nonlocal_symbols
            # new feed
            data_feed = CoinflareRestFeed(UnifiedBinanceClient(market_type=market_type, testnet=use_testnet),
                                          symbols, interval, limit=500, poll_sec=2.0)
            data_feed.start()
            start2 = time.time()
            while True:
                if all(sym in data_feed.data and not data_feed.data[sym].empty for sym in symbols):
                    break
                _refresh_coinflare_now(client, data_feed, symbols, interval, bars=500)
                if time.time() - start2 > 120:
                    logging.warning("[BOOTSTRAP] Timeout waiting for data after adding symbols")
                    break
                time.sleep(2)

        # set local state & brackets
        for (sym, signed, qty_abs, ep_hint) in discovered:
            # in case symbol was trimmed by max_active_symbols
            if sym not in state:
                logging.info(f"[BOOTSTRAP] Skipping {sym}; not in active coverage after cap.")
                continue
            # get a price/ATR snapshot
            _df = _sanitize_df(data_feed.data.get(sym))
            if _df is None or _df.empty:
                # pull fresh bars
                _refresh_coinflare_now(client, data_feed, [sym], interval, bars=500)
                _df = _sanitize_df(data_feed.data.get(sym))
            if _df is None or _df.empty:
                logging.warning(f"[BOOTSTRAP] No bars for {sym}; can't place brackets yet.")
                continue
            if 'ATR' not in _df.columns or _df['ATR'].isna().all():
                _df = indicators(_df, rsi_period=rsi_period)
                data_feed.data[sym] = _df
            last = _df.iloc[-1]
            px = float(last['Close'])
            atr = float(last.get('ATR', 0.0) or 0.0)

            # fill local state
            position = 1 if signed > 0 else -1
            state[sym].update({
                "position": position,
                "qty": float(_fmt_qty(sym, qty_abs)),
                "entry_price": float(ep_hint if ep_hint not in (None, 0) else px),
                "entry_time": datetime.now(timezone.utc),
                "unrealized_pnl": 0.0,
                "disabled": False,
                "small_qty_count": 0,
            })
            # set & place brackets if missing
            _ensure_brackets_for_state(sym, state[sym], px, atr)
            logging.info(f"[BOOTSTRAP] Attached existing position {sym}: pos={position} qty={qty_abs} ep={state[sym]['entry_price']}")

    # Startup reconciliation / flatten (existing behavior) is kept,
    # then we attach/arm any remaining venue positions we want to manage.
    try:
        # You may want flatten_on_start=False so we preserve positions.
        # The next line respects user flag.
        from core.sync.sync_open_positions_on_start import sync_open_positions_on_start as _sync
        _sync(client, data_feed, state, symbols, market_type, sl_mult, tp_mult, flatten_on_start, mode)
    except Exception as e:
        logging.warning(f"Startup position sync failed: {e}")

    # Attach any remaining positions (if not flattened) and set SL/TP
    _bootstrap_existing_positions()

    # ---- HTF data ----
    _htf = None if (htf_interval or "").lower() == "none" else (htf_interval or _HTF_MAP.get(interval.lower()))
    htf_data: Dict[str, pd.DataFrame] = {}
    last_htf_refresh = datetime.min.replace(tzinfo=timezone.utc)
    HTF_REFRESH_SECS = 300.0

    def _refresh_htf_data() -> None:
        if not _htf:
            return
        for sym in list(symbols):
            try:
                raw = client.get_klines(sym, _htf, limit=300)
                df_h = _klines_to_df(raw)
                if not df_h.empty:
                    htf_data[sym] = indicators(df_h, rsi_period=rsi_period)
            except Exception as e:
                logging.debug("[HTF] refresh failed for %s: %s", sym, e)

    _refresh_htf_data()
    if _htf:
        logging.info("[HTF] Using %s as higher-TF trend filter.", _htf)

    # ML
    if use_ml and ml_available():
        ml = RollingML()
        logging.info("ML overlay enabled.")
    else:
        ml = None
        if use_ml:
            logging.info("ML requested but unavailable; running without ML.")

    # ---- Fear & Greed index (refresh daily) ----
    _fg_index: int = -1
    _fg_mult: float = 1.0
    _last_fg_fetch: datetime = datetime.now(timezone.utc) - timedelta(hours=25)

    def _refresh_fg() -> None:
        nonlocal _fg_index, _fg_mult, _last_fg_fetch
        if _sentiment_tuner is None:
            return
        try:
            _fg_index = _sentiment_tuner.get_fear_greed_index()
            _fg_mult = _fg_risk_mult(_fg_index)
            _last_fg_fetch = datetime.now(timezone.utc)
            logging.info("[SENTIMENT] Fear & Greed=%d → risk_mult=%.2f", _fg_index, _fg_mult)
            send_alert("Sentiment Update", {"Fear_Greed": _fg_index, "Risk_Mult": f"{_fg_mult:.2f}"})
        except Exception as e:
            logging.warning("[SENTIMENT] F&G refresh failed: %s", e)

    _refresh_fg()

    # Misc timers/state
    poll_sleep = parse_interval_seconds(interval)
    stale_threshold_sec = max(2 * poll_sleep, 180)
    var_block = False  # when True, block NEW entries but keep managing existing positions
    last_reeval = last_pnl_log = last_autosave = datetime.now(timezone.utc)
    reeval_interval = timedelta(minutes=reeval_interval_minutes)
    autosave_delta = timedelta(seconds=max(10, int(autosave_sec)))
    global_stale_cycles = 0
    ws_backoff_sec = 2
    global_entry_pause_until: Optional[datetime] = None
    forced_used: set[str] = set()

    # Emergency flatten throttle
    last_flatten_attempt: dict[str, float] = {}
    FLATTEN_COOLDOWN_SEC = 5.0

    # -------- LOOP --------
    FAST_SLEEP = 0.25
    POS_POLL_SECS = 1.0
    MAX_ENTRY_SPREAD = 0.003
    _last_pos_poll = time.monotonic()

    try:
        while True:
            now = datetime.now(timezone.utc)

            # Autosave
            if (now - last_autosave) >= autosave_delta:
                try:
                    save_runtime_state(
                        path=state_file, state=state, symbols=symbols,
                        interval=interval, market_type=market_type,
                        meta={"mode": mode, "use_testnet": bool(use_testnet), "last_autosave": now.isoformat()},
                    )
                except Exception as e:
                    logging.warning(f"[STATE] Autosave failed: {e}")
                last_autosave = now

            # HTF periodic refresh
            if _htf and (now - last_htf_refresh).total_seconds() >= HTF_REFRESH_SECS:
                _refresh_htf_data()
                last_htf_refresh = now

            # Dynamic (macro) re-eval of symbol universe
            if dynamic_select and (now - last_reeval) >= reeval_interval:
                last_reeval = now
                logging.info("Running periodic symbol re-evaluation...")
                try:
                    new_syms = auto_select_symbols(client, auto_select, min_volume) or []
                except Exception as e:
                    logging.warning(f"Auto-select failed during re-eval: {e}")
                    new_syms = []

                # Keep any symbols that have open positions unless we explicitly want to close on reselection
                keep_syms = []
                removable_syms = []
                for sname in symbols:
                    if state.get(sname, {}).get("position", 0) != 0 and not close_on_reselect:
                        keep_syms.append(sname)
                    else:
                        removable_syms.append(sname)

                # New target set: take preferred list, but ensure we keep open-position symbols
                target = keep_syms + [s for s in new_syms if s not in keep_syms]
                if max_active_symbols is not None:
                    # never drop a symbol with an open position
                    protected = [s for s in keep_syms]
                    extras = [s for s in target if s not in protected]
                    target = protected + extras[:max_active_symbols - len(protected)]

                if set(target) != set(symbols):
                    # Decide which to remove (flat or we close them if flag set)
                    to_remove = [s for s in symbols if s not in target]
                    for sname in to_remove:
                        if state.get(sname, {}).get("position", 0) != 0 and close_on_reselect:
                            side_close = "SELL_CLOSE" if state[sname]["position"] == 1 else "BUY_CLOSE"
                            try:
                                close_position_fast(client, sname, side_close=side_close, market_type=market_type, leverage_hint=leverage)
                            except Exception as e:
                                logging.warning(f"Close on removal failed for {sname}: {e}")
                        state.get(sname, {})["disabled"] = True

                    symbols = target
                    # refresh feed with the new set
                    try:
                        data_feed.stop()
                    except Exception:
                        pass
                    data_feed = CoinflareRestFeed(UnifiedBinanceClient(market_type=market_type, testnet=use_testnet),
                                                  symbols, interval, limit=500, poll_sec=2.0)
                    data_feed.start()
                    try:
                        save_runtime_state(
                            path=state_file, state=state, symbols=symbols, interval=interval, market_type=market_type,
                            meta={"mode": mode, "use_testnet": bool(use_testnet), "reason": "dynamic_reselect"},
                        )
                    except Exception as e:
                        logging.warning(f"[STATE] Save after re-eval failed: {e}")

            # Daily reset
            if (now - day_start) > timedelta(days=1):
                day_start = now
                starting_balance = max(1.0, get_free_usdt(client, market_type))
                day_realized_pnl = 0.0
                peak_equity = starting_balance
                for s in state.values():
                    s['peak_equity'] = starting_balance
                    s['consecutive_losses'] = 0
                logging.info("Starting new day - reset baseline")
                _refresh_fg()

            # Refresh F&G every 24 h even if the day hasn't rolled (handles first-run + long sessions)
            if (now - _last_fg_fetch).total_seconds() >= 86_400:
                _refresh_fg()

            # Prices & MTM
            current_prices = {sym: float(data_feed.data[sym].iloc[-1]['Close']) for sym in symbols if sym in data_feed.data and not data_feed.data[sym].empty}
            for sym in symbols:
                s = state.setdefault(sym, {"position": 0, "qty": 0.0, "entry_price": None, "entry_time": None,
                                           "stop": None, "take": None, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
                                           "small_qty_count": 0, "disabled": False, "peak_equity": starting_balance,
                                           "consecutive_losses": 0, "stale_count": 0, "last_bar_ts": None,
                                           "cooldown_bars_left": 0, "brackets": None, "scaled_out": False})
                if s['position'] != 0 and s['entry_price'] is not None and s['qty'] > 0:
                    side_in = "BUY" if s['position'] == 1 else "SELL"
                    s['unrealized_pnl'] = trade_pnl(side_in, s['entry_price'], current_prices.get(sym, 0.0), s['qty'])
                else:
                    s['unrealized_pnl'] = 0.0

            total_equity = starting_balance + day_realized_pnl + sum(s['unrealized_pnl'] for s in state.values())
            peak_equity = max(peak_equity, total_equity)
            drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0
            adj_risk = adaptive_risk_scaling(state, risk_per_trade, max(drawdown, 0.0),
                                             max_consec_losses=3, max_drawdown=abs(daily_loss_limit))
            adj_risk = adj_risk * _fg_mult  # Fear & Greed scaling
            equity_gauge.set(total_equity); drawdown_gauge.set(drawdown * 100)

            # Hard stops
            if drawdown > 0.5:
                send_alert("Max Drawdown Alert", {"Drawdown": f"{drawdown:.2%}", "Timestamp": now.isoformat()})
                logging.warning("Max drawdown threshold reached. Halting new trades.")
            port_var = portfolio_var(state, current_prices, data_feed.data, leverage if market_type == 'futures' else 1)

            # Only enforce when var_limit is set to a positive number
            if (var_limit is not None) and (var_limit > 0):
                if port_var > var_limit:
                    if not var_block:
                        send_alert("Portfolio VaR Limit Hit", {"VaR": f"{port_var:.2%}", "Timestamp": now.isoformat()})
                        logging.warning(
                            "Portfolio VaR limit exceeded (%.2f%% > %.2f%%). "
                            "Blocking NEW entries, continuing risk management.",
                            port_var * 100, var_limit * 100
                        )
                    var_block = True
                else:
                    if var_block:
                        send_alert("Portfolio VaR Back Within Limit",
                                   {"VaR": f"{port_var:.2%}", "Timestamp": now.isoformat()})
                        logging.info("Portfolio VaR back within limit. Resuming new entries.")
                    var_block = False

            if day_realized_pnl <= daily_loss_limit * starting_balance:
                send_alert("Daily Loss Limit Hit", {"Loss": f"{day_realized_pnl:.2f}", "Timestamp": now.isoformat()})
                logging.warning("Daily loss limit exceeded. Stopping trading.")
                break

            # Fast TP/SL poll + reconciliation
            if (time.monotonic() - _last_pos_poll) >= POS_POLL_SECS:
                _last_pos_poll = time.monotonic()
                for sym in list(symbols):
                    s = state[sym]

                    # --- Reconcile with venue (bracket could have filled) ---
                    try:
                        net = get_net_position_qty(client, sym)  # signed qty on venue
                    except Exception:
                        net = None

                    if s['position'] != 0 and (net is not None) and abs(float(net)) < 1e-12:
                        last_px = float(current_prices.get(sym, 0.0) or 0.0)
                        side_in = "BUY" if s['position'] == 1 else "SELL"

                        entry_px_val = s.get('entry_price')
                        qty_val = float(s.get('qty') or 0.0)
                        realized = 0.0

                        if entry_px_val is not None and qty_val > 0:
                            try:
                                gross = trade_pnl(side_in, float(entry_px_val), last_px, qty_val)
                                fees = (
                                    order_notional(side_in, float(entry_px_val), qty_val) +
                                    order_notional("SELL" if side_in == "BUY" else "BUY", last_px, qty_val)
                                ) * fee_rate
                                realized = gross - fees
                                s['realized_pnl'] += realized
                                day_realized_pnl += realized
                            except Exception as e:
                                logging.warning(f"[BRACKET-RECON] PnL calc failed for {sym}: {e}")
                        else:
                            logging.warning(f"[BRACKET-RECON] Missing entry/qty for {sym}; sending alert with PnL=0.")

                        # Best-effort cancel sibling bracket orders
                        try:
                            cancel_bracket_orders(client, sym, s.get("brackets"))
                        except Exception as e:
                            logging.warning(f"[BRACKETS] Cancel failed for {sym}: {e}")

                        # Alert
                        try:
                            send_alert(f"Trade Exit: {sym}", {
                                "Event": "Trade Exit",
                                "Action": "CLOSE",
                                "Close Reason": "bracket_fill",
                                "Order Side": "SELL" if side_in == "BUY" else "BUY",
                                "Position Direction": _pos_dir_from_side(side_in),
                                "Symbol": sym,
                                "Quantity": _fmt_qty_str(sym, qty_val),
                                "Entry Price": (None if entry_px_val is None else float(entry_px_val)),
                                "Exit Price": (None if not last_px else float(last_px)),
                                "Net PnL": float(realized),
                                "Timestamp": now.isoformat(),
                            })
                        except Exception as e:
                            logging.warning(f"[ALERT] Failed to send bracket_fill alert for {sym}: {e}")

                        # Normalize local state
                        s.update({
                            "position": 0, "qty": 0.0, "entry_price": None, "entry_time": None,
                            "stop": None, "take": None, "unrealized_pnl": 0.0, "brackets": None
                        })
                        continue

                    # If bot thinks flat but venue shows residual => emergency flatten (throttled)
                    if s['position'] == 0 and (net is not None) and abs(float(net)) >= 1e-12:
                        now_ts = time.monotonic()
                        last_try = last_flatten_attempt.get(sym, 0.0)
                        if (now_ts - last_try) >= FLATTEN_COOLDOWN_SEC:
                            logging.warning(f"[RECON] Bot flat but venue shows net={net:g} on {sym}. Flattening now.")
                            last_flatten_attempt[sym] = now_ts
                            try:
                                close_position_fast(client, sym, side_close=None, market_type=market_type, leverage_hint=leverage)
                            except Exception as e:
                                logging.warning(f"[RECON] Emergency flatten failed on {sym}: {e}")

                    best_bid, best_ask = _best_bid_ask(client, sym)
                    last_close = float(current_prices.get(sym, 0.0) or 0.0)
                    live_px = (best_bid if s['position'] == 1 else best_ask) or last_close

                    # ATR trailing stop: breakeven at 1×ATR profit, trail at 2×ATR
                    _atr_val = 0.0  # initialise here so partial-TP block below always has it
                    if s['position'] != 0 and s['entry_price'] is not None and s['stop'] is not None:
                        ep = float(s['entry_price'])
                        _atr_df = data_feed.data.get(sym)
                        _atr_val = float(_atr_df['ATR'].iloc[-1]) if (_atr_df is not None and 'ATR' in _atr_df.columns and not _atr_df.empty) else 0.0
                        if _atr_val > 0:
                            long = s['position'] == 1
                            profit = (live_px - ep) if long else (ep - live_px)
                            if profit >= 2.0 * _atr_val:
                                # Trail stop to 1×ATR below/above live price
                                trail_stop = (live_px - _atr_val) if long else (live_px + _atr_val)
                                if long and trail_stop > float(s['stop']):
                                    s['stop'] = trail_stop
                                elif not long and trail_stop < float(s['stop']):
                                    s['stop'] = trail_stop
                            elif profit >= 1.0 * _atr_val:
                                # Move stop to breakeven
                                be_stop = ep if long else ep
                                if long and be_stop > float(s['stop']):
                                    s['stop'] = be_stop
                                elif not long and be_stop < float(s['stop']):
                                    s['stop'] = be_stop

                    # Partial TP: scale out 50% at partial_tp_mult × ATR
                    if (partial_tp_mult > 0 and s['position'] != 0
                            and not s.get('scaled_out', False)
                            and s.get('entry_price') is not None
                            and s.get('qty', 0) > 0 and _atr_val > 0):
                        ep = float(s['entry_price'])
                        _long_p = s['position'] == 1
                        profit_pts = (live_px - ep) if _long_p else (ep - live_px)
                        if profit_pts >= partial_tp_mult * _atr_val:
                            half_qty = s['qty'] * 0.5
                            half_str = _fmt_qty_str(sym, half_qty)
                            close_side_p = "SELL" if _long_p else "BUY"
                            if float(half_str) > 0:
                                try:
                                    place_market_order(client, sym, close_side_p, half_str,
                                                       mode, market_type, reduce_only=True)
                                    remaining = max(0.0, s['qty'] - float(half_str))
                                    # Realize PnL on the half
                                    side_in_p = "BUY" if _long_p else "SELL"
                                    from risk_management.risk_management import trade_pnl as _tpnl, order_notional as _otn
                                    g_p = _tpnl(side_in_p, ep, live_px, float(half_str))
                                    f_p = (_otn(side_in_p, ep, float(half_str)) +
                                           _otn(close_side_p, live_px, float(half_str))) * fee_rate
                                    realized_p = g_p - f_p
                                    s['realized_pnl'] = s.get('realized_pnl', 0.0) + realized_p
                                    day_realized_pnl += realized_p
                                    s['qty'] = remaining
                                    s['stop'] = ep  # move to breakeven
                                    s['scaled_out'] = True
                                    # Cancel old brackets, re-place with breakeven stop
                                    try:
                                        cancel_bracket_orders(client, sym, s.get('brackets'))
                                    except Exception:
                                        pass
                                    if market_type == "futures" and remaining > 0 and s.get('take') is not None:
                                        try:
                                            sl_q_p = fmt_price_for(client, sym, exchange, ep)
                                            tp_q_p = fmt_price_for(client, sym, exchange, float(s['take']))
                                            new_br = place_bracket_orders(
                                                client=client, symbol=sym,
                                                side_in="BUY" if _long_p else "SELL",
                                                qty_str=_fmt_qty_str(sym, remaining),
                                                entry_price=fmt_price_for(client, sym, exchange, ep),
                                                sl_px=sl_q_p, tp_px=tp_q_p,
                                                market_type=market_type, leverage_hint=leverage,
                                            )
                                            s['brackets'] = new_br
                                        except Exception as e_br:
                                            logging.warning("[PARTIAL-TP] Re-place bracket failed %s: %s", sym, e_br)
                                    logging.info("[PARTIAL-TP] %s 50%% @ %.4f BE_stop=%.4f rem=%.6f pnl=%.4f",
                                                 sym, live_px, ep, remaining, realized_p)
                                except Exception as e_p:
                                    logging.warning("[PARTIAL-TP] Failed for %s: %s", sym, e_p)

                    tp_px = s['take']
                    sl_px = s['stop']
                    long = s['position'] == 1
                    hit = False
                    EPS = 1e-8
                    if long:
                        hit = (tp_px is not None and live_px >= tp_px - EPS) or (
                                    sl_px is not None and live_px <= sl_px + EPS)
                    else:
                        hit = (tp_px is not None and live_px <= tp_px + EPS) or (
                                    sl_px is not None and live_px >= sl_px - EPS)

                    # ... inside fast poll loop ...
                    if hit and s['position'] != 0 and s['qty'] > 0:
                        if not exchange_healthy(client):
                            logging.warning(f"[HEARTBEAT] Skipping fast-exit on {sym}; exchange not healthy.")
                            continue

                        # Use live book snapshot (bid/ask) with last_close as fallback
                        exit_px = float(live_px or last_close)
                        side_close = "SELL_CLOSE" if long else "BUY_CLOSE"
                        side_in = "BUY" if long else "SELL"

                        # ---- BEFORE snapshot ----
                        dbg_before = {
                            "symbol": sym,
                            "reason": "tp/sl (fast)",
                            "position": ("LONG" if long else "SHORT"),
                            "qty": float(s['qty'] or 0.0),
                            "entry_price": (None if s.get('entry_price') is None else float(s['entry_price'])),
                            "stop": (None if s.get('stop') is None else float(s['stop'])),
                            "take": (None if s.get('take') is None else float(s['take'])),
                            "live_px": float(exit_px),
                            "fee_rate": float(fee_rate),
                        }
                        logging.info("[FAST-EXIT][BEFORE] " + json.dumps(dbg_before))

                        try:
                            close_position_fast(
                                client, sym,
                                side_close=side_close,
                                market_type=market_type,
                                leverage_hint=leverage
                            )

                            # PnL using the price we just saw
                            ep = float(s['entry_price']) if s['entry_price'] is not None else exit_px
                            qty_val = float(s['qty'] or 0.0)
                            gross = trade_pnl(side_in, ep, exit_px, qty_val)
                            fees = (order_notional(side_in, ep, qty_val) +
                                    order_notional("SELL" if long else "BUY", exit_px, qty_val)) * fee_rate
                            realized = gross - fees
                            s['realized_pnl'] += realized
                            day_realized_pnl += realized

                            # cancel any outstanding brackets now that we closed
                            try:
                                cancel_bracket_orders(client, sym, s.get("brackets"))
                            except Exception as e:
                                logging.warning(f"[BRACKETS] Cancel failed for {sym}: {e}")
                            s["brackets"] = None

                            # (optional) check venue net to confirm flat
                            try:
                                net_after = get_net_position_qty(client, sym)
                            except Exception:
                                net_after = None

                            # ---- AFTER snapshot ----
                            dbg_after = {
                                "symbol": sym,
                                "closed_side": side_close,
                                "exit_px": float(exit_px),
                                "gross": float(gross),
                                "fees": float(fees),
                                "realized": float(realized),
                                "venue_net_after": (None if net_after is None else float(net_after)),
                            }
                            logging.info("[FAST-EXIT][AFTER ] " + json.dumps(dbg_after))

                            event = {
                                "Event": "Trade Exit",
                                "Action": "CLOSE",
                                "Close Reason": "tp/sl (fast)",
                                "Order Side": "SELL" if long else "BUY",
                                "Position Direction": _pos_dir_from_sign(1 if long else -1),
                                "Symbol": sym,
                                "Quantity": _fmt_qty_str(sym, qty_val),
                                "Entry Price": float(ep),
                                "Exit Price": float(exit_px),  # <- FIXED: use exit_px
                                "Net PnL": float(realized),
                                "Timestamp": now.isoformat(),
                            }
                            logging.info(json.dumps(event))
                            send_alert(f"Trade Exit: {sym}", event)

                        except Exception as e:
                            logging.warning(f"[EXIT] Close attempt failed for {sym}: {e}")

                        # reset local state; residual cleaner handles dust
                        s.update({
                            "position": 0,
                            "qty": 0.0,
                            "entry_price": None,
                            "entry_time": None,
                            "stop": None,
                            "take": None,
                            "unrealized_pnl": 0.0,
                            "scaled_out": False  # keep this reset as well
                        })

            # Global staleness (refresh/restart feed)
            stale_syms = []
            for sym in list(symbols):
                df = _sanitize_df(data_feed.data.get(sym))
                data_feed.data[sym] = df
                if df is None or df.empty:
                    stale_syms.append(sym)
                    continue
                last_idx = df.index[-1]
                if last_idx.tzinfo is None:
                    last_idx = last_idx.tz_localize(timezone.utc)
                if (datetime.now(timezone.utc) - last_idx).total_seconds() > stale_threshold_sec:
                    stale_syms.append(sym)
            if stale_syms and len(stale_syms) == len(symbols):
                global_stale_cycles += 1
                logging.warning(f"[GLOBAL-STALE] All symbols stale (cycle {global_stale_cycles}). Symbols={stale_syms}")
                try:
                    _refresh_coinflare_now(client, data_feed, symbols, interval, bars=500)
                    global_stale_cycles = 0
                except Exception as e:
                    logging.warning(f"[GLOBAL-STALE] Refresh failed: {e}")
                    try:
                        data_feed.stop(); time.sleep(ws_backoff_sec)
                    except Exception:
                        pass
                    data_feed = CoinflareRestFeed(UnifiedBinanceClient(market_type=market_type, testnet=use_testnet),
                                                  symbols, interval, limit=500, poll_sec=2.0)
                    data_feed.start()
                    ws_backoff_sec = min(ws_backoff_sec * 2, 60)
            else:
                global_stale_cycles = 0; ws_backoff_sec = 2

            # Shock circuit breaker
            if global_entry_pause_until and now >= global_entry_pause_until:
                logging.info("[VOL-SHOCK] Global entry pause ended.")
                global_entry_pause_until = None
            if global_entry_pause_until is None:
                shock_hits = []
                for sym in symbols:
                    df = data_feed.data.get(sym)
                    if df is None or df.empty or 'ATR' not in df.columns or len(df) < 50:
                        continue
                    atr_now = float(df['ATR'].iloc[-1] or 0.0)
                    med50 = float(df['ATR'].rolling(50).median().iloc[-1] or 0.0)
                    if med50 > 0 and (atr_now / med50) >= float(shock_threshold):
                        shock_hits.append(sym)
                if shock_hits:
                    pause_seconds = int(max(1.0, shock_pause_mult) * poll_sleep)
                    global_entry_pause_until = now + timedelta(seconds=pause_seconds)
                    logging.warning(f"[VOL-SHOCK] Pausing new entries for {pause_seconds}s (hits={shock_hits})")

            # ---- Per-symbol (entries/exits) ----
            for sym in list(symbols):
                s = state[sym]
                if s['disabled']:
                    continue
                df = _sanitize_df(data_feed.data.get(sym))
                if df is None or df.empty:
                    continue

                # indicators if missing
                if ('ATR' not in df.columns) or df['ATR'].isna().all():
                    df = indicators(df, rsi_period=rsi_period)
                    data_feed.data[sym] = df
                    if 'ATR' not in df.columns:
                        continue

                row = df.iloc[-1]
                last_ts = df.index[-1]
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize(timezone.utc)

                # bar-close guard
                if bar_close_only and s.get("last_bar_ts") is not None and last_ts <= s["last_bar_ts"]:
                    continue

                # cooldown tick on new bar
                prev_ts = s.get("last_bar_ts")
                if prev_ts is None or last_ts > prev_ts:
                    if s.get("cooldown_bars_left", 0) > 0:
                        s["cooldown_bars_left"] -= 1
                        if s["cooldown_bars_left"] == 0:
                            logging.info(f"[COOLDOWN-ENDED] {sym}")

                price = float(row['Close'])
                atr = float(row.get('ATR', 0.0) or 0.0)
                adx = float(row.get('ADX', 0.0) or 0.0)
                vol = float(row.get('Volume', 0.0) or 0.0)
                bb_u = float(row.get('BB_UPPER', price)); bb_l = float(row.get('BB_LOWER', price))
                bbw = bb_u - bb_l

                # regime filters
                regime_fit = not ((atr < (min_atr_bps/10_000.0)*price) or (adx < float(adx_threshold)) or (bbw < (min_bbw_bps/10_000.0)*price))
                if not regime_fit:
                    reason = ("ATR" if atr < (min_atr_bps/10_000.0)*price
                              else "ADX" if adx < float(adx_threshold)
                              else "BBW")
                    logging.info("[REGIME] %s skip: %s too low (atr=%.2f adx=%.1f bbw=%.2f)", sym, reason, atr, adx, bbw)
                    s["stale_count"] = s.get("stale_count", 0) + 1
                    s["last_bar_ts"] = last_ts
                    # micro-rotation for stale symbols (flat only)
                    if s['position'] == 0 and dynamic_select and (s["stale_count"] >= max(5, int(stale_rotate_bars))):
                        try:
                            candidates = auto_select_symbols(client, auto_select, min_volume) or []
                            new_sym = next((x for x in candidates if x not in symbols), None)
                            if new_sym:
                                logging.info(f"[ROTATE] Replacing stale {sym} with {new_sym}")
                                if sym in symbols and state.get(sym, {}).get("position", 0) == 0:
                                    symbols.remove(sym)
                                    state[sym]["disabled"] = True
                                symbols.append(new_sym)
                                state[new_sym] = {"position": 0, "qty": 0.0, "entry_price": None, "entry_time": None,
                                                  "stop": None, "take": None, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
                                                  "small_qty_count": 0, "disabled": False, "peak_equity": starting_balance,
                                                  "consecutive_losses": 0, "stale_count": 0, "last_bar_ts": None,
                                                  "cooldown_bars_left": 0, "brackets": None, "scaled_out": False}
                                if max_active_symbols is not None and len(symbols) > max_active_symbols:
                                    drop = next((z for z in symbols if state.get(z, {}).get("position", 0) == 0 and state.get(z, {}).get("disabled", False)), None)
                                    if drop:
                                        logging.info(f"[ROTATE] Dropping {drop} to respect max_active_symbols")
                                        symbols.remove(drop)
                                try:
                                    data_feed.stop()
                                except Exception:
                                    pass
                                data_feed = CoinflareRestFeed(UnifiedBinanceClient(market_type=market_type, testnet=use_testnet),
                                                              symbols, interval, limit=500, poll_sec=2.0)
                                data_feed.start()
                        except Exception as e:
                            logging.warning(f"[ROTATE] micro-rotation failed: {e}")
                    continue
                else:
                    s["stale_count"] = 0

                # volume filter
                if volume_filter:
                    try:
                        vma20 = float(df['Volume'].rolling(20).mean().iloc[-1])
                    except Exception:
                        vma20 = 0.0
                    if not ((vma20 > 0 and vol >= 1.2*vma20) or (float(row.get("Vol_Osc", 0.0)) > 0.0)):
                        s["last_bar_ts"] = last_ts
                        continue

                # ML: initial train + periodic retrain + keep close window warm
                if ml and len(df) > 200:
                    ml.update_close(float(row.get("Close", 0) or 0))
                    if not getattr(ml, "trained", False):
                        try:
                            ml.fit(df.iloc[:-1])
                            logging.info("ML trained on initial window.")
                        except Exception as e:
                            logging.warning(f"ML training error: {e}")
                    elif hasattr(ml, "should_retrain") and ml.should_retrain(len(df)):
                        try:
                            ml.fit(df.iloc[:-1])
                            logging.info(f"ML retrained on {len(df)} bars.")
                        except Exception as e:
                            logging.warning(f"ML retrain error: {e}")

                # ---------- FORCE ENTRY (one-time per symbol if requested) ----------
                if force_side in ("BUY", "SELL") and (sym not in forced_used):
                    if exchange_healthy(client):
                        # If venue already has a position, align local state and skip entry
                        try:
                            venue_net = get_net_position_qty(client, sym)
                        except Exception:
                            venue_net = 0.0
                        if abs(float(venue_net)) > 1e-12:
                            state[sym]["position"] = 1 if venue_net > 0 else -1
                            state[sym]["qty"] = abs(float(venue_net))
                            _ensure_brackets_for_state(sym, state[sym], price, atr)
                            s["last_bar_ts"] = last_ts
                            continue

                        eff_min_notional, _, _ = effective_filters(sym)
                        target_usd = max(float(force_size_usd or 0.0), eff_min_notional * 1.01, 10.0)
                        intended_qty = target_usd / max(price, 1e-12)

                        # Optionally enforce exchange filters (snap to grid) before cap/format
                        if ignore_filters:
                            adj_qty = _fmt_qty(sym, intended_qty)
                        else:
                            adj_qty, _, _ = enforce_exchange_filters(
                                client, sym, intended_qty, price,
                                exchange=exchange, market_type=market_type
                            )
                            adj_qty = _fmt_qty(sym, adj_qty)

                        # Cap & format
                        qty, qty_str = snap_cap_and_format(
                            client=client, sym=sym, exchange=exchange, market_type=market_type,
                            price=price, intended_qty=adj_qty, leverage=leverage, safety=0.90
                        )
                        # ensure notional clears minNotional after snapping
                        notional = float(qty) * float(price)
                        if notional < float(eff_min_notional or 10.0):
                            qty = _fmt_qty(sym, (float(eff_min_notional) * 1.01) / max(price, 1e-12))
                            qty_str = _fmt_qty_str(sym, qty)
                            notional = float(qty) * float(price)

                        wallet_free = float(get_free_usdt(client, market_type))
                        hard_cap = wallet_free * float(leverage) * 0.90
                        notional = float(qty) * float(price)
                        min_notional_ok = notional >= float(eff_min_notional or 10.0) - 1e-9
                        under_cap = notional <= hard_cap + 1e-9

                        cap_side = force_side
                        logging.info(f"[CAP-CHECK] {sym} side={cap_side} qty={qty_str} notional={notional:.6f} "
                                     f"hard_cap={hard_cap:.6f} wallet={wallet_free:.6f} lev={leverage} "
                                     f"minNotional={float(eff_min_notional or 10.0):.2f}")

                        if not under_cap:
                            logging.warning(f"[CAP-BLOCK] {sym} order blocked: notional {notional:.6f} > cap {hard_cap:.6f}")
                            s["last_bar_ts"] = last_ts
                            continue
                        if not min_notional_ok or not qty or not qty_str:
                            logging.info(f"[SIZE] Skipping {sym}: below minNotional or zero qty after cap/format.")
                            s["last_bar_ts"] = last_ts
                            continue

                        px = fmt_price_for(client, sym, exchange, price)

                        # Send order (FIX: use force_side here)
                        try:
                            resp = place_market_order(
                                client, sym, force_side, qty_str, mode, market_type,
                                reduce_only=False, last_price_hint=px, leverage_hint=leverage,
                                exchange=exchange,
                            )
                        except Exception as e:
                            logging.error(f"[ORDER] FORCE entry reject {sym} {force_side} {qty_str}: {e}")
                            s["last_bar_ts"] = last_ts
                            continue
                        logging.info(f"[ORDER-RESP] {resp}")

                        # verify fill
                        time.sleep(0.6)
                        net_after = 0.0
                        try:
                            net_after = get_net_position_qty(client, sym)
                        except Exception:
                            pass

                        if abs(float(net_after)) < 1e-12:
                            # Fallback: IOC at top of book (small epsilon so it crosses)
                            try:

                                bid, ask = _best_bid_ask(client, sym)
                                if force_side == "BUY":
                                    px_ioc = (ask if ask > 0 else price) * 1.001
                                else:
                                    px_ioc = (bid if bid > 0 else price) * 0.999

                                place_limit_ioc_order(
                                    client=client, symbol=sym, side=force_side, qty_str=qty_str,
                                    price=px_ioc, market_type=market_type, leverage_hint=leverage,
                                    exchange=exchange,
                                )
                                time.sleep(0.6)
                                try:
                                    net_after = get_net_position_qty(client, sym)
                                except Exception:
                                    net_after = 0.0
                            except Exception as e:
                                logging.warning(f"[ENTRY-FALLBACK] IOC fallback failed for {sym}: {e}")

                        if abs(float(net_after)) < 1e-12:
                            logging.info(f"[ENTRY] No fill for {sym} ({force_side}); skipping state/brackets.")
                            s["last_bar_ts"] = last_ts
                            continue

                        # Use actual filled direction/qty from the venue
                        pos_dir = 1 if float(net_after) > 0 else -1
                        fill_qty = _fmt_qty(sym, abs(float(net_after)))
                        qty_str_filled = _fmt_qty_str(sym, fill_qty)

                        stop_px = price - sl_mult * atr if pos_dir == 1 else price + sl_mult * atr
                        take_px = price + tp_mult * atr if pos_dir == 1 else price - tp_mult * atr
                        s.update({
                            "position": pos_dir,
                            "qty": float(fill_qty),
                            "entry_price": price,
                            "entry_time": now,
                            "stop": stop_px,
                            "take": take_px,
                            "unrealized_pnl": 0.0,
                            "disabled": False,
                            "small_qty_count": 0
                        })

                        # Place brackets (use filled side/size)
                        try:
                            sl_q = fmt_price_for(client, sym, exchange, float(stop_px))
                            tp_q = fmt_price_for(client, sym, exchange, float(take_px))
                            filled_side = "BUY" if pos_dir == 1 else "SELL"
                            br = place_bracket_orders(
                                client=client,
                                symbol=sym,
                                side_in=filled_side,
                                qty_str=qty_str_filled,
                                entry_price=px,
                                sl_px=sl_q,
                                tp_px=tp_q,
                                market_type=market_type,
                                leverage_hint=leverage,
                            )
                            s["brackets"] = br
                        except Exception as e:
                            logging.warning(f"[BRACKETS] Failed to place SL/TP for {sym}: {e}")

                        event = {
                            "Event": "Trade Entry",
                            "Action": "OPEN",
                            "Order Side": filled_side,
                            "Position Direction": _pos_dir_from_sign(pos_dir),
                            "Symbol": sym,
                            "Quantity": qty_str_filled,
                            "Entry Price": fmt_price_for(client, sym, exchange, price),
                            "Stop": fmt_price_for(client, sym, exchange, stop_px),
                            "Take": fmt_price_for(client, sym, exchange, take_px),
                            "Timestamp": now.isoformat(),
                        }

                        logging.info(json.dumps(event)); send_alert(f"Trade Entry: {sym}", event)
                        if trade_log_csv:
                            append_trade_csv(trade_log_csv, {
                                "ts": now.isoformat(),
                                "symbol": sym,
                                "side": filled_side,
                                "qty": qty_str_filled,
                                "entry": event["Entry Price"],
                                "reason": "entry",
                            })

                        if force_once:
                            forced_used.add(sym)
                        s["last_bar_ts"] = last_ts
                        continue  # skip normal entry this bar

                # ---------- EXIT (bar-close path) ----------
                if s['position'] != 0 and s['stop'] is not None and s['take'] is not None:
                    hit_stop = (price <= s['stop']) if s['position'] == 1 else (price >= s['stop'])
                    hit_take = (price >= s['take']) if s['position'] == 1 else (price <= s['take'])
                    time_exit = s['entry_time'] and ((now - s['entry_time']).total_seconds() > max_hours * 3600)
                    if hit_stop or hit_take or time_exit:
                        if not exchange_healthy(client):
                            s["last_bar_ts"] = last_ts
                            continue
                        side_close = "SELL_CLOSE" if s['position'] == 1 else "BUY_CLOSE"
                        try:
                            close_position_fast(client, sym, side_close=side_close,
                                                market_type=market_type, leverage_hint=leverage)
                        except Exception as e:
                            logging.warning(f"[EXIT] close_position_fast failed on {sym}: {e}")
                            # Partial fallback: reduce_only market using snapped current qty
                            qty_to_close = _fmt_qty(sym, s['qty'])
                            if qty_to_close > 0:
                                try:
                                    place_market_order(
                                        client, sym, "SELL" if s['position'] == 1 else "BUY",
                                        _fmt_qty_str(sym, qty_to_close), mode, market_type,
                                        reduce_only=True, last_price_hint=price, leverage_hint=leverage
                                    )
                                except Exception as e2:
                                    logging.error(f"[EXIT] Fallback reduce-only failed on {sym}: {e2}")
                                    s["last_bar_ts"] = last_ts
                                    continue

                        side_in = "BUY" if s['position'] == 1 else "SELL"
                        ep = s['entry_price'] if s['entry_price'] is not None else price
                        gross = trade_pnl(side_in, ep, price, s['qty'])
                        fees = (order_notional(side_in, ep, s['qty']) +
                                order_notional("SELL" if s['position'] == 1 else "BUY", price, s['qty'])) * fee_rate
                        realized = gross - fees
                        s['realized_pnl'] += realized; day_realized_pnl += realized
                        if realized < 0:
                            s['consecutive_losses'] = s.get('consecutive_losses', 0) + 1
                            s['cooldown_bars_left'] = max(s.get('cooldown_bars_left', 0), int(max(0, cooldown_bars)))
                        else:
                            s['consecutive_losses'] = 0
                        try:
                            pnl_gauge.labels(symbol=sym).set(s['realized_pnl'])
                        except Exception:
                            pass

                        cancel_bracket_orders(client, sym, s.get("brackets"))
                        s["brackets"] = None

                        reason = 'stop' if hit_stop else 'take' if hit_take else 'time'
                        event_body = {
                            "Event": "Trade Exit",
                            "Action": "CLOSE",
                            "Close Reason": reason,
                            "Order Side": "SELL" if s['position'] == 1 else "BUY",
                            "Position Direction": _pos_dir_from_sign(s['position']),
                            "Symbol": sym,
                            "Quantity": _fmt_qty_str(sym, s['qty']),
                            "Entry Price": float(ep),
                            "Exit Price": float(price),
                            "Gross PnL": round(gross, 6),
                            "Fees": round(fees, 6),
                            "Net PnL": round(realized, 6),
                            "Timestamp": now.isoformat(),
                        }
                        logging.info(json.dumps(event_body)); send_alert(f"Trade Exit: {sym}", event_body)
                        if trade_log_csv:
                            append_trade_csv(trade_log_csv, {
                                "ts": now.isoformat(),
                                "symbol": sym,
                                "side": event_body["Order Side"],
                                "qty": event_body["Quantity"],
                                "entry": float(ep),
                                "exit": float(price),
                                "reason": reason,
                                "gross": float(gross),
                                "fees": float(fees),
                                "realized": float(realized),
                            })

                        # residual dust
                        if market_type == "futures":
                            clean_up_residual_futures_position(client, sym, mode,
                                                               qty_threshold=residual_qty_threshold,
                                                               max_retries=residual_retries,
                                                               sleep_sec=residual_sleep_sec)
                        s.update({"position": 0, "qty": 0.0, "entry_price": None, "entry_time": None,
                                  "stop": None, "take": None, "unrealized_pnl": 0.0})
                        s["last_bar_ts"] = last_ts
                        try:
                            save_runtime_state(
                                path=state_file, state=state, symbols=symbols, interval=interval,
                                market_type=market_type, meta={"mode": mode, "event": "exit", "ts": now.isoformat()},
                            )
                            last_autosave = now
                        except Exception as e:
                            logging.warning(f"[STATE] Save after exit failed: {e}")
                        continue

                # ---------- NORMAL ENTRY ----------
                if s['position'] == 0:
                    if var_block:
                        s["last_bar_ts"] = last_ts
                        continue
                    if s.get("cooldown_bars_left", 0) > 0:
                        s["last_bar_ts"] = last_ts; continue

                    # Venue double-check: if there's already a net position, sync & skip entry
                    try:
                        venue_net = get_net_position_qty(client, sym)
                    except Exception:
                        venue_net = 0.0
                    if abs(float(venue_net)) > 1e-12:
                        s["position"] = 1 if venue_net > 0 else -1
                        s["qty"] = abs(float(venue_net))
                        _ensure_brackets_for_state(sym, s, price, atr)
                        s["last_bar_ts"] = last_ts
                        continue

                    # max concurrent positions guard
                    open_count = sum(1 for _s in state.values() if _s.get("position", 0) != 0)
                    if open_count >= max_open_positions:
                        s["last_bar_ts"] = last_ts
                        continue

                    # signal — combine all active strategies (any agreement fires)
                    active_strats = strategies or ["trend"]
                    sig = 0
                    for _strat in active_strats:
                        _s = get_signal(row, strategy=_strat, adx_threshold=adx_threshold)
                        if _s != 0:
                            sig = _s
                            break
                    if ml and getattr(ml, "trained", False):
                        try:
                            ml.update_close(float(row.get("Close", 0) or 0))
                            ml_sig = ml.predict_signal(row)
                            # ML can veto but not override: disagreement → skip
                            if ml_sig is not None and ml_sig != 0 and sig != 0 and ml_sig != sig:
                                sig = 0
                        except Exception as e:
                            logging.warning(f"ML predict error: {e}")

                    entry_side = "BUY" if sig == 1 else "SELL" if sig == -1 else None
                    if entry_side is None or atr <= 0:
                        logging.info("[SIGNAL] %s no signal (sig=0, adx=%.1f, atr=%.2f)", sym, adx, atr)
                        s["last_bar_ts"] = last_ts
                        continue

                    # Session/time filter
                    if trade_hours is not None:
                        h0, h1 = trade_hours
                        utc_h = now.hour
                        _in_sess = (h0 <= utc_h < h1) if h0 < h1 else (utc_h >= h0 or utc_h < h1)
                        if not _in_sess:
                            logging.info("[HOURS] %s skip: UTC hour %d outside window %d-%d", sym, utc_h, h0, h1)
                            s["last_bar_ts"] = last_ts
                            continue

                    # HTF trend gate
                    if _htf and sym in htf_data and not htf_data[sym].empty:
                        htf_row = htf_data[sym].iloc[-1]
                        h21 = float(htf_row.get("EMA_21", 0) or 0)
                        h50 = float(htf_row.get("EMA_50", 0) or 0)
                        if h21 > 0 and h50 > 0:
                            if entry_side == "BUY" and h21 < h50 * 0.999:
                                logging.info("[HTF] Skip BUY %s: 1d bearish (EMA21=%.0f < EMA50=%.0f)", sym, h21, h50)
                                s["last_bar_ts"] = last_ts
                                continue
                            if entry_side == "SELL" and h21 > h50 * 1.001:
                                logging.info("[HTF] Skip SELL %s: 1d bullish (EMA21=%.0f > EMA50=%.0f)", sym, h21, h50)
                                s["last_bar_ts"] = last_ts
                                continue

                    # Correlation gate: skip if too correlated with an open position
                    if not _correlation_ok(sym, state, data_feed.data, corr_threshold):
                        logging.info("[CORR] Skipping %s %s: high correlation with open position",
                                     sym, entry_side)
                        s["last_bar_ts"] = last_ts
                        continue

                    if not exchange_healthy(client):
                        s["last_bar_ts"] = last_ts; continue

                    # sizing baseline
                    eff_min_notional, _, eff_min_qty = effective_filters(sym)
                    base_qty = position_size_from_atr(
                        max(0.0, get_free_usdt(client, market_type)),
                        price,
                        atr,
                        adj_risk,
                        leverage if market_type == "futures" else 1,
                        sl_mult=sl_mult,
                    )
                    intended_qty = max(base_qty, eff_min_qty or 0.0, (eff_min_notional or 10.0)/price, 10.0/price)

                    # Optionally enforce exchange filters (snap to grid) before cap/format
                    if ignore_filters:
                        adj_qty = _fmt_qty(sym, intended_qty)
                    else:
                        adj_qty, _, _ = enforce_exchange_filters(
                            client, sym, intended_qty, price,
                            exchange=exchange, market_type=market_type
                        )
                        adj_qty = _fmt_qty(sym, adj_qty)

                    # Cap & format
                    qty, qty_str = snap_cap_and_format(
                        client=client, sym=sym, exchange=exchange, market_type=market_type,
                        price=price, intended_qty=adj_qty, leverage=leverage, safety=0.90
                    )
                    # ensure notional clears minNotional after snapping
                    notional = float(qty) * float(price)
                    if notional < float(eff_min_notional or 10.0):
                        qty = _fmt_qty(sym, (float(eff_min_notional) * 1.01) / max(price, 1e-12))
                        qty_str = _fmt_qty_str(sym, qty)
                        notional = float(qty) * float(price)

                    wallet_free = float(get_free_usdt(client, market_type))
                    hard_cap = wallet_free * float(leverage) * 0.90
                    notional = float(qty) * float(price)
                    min_notional_ok = notional >= float(eff_min_notional or 10.0) - 1e-9
                    under_cap = notional <= hard_cap + 1e-9

                    cap_side = entry_side
                    logging.info(f"[CAP-CHECK] {sym} side={cap_side} qty={qty_str} notional={notional:.6f} "
                                 f"hard_cap={hard_cap:.6f} wallet={wallet_free:.6f} lev={leverage} "
                                 f"minNotional={float(eff_min_notional or 10.0):.2f}")

                    # If we can't meet minNotional within the hard cap, SKIP the trade.
                    if not under_cap:
                        logging.warning(f"[CAP-BLOCK] {sym} order blocked: notional {notional:.6f} > cap {hard_cap:.6f}")
                        s["last_bar_ts"] = last_ts
                        continue

                    if not min_notional_ok:
                        logging.info(f"[SIZE] {sym} below minNotional after cap: {notional:.6f} < {float(eff_min_notional or 10.0):.6f} → skip")
                        s["last_bar_ts"] = last_ts
                        continue
                    if not qty or not qty_str:
                        s['small_qty_count'] += 1
                        if s['small_qty_count'] > 5:
                            s['disabled'] = True
                            logging.warning(f"{sym} disabled due to repeated small qty trades")
                        s["last_bar_ts"] = last_ts
                        continue

                    px = fmt_price_for(client, sym, exchange, price)
                    notional = float(qty) * float(price)
                    if (not ignore_filters) and (notional < float(eff_min_notional or 10.0)):
                        s["last_bar_ts"] = last_ts
                        continue

                    if market_type == "futures" and not _funding_rate_ok(client, sym, entry_side, max_funding_bps):
                        s["last_bar_ts"] = last_ts
                        continue

                    logging.info(f"[SEND] {sym} {entry_side} qty={qty_str} ~notional={notional:.6f}")
                    try:
                        resp = place_market_order(
                            client, sym, entry_side, qty_str, mode, market_type,
                            reduce_only=False, last_price_hint=px, leverage_hint=leverage,
                            exchange=exchange,
                        )
                        logging.info(f"[ORDER-RESP] {resp}")
                    except Exception as e:
                        logging.error(f"[ORDER] API/REJECT {sym} {entry_side} {qty_str}: {e}")
                        s["last_bar_ts"] = last_ts
                        continue

                    # verify fill
                    time.sleep(0.6)
                    net_after = 0.0
                    try:
                        net_after = get_net_position_qty(client, sym)
                    except Exception:
                        pass

                    if abs(float(net_after)) < 1e-12:
                        # Fallback: IOC at top of book (small epsilon so it crosses)
                        try:

                            bid, ask = _best_bid_ask(client, sym)
                            if entry_side == "BUY":
                                px_ioc = (ask if ask > 0 else price) * 1.001
                            else:
                                px_ioc = (bid if bid > 0 else price) * 0.999

                            place_limit_ioc_order(
                                client=client, symbol=sym, side=entry_side, qty_str=qty_str,
                                price=px_ioc, market_type=market_type, leverage_hint=leverage,
                                exchange=exchange,
                            )

                            time.sleep(0.6)
                            try:
                                net_after = get_net_position_qty(client, sym)
                            except Exception:
                                net_after = 0.0
                        except Exception as e:
                            logging.warning(f"[ENTRY-FALLBACK] IOC fallback failed for {sym}: {e}")

                    if abs(float(net_after)) < 1e-12:
                        logging.info(f"[ENTRY] No fill for {sym} ({entry_side}); skipping state/brackets.")
                        s["last_bar_ts"] = last_ts
                        continue

                    # Use actual filled direction/qty from the venue
                    pos_dir = 1 if float(net_after) > 0 else -1
                    fill_qty = _fmt_qty(sym, abs(float(net_after)))
                    qty_str_filled = _fmt_qty_str(sym, fill_qty)

                    stop_px = price - sl_mult * atr if pos_dir == 1 else price + sl_mult * atr
                    take_px = price + tp_mult * atr if pos_dir == 1 else price - tp_mult * atr
                    s.update({
                        "position": pos_dir,
                        "qty": float(fill_qty),
                        "entry_price": price,
                        "entry_time": now,
                        "stop": stop_px,
                        "take": take_px,
                        "unrealized_pnl": 0.0,
                        "disabled": False,
                        "small_qty_count": 0
                    })

                    # Place brackets (use filled side/size)
                    try:
                        sl_q = fmt_price_for(client, sym, exchange, float(stop_px))
                        tp_q = fmt_price_for(client, sym, exchange, float(take_px))
                        filled_side = "BUY" if pos_dir == 1 else "SELL"
                        br = place_bracket_orders(
                            client=client,
                            symbol=sym,
                            side_in=filled_side,
                            qty_str=qty_str_filled,
                            entry_price=px,
                            sl_px=sl_q,
                            tp_px=tp_q,
                            market_type=market_type,
                            leverage_hint=leverage,
                        )
                        s["brackets"] = br
                    except Exception as e:
                        logging.warning(f"[BRACKETS] Failed to place SL/TP for {sym}: {e}")

                    trade_event = {
                        "Event": "Trade Entry",
                        "Action": "OPEN",
                        "Order Side": filled_side,
                        "Position Direction": _pos_dir_from_sign(pos_dir),
                        "Symbol": sym,
                        "Quantity": qty_str_filled,
                        "Entry Price": fmt_price_for(client, sym, exchange, price),
                        "Stop": fmt_price_for(client, sym, exchange, stop_px),
                        "Take": fmt_price_for(client, sym, exchange, take_px),
                        "Order ID": (resp.get("clientOrderId") if isinstance(resp, dict) else "N/A"),
                        "Timestamp": now.isoformat(),
                    }

                    logging.info(json.dumps(trade_event)); send_alert(f"Trade Entry: {sym}", trade_event)
                    if trade_log_csv:
                        append_trade_csv(trade_log_csv, {
                            "ts": now.isoformat(),
                            "symbol": sym,
                            "side": filled_side,
                            "qty": qty_str_filled,
                            "entry": trade_event["Entry Price"],
                            "reason": "entry",
                        })

                    try:
                        save_runtime_state(
                            path=state_file, state=state, symbols=symbols, interval=interval,
                            market_type=market_type, meta={"mode": mode, "event": "entry", "ts": now.isoformat()},
                        )
                        last_autosave = now
                    except Exception as e:
                        logging.warning(f"[STATE] Save after entry failed: {e}")

                # mark bar processed
                s["last_bar_ts"] = last_ts

            # loop pace allows fast exit checks
            time.sleep(FAST_SLEEP)

    except KeyboardInterrupt:
        logging.info("User stopped the bot")

    except Exception as e:
        logging.exception("[LOOP] Unhandled exception in trading loop: %s", e)
    finally:
        # Always stop the feed and persist state on exit
        try:
            data_feed.stop()
        except Exception:
            pass
        try:
            save_runtime_state(
                path=state_file,
                state=state,
                symbols=symbols,
                interval=interval,
                market_type=market_type,
                meta={"mode": mode, "event": "shutdown", "ts": datetime.now(timezone.utc).isoformat()},
            )
            logging.info(f"[STATE] Final state saved to {state_file}")
        except Exception as e:
            logging.warning(f"[STATE] Final save failed: {e}")