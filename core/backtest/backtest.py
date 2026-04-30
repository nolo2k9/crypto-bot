import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from client.binance_client import create_client
from indicators.indicators import indicators, simple_signal, get_signal
from ml_model.ml_model import RollingML, ml_available
from risk_management.risk_management import position_size_from_atr, trade_pnl, order_notional
from utils.utils import fetch_klines_series
from core.logging.logging import setup_logging

_HTF_MAP: Dict[str, str] = {
    "1m": "15m", "3m": "15m", "5m": "1h", "15m": "4h",
    "30m": "4h", "1h": "1d", "2h": "1d", "4h": "1d",
    "6h": "1d", "8h": "1d", "12h": "1d", "1d": "1w",
}


def _in_session(dt: datetime, trade_hours: Optional[Tuple[int, int]]) -> bool:
    if trade_hours is None:
        return True
    h0, h1 = trade_hours
    h = dt.hour
    return (h0 <= h < h1) if h0 < h1 else (h >= h0 or h < h1)


def backtest(
    symbol: str,
    interval: str,
    days: int = 60,
    risk_per_trade: float = 0.01,
    fee_rate: float = 0.001,
    tp_mult: float = 3.0,
    sl_mult: float = 2.0,
    use_ml: bool = True,
    market_type: str = "spot",
    leverage: int = 10,
    min_trade_size: float = 1e-6,
    max_hold_hours: int = 72,
    rsi_period: int = 14,
    slippage_bps_entry: float = 5.0,
    slippage_bps_exit: float = 5.0,
    spread_bps: float = 2.0,
    funding_bps_per_day: float = 0.0,
    htf_interval: Optional[str] = None,
    trade_hours: Optional[Tuple[int, int]] = (6, 22),
    partial_tp_mult: float = 1.5,
    adx_threshold: float = 20.0,
    strategies: Optional[List[str]] = None,
) -> dict:
    def _bps(bps: float) -> float:
        return float(bps) / 10_000.0

    def _fill(price: float, side: str, entry: bool) -> float:
        adj = _bps(spread_bps) / 2.0 + _bps(slippage_bps_entry if entry else slippage_bps_exit)
        return price * (1.0 + adj) if side == "BUY" else price * (1.0 - adj)

    setup_logging(None)
    client = create_client(
        api_key=os.getenv("BINANCE_KEY") or "",
        api_secret=os.getenv("BINANCE_SECRET") or "",
        market_type=market_type,
        testnet=False,
    )
    if market_type == "futures":
        for fn, kw in [
            ("set_position_mode", {"one_way": True}),
            ("set_margin_type", {"symbol": symbol, "margin_type": "ISOLATED"}),
            ("set_leverage", {"symbol": symbol, "leverage": leverage}),
        ]:
            try:
                f = getattr(client, fn, None)
                if callable(f):
                    f(**kw)
            except Exception:
                pass

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        df = fetch_klines_series(client, symbol, interval, start, end)
    except Exception as e:
        logging.error("Error fetching klines: %s", e)
        return {}
    if df is None or df.empty:
        logging.error("No data for %s [%s]", symbol, interval)
        return {}

    logging.info("Fetched %d bars for %s [%s]", len(df), symbol, interval)
    df_ind = indicators(df, rsi_period=rsi_period).dropna(subset=["ATR", "RSI"])
    if df_ind.empty:
        logging.error("No bars after indicator warm-up.")
        return {}

    # ---- HTF data ----
    _htf = None if (htf_interval or "").lower() == "none" else (htf_interval or _HTF_MAP.get(interval.lower()))
    htf_ema21: pd.Series = pd.Series(dtype=float)
    htf_ema50: pd.Series = pd.Series(dtype=float)
    if _htf:
        try:
            df_htf_raw = fetch_klines_series(client, symbol, _htf,
                                              start - timedelta(days=30), end)
            if df_htf_raw is not None and not df_htf_raw.empty:
                df_htf = indicators(df_htf_raw, rsi_period=rsi_period)
                merged = pd.merge_asof(
                    df_ind[[]], df_htf[["EMA_21", "EMA_50"]],
                    left_index=True, right_index=True,
                ).ffill()
                htf_ema21 = merged["EMA_21"]
                htf_ema50 = merged["EMA_50"]
                logging.info("HTF [%s] loaded: %d bars, aligned to %d main bars",
                             _htf, len(df_htf), len(df_ind))
        except Exception as e:
            logging.warning("[BT] HTF load failed: %s", e)

    # ---- Walk-forward ML ----
    # Train on first 70%; only predict on the remaining 30% (true out-of-sample).
    ml_split = int(len(df_ind) * 0.70)
    ml = None
    if use_ml and ml_available() and len(df_ind) > 300:
        ml = RollingML()
        try:
            ml.fit(df_ind.iloc[:ml_split])
            logging.info("ML initial train: bars 0..%d; OOS from bar %d", ml_split - 1, ml_split)
        except Exception as e:
            logging.warning("ML training error: %s", e)
            ml = None

    # ---- Strategy setup ----
    active_strategies = strategies or ["trend"]
    risk_per_strategy = risk_per_trade / len(active_strategies)

    def _blank_pos():
        return {"pos": 0, "qty": None, "entry_price": None, "stop": None,
                "take": None, "entry_time": None, "scaled_out": False}

    strat_state = {s: _blank_pos() for s in active_strategies}

    # ---- Shared P&L state ----
    balance = 10_000.0
    equity_curve: List[float] = []
    peak = balance
    max_dd = 0.0
    trades = wins = losses = 0
    gross_pnl = total_fees = realized_pnl = total_funding = 0.0

    def _size(px: float, atr: float) -> float:
        return max(position_size_from_atr(balance, px, atr, risk_per_strategy, sl_mult=sl_mult), min_trade_size)

    def _record_exit(side_in: str, ep: float, ex: float, q: float, ts_entry, ts_exit):
        nonlocal balance, trades, wins, losses, gross_pnl, total_fees, realized_pnl, total_funding
        g = trade_pnl(side_in, ep, ex, q)
        f = (order_notional(side_in, ep, q) + order_notional(
            "SELL" if side_in == "BUY" else "BUY", ex, q)) * fee_rate
        fund = 0.0
        if ts_entry and market_type == "futures" and funding_bps_per_day > 0:
            dur = max(0.0, (ts_exit - ts_entry).total_seconds() / 86400.0)
            fund = order_notional(side_in, ep, q) * _bps(funding_bps_per_day) * dur
        pnl = g - f - fund
        balance += pnl
        trades += 1; gross_pnl += g; total_fees += f; total_funding += fund; realized_pnl += pnl
        wins += (1 if pnl > 0 else 0); losses += (1 if pnl <= 0 else 0)
        return pnl

    for i in range(len(df_ind) - 1):
        row = df_ind.iloc[i]
        nxt = df_ind.iloc[i + 1]
        price = float(row["Close"])
        atr_val = float(row["ATR"])
        ts = row.name if isinstance(row.name, datetime) else datetime.now(timezone.utc)

        # Process each strategy independently
        for strat_name, st in strat_state.items():
            pos        = st["pos"]
            qty        = st["qty"]
            entry_price = st["entry_price"]
            stop       = st["stop"]
            take       = st["take"]
            entry_time = st["entry_time"]
            scaled_out = st["scaled_out"]

            side_in_cur = "BUY" if pos == 1 else "SELL"

            # ATR trailing stop
            if pos and stop is not None and atr_val > 0:
                if pos == 1:
                    stop = max(stop, price - sl_mult * atr_val)
                else:
                    stop = min(stop, price + sl_mult * atr_val)

            # Partial TP
            if (partial_tp_mult > 0 and pos and not scaled_out
                    and entry_price is not None and qty and atr_val > 0):
                profit = (price - entry_price) if pos == 1 else (entry_price - price)
                if profit >= partial_tp_mult * atr_val:
                    half = qty * 0.5
                    close_px = _fill(price, "SELL" if pos == 1 else "BUY", entry=False)
                    pnl_part = _record_exit(side_in_cur, entry_price, close_px, half, entry_time, ts)
                    qty -= half
                    stop = entry_price
                    scaled_out = True
                    logging.info("PARTIAL-TP @%d [%s]: %s 50%% @ %.4f pnl=%.4f",
                                 i, strat_name, symbol, close_px, pnl_part)

            # Exits
            if pos and stop is not None and take is not None:
                hit_stop = (row["Low"] <= stop) if pos == 1 else (row["High"] >= stop)
                hit_take = (row["High"] >= take) if pos == 1 else (row["Low"] <= take)
                time_exit = entry_time and (ts - entry_time).total_seconds() / 3600 > max_hold_hours
                if hit_stop or hit_take or time_exit:
                    ex_px = _fill(float(nxt["Open"]), "SELL" if pos == 1 else "BUY", entry=False)
                    pnl = _record_exit(side_in_cur, entry_price, ex_px, qty, entry_time, nxt.name)
                    reason = "stop" if hit_stop else "take" if hit_take else "time"
                    logging.info("EXIT @%d [%s] [%s]: %s entry=%.4f exit=%.4f pnl=%.4f",
                                 i, strat_name, reason, symbol, entry_price, ex_px, pnl)
                    st.update(_blank_pos())
                    continue

            # Entries
            if not pos and atr_val > 0:
                if not _in_session(ts, trade_hours):
                    st.update({"pos": pos, "qty": qty, "entry_price": entry_price,
                               "stop": stop, "take": take, "entry_time": entry_time,
                               "scaled_out": scaled_out})
                    continue

                sig = get_signal(row, strategy=strat_name, adx_threshold=adx_threshold)

                # ML gate (trend strategy only)
                if strat_name == "trend" and ml and ml.trained and i >= ml_split:
                    try:
                        ml.update_close(price)
                        if ml.should_retrain(i):
                            ml.fit(df_ind.iloc[:i])
                        ml_sig = ml.predict_signal(row)
                        if ml_sig and sig and ml_sig != sig:
                            sig = 0
                    except Exception as e:
                        logging.warning("ML predict error: %s", e)

                # HTF trend gate (trend + momentum strategies only)
                if sig and strat_name in ("trend", "momentum") and len(htf_ema21) > i:
                    h21 = float(htf_ema21.iloc[i])
                    h50 = float(htf_ema50.iloc[i])
                    if h21 > 0 and h50 > 0 and not pd.isna(h21) and not pd.isna(h50):
                        if sig == 1 and h21 < h50 * 0.999:
                            sig = 0
                        if sig == -1 and h21 > h50 * 1.001:
                            sig = 0

                if sig:
                    side_in = "BUY" if sig == 1 else "SELL"
                    en_px = _fill(float(nxt["Open"]), side_in, entry=True)
                    q = _size(en_px, atr_val)
                    if q > 0:
                        pos = 1 if sig == 1 else -1
                        qty = q
                        entry_price = en_px
                        entry_time = nxt.name if isinstance(nxt.name, datetime) else ts
                        stop = en_px - sl_mult * atr_val if pos == 1 else en_px + sl_mult * atr_val
                        take = en_px + tp_mult * atr_val if pos == 1 else en_px - tp_mult * atr_val
                        scaled_out = False
                        logging.info("ENTRY @%d [%s]: %s %s en=%.4f qty=%.6f",
                                     i, strat_name, symbol, side_in, en_px, qty)

            st.update({"pos": pos, "qty": qty, "entry_price": entry_price,
                       "stop": stop, "take": take, "entry_time": entry_time,
                       "scaled_out": scaled_out})

        # MTM equity across all open strategy positions
        mtm = sum(
            trade_pnl("BUY" if st["pos"] == 1 else "SELL", st["entry_price"], price, st["qty"])
            for st in strat_state.values()
            if st["pos"] and st["entry_price"]
        )
        eq = balance + mtm
        equity_curve.append(eq)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak if peak > 0 else 0.0)

    # Force-close any open positions
    for strat_name, st in strat_state.items():
        if st["pos"] and st["entry_price"]:
            last_px = float(df_ind.iloc[-1]["Close"])
            side_in = "BUY" if st["pos"] == 1 else "SELL"
            ex_px = _fill(last_px, "SELL" if st["pos"] == 1 else "BUY", entry=False)
            pnl = _record_exit(side_in, st["entry_price"], ex_px, st["qty"],
                               st["entry_time"], df_ind.index[-1])
            logging.info("FINAL EXIT [%s]: %s entry=%.4f exit=%.4f pnl=%.4f",
                         strat_name, symbol, st["entry_price"], ex_px, pnl)

    win_rate = wins / trades if trades else 0.0
    summary = {
        "event": "backtest_summary",
        "symbol": symbol, "interval": interval, "htf": _htf,
        "market_type": market_type,
        "leverage": leverage if market_type == "futures" else "N/A",
        "strategies": active_strategies,
        "tp_mult": tp_mult, "sl_mult": sl_mult,
        "partial_tp_mult": partial_tp_mult,
        "adx_threshold": adx_threshold,
        "trade_hours": list(trade_hours) if trade_hours else None,
        "bars": len(df_ind),
        "ml_oos_from": ml_split if ml else None,
        "trades": trades, "wins": wins, "losses": losses,
        "win_rate": round(win_rate, 4),
        "gross_pnl": round(gross_pnl, 4),
        "fees": round(total_fees, 4),
        "funding": round(total_funding, 4),
        "realized_pnl": round(realized_pnl, 4),
        "start_balance": 10_000.0,
        "end_balance": round(balance, 4),
        "return_%": round((balance / 10_000.0 - 1) * 100, 4),
        "max_drawdown_%": round(max_dd * 100, 4),
    }
    logging.info(json.dumps(summary, indent=2))
    return summary


def grid_search_backtest(
    symbol: str,
    interval: str,
    days: int = 90,
    tp_mults: Optional[List[float]] = None,
    sl_mults: Optional[List[float]] = None,
    adx_thresholds: Optional[List[float]] = None,
    **base_kwargs,
) -> None:
    tp_mults = tp_mults or [2.0, 2.5, 3.0, 3.5]
    sl_mults = sl_mults or [1.5, 2.0, 2.5]
    adx_thresholds = adx_thresholds or [15.0, 20.0, 25.0]

    combos = [(tp, sl, adx) for tp in tp_mults for sl in sl_mults for adx in adx_thresholds]
    logging.info("Grid search: %d combinations for %s [%s] over %dd", len(combos), symbol, interval, days)

    results = []
    for n, (tp, sl, adx) in enumerate(combos, 1):
        logging.info("[%d/%d] tp=%.1f sl=%.1f adx=%.0f", n, len(combos), tp, sl, adx)
        try:
            r = backtest(symbol, interval, days=days, tp_mult=tp, sl_mult=sl,
                         adx_threshold=adx, **base_kwargs)
            if r:
                results.append(r)
        except Exception as e:
            logging.warning("Grid run failed tp=%.1f sl=%.1f adx=%.0f: %s", tp, sl, adx, e)

    results.sort(key=lambda x: x.get("return_%", -999), reverse=True)
    logging.info("=== GRID SEARCH RESULTS (by return) ===")
    for i, r in enumerate(results[:15], 1):
        logging.info(
            "#%2d  tp=%.1f sl=%.1f adx=%.0f  ret=%+.2f%%  dd=%.2f%%  trades=%d  wr=%.2f",
            i, r["tp_mult"], r["sl_mult"], r.get("adx_threshold", 0),
            r.get("return_%", 0), r.get("max_drawdown_%", 0),
            r.get("trades", 0), r.get("win_rate", 0),
        )
    if results:
        b = results[0]
        logging.info("BEST → tp=%.1f sl=%.1f adx=%.0f ret=%+.2f%% dd=%.2f%%",
                     b["tp_mult"], b["sl_mult"], b.get("adx_threshold", 0),
                     b.get("return_%", 0), b.get("max_drawdown_%", 0))
