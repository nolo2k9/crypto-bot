import logging
from typing import Dict, List
from core.helpers.futures_position_helpers.futures_Position_helpers import _get_futures_position_info
from order_manager.order_manager import (
    enforce_exchange_filters,
    place_market_order,
)
from datetime import datetime, timedelta, timezone


# -------- Startup sync --------
def sync_open_positions_on_start(client, data_feed: "DataFeed", state: Dict[str, dict],
                                 symbols: List[str], market_type: str,
                                 sl_mult: float, tp_mult: float,
                                 flatten_on_start: bool, mode: str) -> None:
    if market_type != "futures":
        logging.info("Startup sync: spot mode — no external positions to sync.")
        return
    synced, flattened = [], []
    for sym in symbols:
        try:
            net_qty, entry_price, upd_ms = _get_futures_position_info(client, sym)
        except Exception as e:
            logging.warning(f"Startup sync: failed to fetch futures position for {sym}: {e}")
            continue
        if abs(net_qty) < 1e-12:
            continue
        side = "LONG" if net_qty > 0 else "SHORT"
        abs_qty = abs(net_qty)
        if flatten_on_start:
            exit_side = "SELL" if net_qty > 0 else "BUY"
            adj_qty, _, _ = enforce_exchange_filters(client, sym, abs_qty, None)
            qty_str = str(adj_qty)
            try:
                place_market_order(client, sym, exit_side, qty_str, mode, "futures", reduce_only=True)
                logging.info(f"Startup flatten: {sym} {side} {qty_str} closed.")
                flattened.append(sym)
                s = state[sym]
                s.update({"position": 0, "qty": 0.0, "entry_price": None, "entry_time": None,
                          "stop": None, "take": None, "unrealized_pnl": 0.0})
            except Exception as e:
                logging.warning(f"Startup flatten: failed to close {sym} {side}: {e}")
            continue

        df = data_feed.data.get(sym)
        if df is None or df.empty or 'ATR' not in df.columns:
            logging.warning(f"Startup sync: no ATR data yet for {sym}; leaving stops unset.")
            atr = 0.0
        else:
            atr = float(df.iloc[-1].get('ATR', 0.0)) or 0.0

        s = state[sym]
        s['position'] = 1 if net_qty > 0 else -1
        s['qty'] = abs_qty
        s['entry_price'] = float(entry_price) if entry_price else None
        s['entry_time'] = (datetime.fromtimestamp(upd_ms / 1000.0, tz=timezone.utc)
                           if upd_ms else datetime.now(timezone.utc))
        if atr > 0.0 and s['entry_price'] is not None:
            if s['position'] == 1:
                s['stop'] = s['entry_price'] - sl_mult * atr
                s['take'] = s['entry_price'] + tp_mult * atr
            else:
                s['stop'] = s['entry_price'] + sl_mult * atr
                s['take'] = s['entry_price'] - tp_mult * atr
        else:
            s['stop'] = None
            s['take'] = None
        s['unrealized_pnl'] = 0.0
        s['disabled'] = False
        s['small_qty_count'] = 0
        synced.append(f"{sym} {side} qty={abs_qty:g} ep={entry_price:g}")
    if flatten_on_start and flattened:
        logging.info(f"Startup flatten complete: {flattened}")
    if (not flatten_on_start) and synced:
        logging.info("Startup position sync (futures): " + "; ".join(synced))
