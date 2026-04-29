import logging
import time
from typing import Optional, Tuple

from order_manager.order_manager import enforce_exchange_filters, place_market_order


def _get_futures_position_info(client, symbol: str) -> Tuple[float, float, Optional[int]]:
    """Return (net_qty, avg_entry_price, last_update_ms) from Binance futures."""
    try:
        rows = client.futures_position_information(symbol=symbol) or []
        net_amt = 0.0
        w_num = w_den = 0.0
        last_update = None
        for r in rows:
            amt = float(r.get("positionAmt", 0.0))
            ep  = float(r.get("entryPrice",  0.0))
            if amt:
                net_amt += amt
                w = abs(amt)
                w_num += w * ep
                w_den += w
            upd = r.get("updateTime")
            if upd:
                try:
                    upd = int(upd)
                    last_update = upd if last_update is None else max(last_update, upd)
                except Exception:
                    pass
        avg_ep = (w_num / w_den) if w_den > 0 else 0.0
        return net_amt, avg_ep, last_update
    except Exception as e:
        logging.debug("[POS-HELPER] futures_position_information failed for %s: %s", symbol, e)
        return 0.0, 0.0, None


def clean_up_residual_futures_position(
    client, symbol: str, mode: str,
    qty_threshold: float = 1e-6,
    max_retries: int = 3,
    sleep_sec: float = 0.5,
) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            net_qty, _, _ = _get_futures_position_info(client, symbol)
            residual = abs(net_qty)
            if residual <= qty_threshold:
                if attempt > 1:
                    logging.info("[RESIDUAL] %s flattened after %d attempt(s).", symbol, attempt - 1)
                return
            exit_side = "SELL" if net_qty > 0 else "BUY"
            adj_qty, _, _ = enforce_exchange_filters(client, symbol, residual, None)
            if adj_qty <= 0:
                logging.info("[RESIDUAL] %s residual %g below filters; stopping.", symbol, residual)
                return
            place_market_order(client, symbol, exit_side, str(adj_qty),
                               mode, "futures", reduce_only=True)
            logging.info("[RESIDUAL] Attempt %d/%d: %s %s %g reduceOnly",
                         attempt, max_retries, symbol, exit_side, adj_qty)
        except Exception as e:
            logging.warning("[RESIDUAL] Attempt %d failed for %s: %s", attempt, symbol, e)
        time.sleep(max(0.0, sleep_sec))


def get_net_position_qty(client, symbol: str) -> float:
    """Signed net position qty for symbol from Binance futures."""
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
        logging.debug("[POS-HELPER] get_net_position_qty failed for %s: %s", symbol, e)
        return 0.0
