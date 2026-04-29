import logging


def place_bracket_orders(
    client,
    symbol: str,
    side_in: str,        # "BUY" | "SELL"
    qty_str: str,
    entry_price: float,
    sl_px: float,
    tp_px: float,
    market_type: str = "futures",
    leverage_hint: int | None = None,
):
    """
    Binance USDT-M Futures bracket: TAKE_PROFIT_MARKET (TP) + STOP_MARKET (SL).
    closePosition=True closes the entire position so we don't need exact qty.
    """
    if market_type != "futures":
        logging.info("[BRACKETS] Spot mode: skipping bracket placement.")
        return {"tp": None, "sl": None}

    close_side = "SELL" if side_in.upper() in ("BUY", "BUY_OPEN") else "BUY"

    tp_resp = sl_resp = None
    try:
        tp_resp = client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=f"{float(tp_px):.8g}",
            closePosition=True,
            workingType="MARK_PRICE",
            priceProtect=True,
        )
        logging.info("[BRACKETS] TP submitted for %s at %s", symbol, tp_px)
    except Exception as e:
        logging.warning("[BRACKETS] TP failed for %s: %s", symbol, e)

    try:
        sl_resp = client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type="STOP_MARKET",
            stopPrice=f"{float(sl_px):.8g}",
            closePosition=True,
            workingType="MARK_PRICE",
            priceProtect=True,
        )
        logging.info("[BRACKETS] SL submitted for %s at %s", symbol, sl_px)
    except Exception as e:
        logging.warning("[BRACKETS] SL failed for %s: %s", symbol, e)

    return {"tp": tp_resp, "sl": sl_resp}


def cancel_bracket_orders(client, symbol: str, bracket_handles: dict | None):
    """Best-effort cancel of TP/SL orders by orderId."""
    if not bracket_handles:
        return
    for key in ("tp", "sl"):
        resp = bracket_handles.get(key) or {}
        order_id = resp.get("orderId") or resp.get("clientOrderId")
        if not order_id:
            continue
        try:
            client.futures_cancel_order(symbol=symbol, orderId=order_id)
            logging.info("[BRACKETS] Cancelled %s %s on %s", key, order_id, symbol)
        except Exception as e:
            logging.warning("[BRACKETS] Cancel %s failed on %s: %s", key, symbol, e)
