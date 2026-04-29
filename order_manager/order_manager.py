#!/usr/bin/env python3
"""
order_manager/order_manager.py  —  Binance (spot + USDT-M futures)

Provides:
  get_symbol_filters, enforce_exchange_filters, format_quantity
  place_market_order, place_limit_ioc_order
  close_position_fast
  cap_qty_by_limits, calc_total_pnl
  _best_bid_ask
"""

from __future__ import annotations

import logging
import math
import secrets
import time
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, Tuple


# ------------------------------------------------------------------ #
#  Tiny helpers                                                        #
# ------------------------------------------------------------------ #

def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _decimal_truncate(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    d_step = Decimal(str(step))
    places = max(0, -d_step.as_tuple().exponent)
    q = Decimal(10) ** (-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_DOWN))


def _dec_trunc_to_dp(x: float, dp: int) -> Decimal:
    q = Decimal(1).scaleb(-dp)
    return Decimal(str(x)).quantize(q, rounding=ROUND_DOWN)


def cid_gen(symbol: str = "") -> str:
    ts = int(time.time() * 1000)
    rand = secrets.token_hex(3)
    sym = "".join(ch for ch in (symbol or "") if ch.isalnum())[:8]
    return (f"sb_{sym}_{ts}_{rand}" if sym else f"sb_{ts}_{rand}")[:32]


def _price_to_str(px: float, filters: dict) -> str:
    pp = filters.get("pricePrecision")
    if pp is not None:
        return f"{px:.{int(pp)}f}"
    tick = filters.get("tickSize")
    if tick:
        q = Decimal(str(tick))
        return str(Decimal(str(px)).quantize(q, rounding=ROUND_DOWN))
    return str(px)


def _best_bid_ask(client, symbol: str) -> tuple[float, float]:
    """Return (best_bid, best_ask) from Binance order book, or (0.0, 0.0)."""
    try:
        # Futures
        ob = client.futures_order_book(symbol=symbol, limit=5)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        bid = float(bids[0][0]) if bids else 0.0
        ask = float(asks[0][0]) if asks else 0.0
        if bid or ask:
            return bid, ask
    except Exception:
        pass
    try:
        # Spot fallback
        ob = client.get_order_book(symbol=symbol, limit=5)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        bid = float(bids[0][0]) if bids else 0.0
        ask = float(asks[0][0]) if asks else 0.0
        return bid, ask
    except Exception:
        return 0.0, 0.0


def _signed_net_qty(client, symbol: str) -> float:
    """Signed net position from Binance futures_position_information."""
    try:
        rows = client.futures_position_information(symbol=symbol)
        net = 0.0
        for r in rows if isinstance(rows, list) else []:
            try:
                net += float(r.get("positionAmt", 0.0))
            except Exception:
                pass
        return net
    except Exception:
        return 0.0


# ------------------------------------------------------------------ #
#  Symbol filters                                                      #
# ------------------------------------------------------------------ #

def get_symbol_filters(client, symbol: str, exchange: str = "binance") -> Dict:
    try:
        info = client.get_symbol_info(symbol)
    except Exception:
        info = None

    if not info:
        return {
            "pricePrecision": None, "quantityPrecision": None,
            "tickSize": None, "stepSize": None,
            "minNotional": 10.0, "minQty": 0.0, "maxQty": float("inf"),
        }

    filters = {f.get("filterType"): f for f in info.get("filters", [])}
    lot = filters.get("LOT_SIZE", {})
    price_f = filters.get("PRICE_FILTER", {})
    mn_f = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))

    return {
        "pricePrecision":    int(info.get("pricePrecision", 0)),
        "quantityPrecision": int(info.get("quantityPrecision", 0)),
        "tickSize":          _safe_float(price_f.get("tickSize"), 0.0),
        "stepSize":          _safe_float(lot.get("stepSize"), 0.0),
        "minNotional":       _safe_float(mn_f.get("minNotional", mn_f.get("notional", 10.0)), 10.0),
        "minQty":            _safe_float(lot.get("minQty"), 0.0),
        "maxQty":            _safe_float(lot.get("maxQty"), float("inf")),
    }


def enforce_exchange_filters(
    client, symbol: str, qty: float, price: Optional[float],
    exchange: str = "binance", market_type: str = "futures",
) -> Tuple[float, Optional[float], float]:
    f = get_symbol_filters(client, symbol, exchange)

    adj_qty = float(qty)
    step = f.get("stepSize")
    if not step or step <= 0:
        qp = f.get("quantityPrecision")
        step = (10 ** -(qp or 0)) if qp is not None else (0.001 if adj_qty < 1.0 else 1.0)
    if step and step > 0:
        adj_qty = math.floor(adj_qty / step) * step
    adj_qty = max(adj_qty, f.get("minQty", 0.0))
    max_q = f.get("maxQty", float("inf"))
    if adj_qty > max_q:
        adj_qty = max_q

    adj_price = price
    if price is not None:
        tick = f.get("tickSize") or (10 ** -(f.get("pricePrecision") or 0))
        if tick and tick > 0:
            adj_price = max(math.floor(price / tick) * tick, tick)

    return adj_qty, adj_price, f.get("minNotional", 0.0)


def format_quantity(qty: float, step_or_precision, *, precision: Optional[int] = None) -> str:
    q = float(qty)
    if step_or_precision and isinstance(step_or_precision, (int, float)) and step_or_precision > 0:
        q = _decimal_truncate(q, float(step_or_precision))
    if precision is not None and precision >= 0:
        return f"{q:.{precision}f}"
    s = f"{q:.12f}"
    return s.rstrip("0").rstrip(".")


def _format_qty_for_exchange(client, symbol: str, exchange: Optional[str], qty: float) -> str:
    f = get_symbol_filters(client, symbol, exchange or "binance")
    step = f.get("stepSize")
    qp = f.get("quantityPrecision")
    if step and step > 0:
        q = _decimal_truncate(qty, float(step))
        dp = max(0, -Decimal(str(step)).as_tuple().exponent)
        return f"{q:.{dp}f}"
    if qp is not None:
        qd = _dec_trunc_to_dp(qty, int(qp))
        return f"{qd:.{int(qp)}f}"
    for dp in (3, 2, 1, 0):
        qd = _dec_trunc_to_dp(qty, dp)
        if qd > 0:
            return f"{qd:.{dp}f}"
    return "0"


# ------------------------------------------------------------------ #
#  Order placement                                                     #
# ------------------------------------------------------------------ #

def _binance_create(client, payload: dict, market_type: str = "futures"):
    """Send an order via Binance python-binance client."""
    if market_type == "futures":
        return client.futures_create_order(**payload)
    return client.create_order(**payload)


def place_market_order(
    client, symbol: str, side: str, qty_str: str, mode: str, market_type: str,
    reduce_only: bool = False, last_price_hint=None, leverage_hint=None,
    exchange: str = "binance",
):
    sym = symbol.upper()
    side_u = side.upper().replace("_OPEN", "").replace("_CLOSE", "")
    qty_fmt = _format_qty_for_exchange(client, sym, exchange, float(qty_str))

    payload: Dict = {"symbol": sym, "side": side_u, "type": "MARKET", "quantity": qty_fmt}
    if reduce_only:
        payload["reduceOnly"] = "true"

    if mode == "paper":
        # Simulate taker slippage: 3 bps adverse to direction
        ref = float(last_price_hint) if last_price_hint else 0.0
        slip = ref * 0.0003 if ref > 0 else 0.0
        fill_px = (ref + slip) if side_u == "BUY" else (ref - slip)
        logging.info("[PAPER] MARKET %s %s qty=%s fill=%.6f (slip=%.1f bps)",
                     sym, side_u, qty_fmt, fill_px, 3.0)
        return {
            "clientOrderId": cid_gen(sym), "status": "FILLED",
            "paper": True, "avgPrice": str(fill_px), "executedQty": qty_fmt,
        }

    return _binance_create(client, payload, market_type)


def place_limit_ioc_order(
    client, symbol: str, side: str, qty_str: str, price: float,
    market_type: str, leverage_hint=None, exchange: str = "binance",
):
    sym = symbol.upper()
    side_u = side.upper().replace("_OPEN", "").replace("_CLOSE", "")
    f = get_symbol_filters(client, sym, exchange)
    qty_fmt = _format_qty_for_exchange(client, sym, exchange, float(qty_str))
    px_str = _price_to_str(float(price), f)

    payload: Dict = {
        "symbol": sym, "side": side_u, "type": "LIMIT",
        "timeInForce": "IOC", "quantity": qty_fmt, "price": px_str,
    }
    return _binance_create(client, payload, market_type)


def close_position_fast(
    client, symbol: str, side_close: Optional[str] = None,
    market_type: str = "futures", leverage_hint=None, exchange: str = "binance",
):
    """Close the full net position with a MARKET reduceOnly order."""
    net = _signed_net_qty(client, symbol)
    if abs(net) < 1e-12:
        return {"ok": True, "status": "flat"}

    side = "SELL" if net > 0 else "BUY"
    qty_fmt = _format_qty_for_exchange(client, symbol, exchange, abs(net))

    payload: Dict = {
        "symbol": symbol.upper(),
        "side": side,
        "type": "MARKET",
        "quantity": qty_fmt,
        "reduceOnly": "true",
    }
    return _binance_create(client, payload, market_type)


# ------------------------------------------------------------------ #
#  Sizing clamp                                                        #
# ------------------------------------------------------------------ #

def cap_qty_by_limits(
    qty: float, price: float, *,
    wallet_balance: float, leverage: float, filters: Dict, safety: float = 0.90,
) -> float:
    step         = Decimal(str(filters.get("stepSize") or 0))
    min_qty      = Decimal(str(filters.get("minQty") or 0))
    min_notional = Decimal(str(filters.get("minNotional") or 0))
    max_qty      = Decimal(str(filters.get("maxQty") or float("inf")))
    px           = Decimal(str(price))
    q            = Decimal(str(qty))

    hard_cap = Decimal(str(wallet_balance)) * Decimal(str(leverage)) * Decimal(str(safety))
    if px > 0 and q * px > hard_cap:
        q = hard_cap / px
    if step and step > 0:
        q = (q // step) * step
    if max_qty and q > max_qty:
        q = (max_qty // step) * step if step and step > 0 else max_qty
    if min_qty and q < min_qty:
        q = min_qty
    if min_notional and px > 0 and q * px < min_notional:
        need = min_notional / px
        q = ((need // step) * step) if step and step > 0 else need
    if px > 0 and q * px > hard_cap:
        return 0.0
    return float(q)


def calc_total_pnl(state_dict: dict) -> float:
    try:
        realized   = sum(float(s.get("realized_pnl",   0.0)) for s in state_dict.values())
        unrealized = sum(float(s.get("unrealized_pnl", 0.0)) for s in state_dict.values())
        return realized + unrealized
    except Exception:
        return 0.0
