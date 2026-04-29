from typing import Optional
from decimal import Decimal, ROUND_DOWN
import math


def _quantize_price(p: float, price_precision: Optional[int]) -> float:
    if price_precision is None:
        return float(Decimal(str(p)))
    q = Decimal(1).scaleb(-price_precision)
    return float(Decimal(str(p)).quantize(q, rounding=ROUND_DOWN))


def _quantize(x: float, step: Optional[float], precision: Optional[int]) -> float:
    """
    Floor 'x' to either a LOT_SIZE step, or to a decimal precision if provided.
    """
    if step and step > 0:
        q = Decimal(str(step))
        return float((Decimal(str(x)) // q) * q)
    if precision is not None:
        q = Decimal(1).scaleb(-precision)  # 10^-precision
        return float(Decimal(str(x)).quantize(q, rounding=ROUND_DOWN))
    return float(x)


def _price_precision_for(client, symbol: str, exchange: str) -> Optional[int]:
    """Infer price precision from client symbol info or tick size; avoid importing order_manager to prevent cycles."""
    info = None
    try:
        if hasattr(client, "get_symbol_info"):
            info = client.get_symbol_info(symbol)
    except Exception:
        info = None

    # Fallback: search contract listings
    if not info and hasattr(client, "contract"):
        for fn_name in ("get_contracts", "contracts", "symbols", "instruments"):
            fn = getattr(client.contract, fn_name, None)
            if callable(fn):
                try:
                    rows = fn()
                    rows = rows.get("data", rows) if isinstance(rows, dict) else rows
                    if isinstance(rows, list):
                        for r in rows:
                            sym_id = str(r.get("symbol") or r.get("symbolId") or r.get("symbol_id") or r.get("id") or "").upper()
                            if sym_id == symbol.upper():
                                info = r
                                break
                except Exception:
                    pass
            if info:
                break

    # Explicit precision fields
    for key in ("pricePrecision", "price_precision", "pxPrecision", "px_precision"):
        if isinstance(info, dict) and info.get(key) is not None:
            try:
                return int(info.get(key))
            except Exception:
                pass

    # Derive from tick size
    tick = None
    if isinstance(info, dict):
        for f in (info.get("filters") or []):
            if isinstance(f, dict) and f.get("filterType") in ("PRICE_FILTER", "PRICE_FILTERS"):
                try:
                    tick = float(f.get("tickSize") or f.get("priceTick") or f.get("minPriceTick"))
                    break
                except Exception:
                    pass
        if tick is None:
            for k in ("tickSize","priceTick","minPriceTick"):
                if info.get(k) is not None:
                    try:
                        tick = float(info[k]); break
                    except Exception:
                        pass
    if tick and tick > 0:
        try:
            return max(0, int(round(-math.log10(tick))))
        except Exception:
            pass
    return None


def fmt_price_for(client, symbol: str, exchange: str, px: float) -> float:
    pp = _price_precision_for(client, symbol, exchange)
    return _quantize_price(px, pp)


def fmt_price_str_for(client, symbol: str, exchange: str, px: float) -> str:
    pp = _price_precision_for(client, symbol, exchange)
    if pp is None:
        return format(Decimal(str(px)).normalize(), "f")
    return f"{px:.{pp}f}"