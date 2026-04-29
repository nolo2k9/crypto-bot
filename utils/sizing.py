from utils.filters import normalize_filters
from utils.precision import format_quantity
from utils.balance import get_free_usdt
from order_manager.order_manager import get_symbol_filters, cap_qty_by_limits


def snap_cap_and_format(client, sym, exchange, market_type, price, intended_qty, leverage, safety=0.90):
    f = normalize_filters(get_symbol_filters(client, sym, exchange))
    free_usdt = max(0.0, get_free_usdt(client, market_type))
    qty_capped = cap_qty_by_limits(
        qty=float(intended_qty),
        price=float(price),
        wallet_balance=float(free_usdt),
        leverage=float(leverage if market_type == "futures" else 1),
        filters={"minNotional": f["min_notional"], "stepSize": f["step_size"], "minQty": f["min_qty"]},
        safety=float(safety),
    )
    if qty_capped <= 0:
        return 0.0, None
    qty_str = format_quantity(qty_capped, f["step_size"], f["qty_prec"])
    return float(qty_str) if qty_str else 0.0, qty_str
