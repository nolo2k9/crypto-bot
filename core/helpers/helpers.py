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
    """Return (minNotional, stepSize, minQty) with CLI overrides applied."""
    f = get_symbol_filters(client, sym, exchange)
    eff_min_notional = float(
        min_notional_override if min_notional_override is not None else f.get("minNotional", 10.0) or 10.0)
    eff_step = float(step_size_override) if step_size_override is not None else f.get("stepSize")
    eff_min_qty = float(min_qty_override if min_qty_override is not None else (f.get("minQty") or 0.0))
    logging.info(f"[FILTERS] {sym} eff_min_notional={eff_min_notional}, eff_step={eff_step}, eff_min_qty={eff_min_qty}")
    return eff_min_notional, eff_step, eff_min_qty