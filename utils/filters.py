def normalize_filters(raw: dict) -> dict:
    f = raw or {}
    def _g(*names, cast=float, default=None):
        for n in names:
            if n in f and f[n] is not None:
                try:
                    return cast(f[n])
                except Exception:
                    pass
        return default

    d = {
        "min_notional": _g("minNotional","notional","quoteMinNotional","minNotionalValue"),
        "step_size":    _g("stepSize","step_size"),
        "min_qty":      _g("minQty","min_quantity"),
        "qty_prec":     _g("quantityPrecision","qtyPrecision", cast=int, default=None),
        "px_prec":      _g("pricePrecision","pxPrecision", cast=int, default=None),
    }

    # derive step_size from quantityPrecision when missing
    if (d["step_size"] in (None, 0)) and (d["qty_prec"] is not None):
        d["step_size"] = 10 ** (-int(d["qty_prec"]))

    # if min_qty missing, use one step
    if (d["min_qty"] in (None, 0)) and d["step_size"]:
        d["min_qty"] = d["step_size"]

    return d
