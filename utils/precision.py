# CryptoBot/utils/precision.py
from decimal import Decimal, ROUND_DOWN

def floor_to_step(value: float, step: float | None, prec: int | None) -> float:
    if step and step > 0:
        q = Decimal(str(step))
        return float((Decimal(str(value)) // q) * q)
    if prec is not None:
        q = Decimal(1).scaleb(-int(prec))
        return float(Decimal(str(value)).quantize(q, rounding=ROUND_DOWN))
    return float(value)

def format_quantity(qty: float, step: float | None, prec: int | None) -> str | None:
    q = floor_to_step(qty, step, prec)
    if q <= 0:
        return None
    if step and step > 0:
        # format to the number of decimals in step (e.g., step=0.001 -> 3)
        s = str(step)
        dp = len(s.split(".")[1]) if "." in s else 0
        return f"{q:.{dp}f}"
    if prec is not None:
        return f"{q:.{prec}f}"
    return str(q)
