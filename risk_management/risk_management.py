import numpy as np
import pandas as pd


def position_size_from_atr(
    balance_usdt: float, price: float, atr: float, risk_per_trade: float,
    leverage: int = 1, sl_mult: float = 1.0,
) -> float:
    if atr <= 0 or price <= 0:
        return 0.0
    # Correct ATR sizing: risk exactly risk_per_trade×balance when stop is hit.
    # Stop is placed at sl_mult×ATR from entry, so divide by that full distance.
    risk_dollars = balance_usdt * risk_per_trade
    stop_distance = max(atr * max(sl_mult, 1e-6), 1e-9)
    qty = risk_dollars / stop_distance
    return max(qty, 0.0)


def trade_pnl(side: str, entry: float, exit: float, qty: float) -> float:
    if side == "BUY":
        return (exit - entry) * qty
    return (entry - exit) * qty


def order_notional(side: str, price: float, qty: float) -> float:
    return abs(price * qty)


def portfolio_var(positions: dict, prices: dict, df: dict, leverage: int = 1) -> float:
    weights = []
    returns = []
    total_notional = sum(
        p["qty"] * prices.get(s, p.get("entry_price") or 0) * leverage
        for s, p in positions.items() if p.get("qty", 0) > 0
    )
    if total_notional <= 0:
        return 0.0

    for sym, pos in positions.items():
        if pos.get("qty", 0) > 0:
            price = prices.get(sym) or pos.get("entry_price") or 0
            if price <= 0:
                continue
            notional = pos["qty"] * price * leverage
            weights.append(notional / total_notional)
            rets = (
                df[sym]["Close"].pct_change().dropna().tail(252)
                if sym in df and not df[sym].empty
                else pd.Series(dtype=float)
            )
            returns.append(rets)

    if not weights or not returns:
        return 0.0

    # Single position: variance only, no covariance needed
    if len(returns) == 1:
        var = float(returns[0].var()) * 252
        return float(np.sqrt(max(var, 0.0)))

    # Align all series to the same length before computing covariance
    min_len = min(len(r) for r in returns)
    if min_len < 2:
        return 0.0
    matrix = np.array([r.iloc[-min_len:].values for r in returns])
    cov_matrix = np.cov(matrix)
    w = np.array(weights)
    variance = float(np.dot(w, np.dot(cov_matrix * 252, w)))
    return float(np.sqrt(max(variance, 0.0)))



def adaptive_risk_scaling(state, base_risk, drawdown, max_consec_losses=3, max_drawdown=0.10):
    """
    Scale down risk dynamically based on drawdown and consecutive losses.
    - Reduce risk linearly as drawdown approaches max_drawdown.
    - Reduce risk further if consecutive losses exceed threshold.
    """
    if max_drawdown <= 0:
        max_drawdown = 1e-6

    # Existing logic:
    risk_scale = 1.0
    if drawdown > max_drawdown:
        risk_scale *= max(0.0, 1.0 - (drawdown - max_drawdown) / max_drawdown)

    # Other logic based on loss streaks, etc.

    adjusted_risk = base_risk * risk_scale
    return max(0.001, adjusted_risk)
