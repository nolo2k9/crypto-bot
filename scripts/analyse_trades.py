#!/usr/bin/env python3
"""Analyse trades.csv and print a performance report."""
import sys
import math
import csv
from collections import defaultdict
from datetime import datetime


def load_trades(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def analyse(trades: list[dict]) -> None:
    if not trades:
        print("No trades found.")
        return

    pnls = []
    by_symbol = defaultdict(list)

    for t in trades:
        try:
            pnl = float(t.get("pnl", 0) or 0)
        except ValueError:
            continue
        pnls.append(pnl)
        sym = t.get("symbol", "UNKNOWN")
        by_symbol[sym].append(pnl)

    if not pnls:
        print("No PnL data found in trades.")
        return

    total      = sum(pnls)
    wins       = [p for p in pnls if p > 0]
    losses     = [p for p in pnls if p <= 0]
    win_rate   = len(wins) / len(pnls) * 100
    avg_win    = sum(wins) / len(wins) if wins else 0
    avg_loss   = sum(losses) / len(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    equity = 5000.0
    peak   = equity
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak * 100
        max_dd  = max(max_dd, dd)

    mean  = total / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / len(pnls)
    std   = math.sqrt(variance) if variance > 0 else 0
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0

    print("=" * 50)
    print("  SPIDERBOT PERFORMANCE REPORT")
    print("=" * 50)
    print(f"  Total trades    : {len(pnls)}")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Total PnL       : ${total:+.2f}")
    print(f"  Avg win         : ${avg_win:+.2f}")
    print(f"  Avg loss        : ${avg_loss:+.2f}")
    print(f"  Profit factor   : {profit_factor:.2f}")
    print(f"  Max drawdown    : {max_dd:.2f}%")
    print(f"  Sharpe ratio    : {sharpe:.2f}")
    print()
    print("  BY SYMBOL:")
    for sym, sym_pnls in sorted(by_symbol.items()):
        sym_total   = sum(sym_pnls)
        sym_wr      = len([p for p in sym_pnls if p > 0]) / len(sym_pnls) * 100
        print(f"    {sym:12s}  trades={len(sym_pnls):3d}  wr={sym_wr:5.1f}%  pnl=${sym_total:+.2f}")
    print("=" * 50)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/trades.csv"
    try:
        trades = load_trades(path)
        analyse(trades)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)
