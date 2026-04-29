import logging
from typing import List


def auto_select_symbols(client, mode: str, min_volume: float, **_) -> List[str]:
    """
    Pick top-gainer / top-loser / both from Binance USDT-M futures tickers.
    mode: "top_gainer" | "top_loser" | "both"
    min_volume: minimum 24h quote volume in USDT.
    """
    try:
        tickers = client.futures_ticker()
    except Exception:
        try:
            tickers = client.get_ticker()
        except Exception:
            tickers = []

    if not tickers:
        logging.warning("[SELECT] No tickers returned; defaulting to BTCUSDT")
        return ["BTCUSDT"]

    pairs = [
        t for t in tickers
        if str(t.get("symbol", "")).upper().endswith("USDT")
        and float(t.get("quoteVolume", 0) or 0) >= min_volume
    ]
    if not pairs:
        return ["BTCUSDT"]

    pairs.sort(key=lambda x: float(x.get("priceChangePercent", 0) or 0), reverse=True)

    if mode == "top_gainer":
        return [pairs[0]["symbol"]]
    if mode == "top_loser":
        return [pairs[-1]["symbol"]]
    if mode == "both":
        return [pairs[0]["symbol"], pairs[-1]["symbol"]]
    return [pairs[0]["symbol"]]