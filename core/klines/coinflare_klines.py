import logging
import pandas as pd
from typing import List, Optional


def _fetch_df(client, symbol: str, interval: str, bars: int = 500) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from Binance and return a UTC-indexed DataFrame."""
    try:
        raw = client.get_klines(symbol, interval, limit=int(bars))
    except Exception as e:
        logging.warning("[KLINES] get_klines failed for %s: %s", symbol, e)
        return None

    if not raw:
        return None

    try:
        rows = []
        for r in raw:
            if isinstance(r, dict):
                ts = r.get("openTime") or r.get("t") or r.get("time")
                o, h, l, c, v = r.get("open") or r.get("o"), r.get("high") or r.get("h"), \
                                  r.get("low") or r.get("l"), r.get("close") or r.get("c"), \
                                  r.get("volume") or r.get("v")
            else:
                ts, o, h, l, c, v = r[0], r[1], r[2], r[3], r[4], r[5]
            if None in (ts, o, h, l, c, v):
                continue
            ts_int = int(ts)
            if ts_int < 10**10:
                ts_int *= 1000
            rows.append((pd.Timestamp(ts_int, unit="ms", tz="UTC"),
                         float(o), float(h), float(l), float(c), float(v)))
        if not rows:
            return None
        df = (pd.DataFrame(rows, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
              .sort_values("Time").drop_duplicates("Time").set_index("Time"))
        return df
    except Exception as e:
        logging.warning("[KLINES] Parse failed for %s: %s", symbol, e)
        return None


def _refresh_klines_now(client, data_feed, symbols: List[str], interval: str, bars: int = 500) -> None:
    for sym in symbols:
        try:
            fresh = _fetch_df(client, sym, interval, bars=bars)
            if fresh is not None and not fresh.empty:
                data_feed.data[sym] = fresh
                logging.info("[KLINES] Refreshed %s: %d rows, last=%s", sym, len(fresh), fresh.index[-1])
            else:
                logging.warning("[KLINES] Empty refresh for %s", sym)
        except Exception as e:
            logging.warning("[KLINES] Refresh failed for %s: %s", sym, e)


# Legacy alias so any old import still works
_refresh_coinflare_now = _refresh_klines_now
