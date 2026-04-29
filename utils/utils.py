import os
import pandas as pd
import logging


_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
}
_BINANCE_MAX_LIMIT = 1500


def _parse_raw_kline(r):
    if isinstance(r, dict):
        t  = r.get("t") or r.get("openTime") or r.get("open_time")
        op = r.get("o") or r.get("open")
        hi = r.get("h") or r.get("high")
        lo = r.get("l") or r.get("low")
        cl = r.get("c") or r.get("close")
        vo = r.get("v") or r.get("volume")
    else:
        t  = r[0] if len(r) > 0 else None
        op = r[1] if len(r) > 1 else None
        hi = r[2] if len(r) > 2 else None
        lo = r[3] if len(r) > 3 else None
        cl = r[4] if len(r) > 4 else None
        vo = r[5] if len(r) > 5 else None
    return t, op, hi, lo, cl, vo


def fetch_klines_series(client, symbol, interval, start_dt, end_dt, limit=None):
    """
    Paginated klines fetch (handles >1500-bar requests automatically).
    Passes start/end as keyword args to avoid positional-order mismatch.
    """
    try:
        start_ms = int(start_dt.timestamp() * 1000) if start_dt else None
        end_ms   = int(end_dt.timestamp() * 1000)   if end_dt   else None
        iv_ms    = _INTERVAL_MS.get(str(interval).lower(), 60_000)

        all_rows = []
        cursor   = start_ms

        while True:
            page_limit = _BINANCE_MAX_LIMIT
            raw = client.get_klines(
                symbol, interval,
                start_str=cursor, end_str=end_ms,
                limit=page_limit,
            )
            if not raw:
                break

            for r in raw:
                t, op, hi, lo, cl, vo = _parse_raw_kline(r)
                if None in (t, op, hi, lo, cl, vo):
                    continue
                ts = int(t)
                if end_ms and ts >= end_ms:
                    continue
                all_rows.append([pd.to_datetime(ts, unit="ms"), float(op), float(hi),
                                 float(lo), float(cl), float(vo)])

            # Binance returns at most page_limit bars; if fewer, we're done
            if len(raw) < page_limit:
                break

            # Advance cursor past the last returned bar
            last_t = _parse_raw_kline(raw[-1])[0]
            if last_t is None:
                break
            cursor = int(last_t) + iv_ms
            if end_ms and cursor >= end_ms:
                break

        if not all_rows:
            return pd.DataFrame()

        df = (pd.DataFrame(all_rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
                .set_index("Date")
                .drop_duplicates()
                .sort_index())
        return df
    except Exception as e:
        logging.warning("Error fetching klines for %s: %s", symbol, e)
        return pd.DataFrame()



def append_trade_csv(csv_path: str, record: dict) -> None:
    try:
        df = pd.DataFrame([record])
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Failed to write trade CSV: {e}")
