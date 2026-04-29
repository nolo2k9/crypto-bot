import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np

# -------- Time/index helpers --------
def parse_interval_seconds(interval: str) -> int:
    try:
        if interval.endswith('m'):
            return max(5, int(interval[:-1] or 1) * 60)
        if interval.endswith('h'):
            return max(60, int(interval[:-1] or 1) * 3600)
        if interval.endswith('s'):
            return max(1, int(interval[:-1]))
    except Exception:
        pass
    return 30


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone.utc)
        return df
    time_cols = ["Time", "time", "openTime", "open_time", "t", "startTime", "start_time", "closeTime", "close_time"]
    for col in time_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                mx = vals.max()
                unit = "ms" if (pd.notna(mx) and mx > 10**10) else "s"
                dt = pd.to_datetime(vals, unit=unit, utc=True)
                df = df.assign(_Time=dt).dropna(subset=["_Time"]).sort_values("_Time").set_index("_Time")
                df.index.name = None
                return df
    # Try epoch in index
    try:
        idx = pd.to_numeric(pd.Index(df.index), errors="coerce")
        if np.isfinite(idx).all():
            mx = np.nanmax(idx)
            unit = "ms" if (pd.notna(mx) and mx > 10**10) else "s"
            df.index = pd.to_datetime(idx, unit=unit, utc=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_dt_index(df)
    if df is None or df.empty:
        return pd.DataFrame()
    now = datetime.now(timezone.utc)
    start = pd.Timestamp("2018-01-01", tz="UTC")
    end = now + timedelta(minutes=10)
    mask = (df.index >= start) & (df.index <= end)
    df = df.loc[mask]
    if df.empty:
        return df
    return df[~df.index.duplicated(keep="last")].sort_index()