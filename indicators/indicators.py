import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple


def indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    adx_period: int = 14,
    atr_period: int = 14,
    bb_window: int = 20,
    bb_std: float = 2.0,
    stoch_k: int = 14,
    stoch_smooth_k: int = 3,
    stoch_smooth_d: int = 3,
    vwap: bool = True,
    volume_osc: Tuple[int, int] = (5, 10),
) -> pd.DataFrame:
    """
    Compute standard indicators plus regime/trend filters.
    Output columns:
      ATR, ATR_pct, RSI, ADX, MACD, MACD_SIGNAL, MACD_HIST,
      BB_UPPER, BB_MID, BB_LOWER, BB_PCT, VWAP,
      EMA_21, EMA_50, EMA_200,
      Stoch_RSI, Stoch_RSI_K, Stoch_RSI_D,
      Vol_Osc, Volume_MA
    """
    out = df.copy()
    high = out["High"]
    low = out["Low"]
    close = out["Close"]
    prev_close = close.shift()

    # --- ATR ---
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(window=atr_period, min_periods=1).mean()
    out["ATR_pct"] = (out["ATR"] / close.replace(0, np.nan)) * 100.0

    # --- RSI ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # --- ADX ---
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    trN = tr.rolling(adx_period).sum()
    plus_di = 100 * pd.Series(plus_dm, index=out.index).rolling(adx_period).sum() / trN
    minus_di = 100 * pd.Series(minus_dm, index=out.index).rolling(adx_period).sum() / trN
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    out["ADX"] = dx.rolling(adx_period).mean()

    # --- MACD ---
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = exp1 - exp2
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    # --- Bollinger Bands ---
    ma = close.rolling(window=bb_window).mean()
    std = close.rolling(window=bb_window).std()
    out["BB_MID"] = ma
    out["BB_UPPER"] = ma + bb_std * std
    out["BB_LOWER"] = ma - bb_std * std
    band_width = (out["BB_UPPER"] - out["BB_LOWER"]).replace(0, np.nan)
    out["BB_PCT"] = ((close - out["BB_LOWER"]) / band_width).clip(0.0, 1.0)

    # --- EMAs (regime / trend filters) ---
    out["EMA_21"] = close.ewm(span=21, adjust=False).mean()
    out["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA_200"] = close.ewm(span=200, adjust=False).mean()

    # --- VWAP — resets daily at UTC midnight using typical price ---
    if vwap:
        tp = (out["High"] + out["Low"] + out["Close"]) / 3.0
        try:
            day = out.index.normalize() if hasattr(out.index, "normalize") else pd.to_datetime(out.index).normalize()
            out["VWAP"] = (
                (tp * out["Volume"]).groupby(day).cumsum()
                / out["Volume"].groupby(day).cumsum()
            )
        except Exception:
            out["VWAP"] = (tp * out["Volume"]).cumsum() / out["Volume"].cumsum()
    else:
        out["VWAP"] = np.nan

    # --- Stochastic RSI ---
    rsi = out["RSI"]
    rsi_low = rsi.rolling(stoch_k).min()
    rsi_high = rsi.rolling(stoch_k).max()
    stoch_rsi = 100 * (rsi - rsi_low) / (rsi_high - rsi_low + 1e-9)
    out["Stoch_RSI"] = stoch_rsi
    out["Stoch_RSI_K"] = stoch_rsi.rolling(stoch_smooth_k).mean()
    out["Stoch_RSI_D"] = out["Stoch_RSI_K"].rolling(stoch_smooth_d).mean()

    # --- Volume ---
    short_n, long_n = volume_osc
    short_vol = out["Volume"].ewm(span=short_n, adjust=False).mean()
    long_vol = out["Volume"].ewm(span=long_n, adjust=False).mean()
    out["Vol_Osc"] = 100 * (short_vol - long_vol) / long_vol.replace(0, np.nan)
    out["Volume_MA"] = out["Volume"].rolling(20, min_periods=1).mean()

    # --- Rate of Change (for momentum strategy) ---
    out["ROC_10"] = close.pct_change(periods=10) * 100.0

    # --- Rolling high/low (for breakout strategy) ---
    out["rolling_high_20"] = high.shift(1).rolling(20, min_periods=5).max()
    out["rolling_low_20"]  = low.shift(1).rolling(20, min_periods=5).min()

    # --- Fill critical cols ---
    critical_cols = [
        "ATR", "ATR_pct", "RSI", "ADX", "MACD", "MACD_SIGNAL", "MACD_HIST",
        "BB_UPPER", "BB_MID", "BB_LOWER", "BB_PCT",
        "EMA_21", "EMA_50", "EMA_200",
        "VWAP",
        "Stoch_RSI", "Stoch_RSI_K", "Stoch_RSI_D",
        "Vol_Osc", "Volume_MA",
        "ROC_10", "rolling_high_20", "rolling_low_20",
    ]
    out[critical_cols] = out[critical_cols].ffill().bfill()
    total_nans = int(out[critical_cols].isna().sum().sum())
    if total_nans:
        logging.debug("Indicators: filled %d NaN values after warm-up", total_nans)

    return out


def simple_signal(row, adx_threshold: float = 20.0) -> int:
    """
    Multi-factor signal with EMA regime filter.
    Returns +1 (long), -1 (short), 0 (flat).
    """
    close = float(row.get("Close", 0) or 0)
    if close <= 0:
        return 0

    rsi = float(row.get("RSI", 50) or 50)
    adx = float(row.get("ADX", 0) or 0)
    macd_hist = float(row.get("MACD_HIST", 0) or 0)
    macd = float(row.get("MACD", 0) or 0)
    macd_sig = float(row.get("MACD_SIGNAL", 0) or 0)
    ema21 = float(row.get("EMA_21", close) or close)
    ema50 = float(row.get("EMA_50", close) or close)
    ema200_raw = row.get("EMA_200")
    stoch_k = float(row.get("Stoch_RSI_K", 50) or 50)
    stoch_d = float(row.get("Stoch_RSI_D", 50) or 50)
    bb_pct = float(row.get("BB_PCT", 0.5) or 0.5)
    vwap = float(row.get("VWAP", close) or close)
    vol = float(row.get("Volume", 0) or 0)
    vol_ma = float(row.get("Volume_MA", 0) or 0)
    vol_osc = float(row.get("Vol_Osc", 0) or 0)

    # Regime gate: require price on the right side of EMA_200
    long_ok = True
    short_ok = True
    if ema200_raw is not None and not (isinstance(ema200_raw, float) and np.isnan(ema200_raw)):
        ema200 = float(ema200_raw)
        if ema200 > 0:
            long_ok = close > ema200 * 0.995   # price must be near or above EMA_200 for longs
            short_ok = close < ema200 * 1.005  # price must be near or below EMA_200 for shorts

    # Trend alignment: fast EMA vs slow EMA
    trend_up = ema21 > ema50
    trend_dn = ema21 < ema50

    # Trend strength gate
    if adx < adx_threshold:
        return 0

    # Volume confirmation (only skip if volume MA has data)
    vol_ok = (vol_ma <= 0) or (vol > vol_ma * 0.8)

    bullish = (
        long_ok
        and trend_up
        and macd_hist > 0
        and macd > macd_sig
        and 40 < rsi < 72
        and stoch_k > stoch_d
        and stoch_k < 85
        and close > vwap * 0.997
        and bb_pct > 0.3
        and (vol_osc > -10 or vol_ok)
    )

    bearish = (
        short_ok
        and trend_dn
        and macd_hist < 0
        and macd < macd_sig
        and 28 < rsi < 60
        and stoch_k < stoch_d
        and stoch_k > 15
        and close < vwap * 1.003
        and bb_pct < 0.7
        and (vol_osc > -10 or vol_ok)
    )

    if bullish and not bearish:
        return 1
    if bearish and not bullish:
        return -1
    import logging as _log
    _log.info("[TREND-DEBUG] sig=0 long_ok=%s short_ok=%s trend_up=%s trend_dn=%s "
              "macd_hist=%.2f rsi=%.1f stoch_k=%.1f bb_pct=%.2f close_vs_vwap=%.4f",
              long_ok, short_ok, trend_up, trend_dn,
              macd_hist, rsi, stoch_k, bb_pct, (close / vwap - 1) if vwap else 0)
    return 0


def mean_reversion_signal(row, adx_threshold: float = 30.0) -> int:
    """
    Mean reversion: buy oversold bounces, sell overbought exhaustion.
    Best in ranging/choppy markets (ADX below threshold).
    Returns +1 (long), -1 (short), 0 (flat).
    """
    close = float(row.get("Close", 0) or 0)
    if close <= 0:
        return 0

    rsi        = float(row.get("RSI", 50) or 50)
    adx        = float(row.get("ADX", 0) or 0)
    bb_pct     = float(row.get("BB_PCT", 0.5) or 0.5)
    stoch_k    = float(row.get("Stoch_RSI_K", 50) or 50)
    stoch_d    = float(row.get("Stoch_RSI_D", 50) or 50)
    ema200_raw = row.get("EMA_200")

    if adx > adx_threshold:
        return 0

    long_ok = short_ok = True
    if ema200_raw is not None and not (isinstance(ema200_raw, float) and np.isnan(ema200_raw)):
        ema200 = float(ema200_raw)
        if ema200 > 0:
            long_ok  = close > ema200 * 0.90
            short_ok = close < ema200 * 1.10

    bullish = (
        long_ok
        and rsi < 35
        and bb_pct < 0.15
        and stoch_k < 25
        and stoch_k > stoch_d
    )
    bearish = (
        short_ok
        and rsi > 65
        and bb_pct > 0.85
        and stoch_k > 75
        and stoch_k < stoch_d
    )

    if bullish and not bearish:
        return 1
    if bearish and not bullish:
        return -1
    return 0


def breakout_signal(row) -> int:
    """
    Breakout: enter when price closes beyond recent 20-bar high/low with volume confirmation.
    Returns +1 (long), -1 (short), 0 (flat).
    """
    close = float(row.get("Close", 0) or 0)
    if close <= 0:
        return 0

    rolling_high = float(row.get("rolling_high_20", 0) or 0)
    rolling_low  = float(row.get("rolling_low_20", float("inf")) or float("inf"))
    vol          = float(row.get("Volume", 0) or 0)
    vol_ma       = float(row.get("Volume_MA", 0) or 0)
    adx          = float(row.get("ADX", 0) or 0)
    rsi          = float(row.get("RSI", 50) or 50)
    ema200_raw   = row.get("EMA_200")

    if adx < 15:
        return 0

    vol_spike = vol_ma <= 0 or vol > vol_ma * 1.2

    long_ok = short_ok = True
    if ema200_raw is not None and not (isinstance(ema200_raw, float) and np.isnan(ema200_raw)):
        ema200 = float(ema200_raw)
        if ema200 > 0:
            long_ok  = close > ema200 * 0.99
            short_ok = close < ema200 * 1.01

    bullish = (
        long_ok
        and rolling_high > 0
        and close > rolling_high * 1.001
        and vol_spike
        and rsi < 80
    )
    bearish = (
        short_ok
        and rolling_low < float("inf")
        and close < rolling_low * 0.999
        and vol_spike
        and rsi > 20
    )

    if bullish and not bearish:
        return 1
    if bearish and not bullish:
        return -1
    return 0


def momentum_signal(row) -> int:
    """
    Momentum: enter in direction of strong price momentum with trend confirmation.
    Uses 10-bar rate of change + EMA alignment + MACD.
    Returns +1 (long), -1 (short), 0 (flat).
    """
    close = float(row.get("Close", 0) or 0)
    if close <= 0:
        return 0

    roc        = float(row.get("ROC_10", 0) or 0)
    adx        = float(row.get("ADX", 0) or 0)
    rsi        = float(row.get("RSI", 50) or 50)
    ema21      = float(row.get("EMA_21", close) or close)
    ema50      = float(row.get("EMA_50", close) or close)
    macd_hist  = float(row.get("MACD_HIST", 0) or 0)
    vol        = float(row.get("Volume", 0) or 0)
    vol_ma     = float(row.get("Volume_MA", 0) or 0)
    ema200_raw = row.get("EMA_200")

    if adx < 20:
        return 0

    vol_ok = vol_ma <= 0 or vol > vol_ma * 0.9

    long_ok = short_ok = True
    if ema200_raw is not None and not (isinstance(ema200_raw, float) and np.isnan(ema200_raw)):
        ema200 = float(ema200_raw)
        if ema200 > 0:
            long_ok  = close > ema200 * 0.995
            short_ok = close < ema200 * 1.005

    trend_up = ema21 > ema50
    trend_dn = ema21 < ema50

    bullish = (
        long_ok and trend_up
        and roc > 2.0
        and 50 < rsi < 75
        and macd_hist > 0
        and vol_ok
    )
    bearish = (
        short_ok and trend_dn
        and roc < -2.0
        and 25 < rsi < 50
        and macd_hist < 0
        and vol_ok
    )

    if bullish and not bearish:
        return 1
    if bearish and not bullish:
        return -1
    return 0


def get_signal(row, strategy: str = "trend", adx_threshold: float = 20.0) -> int:
    """Dispatch to the correct signal function by strategy name."""
    if strategy == "trend":
        return simple_signal(row, adx_threshold=adx_threshold)
    if strategy == "mean_reversion":
        return mean_reversion_signal(row, adx_threshold=adx_threshold)
    if strategy == "breakout":
        return breakout_signal(row)
    if strategy == "momentum":
        return momentum_signal(row)
    return 0