"""
data_feed/ws_feed.py

Binance Futures WebSocket kline feed.
Drop-in replacement for the old CoinflareRestFeed — exposes:
  self.data: Dict[str, pd.DataFrame]  (OHLCV, UTC DatetimeIndex)

Architecture
============
1. Seed each symbol with `limit` bars via Binance REST.
2. Open a single WebSocket, subscribe all symbols via JSON SUBSCRIBE message.
3. On each *closed* candle (k.x == true): append row, trim to `limit`.
4. Exponential back-off reconnect (1 → 60 s).
5. Fallback REST polling kicks in if no WS message for `fallback_sec` seconds;
   retires automatically once WebSocket recovers.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional

import pandas as pd
import websocket

log = logging.getLogger(__name__)

_WS_LIVE    = "wss://fstream.binance.com/ws"
_WS_TESTNET = "wss://stream.binancefuture.com/ws"


def _df_from_rest(client, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    try:
        raw = client.get_klines(symbol, interval, limit=limit)
    except Exception as e:
        log.warning("[WS-FEED] REST seed failed for %s: %s", symbol, e)
        return pd.DataFrame()

    if not raw:
        return pd.DataFrame()

    rows = []
    try:
        for r in raw:
            if isinstance(r, dict):
                ts = int(r.get("openTime") or r.get("t") or 0)
                o, h, l, c, v = (float(r.get(k, 0)) for k in ("open", "high", "low", "close", "volume"))
            else:
                ts = int(r[0])
                o, h, l, c, v = float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
            if ts < 10**10:
                ts *= 1000
            rows.append((pd.Timestamp(ts, unit="ms").tz_localize("UTC"), o, h, l, c, v))
    except Exception as e:
        log.warning("[WS-FEED] REST parse error for %s: %s", symbol, e)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    return df.sort_values("Time").drop_duplicates("Time").set_index("Time")


class BinanceWsFeed:
    """
    Binance USDT-M Futures real-time kline feed with REST seed + REST fallback.

    Parameters
    ----------
    client       : UnifiedBinanceClient (for REST seed / fallback).
    symbols      : e.g. ["BTCUSDT", "ETHUSDT"]
    interval     : e.g. "5m", "1m"
    limit        : max candles to keep per symbol (default 500)
    poll_sec     : REST fallback poll interval in seconds (default 2.0)
    fallback_sec : seconds without a WS update before REST fallback activates (default 20.0)
    testnet      : use Binance Futures testnet WebSocket URL
    """

    def __init__(
        self,
        client,
        symbols: List[str],
        interval: str,
        limit: int = 500,
        poll_sec: float = 2.0,
        fallback_sec: float = 20.0,
        testnet: bool = False,
    ):
        self.client = client
        self.symbols = [s.upper() for s in symbols]
        self.interval = interval
        self.limit = int(limit)
        self.poll_sec = float(poll_sec)
        self.fallback_sec = float(fallback_sec)
        self.ws_url = _WS_TESTNET if testnet else _WS_LIVE

        self.data: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in self.symbols}

        self._stop = threading.Event()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._fallback_thread: Optional[threading.Thread] = None

        self._ws_connected = False
        self._last_ws_msg: float = 0.0
        self._backoff: float = 1.0
        self._fallback_active = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._stop.clear()
        self._seed_rest(self.symbols)
        self._start_ws()
        self._start_fallback_watcher()
        log.info("[WS-FEED] Started for %d symbol(s) on %s", len(self.symbols), self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        for t in (self._ws_thread, self._fallback_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)
        log.info("[WS-FEED] Stopped.")

    # ------------------------------------------------------------------ #
    #  REST seed                                                           #
    # ------------------------------------------------------------------ #

    def _seed_rest(self, syms: List[str]) -> None:
        for sym in syms:
            if sym not in self.data or self.data[sym].empty:
                df = _df_from_rest(self.client, sym, self.interval, self.limit)
                if not df.empty:
                    self.data[sym] = df
                    log.info("[WS-FEED] Seeded %s: %d bars", sym, len(df))
                else:
                    log.warning("[WS-FEED] Empty REST seed for %s", sym)

    # ------------------------------------------------------------------ #
    #  WebSocket                                                           #
    # ------------------------------------------------------------------ #

    def _start_ws(self) -> None:
        self._ws_thread = threading.Thread(
            target=self._ws_loop, name="BinanceWsFeed-ws", daemon=True
        )
        self._ws_thread.start()

    def _ws_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                log.info("[WS-FEED] Connecting → %s", self.ws_url)
                self._ws.run_forever(ping_interval=25, ping_timeout=10)
            except Exception as e:
                log.warning("[WS-FEED] run_forever error: %s", e)

            if self._stop.is_set():
                break

            log.info("[WS-FEED] Reconnecting in %.1fs…", self._backoff)
            self._stop.wait(self._backoff)
            self._backoff = min(self._backoff * 2.0, 60.0)

    def _on_open(self, ws) -> None:
        self._ws_connected = True
        self._backoff = 1.0
        streams = [f"{s.lower()}@kline_{self.interval}" for s in self.symbols]
        msg = json.dumps({"method": "SUBSCRIBE", "params": streams, "id": 1})
        try:
            ws.send(msg)
            log.info("[WS-FEED] Subscribed: %s", streams)
        except Exception as e:
            log.warning("[WS-FEED] Subscribe failed: %s", e)

    def _on_message(self, ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return

        if not isinstance(msg, dict):
            return

        # Subscription ack
        if "result" in msg and "id" in msg:
            return

        k = msg.get("k")
        if k is None:
            return

        if not k.get("x", False):
            return  # candle not closed yet

        sym = str(k.get("s", "")).upper()
        if sym not in self.data:
            return

        try:
            ts_ms = int(k["t"])
            o, h, l, c, v = float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]), float(k["v"])
        except (KeyError, TypeError, ValueError):
            return

        self._append_candle(sym, ts_ms, o, h, l, c, v)
        self._last_ws_msg = time.monotonic()

    def _on_error(self, ws, error) -> None:
        log.warning("[WS-FEED] Error: %s", error)
        self._ws_connected = False

    def _on_close(self, ws, code, msg) -> None:
        self._ws_connected = False
        log.info("[WS-FEED] Closed (code=%s msg=%s)", code, msg)

    # ------------------------------------------------------------------ #
    #  Candle management                                                   #
    # ------------------------------------------------------------------ #

    def _append_candle(self, sym: str, ts_ms: int, o: float, h: float,
                       l: float, c: float, v: float) -> None:
        ts = pd.Timestamp(ts_ms, unit="ms").tz_localize("UTC")
        new_row = pd.DataFrame(
            [[o, h, l, c, v]],
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([ts], name="Time"),
        )
        existing = self.data.get(sym)
        if existing is None or existing.empty:
            self.data[sym] = new_row
        elif ts in existing.index:
            existing.loc[ts] = new_row.iloc[0]
            self.data[sym] = existing
        else:
            self.data[sym] = pd.concat([existing, new_row]).tail(self.limit)

    # ------------------------------------------------------------------ #
    #  Fallback watcher                                                    #
    # ------------------------------------------------------------------ #

    def _start_fallback_watcher(self) -> None:
        self._fallback_thread = threading.Thread(
            target=self._fallback_loop, name="BinanceWsFeed-fallback", daemon=True
        )
        self._fallback_thread.start()

    def _fallback_loop(self) -> None:
        _rest_stop = threading.Event()
        _rest_thr: Optional[threading.Thread] = None

        def _start_rest():
            nonlocal _rest_thr
            if _rest_thr and _rest_thr.is_alive():
                return
            _rest_stop.clear()
            _rest_thr = threading.Thread(
                target=self._rest_poll_loop, args=(_rest_stop,),
                name="BinanceWsFeed-rest", daemon=True,
            )
            _rest_thr.start()
            self._fallback_active = True
            log.warning("[WS-FEED] WS stale — falling back to REST polling")

        def _stop_rest():
            nonlocal _rest_thr
            _rest_stop.set()
            self._fallback_active = False
            if _rest_thr and _rest_thr.is_alive():
                _rest_thr.join(timeout=2.0)
            _rest_thr = None
            log.info("[WS-FEED] WS recovered — REST fallback retired")

        # Give WS a chance to connect before we start watching
        self._stop.wait(5.0)

        while not self._stop.is_set():
            stale = (time.monotonic() - self._last_ws_msg) > self.fallback_sec
            if stale and not self._fallback_active:
                _start_rest()
            elif not stale and self._fallback_active:
                _stop_rest()
            self._stop.wait(5.0)

        _rest_stop.set()

    def _rest_poll_loop(self, stop: threading.Event) -> None:
        while not stop.is_set() and not self._stop.is_set():
            for sym in list(self.symbols):
                try:
                    df = _df_from_rest(self.client, sym, self.interval, self.limit)
                    if not df.empty:
                        self.data[sym] = df
                except Exception as e:
                    log.warning("[WS-FEED][REST] %s: %s", sym, e)
            stop.wait(self.poll_sec)


# Alias — loop.py imports as CoinflareRestFeed for now; we swap it here.
CoinflareRestFeed = BinanceWsFeed
