"""
client/binance_client.py

Exports:
  UnifiedBinanceClient  — thin wrapper around python-binance for spot + USDT-M futures
  create_client(api_key, api_secret, market_type, testnet) -> UnifiedBinanceClient
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from binance.client import Client as _BinanceClient


class UnifiedBinanceClient:
    """
    Normalises spot/futures calls into a single interface the bot expects:
      - get_symbol_info(symbol)   → filters dict
      - get_klines(symbol, interval, limit, ...)
      - futures_position_information(symbol)
      - futures_create_order(**params)
      - futures_cancel_order(symbol, orderId)
      - futures_order_book(symbol, limit)
      - futures_account_balance()
      - futures_ticker()
      - set_leverage / set_margin_type / set_position_mode / get_position_mode
      - ping / get_server_time
    Everything else is proxied to the underlying python-binance Client via __getattr__.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        market_type: str = "futures",
        testnet: bool = False,
    ):
        self.market_type = (market_type or "futures").lower()
        key = api_key or os.getenv("BINANCE_KEY", "")
        sec = api_secret or os.getenv("BINANCE_SECRET", "")
        self._client = _BinanceClient(key, sec)

        if testnet:
            self._client.API_URL = "https://testnet.binance.vision/api"
            setattr(self._client, "FAPI_URL", "https://testnet.binancefuture.com/fapi")
            setattr(self._client, "FURL",     "https://testnet.binancefuture.com/fapi")

        self._futures_info_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    #  Health                                                              #
    # ------------------------------------------------------------------ #

    def ping(self) -> None:
        if self.market_type == "futures":
            self._client.futures_ping()
        else:
            self._client.ping()

    def get_server_time(self) -> Dict[str, int]:
        try:
            return self._client.get_server_time()
        except Exception:
            t = self._client.futures_time()
            return {"serverTime": int(t.get("serverTime", 0))}

    # ------------------------------------------------------------------ #
    #  Symbol info                                                         #
    # ------------------------------------------------------------------ #

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        if self.market_type == "spot":
            return self._client.get_symbol_info(symbol) or {}
        info = self._futures_exchange_info()
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol.upper():
                return s
        # Graceful default so the bot can still run with a warning
        return {"symbol": symbol, "filters": [], "pricePrecision": 2, "quantityPrecision": 3}

    @lru_cache(maxsize=1)
    def _futures_exchange_info(self) -> Dict[str, Any]:
        return self._client.futures_exchange_info() or {}

    # ------------------------------------------------------------------ #
    #  Market data                                                         #
    # ------------------------------------------------------------------ #

    def get_klines(self, symbol: str, interval: str, limit: int = 500,
                   start_str=None, end_str=None, **_) -> List:
        if self.market_type == "futures":
            return self._client.futures_klines(
                symbol=symbol, interval=interval, limit=limit,
                startTime=start_str, endTime=end_str,
            )
        return self._client.get_klines(
            symbol=symbol, interval=interval, limit=limit,
            startTime=start_str, endTime=end_str,
        )

    def futures_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        return self._client.futures_order_book(symbol=symbol, limit=limit)

    def futures_ticker(self) -> List[Dict[str, Any]]:
        rows = self._client.futures_ticker() or []
        return [
            {"symbol": r.get("symbol"), "priceChangePercent": r.get("priceChangePercent", "0"),
             "quoteVolume": r.get("quoteVolume", "0")}
            for r in rows
        ]

    def get_ticker(self) -> List[Dict[str, Any]]:
        return self._client.get_ticker() or []

    # ------------------------------------------------------------------ #
    #  Account / balance                                                   #
    # ------------------------------------------------------------------ #

    def futures_account_balance(self) -> List[Dict[str, Any]]:
        return self._client.futures_account_balance()

    def futures_account(self) -> Dict[str, Any]:
        return self._client.futures_account()

    # ------------------------------------------------------------------ #
    #  Positions                                                           #
    # ------------------------------------------------------------------ #

    def futures_position_information(self, symbol: str = "") -> List[Dict[str, Any]]:
        kwargs = {"symbol": symbol} if symbol else {}
        return self._client.futures_position_information(**kwargs) or []

    # Alias so position-helper code that calls .positions() still works
    def positions(self, symbol: str = "", **_) -> List[Dict[str, Any]]:
        return self.futures_position_information(symbol) if symbol else \
               self._client.futures_position_information() or []

    # ------------------------------------------------------------------ #
    #  Orders                                                              #
    # ------------------------------------------------------------------ #

    def futures_create_order(self, **params) -> Dict[str, Any]:
        return self._client.futures_create_order(**params)

    def futures_cancel_order(self, symbol: str, orderId) -> Dict[str, Any]:
        return self._client.futures_cancel_order(symbol=symbol, orderId=orderId)

    def create_order(self, **params) -> Dict[str, Any]:
        return self._client.create_order(**params)

    # ------------------------------------------------------------------ #
    #  Account configuration                                               #
    # ------------------------------------------------------------------ #

    def set_leverage(self, symbol: str, leverage: int):
        return self._client.futures_change_leverage(symbol=symbol, leverage=int(leverage))

    def set_margin_type(self, symbol: str, margin_type: str):
        try:
            return self._client.futures_change_margin_type(
                symbol=symbol, marginType=margin_type.upper()
            )
        except Exception as e:
            # Binance raises if margin type is already set — ignore
            if "No need to change" in str(e) or "-4046" in str(e):
                return {"msg": "already set"}
            raise

    def set_position_mode(self, one_way: bool = True):
        try:
            return self._client.futures_change_position_mode(dualSidePosition=not one_way)
        except Exception as e:
            if "No need to change" in str(e) or "-4059" in str(e):
                return {"msg": "already set"}
            raise

    def get_position_mode(self) -> str:
        try:
            r = self._client.futures_get_position_mode()
            return "ONE_WAY" if not r.get("dualSidePosition", False) else "HEDGE"
        except Exception:
            return "UNKNOWN"

    # ------------------------------------------------------------------ #
    #  Proxy everything else                                               #
    # ------------------------------------------------------------------ #

    def __getattr__(self, name: str):
        return getattr(self._client, name)


def create_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    market_type: str = "futures",
    testnet: bool = False,
) -> UnifiedBinanceClient:
    return UnifiedBinanceClient(api_key, api_secret, market_type=market_type, testnet=testnet)


__all__ = ["UnifiedBinanceClient", "create_client"]
