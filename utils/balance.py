import logging


def get_free_usdt(client, market_type: str = "futures") -> float:
    """Return available USDT balance from a Binance client."""
    if market_type == "futures":
        try:
            balances = client.futures_account_balance()
            if isinstance(balances, list):
                for b in balances:
                    if str(b.get("asset", "")).upper() == "USDT":
                        for k in ("availableBalance", "withdrawAvailable", "crossWalletBalance", "balance"):
                            if b.get(k) is not None:
                                return float(b[k])
        except Exception as e:
            logging.debug("[BAL] futures_account_balance failed: %s", e)

        try:
            acct = client.futures_account()
            if isinstance(acct, dict):
                v = acct.get("availableBalance") or acct.get("totalAvailableBalance")
                if v is not None:
                    return float(v)
                for a in acct.get("assets", []):
                    if str(a.get("asset", "")).upper() == "USDT":
                        for k in ("availableBalance", "walletBalance"):
                            if a.get(k) is not None:
                                return float(a[k])
        except Exception as e:
            logging.debug("[BAL] futures_account failed: %s", e)

    # Spot fallback
    try:
        bal = client.get_asset_balance(asset="USDT") or {}
        v = bal.get("free") or bal.get("available")
        if v is not None:
            return float(v)
    except Exception:
        pass

    return 0.0