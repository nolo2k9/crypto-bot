import logging
import time


def exchange_healthy(client, max_skew_ms: int = 2500) -> bool:
    """Binance heartbeat: ping + optional clock-skew check."""
    try:
        getter = getattr(client, "get_server_time", None)
        if callable(getter):
            resp = getter()
            server_ms = int(resp.get("serverTime", 0)) if isinstance(resp, dict) else int(resp)
            skew = abs(server_ms - int(time.time() * 1000))
            if skew > max_skew_ms:
                logging.warning("[HEARTBEAT] Clock skew %d ms exceeds %d ms", skew, max_skew_ms)
            return True
        pinger = getattr(client, "ping", None)
        if callable(pinger):
            pinger()
        return True
    except Exception as e:
        logging.warning("[HEARTBEAT] Exchange healthcheck failed: %s", e)
        return False