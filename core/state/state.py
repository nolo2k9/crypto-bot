import os
import logging
import json
import math
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

def _safe_ratio(num, den, default=0.0):
    try:
        den = float(den)
        return float(num)/den if den != 0.0 else default
    except Exception:
        return default

def _clean_num(x, default=None) -> Optional[float]:
    """Coerce to float; return default for None/NaN/Inf/invalid."""
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default

def _clean_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def _iso(dt) -> Optional[str]:
    """Safe ISO timestamp (or None)."""
    try:
        if dt is None:
            return None
        if hasattr(dt, "isoformat"):
            return dt.isoformat()
        # fallback for strings or other reprs
        return str(dt)
    except Exception:
        return None

def _bracket_ids(br) -> Optional[Dict[str, Any]]:
    """
    Reduce bracket structure to JSON-safe identifiers only.
    Accepts various shapes (dicts/lists) and extracts likely order IDs.
    """
    try:
        if br is None:
            return None
        if isinstance(br, dict):
            out = {}
            # common keys we might want to persist
            for k in ("sl_order_id", "tp_order_id", "entry_order_id", "reduce_only_ids"):
                if k in br:
                    out[k] = br[k]
            # sometimes "orders": [{"id":...,"type":"TP"}, ...]
            if "orders" in br and isinstance(br["orders"], list):
                out["orders"] = []
                for o in br["orders"]:
                    if isinstance(o, dict):
                        out["orders"].append({
                            "id": o.get("id") or o.get("orderId") or o.get("clientOrderId"),
                            "kind": o.get("kind") or o.get("type"),
                            "side": o.get("side"),
                        })
            return out or None
        if isinstance(br, list):
            # assume list of order dicts
            slim = []
            for o in br:
                if isinstance(o, dict):
                    slim.append({
                        "id": o.get("id") or o.get("orderId") or o.get("clientOrderId"),
                        "kind": o.get("kind") or o.get("type"),
                        "side": o.get("side"),
                    })
            return {"orders": slim} if slim else None
        # unknown shape → stringify
        return {"repr": str(br)}
    except Exception:
        return None

def _ensure_parent_dir(path: str) -> None:
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        # non-fatal; we'll attempt to write anyway
        pass

# -------- State I/O --------
def save_runtime_state(path: str, state: Dict[str, dict], *,
                       symbols: List[str], interval: str, market_type: str,
                       meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Persist a JSON-safe snapshot of runtime state. Never raises.
    - Sanitizes NaN/Inf/NumPy scalars.
    - Ignores per-symbol serialization errors instead of failing whole save.
    - Stores only lightweight bracket identifiers.
    - Atomic write via .tmp + os.replace.
    """
    try:
        _ensure_parent_dir(path)

        out: Dict[str, Any] = {
            "version": 2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": list(symbols or []),
            "interval": str(interval),
            "market_type": str(market_type),
            "meta": meta or {},
            "state": {}
        }

        for sym, s in (state or {}).items():
            try:
                # base fields with sanitization
                pos      = _clean_int(s.get("position", 0), 0)
                qty      = _clean_num(s.get("qty", 0.0), 0.0)
                ep       = _clean_num(s.get("entry_price"), None)
                etime    = _iso(s.get("entry_time"))
                stop     = _clean_num(s.get("stop"), None)
                take     = _clean_num(s.get("take"), None)
                realized = _clean_num(s.get("realized_pnl", 0.0), 0.0)
                unreal   = _clean_num(s.get("unrealized_pnl", 0.0), 0.0)
                consec   = _clean_int(s.get("consecutive_losses", 0), 0)
                stale    = _clean_int(s.get("stale_count", 0), 0)
                last_ts  = _iso(s.get("last_bar_ts"))
                cool     = _clean_int(s.get("cooldown_bars_left", 0), 0)
                disabled = bool(s.get("disabled", False))
                peak_eq  = _clean_num(s.get("peak_equity", 0.0), 0.0)
                small_ct = _clean_int(s.get("small_qty_count", 0), 0)
                scaled   = bool(s.get("scaled_out", False))
                brackets = _bracket_ids(s.get("brackets"))

                out["state"][str(sym)] = {
                    "position": pos,
                    "qty": qty,
                    "entry_price": ep,
                    "entry_time": etime,
                    "stop": stop,
                    "take": take,
                    "realized_pnl": realized,
                    "unrealized_pnl": unreal,
                    "consecutive_losses": consec,
                    "stale_count": stale,
                    "last_bar_ts": last_ts,
                    "cooldown_bars_left": cool,
                    "disabled": disabled,
                    "peak_equity": peak_eq,
                    "small_qty_count": small_ct,
                    "scaled_out": scaled,
                    "brackets": brackets,  # slim identifiers only
                }
            except Exception as e_sym:
                logging.warning(f"[STATE] Skipped symbol '{sym}' during save: {e_sym}")

        tmp = f"{path}.tmp"
        # ensure_ascii=False to preserve any unicode in meta, allow_nan=False to
        # catch any missed NaN/Inf (shouldn't happen due to _clean_num).
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
        logging.info(f"[STATE] Runtime state saved to {path}")
    except Exception as e:
        logging.warning(f"[STATE] Failed to save runtime state to {path}: {e}")

def load_runtime_state(path: str) -> Optional[dict]:
    """Load state JSON; tolerant to missing file and minor schema drift."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        # light schema normalization
        if "version" not in doc:
            doc["version"] = 1
        if "state" not in doc or not isinstance(doc["state"], dict):
            doc["state"] = {}
        return doc
    except Exception as e:
        logging.warning(f"[STATE] Failed to read runtime state from {path}: {e}")
        return None
