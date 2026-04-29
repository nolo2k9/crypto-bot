import os
from typing import Dict, List, Optional
from logging.handlers import RotatingFileHandler
import logging
from prometheus_client import start_http_server
import sys

# -------- Logging --------
def setup_logging(log_file: Optional[str] = None) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Remove any handlers added by earlier basicConfig calls
    for h in root.handlers[:]:
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    port = int(os.getenv("METRICS_PORT", "8000"))
    try:
        start_http_server(port)
        logging.info(f"Prometheus metrics available on :{port}")
    except Exception as e:
        logging.warning(f"Failed to start Prometheus server: {e}")