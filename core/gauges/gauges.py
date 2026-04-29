from prometheus_client import Gauge
equity_gauge = Gauge("bot_equity_usdt", "Current portfolio equity")
pnl_gauge = Gauge("bot_realized_pnl_usdt", "Realized PnL", ["symbol"])
drawdown_gauge = Gauge("bot_max_drawdown_pct", "Maximum drawdown percentage")