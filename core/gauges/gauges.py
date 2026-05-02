from prometheus_client import Gauge
equity_gauge = Gauge("bot_equity_usdt", "Current portfolio equity")
pnl_gauge = Gauge("bot_realized_pnl_usdt", "Realized PnL", ["symbol"])
drawdown_gauge = Gauge("bot_max_drawdown_pct", "Maximum drawdown percentage")
open_positions_gauge = Gauge("bot_open_positions", "Number of open positions")
unrealized_pnl_gauge = Gauge("bot_unrealized_pnl_usdt", "Total unrealized PnL across all positions")
trade_count_gauge = Gauge("bot_trade_count", "Total trades entered since bot start")
adx_gauge = Gauge("bot_adx", "Current ADX value", ["symbol"])
fear_greed_gauge = Gauge("bot_fear_greed", "Fear and Greed index (0-100)")