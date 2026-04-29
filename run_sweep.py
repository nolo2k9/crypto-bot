import subprocess
import csv
import json
import re
from itertools import product
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- parameter ranges ---
risk_values = [0.01, 0.05, 0.1, 0.2]
sl_mult_values = [1.5, 2.0, 2.5, 3.0]
tp_mult_values = [2.0, 2.5, 3.0, 4.0, 5.0]
rsi_periods = [10, 14, 20, 30]
adx_thresholds = [20, 25, 30, 35]
volume_filter_enabled = [True, False]
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
intervals = ["1h", "4h"]

base_cmd = [
    "python", "spider_bot.py",
    "--mode", "backtest",
    "--days", "365",
    "--fee", "0.001",
    "--market-type", "spot",
    "--use-testnet"
]

csv_file = "backtest_sweep_results.csv"
top_csv_file = "top_strategies.csv"

header = [
    "symbol", "interval", "risk", "sl_mult", "tp_mult", "rsi_period", "adx_threshold", "volume_filter",
    "trades", "wins", "losses", "win_rate", "gross_pnl", "fees",
    "realized_pnl", "start_balance", "end_balance", "return_%", "max_drawdown_%"
]

def parse_summary(output: str):
    """Extracts the backtest summary JSON block from bot output."""
    try:
        matches = re.findall(r'({.*?"event"\s*:\s*"backtest_summary".*?})', output, flags=re.DOTALL)
        for m in matches:
            try:
                summary = json.loads(m)
                if summary.get("event") == "backtest_summary":
                    return summary
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"Parsing error: {e}")
    return None

def run_backtest(params):
    symbol, interval, risk, sl_mult, tp_mult, rsi_period, adx_threshold, volume_filter = params
    vol_filter_str = "true" if volume_filter else "false"
    cmd = base_cmd + [
        "--symbols", symbol,
        "--interval", interval,
        "--risk", str(risk),
        "--sl-mult", str(sl_mult),
        "--tp-mult", str(tp_mult),
        "--rsi-period", str(rsi_period),
        "--adx-threshold", str(adx_threshold),
        "--volume-filter", vol_filter_str,
    ]

    print(f"Running backtest: {symbol} {interval} | risk={risk}, sl={sl_mult}, tp={tp_mult}, "
          f"rsi={rsi_period}, adx={adx_threshold}, vol_filter={volume_filter}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    summary = parse_summary(result.stdout) or parse_summary(result.stderr)
    if summary:
        return {
            "symbol": symbol,
            "interval": interval,
            "risk": risk,
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "rsi_period": rsi_period,
            "adx_threshold": adx_threshold,
            "volume_filter": volume_filter,
            "trades": summary.get("trades"),
            "wins": summary.get("wins"),
            "losses": summary.get("losses"),
            "win_rate": summary.get("win_rate"),
            "gross_pnl": summary.get("gross_pnl"),
            "fees": summary.get("fees"),
            "realized_pnl": summary.get("realized_pnl"),
            "start_balance": summary.get("start_balance"),
            "end_balance": summary.get("end_balance"),
            "return_%": summary.get("return_%"),
            "max_drawdown_%": summary.get("max_drawdown_%"),
        }
    else:
        print("⚠️ No summary found.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

if __name__ == "__main__":
    all_params = list(product(
        symbols, intervals, risk_values, sl_mult_values, tp_mult_values,
        rsi_periods, adx_thresholds, volume_filter_enabled
    ))

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run_backtest, p): p for p in all_params}
            for future in as_completed(futures):
                row = future.result()
                if row:
                    writer.writerow(row)
                    print("✅ Results saved.")

    # === Ranking phase ===
    print("\n📊 Ranking Top Strategies...")

    df = pd.read_csv(csv_file)

    numeric_cols = ["return_%", "win_rate", "max_drawdown_%", "trades"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["trades"] >= 30]
    df["score"] = (df["return_%"] * 0.6) + (df["win_rate"] * 0.3) - (df["max_drawdown_%"] * 0.1)

    top_strategies = df.sort_values("score", ascending=False).head(10)

    print("\n🏆 Top 10 Strategies:")
    print(top_strategies[[
        "symbol", "interval", "risk", "sl_mult", "tp_mult", "rsi_period",
        "adx_threshold", "volume_filter", "return_%", "win_rate", "max_drawdown_%", "trades", "score"
    ]])

    top_strategies.to_csv(top_csv_file, index=False)
    print(f"\n💾 Top 10 strategies saved to: {top_csv_file}")
