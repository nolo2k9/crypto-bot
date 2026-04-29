FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Runtime output lands in /data (mounted as a volume)
ENV LOG_FILE=/data/trader.log \
    TRADE_LOG_CSV=/data/trades.csv \
    STATE_FILE=/data/bot_state.json

# Prometheus metrics
EXPOSE 8000

# Non-root user
RUN useradd -m botuser && mkdir -p /data && chown botuser /data
USER botuser

ENTRYPOINT ["python", "spider_bot.py"]
CMD ["--mode", "paper", "--symbols", "BTCUSDT", "ETHUSDT", "SOLUSDT", \
     "--interval", "4h", "--leverage", "10", "--risk", "0.01", \
     "--tp-mult", "2.5", "--sl-mult", "1.5", "--adx-threshold", "25", \
     "--fee", "0.0005"]