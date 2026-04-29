from dotenv import load_dotenv
import logging
import os
import requests
from binance.client import Client
from textblob import TextBlob
from datetime import datetime, timedelta
from pprint import pprint

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))

log = logging.getLogger(__name__)


class SentimentTuner:
    class SantimentClient:
        BASE_URL = "https://api.santiment.net/graphql"

        def __init__(self, api_key=None):
            self.api_key = api_key or os.getenv('SANTIMENT_API_KEY')
            if not self.api_key:
                raise ValueError("Santiment API key is missing in environment variables.")

        def _query(self, query, variables=None):
            headers = {"Authorization": f"Apikey {self.api_key}"}
            payload = {"query": query}
            if variables:
                payload["variables"] = variables
            response = requests.post(self.BASE_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

        def get_social_sentiment(self, slug, from_date, to_date):
            query = """
            query Sentiment($slug: String!, $from: DateTime!, $to: DateTime!) {
              getMetric(metric: "social_sentiment_net_sentiment") {
                timeseriesData(
                  slug: $slug,
                  from: $from,
                  to: $to,
                  interval: "1d"
                ) {
                  datetime
                  value
                }
              }
            }
            """
            variables = {
                "slug": slug,
                "from": from_date,
                "to": to_date
            }
            result = self._query(query, variables)
            timeseries = result.get('data', {}).get('getMetric', {}).get('timeseriesData', [])
            return timeseries

    def __init__(self):
        binance_key = os.getenv("BINANCE_KEY")

        binance_secret = os.getenv("BINANCE_SECRET")
        self.binance_client = Client(binance_key, binance_secret) if binance_key and binance_secret else None

        try:
            self.santiment_client = self.SantimentClient()
        except ValueError as e:
            log.warning("[SentimentTuner] %s", e)
            self.santiment_client = None

        self.cryptopanic_key = os.getenv('CRYPTOPANIC_API_KEY')

    def get_fear_greed_index(self) -> int:
        try:
            resp = requests.get("https://api.alternative.me/fng/", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return int(data['data'][0]['value'])
        except Exception as e:
            log.warning("[SentimentTuner] Error fetching Fear & Greed: %s", e)
            return -1

    def recommend_params_from_fg(self, fg: int):
        if fg < 30:
            return dict(risk=0.005, sl_mult=1.5, tp_mult=2.0)
        elif fg > 70:
            return dict(risk=0.02, sl_mult=2.5, tp_mult=3.5)
        else:
            return dict(risk=0.01, sl_mult=2.0, tp_mult=3.0)

    def fetch_cryptopanic_news(self, max_items=20):
        if not self.cryptopanic_key:
            log.warning("[SentimentTuner] No CryptoPanic API key set.")
            return []
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_key}&kind=news&public=true&region=us&page_size={max_items}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            posts = data.get('results', [])
            if not posts:
                log.debug("[SentimentTuner] No posts found in CryptoPanic response")
            return [f"{p['title']} - {p.get('domain', 'unknown')}" for p in posts]
        except Exception as e:
            log.warning("[SentimentTuner] Error fetching CryptoPanic news: %s", e)
            return []

    def fetch_news_sentiment(self, max_items=20) -> float:
        """TextBlob sentiment of CryptoPanic headlines (-1 to +1)."""
        headlines = self.fetch_cryptopanic_news(max_items)
        if not headlines:
            return 0.0
        try:
            scores = [TextBlob(h).sentiment.polarity for h in headlines]
            return sum(scores) / len(scores)
        except Exception as e:
            log.warning("[SentimentTuner] TextBlob scoring error: %s", e)
            return 0.0

    def fetch_coingecko_data(self):
        url = ("https://api.coingecko.com/api/v3/coins/markets"
               "?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false&price_change_percentage=24h")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            coins = []
            for d in data:
                community_score = d.get('community_score')
                sentiment_up = d.get('sentiment_votes_up_percentage')
                social_score = 0.0
                if community_score is not None:
                    social_score = community_score
                elif sentiment_up is not None:
                    social_score = sentiment_up / 100.0
                coins.append({
                    'symbol': d['symbol'].upper() + "USDT",
                    'volume': d.get('total_volume', 0),
                    'price_change': abs(d.get('price_change_percentage_24h', 0)),
                    'social_score': social_score
                })
            return coins
        except Exception as e:
            log.warning("[SentimentTuner] Error fetching CoinGecko data: %s", e)
            return []

    def get_santiment_sentiment(self, slug='bitcoin') -> float:
        if not self.santiment_client:
            log.debug("[SentimentTuner] Santiment client not initialized.")
            return 0.0
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=7)
        try:
            data = self.santiment_client.get_social_sentiment(slug, from_date.isoformat(), to_date.isoformat())
            if not data:
                return 0.0
            avg_sentiment = sum(item.get('value', 0) for item in data) / len(data)
            return avg_sentiment
        except Exception as e:
            log.warning("[SentimentTuner] Santiment API error: %s", e)
            return 0.0

    def analyze_trending_symbols(self, news_list, news_sentiment, coingecko_coins):
        if not self.binance_client:
            log.warning("[SentimentTuner] Binance client not initialized; cannot fetch tickers.")
            return []

        common_symbols = [coin['symbol'] for coin in coingecko_coins]
        mention_count = {sym: 0 for sym in common_symbols}

        for headline in news_list:
            upper = headline.upper()
            for sym in common_symbols:
                base_sym = sym.replace("USDT", "")
                if base_sym in upper:
                    mention_count[sym] += 1

        combined_scores = []
        for coin in coingecko_coins:
            sym = coin['symbol']
            try:
                ticker = self.binance_client.get_ticker(symbol=sym)
                vol = float(ticker.get('volume', 0))
                change = coin['price_change']
                social_score = coin['social_score']
                social_score = social_score if social_score > 0 else 0.1
                adjusted_sentiment = news_sentiment if news_sentiment > 0 else 0.1
                mentions = mention_count.get(sym, 0)
                score = vol * change * (mentions + 1) * (social_score + 1) * (adjusted_sentiment + 1)
                combined_scores.append((sym, score))
            except Exception as e:
                log.warning("[SentimentTuner] Binance API error for %s: %s", sym, e)

        filtered = [x for x in combined_scores if x[1] > 0]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in filtered[:5]]

    def daily_recommendations(self):
        fg = self.get_fear_greed_index()
        params = self.recommend_params_from_fg(fg) if fg >= 0 else None

        santiment_sentiment = self.get_santiment_sentiment('bitcoin')
        news_list = self.fetch_cryptopanic_news() if self.cryptopanic_key else []
        news_sentiment = self.fetch_news_sentiment() if news_list else 0.0
        coingecko_coins = self.fetch_coingecko_data()

        recommended_symbols = []
        if news_list and self.binance_client:
            recommended_symbols = self.analyze_trending_symbols(news_list, news_sentiment, coingecko_coins)

        return {
            'fear_greed_index': fg,
            'recommended_params': params,
            'top_news': news_list,
            'twitter_sentiment': news_sentiment,   # kept key for CLI compat
            'santiment_sentiment': santiment_sentiment,
            'coingecko_coins': coingecko_coins,
            'trending_symbols': recommended_symbols
        }


if __name__ == "__main__":
    tuner = SentimentTuner()
    recs = tuner.daily_recommendations()
    pprint(recs)
