from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import asyncio
import logging
from dotenv import load_dotenv
import os
import redis

load_dotenv()
logging.basicConfig(level=logging.INFO)

class NewsAnalyzer:
    def __init__(self, redis_client):
        self.analyzer = SentimentIntensityAnalyzer()
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        self.redis = redis_client
        self.cache_key = 'news_sentiment'
        self.cache_ttl = 300  # 5 минут

    async def analyze_news_async(self):
        try:
            # Проверяем кэш в Redis
            cached_sentiment = self.redis.get(self.cache_key)
            if cached_sentiment:
                logging.info("Using cached sentiment from Redis")
                return float(cached_sentiment)

            # Если нет в кэше — запрашиваем новости
            loop = asyncio.get_event_loop()
            articles = await loop.run_in_executor(None, self._fetch_news)
            
            if not articles:
                logging.warning("No news articles fetched")
                return 0.0
            
            # Анализ сентимента
            sentiments = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    scores = self.analyzer.polarity_scores(text)
                    sentiments.append(scores['compound'])
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            
            # Кэшируем результат в Redis
            self.redis.setex(self.cache_key, self.cache_ttl, str(avg_sentiment))
            logging.info(f"Fetched and cached sentiment: {avg_sentiment}")
            return avg_sentiment
        except Exception as e:
            logging.error(f"Error analyzing news: {e}")
            return 0.0

    def _fetch_news(self):
        try:
            response = self.newsapi.get_everything(
                q='bitcoin OR crypto OR BTC',
                language='en',
                sort_by='publishedAt',
                page_size=10
            )
            return response.get('articles', [])
        except Exception as e:
            logging.error(f"Error fetching news: {e}")
            return []
