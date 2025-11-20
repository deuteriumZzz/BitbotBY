from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import asyncio
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)

class NewsAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

    async def analyze_news_async(self):
        try:
            # Асинхронный запрос новостей о крипте (BTC)
            loop = asyncio.get_event_loop()
            articles = await loop.run_in_executor(None, self._fetch_news)
            
            if not articles:
                logging.warning("No news articles fetched")
                return 0.0
            
            # Анализ сентимента: усредняем compound score по заголовкам и описаниям
            sentiments = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    scores = self.analyzer.polarity_scores(text)
                    sentiments.append(scores['compound'])
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            logging.info(f"Analyzed {len(articles)} articles, average sentiment: {avg_sentiment}")
            return avg_sentiment
        except Exception as e:
            logging.error(f"Error analyzing news: {e}")
            return 0.0

    def _fetch_news(self):
        # Синхронный запрос к NewsAPI (запускается в executor)
        try:
            response = self.newsapi.get_everything(
                q='bitcoin OR crypto OR BTC',  # Ключевые слова
                language='en',
                sort_by='publishedAt',
                page_size=10  # Ограничение для бесплатного плана
            )
            return response.get('articles', [])
        except Exception as e:
            logging.error(f"Error fetching news: {e}")
            return []
