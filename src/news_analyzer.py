import asyncio
import logging
import os

from dotenv import load_dotenv
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
logging.basicConfig(level=logging.INFO)


class NewsAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

    async def analyze_news_async(self):
        try:
            # Запрашиваем новости
            articles = await self._fetch_news_async()

            if not articles:
                logging.warning("No news articles fetched")
                return 0.0

            # Анализ сентимента
            sentiments = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    scores = self.analyzer.polarity_scores(text)
                    sentiments.append(scores["compound"])

            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            logging.info(f"News sentiment: {avg_sentiment}")
            return avg_sentiment

        except Exception as e:
            logging.error(f"Error analyzing news: {e}")
            return 0.0

    async def _fetch_news_async(self):
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.newsapi.get_everything(
                    q="bitcoin OR crypto OR BTC",
                    language="en",
                    sort_by="publishedAt",
                    page_size=10,
                ),
            )
            return response.get("articles", [])
        except Exception as e:
            logging.error(f"Error fetching news: {e}")
            return []
