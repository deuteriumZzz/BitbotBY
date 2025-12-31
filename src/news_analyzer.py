import asyncio
import logging
import os

from dotenv import load_dotenv
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
logging.basicConfig(level=logging.INFO)


class NewsAnalyzer:
    """
    Класс для анализа сентимента новостей по теме криптовалют.

    Использует NewsAPI для получения новостей и VADER Sentiment для анализа настроений.
    Предоставляет асинхронные методы для обработки данных.
    """

    def __init__(self):
        """
        Инициализирует экземпляр NewsAnalyzer.

        Загружает API-ключ из переменных окружения и создает экземпляры анализатора сентимента и клиента NewsAPI.

        :raises ValueError: Если NEWS_API_KEY не найден в переменных окружения.
        """
        self.analyzer = SentimentIntensityAnalyzer()
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables")
        self.newsapi = NewsApiClient(api_key=api_key)

    async def analyze_news_async(self):
        """
        Асинхронно анализирует сентимент новостей по теме криптовалют.

        Получает последние новости, анализирует их сентимент с использованием VADER и возвращает средний сентимент.
        Логирует результаты и ошибки.

        :return: Средний сентимент новостей (float от -1 до 1, где 1 - положительный, -1 - отрицательный).
        :raises Exception: В случае ошибок при анализе (логируется в logger).
        """
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
        """
        Асинхронно получает новости с помощью NewsAPI.

        Запрашивает новости по ключевым словам "bitcoin OR crypto OR BTC" на английском языке,
        отсортированные по дате публикации, с ограничением на 10 статей.

        :return: Список статей (dict с данными о новостях) или пустой список в случае ошибки.
        :raises Exception: В случае ошибок при запросе (логируется в logger).
        """
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
