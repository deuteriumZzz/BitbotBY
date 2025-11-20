from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

class NewsAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    async def analyze_news_async(self, news_list):
        try:
            if not news_list:
                return 0.0
            # Простой анализ (можно интегрировать API новостей)
            text = ' '.join(news_list)
            scores = self.analyzer.polarity_scores(text)
            return scores['compound']
        except Exception as e:
            logging.error(f"Error analyzing news: {e}")
            return 0.0
