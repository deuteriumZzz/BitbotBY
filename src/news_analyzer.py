from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Инициализируем анализатор заранее
analyzer = SentimentIntensityAnalyzer()


def analyze_news_sentiment(text: str) -> float:
    """
    Быстрый анализ настроения новости (VADER: ~10ms).
    Возвращает compound score от -1 (негатив) до 1 (позитив).
    """
    if not text:
        return 0.0
    scores = analyzer.polarity_scores(text)
    return scores["compound"]  # Используем compound для простоты
