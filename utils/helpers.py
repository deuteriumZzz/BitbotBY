import asyncio
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import talib


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает технические индикаторы для заданного DataFrame с рыночными данными.

    Функция вычисляет различные технические индикаторы, такие как RSI, MACD, Bollinger Bands,
    скользящие средние, стохастический осциллятор, индикаторы объема, волатильности и тренда,
    используя библиотеку TA-Lib. Если DataFrame пуст, возвращает его без изменений.
    Добавляет новые столбцы в DataFrame с рассчитанными индикаторами.

    :param df: DataFrame с рыночными данными, содержащий столбцы 'close', 'high', 'low', 'volume' (pd.DataFrame).
    :return: DataFrame с добавленными техническими индикаторами (pd.DataFrame).
    :raises ValueError: Если в DataFrame отсутствуют необходимые столбцы.
    """
    if df.empty:
        return df

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # RSI
    df["rsi"] = talib.RSI(close, timeperiod=14)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(close)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # Bollinger Bands
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(close, timeperiod=20)

    # Moving Averages
    df["sma_20"] = talib.SMA(close, timeperiod=20)
    df["sma_50"] = talib.SMA(close, timeperiod=50)
    df["ema_12"] = talib.EMA(close, timeperiod=12)
    df["ema_26"] = talib.EMA(close, timeperiod=26)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = talib.STOCH(high, low, close)

    # Volume indicators
    df["volume_sma"] = talib.SMA(volume, timeperiod=20)
    df["obv"] = talib.OBV(close, volume)

    # Volatility
    df["atr"] = talib.ATR(high, low, close, timeperiod=14)
    df["natr"] = talib.NATR(high, low, close, timeperiod=14)

    # Trend indicators
    df["adx"] = talib.ADX(high, low, close, timeperiod=14)
    df["cci"] = talib.CCI(high, low, close, timeperiod=14)

    return df


def generate_signals(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Генерирует торговые сигналы на основе технических индикаторов.

    Анализирует последние значения индикаторов в DataFrame и определяет сигналы,
    такие как перекупленность/перепроданность RSI, бычьи/медвежьи сигналы MACD,
    пересечения скользящих средних и другие. Если DataFrame пуст, возвращает пустой словарь.
    Сигналы представлены в виде словаря булевых значений.

    :param df: DataFrame с рассчитанными техническими индикаторами (pd.DataFrame).
    :return: Словарь с торговыми сигналами, где ключи - названия сигналов, значения - булевы флаги (Dict[str, bool]).
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]

    signals = {
        "rsi_overbought": latest["rsi"] > 70 if pd.notna(latest["rsi"]) else False,
        "rsi_oversold": latest["rsi"] < 30 if pd.notna(latest["rsi"]) else False,
        "macd_bullish": latest["macd"] > latest["macd_signal"]
        if pd.notna(latest["macd"])
        else False,
        "macd_bearish": latest["macd"] < latest["macd_signal"]
        if pd.notna(latest["macd"])
        else False,
        "price_above_sma20": latest["close"] > latest["sma_20"]
        if pd.notna(latest["sma_20"])
        else False,
        "price_below_sma20": latest["close"] < latest["sma_20"]
        if pd.notna(latest["sma_20"])
        else False,
        "golden_cross": latest["sma_20"] > latest["sma_50"]
        if pd.notna(latest["sma_50"])
        else False,
        "death_cross": latest["sma_20"] < latest["sma_50"]
        if pd.notna(latest["sma_50"])
        else False,
        "stoch_overbought": latest["stoch_k"] > 80
        if pd.notna(latest["stoch_k"])
        else False,
        "stoch_oversold": latest["stoch_k"] < 20
        if pd.notna(latest["stoch_k"])
        else False,
    }

    return signals


def calculate_position_size(
    account_balance: float, risk_per_trade: float, entry_price: float, stop_loss: float
) -> float:
    """
    Рассчитывает размер позиции на основе управления рисками.

    Определяет размер позиции, чтобы риск на сделку не превышал заданный процент от баланса аккаунта.
    Риск рассчитывается как разница между ценой входа и стоп-лоссом. Если риск на акцию равен нулю,
    возвращает 0.

    :param account_balance: Текущий баланс аккаунта (float).
    :param risk_per_trade: Процент риска на сделку (от 0 до 1) (float).
    :param entry_price: Цена входа в позицию (float).
    :param stop_loss: Цена стоп-лосса (float).
    :return: Размер позиции в единицах актива (float).
    """
    risk_amount = account_balance * risk_per_trade
    risk_per_share = abs(entry_price - stop_loss)

    if risk_per_share == 0:
        return 0

    position_size = risk_amount / risk_per_share
    return position_size


def format_price(price: float) -> str:
    """
    Форматирует цену с соответствующей точностью.

    В зависимости от величины цены применяет разное количество десятичных знаков:
    для цен >= 1000 - 2 знака, >= 1 - 4 знака, иначе - 8 знаков.

    :param price: Цена для форматирования (float).
    :return: Отформатированная строка цены (str).
    """
    if price >= 1000:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    else:
        return f"{price:.8f}"


async def async_retry(func, max_retries: int = 3, delay: float = 1.0, **kwargs):
    """
    Повторяет асинхронную функцию с экспоненциальной задержкой.

    Выполняет функцию до max_retries раз, увеличивая задержку между попытками экспоненциально.
    Если все попытки исчерпаны, поднимает последнее исключение.

    :param func: Асинхронная функция для повторения (callable).
    :param max_retries: Максимальное количество попыток (int, по умолчанию 3).
    :param delay: Начальная задержка в секундах (float, по умолчанию 1.0).
    :param kwargs: Дополнительные аргументы для функции.
    :return: Результат выполнения функции или None, если все попытки неудачны.
    :raises Exception: Последнее исключение, если все попытки исчерпаны.
    """
    for attempt in range(max_retries):
        try:
            return await func(**kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(delay * (2**attempt))
    return None


def calculate_volatility(df: pd.DataFrame, period: int = 20) -> float:
    """
    Рассчитывает волатильность цены.

    Вычисляет стандартное отклонение процентных изменений цены за заданный период.
    Если данных недостаточно, возвращает 0.

    :param df: DataFrame с ценовыми данными, содержащий столбец 'close' (pd.DataFrame).
    :param period: Период для расчета волатильности (int, по умолчанию 20).
    :return: Волатильность как стандартное отклонение (float).
    """
    if len(df) < period:
        return 0

    returns = df["close"].pct_change().dropna()
    if len(returns) < period:
        return 0

    return returns.tail(period).std()


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Нормализует данные в диапазон [0, 1].

    Приводит массив данных к диапазону от 0 до 1 путем вычитания минимума и деления на размах.
    Если все значения одинаковы, возвращает массив нулей.

    :param data: Массив данных для нормализации (np.ndarray).
    :return: Нормализованный массив (np.ndarray).
    """
    if len(data) == 0:
        return data

    min_val = np.min(data)
    max_val = np.max(data)

    if max_val == min_val:
        return np.zeros_like(data)

    return (data - min_val) / (max_val - min_val)


def create_lagged_features(
    df: pd.DataFrame, columns: List[str], lags: List[int]
) -> pd.DataFrame:
    """
    Создает лаговые признаки для временных рядов.

    Добавляет новые столбцы с лаговыми значениями указанных столбцов на заданные лаги.
    Используется для создания признаков в моделях машинного обучения.

    :param df: DataFrame с данными (pd.DataFrame).
    :param columns: Список столбцов для создания лагов (List[str]).
    :param lags: Список лагов (List[int]).
    :return: DataFrame с добавленными лаговыми признаками (pd.DataFrame).
    """
    df_copy = df.copy()

    for col in columns:
        for lag in lags:
            df_copy[f"{col}_lag_{lag}"] = df_copy[col].shift(lag)

    return df_copy


def calculate_correlation_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает корреляционную матрицу для нескольких символов.

    Вычисляет корреляцию процентных изменений цен между символами.
    Используется для анализа взаимосвязей активов.

    :param prices_df: DataFrame с ценами для нескольких символов (pd.DataFrame).
    :return: Корреляционная матрица (pd.DataFrame).
    """
    returns = prices_df.pct_change().dropna()
    return returns.corr()


def safe_divide(a: float, b: float) -> float:
    """
    Безопасное деление с проверкой на ноль.

    Выполняет деление a на b, возвращая 0, если b равно 0, чтобы избежать деления на ноль.

    :param a: Делимое (float).
    :param b: Делитель (float).
    :return: Результат деления или 0 (float).
    """
    if b == 0:
        return 0
    return a / b


def format_timestamp(timestamp: Any) -> str:
    """
    Форматирует временную метку в строку ISO.

    Преобразует различные типы временных меток (строка, datetime, timestamp) в строку ISO.
    Если тип неизвестен, возвращает текущую временную метку.

    :param timestamp: Временная метка для форматирования (Any).
    :return: Строка в формате ISO (str).
    """
    if isinstance(timestamp, str):
        return timestamp
    elif isinstance(timestamp, datetime):
        return timestamp.isoformat()
    elif isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp).isoformat()
    else:
        return datetime.now().isoformat()


def validate_market_data(data: Dict) -> bool:
    """
    Проверяет структуру рыночных данных.

    Убеждается, что словарь содержит все необходимые поля для рыночных данных.

    :param data: Словарь с рыночными данными (Dict).
    :return: True, если все поля присутствуют, иначе False (bool).
    """
    required_fields = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    return all(field in data for field in required_fields)


def calculate_performance_metrics(trades: List[Dict]) -> Dict[str, float]:
    """
    Рассчитывает метрики производительности торговли.

    Анализирует список сделок и вычисляет метрики, такие как общее количество сделок,
    количество выигрышных и проигрышных, коэффициент выигрыша, общий P&L и другие.
    Если список сделок пуст, возвращает пустой словарь.

    :param trades: Список словарей с данными о сделках (List[Dict]).
    :return: Словарь с метриками производительности (Dict[str, float]).
    """
    if not trades:
        return {}

    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

    total_pnl = sum(t.get("pnl", 0) for t in trades)
    avg_win = (
        np.mean([t.get("pnl", 0) for t in winning_trades]) if winning_trades else 0
    )
    avg_loss = np.mean([t.get("pnl", 0) for t in losing_trades]) if losing_trades else 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": len(winning_trades) / len(trades) if trades else 0,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
    }
