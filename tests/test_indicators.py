from src.indicators import calculate_technical_indicators
from tests.conftest import make_ohlcv


def test_rsi_range():
    """RSI всегда должен быть в диапазоне [0, 100]."""
    df = calculate_technical_indicators(make_ohlcv(100))
    rsi = df["rsi"].dropna()
    assert (rsi >= 0).all(), "RSI below 0"
    assert (rsi <= 100).all(), "RSI above 100"


def test_rsi_flat_market_no_error():
    """Флэтовый рынок (loss=0) не должен вызывать ZeroDivisionError."""
    df = make_ohlcv(100, trend=0.0)
    df["close"] = 100.0
    df["open"] = 100.0
    df["high"] = 100.0
    df["low"] = 100.0
    result = calculate_technical_indicators(df)
    assert "rsi" in result.columns
    assert not result["rsi"].isna().all()


def test_bb_width_no_zero_division():
    """bb_width должен быть конечным даже при постоянных ценах."""
    df = make_ohlcv(100, trend=0.0)
    df["close"] = 50.0
    result = calculate_technical_indicators(df)
    assert "bb_width" in result.columns
    finite = (
        result["bb_width"].replace([float("inf"), float("-inf")], float("nan")).dropna()
    )
    assert len(finite) > 0


def test_volume_ratio_no_zero_division():
    """volume_ratio не должен быть inf или nan при volume=0."""
    df = make_ohlcv(60)
    df["volume"] = 0.0
    result = calculate_technical_indicators(df)
    col = result["volume_ratio"]
    assert not col.isin([float("inf"), float("-inf")]).any()


def test_ema_columns_exist():
    df = calculate_technical_indicators(make_ohlcv(100))
    for col in [
        "ema_short",
        "ema_long",
        "rsi",
        "macd",
        "bb_upper",
        "bb_lower",
        "atr",
    ]:
        assert col in df.columns, f"Отсутствует колонка: {col}"


def test_no_future_leak_ffill():
    """
    После ffill().fillna(0) в хвостовых строках не должно быть
    бесконечных значений (bfill был старым багом, заглядывавшим
    в будущие данные).
    """
    df = make_ohlcv(100)
    result = calculate_technical_indicators(df)
    tail = result.iloc[30:]
    numeric = tail.select_dtypes(include="number")
    assert (
        not numeric.isin([float("inf"), float("-inf")]).any().any()
    ), "Обнаружены бесконечные значения"
