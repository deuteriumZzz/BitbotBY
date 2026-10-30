"""Тесты логики TTL-кэша режима, извлечённой из торгового цикла TradingBot."""

import time
from unittest.mock import MagicMock

TTL = 300.0


def _predict(cache: dict, sym: str, predict_fn, ttl: float) -> str:
    """Чистый TTL-кэш поиска — зеркалирует логику торгового цикла TradingBot."""
    now = time.monotonic()
    cached_regime, cached_ts = cache.get(sym, ("unknown", float("-inf")))
    if now - cached_ts < ttl:
        return cached_regime
    regime = predict_fn()
    cache[sym] = (regime, now)
    return regime


class TestRegimeCache:
    def setup_method(self):
        self.cache: dict = {}
        self.predict = MagicMock(return_value="trending_up")

    def test_cold_cache_calls_predict(self):
        result = _predict(self.cache, "BTC/USDT", self.predict, TTL)
        self.predict.assert_called_once()
        assert result == "trending_up"

    def test_warm_cache_skips_predict(self):
        self.cache["ETH/USDT"] = ("ranging", time.monotonic())
        result = _predict(self.cache, "ETH/USDT", self.predict, TTL)
        self.predict.assert_not_called()
        assert result == "ranging"

    def test_expired_cache_calls_predict_again(self):
        self.predict.return_value = "trending_down"
        self.cache["SOL/USDT"] = ("ranging", time.monotonic() - (TTL + 100))
        result = _predict(self.cache, "SOL/USDT", self.predict, TTL)
        self.predict.assert_called_once()
        assert result == "trending_down"
        assert self.cache["SOL/USDT"][0] == "trending_down"

    def test_cache_entry_updated_after_predict(self):
        before = time.monotonic()
        _predict(self.cache, "XRP/USDT", self.predict, TTL)
        regime, ts = self.cache["XRP/USDT"]
        assert regime == "trending_up"
        assert ts >= before

    def test_different_symbols_cached_independently(self):
        self.predict.side_effect = ["trending_up", "ranging"]
        _predict(self.cache, "BTC/USDT", self.predict, TTL)
        _predict(self.cache, "ETH/USDT", self.predict, TTL)
        assert self.cache["BTC/USDT"][0] == "trending_up"
        assert self.cache["ETH/USDT"][0] == "ranging"

    def test_zero_ttl_always_calls_predict(self):
        self.cache["ADA/USDT"] = ("ranging", time.monotonic())
        _predict(self.cache, "ADA/USDT", self.predict, ttl=0.0)
        self.predict.assert_called_once()
