"""
Тесты для src/redis_client.py.

Используют mock вместо fakeredis, так как fakeredis не в requirements.txt.
Redis-клиент мокается на уровне класса через MagicMock с нужными
возвращаемыми значениями для каждого метода.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# conftest.py уже мокает redis на уровне sys.modules
from src.redis_client import RedisClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(ping_ok: bool = True) -> RedisClient:
    """Создаёт RedisClient с полностью замоканным redis.Redis."""
    with patch("src.redis_client.redis.Redis") as MockRedis:
        mock_conn = MagicMock()
        if ping_ok:
            mock_conn.ping.return_value = True
        else:
            mock_conn.ping.side_effect = ConnectionError("redis down")
        mock_conn.connection_pool.connection_kwargs = {"host": "redis", "port": 6379}
        MockRedis.return_value = mock_conn
        client = RedisClient(host="localhost", port=6379)
        client._mock_conn = mock_conn  # сохраняем для проверок
    return client


def _make_unavailable_client() -> RedisClient:
    """Клиент, у которого _available=False (Redis недоступен)."""
    return _make_client(ping_ok=False)


# ---------------------------------------------------------------------------
# _available=False — все методы возвращают None/False/{}
# ---------------------------------------------------------------------------

class TestUnavailableGuard:
    """Когда Redis недоступен, методы должны возвращать safe defaults."""

    def setup_method(self):
        self.client = _make_unavailable_client()

    def test_save_market_data_returns_none_when_unavailable(self):
        """save_market_data не падает и ничего не пишет при _available=False."""
        df = pd.DataFrame({"close": [1.0, 2.0]})
        result = self.client.save_market_data("key", df)
        assert result is None
        self.client._mock_conn.setex.assert_not_called()

    def test_load_market_data_returns_none_when_unavailable(self):
        """load_market_data возвращает None при _available=False."""
        result = self.client.load_market_data("key")
        assert result is None

    def test_save_trading_state_returns_none_when_unavailable(self):
        """save_trading_state не падает при _available=False."""
        result = self.client.save_trading_state("key", {"foo": 1})
        assert result is None

    def test_load_trading_state_returns_none_when_unavailable(self):
        """load_trading_state возвращает None при _available=False."""
        result = self.client.load_trading_state("key")
        assert result is None

    def test_save_model_returns_none_when_unavailable(self):
        """save_model не падает при _available=False."""
        result = self.client.save_model("my_strategy", {"weights": [1, 2]})
        assert result is None

    def test_load_model_returns_none_when_unavailable(self):
        """load_model возвращает None при _available=False."""
        result = self.client.load_model("my_strategy")
        assert result is None

    def test_acquire_lock_returns_true_when_unavailable(self):
        """acquire_lock возвращает True при _available=False (fail-open).

        Redis недоступен → распределённая блокировка пропускается,
        но внутрипроцессный asyncio.Lock в OrderExecutor всё равно
        защищает от параллельного входа в сделки.
        """
        result = self.client.acquire_lock("my_lock")
        assert result is True

    def test_release_lock_cleans_up_token_when_unavailable(self):
        """release_lock чистит кэш токенов даже при _available=False."""
        self.client._lock_tokens["my_lock"] = "some-token"
        self.client.release_lock("my_lock")
        assert "my_lock" not in self.client._lock_tokens

    def test_publish_signal_returns_none_when_unavailable(self):
        """publish_signal не падает при _available=False."""
        result = self.client.publish_signal({"action": "buy"})
        assert result is None

    def test_update_performance_stats_returns_none_when_unavailable(self):
        """update_performance_stats не падает при _available=False."""
        result = self.client.update_performance_stats({"pnl": 100})
        assert result is None

    def test_get_performance_stats_returns_empty_dict_when_unavailable(self):
        """get_performance_stats возвращает {} при _available=False."""
        result = self.client.get_performance_stats()
        assert result == {}


# ---------------------------------------------------------------------------
# save_market_data / load_market_data round-trip
# ---------------------------------------------------------------------------

class TestMarketDataRoundTrip:
    """Round-trip тест: сохранение и загрузка DataFrame через Redis."""

    def setup_method(self):
        self.client = _make_client(ping_ok=True)

    def test_save_calls_setex_with_serialized_df(self):
        """save_market_data сериализует DataFrame и вызывает setex."""
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        self.client.save_market_data("btc:ohlcv", df)
        self.client._mock_conn.setex.assert_called_once()
        call_args = self.client._mock_conn.setex.call_args[0]
        assert call_args[0] == "btc:ohlcv"
        # значение должно быть bytes
        assert isinstance(call_args[2], bytes)

    def test_load_returns_dataframe_with_correct_data(self):
        """load_market_data десериализует bytes и возвращает корректный DataFrame."""
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        serialized = df.to_json(orient="split").encode()
        self.client._mock_conn.get.return_value = serialized

        result = self.client.load_market_data("btc:ohlcv")
        assert result is not None
        assert list(result.columns) == ["close"]
        assert list(result["close"]) == [100.0, 101.0, 102.0]

    def test_load_returns_none_on_cache_miss(self):
        """load_market_data возвращает None когда ключ не найден."""
        self.client._mock_conn.get.return_value = None
        result = self.client.load_market_data("missing:key")
        assert result is None

    def test_save_sets_available_false_on_redis_error(self):
        """Ошибка Redis при записи устанавливает _available=False."""
        self.client._mock_conn.setex.side_effect = Exception("connection lost")
        df = pd.DataFrame({"close": [1.0]})
        self.client.save_market_data("key", df)
        assert self.client._available is False

    def test_load_sets_available_false_on_redis_error(self):
        """Ошибка Redis при чтении устанавливает _available=False."""
        self.client._mock_conn.get.side_effect = Exception("connection lost")
        result = self.client.load_market_data("key")
        assert result is None
        assert self.client._available is False


# ---------------------------------------------------------------------------
# save_trading_state / load_trading_state round-trip
# ---------------------------------------------------------------------------

class TestTradingStateRoundTrip:
    """Round-trip для торгового состояния (dict → JSON → dict)."""

    def setup_method(self):
        self.client = _make_client(ping_ok=True)

    def test_save_serializes_and_calls_setex(self):
        """save_trading_state сериализует dict в JSON и вызывает setex."""
        state = {"balance": 10000.0, "positions": {"BTC": 0.5}}
        self.client.save_trading_state("state:test", state)
        self.client._mock_conn.setex.assert_called_once()

    def test_load_deserializes_json(self):
        """load_trading_state корректно десериализует JSON обратно в dict."""
        state = {"balance": 10000.0, "positions": {"BTC": 0.5}}
        self.client._mock_conn.get.return_value = json.dumps(state).encode("utf-8")

        result = self.client.load_trading_state("state:test")
        assert result == state

    def test_load_returns_none_on_miss(self):
        """load_trading_state возвращает None при cache miss."""
        self.client._mock_conn.get.return_value = None
        result = self.client.load_trading_state("state:missing")
        assert result is None


# ---------------------------------------------------------------------------
# save_model / load_model с numpy arrays
# ---------------------------------------------------------------------------

class TestModelRoundTrip:
    """Тест сохранения/загрузки модели с numpy arrays."""

    def setup_method(self):
        self.client = _make_client(ping_ok=True)

    def test_save_model_serializes_numpy_arrays(self):
        """save_model конвертирует numpy arrays в lists через .tolist()."""
        model_data = {"weights": np.array([1.0, 2.0, 3.0]), "bias": np.float32(0.5)}
        self.client.save_model("test_strategy", model_data)
        self.client._mock_conn.setex.assert_called_once()
        key, ttl, value = self.client._mock_conn.setex.call_args[0]
        assert key == "model:test_strategy"
        # Значение должно быть корректным JSON
        parsed = json.loads(value.decode())
        assert parsed["weights"] == [1.0, 2.0, 3.0]

    def test_load_model_deserializes_json(self):
        """load_model десериализует JSON обратно в dict."""
        model_data = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}
        self.client._mock_conn.get.return_value = json.dumps(model_data).encode()

        result = self.client.load_model("test_strategy")
        assert result == model_data

    def test_load_model_uses_correct_key_prefix(self):
        """load_model использует ключ model:{strategy_name}."""
        self.client._mock_conn.get.return_value = None
        self.client.load_model("my_strat")
        self.client._mock_conn.get.assert_called_with("model:my_strat")

    def test_load_model_returns_none_on_miss(self):
        """load_model возвращает None при cache miss."""
        self.client._mock_conn.get.return_value = None
        result = self.client.load_model("unknown_strat")
        assert result is None


# ---------------------------------------------------------------------------
# acquire_lock / release_lock
# ---------------------------------------------------------------------------

class TestLocking:
    """Тесты распределённых блокировок."""

    def setup_method(self):
        self.client = _make_client(ping_ok=True)

    def test_acquire_lock_returns_true_on_success(self):
        """acquire_lock возвращает True когда SET NX успешен."""
        self.client._mock_conn.set.return_value = True
        result = self.client.acquire_lock("trade_lock", timeout=10)
        assert result is True
        assert "trade_lock" in self.client._lock_tokens

    def test_acquire_lock_returns_false_when_already_held(self):
        """acquire_lock возвращает False когда замок уже захвачен."""
        self.client._mock_conn.set.return_value = None  # Redis вернёт None для NX
        result = self.client.acquire_lock("trade_lock")
        assert result is False
        assert "trade_lock" not in self.client._lock_tokens

    def test_acquire_lock_stores_token(self):
        """После успешного acquire_lock токен хранится в _lock_tokens."""
        self.client._mock_conn.set.return_value = True
        self.client.acquire_lock("trade_lock")
        assert "trade_lock" in self.client._lock_tokens
        assert isinstance(self.client._lock_tokens["trade_lock"], str)

    def test_release_lock_calls_eval_with_lua_script(self):
        """release_lock вызывает Lua-скрипт для атомарного удаления."""
        self.client._mock_conn.set.return_value = True
        self.client.acquire_lock("trade_lock")
        self.client.release_lock("trade_lock")
        self.client._mock_conn.eval.assert_called_once()
        # Токен должен быть удалён из кэша
        assert "trade_lock" not in self.client._lock_tokens

    def test_release_lock_noop_when_no_token(self):
        """release_lock не вызывает eval если токен не известен."""
        self.client.release_lock("unknown_lock")
        self.client._mock_conn.eval.assert_not_called()

    def test_acquire_lock_sets_available_false_on_error(self):
        """Ошибка при acquire_lock устанавливает _available=False, но возвращает True (fail-open)."""
        self.client._mock_conn.set.side_effect = Exception("redis error")
        result = self.client.acquire_lock("trade_lock")
        # fail-open: блокировку пропускаем, торговля продолжается
        assert result is True
        assert self.client._available is False


# ---------------------------------------------------------------------------
# reconnect()
# ---------------------------------------------------------------------------

class TestReconnect:
    """Тесты логики переподключения."""

    def test_reconnect_sets_available_true_when_redis_up(self):
        """reconnect() устанавливает _available=True при успешном ping."""
        client = _make_unavailable_client()
        assert client._available is False
        # Теперь симулируем что Redis снова доступен
        client._mock_conn.ping.return_value = True
        client._mock_conn.ping.side_effect = None
        result = client.reconnect()
        assert result is True
        assert client._available is True

    def test_reconnect_returns_false_when_redis_still_down(self):
        """reconnect() возвращает False если Redis всё ещё недоступен."""
        client = _make_unavailable_client()
        result = client.reconnect()
        assert result is False
        assert client._available is False


# ---------------------------------------------------------------------------
# publish_signal
# ---------------------------------------------------------------------------

class TestPublishSignal:
    """Тесты публикации торгового сигнала."""

    def setup_method(self):
        self.client = _make_client(ping_ok=True)

    def test_publish_signal_calls_publish_with_json(self):
        """publish_signal публикует корректный JSON в канал trading_signals."""
        signal = {"action": "buy", "symbol": "BTC/USDT", "confidence": 0.9}
        self.client.publish_signal(signal)

        self.client._mock_conn.publish.assert_called_once()
        channel, payload = self.client._mock_conn.publish.call_args[0]
        assert channel == "trading_signals"
        parsed = json.loads(payload)
        assert parsed == signal

    def test_publish_signal_sets_available_false_on_error(self):
        """Ошибка при публикации устанавливает _available=False."""
        self.client._mock_conn.publish.side_effect = Exception("broken pipe")
        self.client.publish_signal({"action": "sell"})
        assert self.client._available is False


# ---------------------------------------------------------------------------
# update_performance_stats / get_performance_stats round-trip
# ---------------------------------------------------------------------------

class TestPerformanceStats:
    """Round-trip для статистики производительности."""

    def setup_method(self):
        self.client = _make_client(ping_ok=True)

    def test_update_performance_stats_calls_set(self):
        """update_performance_stats вызывает redis set с корректными данными."""
        stats = {"win_rate": 0.65, "total_pnl": 1234.56, "trades": 42}
        self.client.update_performance_stats(stats)
        self.client._mock_conn.set.assert_called_once()
        args = self.client._mock_conn.set.call_args
        key = args[0][0]
        assert key == "performance_stats"

    def test_get_performance_stats_returns_deserialized_dict(self):
        """get_performance_stats десериализует JSON в dict."""
        stats = {"win_rate": 0.65, "total_pnl": 1234.56}
        self.client._mock_conn.get.return_value = json.dumps(stats).encode("utf-8")

        result = self.client.get_performance_stats()
        assert result == stats

    def test_get_performance_stats_returns_empty_dict_on_miss(self):
        """get_performance_stats возвращает {} при отсутствии данных."""
        self.client._mock_conn.get.return_value = None
        result = self.client.get_performance_stats()
        assert result == {}

    def test_get_performance_stats_sets_available_false_on_error(self):
        """Ошибка при чтении устанавливает _available=False."""
        self.client._mock_conn.get.side_effect = Exception("timeout")
        result = self.client.get_performance_stats()
        assert result == {}
        assert self.client._available is False
