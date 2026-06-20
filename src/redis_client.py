"""
Redis-клиент: кэш рыночных данных, торговое состояние, хранилище модели, блокировки.

Настраивается через переменные окружения REDIS_HOST / REDIS_PORT / REDIS_PASSWORD.
Если Redis недоступен — бот работает без персистентного состояния (логгирует
предупреждение, не бросает исключение).
"""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

import pandas as pd
import redis

from src.constants import (
    REDIS_TTL_MARKET_DATA,
    REDIS_TTL_MODEL,
    REDIS_TTL_TRADING_STATE,
)


class RedisClient:
    """
    Redis-клиент торгового бота.

    Обеспечивает:
    - кэширование рыночных данных (DataFrame ↔ JSON split-формат);
    - персистентность торгового состояния (позиции, портфель);
    - хранение весов модели (numpy-массивы сериализуются через .tolist());
    - распределённые блокировки с UUID-токенами и Lua atomic release;
    - публикацию сигналов (Pub/Sub канал trading_signals);
    - статистику производительности.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 6379,
        password: Optional[str] = None,
    ) -> None:
        """
        Подключается к Redis. Не бросает исключение при недоступности.

        :param host: Хост Redis (по умолчанию: env REDIS_HOST или "redis").
        :param port: Порт Redis (по умолчанию: env REDIS_PORT или 6379).
        :param password: Пароль Redis (по умолчанию: env REDIS_PASSWORD).
        """
        redis_host = host or os.getenv("REDIS_HOST", "redis")
        redis_port = port or int(os.getenv("REDIS_PORT", 6379))
        redis_password = password or os.getenv("REDIS_PASSWORD")

        # decode_responses=False: держим bytes для корректного round-trip JSON
        # DataFrame; строки декодируются вручную через .decode() + json.loads().
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=False,
        )
        self.logger = logging.getLogger(__name__)
        self._available: bool = False
        self._lock_tokens: Dict[str, str] = {}
        self._test_connection()

    def _test_connection(self) -> None:
        """Ping Redis; при ошибке устанавливает _available=False и логгирует warning."""
        try:
            self.redis_client.ping()
            self._available = True
            kwargs = self.redis_client.connection_pool.connection_kwargs
            self.logger.info(
                "Connected to Redis: %s:%s", kwargs["host"], kwargs["port"]
            )
        except Exception as e:
            self._available = False
            self.logger.warning(
                "Redis unavailable: %s. Running without state persistence.", e
            )

    def reconnect(self) -> bool:
        """
        Пробует переподключиться к Redis.

        Вызывать периодически (например, каждый цикл бота) для восстановления
        персистентности после кратковременного сбоя Redis без перезапуска бота.

        :return: True если подключение установлено.
        """
        self._test_connection()
        return self._available

    def save_market_data(self, key: str, data: pd.DataFrame) -> None:
        """
        Сериализует DataFrame в JSON (split-ориентация) и сохраняет с TTL.

        :param key: Ключ Redis.
        :param data: DataFrame с OHLCV-данными.
        """
        if not self._available:
            return
        try:
            serialized = data.to_json(orient="split").encode()
            self.redis_client.setex(key, REDIS_TTL_MARKET_DATA, serialized)
            self.logger.debug("Saved market data: %s", key)
        except Exception as e:
            self._available = False
            self.logger.warning("Redis write failed (%s): %s", key, e)

    def load_market_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Загружает и десериализует DataFrame из Redis.

        :param key: Ключ Redis.
        :return: DataFrame или None при промахе кэша / ошибке.
        """
        if not self._available:
            return None
        try:
            raw = self.redis_client.get(key)
            if raw:
                self.logger.debug("Loaded market data: %s", key)
                return pd.read_json(io.StringIO(raw.decode()), orient="split")
        except Exception as e:
            self._available = False
            self.logger.warning("Redis read failed (%s): %s", key, e)
        return None

    def save_trading_state(self, key: str, state: Dict[str, Any]) -> None:
        """
        Сохраняет торговое состояние (словарь) как JSON с TTL REDIS_TTL_TRADING_STATE.

        :param key: Ключ Redis.
        :param state: Сериализуемый словарь (позиции, портфель и т.д.).
        """
        if not self._available:
            return
        try:
            serialized = json.dumps(state)
            self.redis_client.setex(key, REDIS_TTL_TRADING_STATE, serialized)
            self.logger.debug("Saved trading state: %s", key)
        except Exception as e:
            self._available = False
            self.logger.warning("Redis write failed (%s): %s", key, e)

    def load_trading_state(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Загружает и десериализует торговое состояние из Redis.

        :param key: Ключ Redis.
        :return: Словарь или None если ключ отсутствует / декодирование не удалось.
        """
        if not self._available:
            return None
        try:
            state = self.redis_client.get(key)
            if state:
                self.logger.debug("Loaded trading state: %s", key)
                return json.loads(state.decode("utf-8"))
        except Exception as e:
            self._available = False
            self.logger.warning("Redis read failed (%s): %s", key, e)
        return None

    def save_model(
        self,
        strategy_name: str,
        model_data: Dict[str, Any],
    ) -> None:
        """
        Сохраняет веса модели как JSON с TTL REDIS_TTL_MODEL.

        numpy-массивы сериализуются через .tolist() для корректного round-trip
        без потери точности (default=str превратил бы массивы в строку repr).

        :param strategy_name: Имя стратегии (ключ Redis: model:<strategy_name>).
        :param model_data: Словарь с весами модели.
        """
        if not self._available:
            return

        def _numpy_safe(obj: Any) -> Any:
            if hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(type(obj))

        try:
            serialized = json.dumps(model_data, default=_numpy_safe).encode()
            self.redis_client.setex(
                f"model:{strategy_name}", REDIS_TTL_MODEL, serialized
            )
            self.logger.debug("Saved model: %s", strategy_name)
        except Exception as e:
            self._available = False
            self.logger.warning("Redis write failed (model:%s): %s", strategy_name, e)

    def load_model(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Загружает и десериализует веса модели из Redis.

        :param strategy_name: Имя стратегии.
        :return: Словарь с весами или None при промахе / ошибке.
        """
        if not self._available:
            return None
        try:
            raw = self.redis_client.get(f"model:{strategy_name}")
            if raw:
                self.logger.debug("Loaded model: %s", strategy_name)
                return json.loads(raw.decode())
        except Exception as e:
            self._available = False
            self.logger.warning("Redis read failed (model:%s): %s", strategy_name, e)
        return None

    # Атомарная проверка владельца блокировки: удаляем ключ только если
    # сохранённый токен совпадает с нашим. Plain DEL освободил бы блокировку
    # чужого процесса после истечения TTL.
    _RELEASE_SCRIPT = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        return redis.call('del', KEYS[1])
    end
    return 0
    """

    def acquire_lock(self, lock_name: str, timeout: int = 10) -> bool:
        """
        Захватывает распределённую блокировку через SET NX с UUID-токеном владельца.

        :param lock_name: Ключ Redis для блокировки.
        :param timeout: TTL блокировки в секундах (авто-освобождение при краше).
        :return: True если блокировка захвачена, False если уже занята.
        """
        if not self._available:
            # Redis down — distributed lock unavailable, but in-process asyncio.Lock
            # in OrderExecutor still prevents concurrent access within this process.
            # Returning False would silently block ALL order creation, which is far
            # worse than operating without the cross-process lock.
            self.logger.warning(
                "Redis unavailable — distributed lock skipped for %s", lock_name
            )
            return True
        try:
            token = str(uuid.uuid4())
            result = bool(self.redis_client.set(lock_name, token, nx=True, ex=timeout))
            if result:
                self._lock_tokens[lock_name] = token
                self.logger.debug("Lock acquired: %s", lock_name)
            return result
        except Exception as e:
            self._available = False
            self.logger.warning("Redis lock failed (%s): %s", lock_name, e)
            return True  # fail-open: prefer trading over silently halting

    def release_lock(self, lock_name: str) -> None:
        """
        Освобождает блокировку только если этот процесс является её владельцем.

        Использует Lua-скрипт для атомарной проверки токена и удаления ключа.

        :param lock_name: Ключ Redis для блокировки.
        """
        if not self._available:
            self._lock_tokens.pop(lock_name, None)
            return
        try:
            token = self._lock_tokens.pop(lock_name, None)
            if token is None:
                return
            self.redis_client.eval(self._RELEASE_SCRIPT, 1, lock_name, token)
            self.logger.debug("Lock released: %s", lock_name)
        except Exception as e:
            self._available = False
            self.logger.warning("Redis lock release failed (%s): %s", lock_name, e)

    def publish_signal(self, signal_data: Dict[str, Any]) -> None:
        """
        Публикует торговый сигнал в Pub/Sub канал 'trading_signals'.

        :param signal_data: Сериализуемый словарь с данными сигнала.
        """
        if not self._available:
            return
        try:
            self.redis_client.publish("trading_signals", json.dumps(signal_data))
            self.logger.debug("Signal published")
        except Exception as e:
            self._available = False
            self.logger.warning("Redis publish failed: %s", e)

    def update_performance_stats(self, stats: Dict[str, Any]) -> None:
        """
        Записывает статистику производительности в 'performance_stats' с TTL 24 часа.

        :param stats: Словарь с метриками производительности.
        """
        if not self._available:
            return
        try:
            self.redis_client.set(
                "performance_stats",
                json.dumps(stats),
                ex=REDIS_TTL_TRADING_STATE,
            )
            self.logger.debug("Performance stats updated: %s", stats)
        except Exception as e:
            self._available = False
            self.logger.warning("Redis write failed (performance_stats): %s", e)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Возвращает актуальную статистику производительности из Redis.

        :return: Словарь с метриками или {} если ключ отсутствует / Redis недоступен.
        """
        if not self._available:
            return {}
        try:
            data = self.redis_client.get("performance_stats")
            if data:
                return json.loads(data.decode("utf-8"))
            return {}
        except Exception as e:
            self._available = False
            self.logger.warning("Redis read failed (performance_stats): %s", e)
            return {}
