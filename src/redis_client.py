"""
Клиент Redis для кэширования рыночных данных, состояний торговли и блокировок.
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


class RedisClient:
    """
    Клиент Redis для торгового бота.

    Предоставляет методы для сохранения и загрузки рыночных данных,
    состояний торговли, моделей, управления блокировками, публикации
    сигналов и обновления статистики производительности.
    Конфигурируется через переменные окружения REDIS_HOST/PORT/PASSWORD.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 6379,
        password: Optional[str] = None,
    ) -> None:
        """
        Инициализирует клиент Redis и проверяет соединение.

        :param host: Хост Redis (по умолчанию из REDIS_HOST или "redis").
        :param port: Порт Redis (по умолчанию из REDIS_PORT или 6379).
        :param password: Пароль (по умолчанию из REDIS_PASSWORD).
        :raises redis.ConnectionError: Если подключение не удалось.
        """
        redis_host = host or os.getenv("REDIS_HOST", "redis")
        redis_port = port or int(os.getenv("REDIS_PORT", 6379))
        redis_password = password or os.getenv("REDIS_PASSWORD")

        # decode_responses=False: нужно для хранения pickle-байтов.
        # JSON-ответы декодируются вручную через .decode() + json.loads().
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
        """Проверяет соединение с Redis.

        Не бросает исключение — бот работает без Redis.
        """
        try:
            self.redis_client.ping()
            self._available = True
            kwargs = self.redis_client.connection_pool.connection_kwargs
            self.logger.info(f"Подключено к Redis: {kwargs['host']}:{kwargs['port']}")
        except Exception as e:
            self._available = False
            self.logger.warning(
                f"Redis недоступен: {e}. " "Бот работает без персистентности состояния."
            )

    def save_market_data(self, key: str, data: pd.DataFrame) -> None:
        """Сохраняет рыночные данные (DataFrame) в Redis как JSON. TTL — 5 минут."""
        try:
            serialized = data.to_json(orient="split").encode()
            self.redis_client.setex(key, 300, serialized)
            self.logger.debug(f"Сохранены данные: {key}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения рыночных данных: {e}")

    def load_market_data(self, key: str) -> Optional[pd.DataFrame]:
        """Загружает рыночные данные из Redis."""
        try:
            raw = self.redis_client.get(key)
            if raw:
                self.logger.debug(f"Загружены данные: {key}")
                return pd.read_json(io.StringIO(raw.decode()), orient="split")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки рыночных данных: {e}")
        return None

    def save_trading_state(self, key: str, state: Dict[str, Any]) -> None:
        """
        Сохраняет состояние торговли в Redis как JSON.

        TTL — 24 часа.

        :param key: Ключ для сохранения.
        :param state: Словарь состояния.
        """
        try:
            serialized = json.dumps(state)
            self.redis_client.setex(key, 86400, serialized)
            self.logger.debug(f"Сохранено состояние: {key}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения состояния: {e}")

    def load_trading_state(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Загружает состояние торговли из Redis.

        :param key: Ключ для загрузки.
        :return: Словарь состояния или None если не найден.
        """
        try:
            state = self.redis_client.get(key)
            if state:
                self.logger.debug(f"Загружено состояние: {key}")
                return json.loads(state.decode("utf-8"))
        except Exception as e:
            self.logger.error(f"Ошибка загрузки состояния: {e}")
        return None

    def save_model(
        self,
        strategy_name: str,
        model_data: Dict[str, Any],
    ) -> None:
        """Сохраняет данные модели в Redis как JSON. TTL — 7 дней."""
        try:
            serialized = json.dumps(model_data, default=str).encode()
            self.redis_client.setex(f"model:{strategy_name}", 604800, serialized)
            self.logger.debug(f"Сохранена модель: {strategy_name}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели: {e}")

    def load_model(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Загружает данные модели из Redis."""
        try:
            raw = self.redis_client.get(f"model:{strategy_name}")
            if raw:
                self.logger.debug(f"Загружена модель: {strategy_name}")
                return json.loads(raw.decode())
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
        return None

    # Lua script: delete key only if value matches token (atomic ownership check)
    _RELEASE_SCRIPT = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        return redis.call('del', KEYS[1])
    end
    return 0
    """

    def acquire_lock(self, lock_name: str, timeout: int = 10) -> bool:
        """
        Захватывает распределённую блокировку через SET NX с UUID-токеном.

        :param lock_name: Название блокировки.
        :param timeout: TTL блокировки в секундах (по умолчанию 10).
        :return: True если блокировка захвачена, False иначе.
        """
        try:
            token = str(uuid.uuid4())
            result = bool(self.redis_client.set(lock_name, token, nx=True, ex=timeout))
            if result:
                self._lock_tokens[lock_name] = token
                self.logger.debug(f"Блокировка захвачена: {lock_name}")
            return result
        except Exception as e:
            self.logger.error(f"Ошибка захвата блокировки: {e}")
            return False

    def release_lock(self, lock_name: str) -> None:
        """
        Освобождает блокировку только если текущий процесс является её владельцем.

        :param lock_name: Название блокировки.
        """
        try:
            token = self._lock_tokens.pop(lock_name, None)
            if token is None:
                return
            self.redis_client.eval(self._RELEASE_SCRIPT, 1, lock_name, token)
            self.logger.debug(f"Блокировка освобождена: {lock_name}")
        except Exception as e:
            self.logger.error(f"Ошибка освобождения блокировки: {e}")

    def publish_signal(self, signal_data: Dict[str, Any]) -> None:
        """
        Публикует торговый сигнал через Redis Pub/Sub.

        Канал: "trading_signals".

        :param signal_data: Данные сигнала для публикации.
        """
        try:
            self.redis_client.publish(
                "trading_signals",
                json.dumps(signal_data),
            )
            self.logger.debug("Сигнал опубликован")
        except Exception as e:
            self.logger.error(f"Ошибка публикации сигнала: {e}")

    def update_performance_stats(self, stats: Dict[str, Any]) -> None:
        """
        Обновляет статистику производительности в Redis.

        TTL — 24 часа. Ключ: "performance_stats".

        :param stats: Словарь со статистикой.
        """
        try:
            self.redis_client.set(
                "performance_stats",
                json.dumps(stats),
                ex=86400,
            )
            self.logger.debug(f"Статистика обновлена: {stats}")
        except Exception as e:
            self.logger.error(f"Ошибка обновления статистики: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику производительности из Redis.

        :return: Словарь статистики или пустой dict если нет данных.
        """
        try:
            data = self.redis_client.get("performance_stats")
            if data:
                return json.loads(data.decode("utf-8"))
            return {}
        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            return {}
