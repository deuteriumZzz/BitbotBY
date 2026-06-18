"""
Клиент Redis для кэширования рыночных данных, состояний торговли и блокировок.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
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
        self._test_connection()

    def _test_connection(self) -> None:
        """
        Проверяет соединение с Redis через ping.

        :raises Exception: Если подключение не удалось.
        """
        try:
            self.redis_client.ping()
            kwargs = self.redis_client.connection_pool.connection_kwargs
            self.logger.info(
                f"Подключено к Redis: " f"{kwargs['host']}:{kwargs['port']}"
            )
        except Exception as e:
            self.logger.error(f"Не удалось подключиться к Redis: {e}")
            raise

    def save_market_data(self, key: str, data: pd.DataFrame) -> None:
        """
        Сохраняет рыночные данные (DataFrame) в Redis через pickle.

        TTL — 5 минут.

        :param key: Ключ для сохранения.
        :param data: DataFrame с рыночными данными.
        """
        try:
            serialized = pickle.dumps(data)
            self.redis_client.setex(key, 300, serialized)
            self.logger.debug(f"Сохранены данные: {key}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения рыночных данных: {e}")

    def load_market_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Загружает рыночные данные из Redis.

        :param key: Ключ для загрузки.
        :return: DataFrame или None если ключ не найден.
        """
        try:
            data = self.redis_client.get(key)
            if data:
                self.logger.debug(f"Загружены данные: {key}")
                return pickle.loads(data)
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
        """
        Сохраняет данные модели в Redis через pickle.

        TTL — 7 дней. Ключ: "model:{strategy_name}".

        :param strategy_name: Название стратегии.
        :param model_data: Данные модели.
        """
        try:
            serialized = pickle.dumps(model_data)
            self.redis_client.setex(f"model:{strategy_name}", 604800, serialized)
            self.logger.debug(f"Сохранена модель: {strategy_name}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели: {e}")

    def load_model(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Загружает данные модели из Redis.

        :param strategy_name: Название стратегии.
        :return: Данные модели или None если не найдены.
        """
        try:
            model_data = self.redis_client.get(f"model:{strategy_name}")
            if model_data:
                self.logger.debug(f"Загружена модель: {strategy_name}")
                return pickle.loads(model_data)
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
        return None

    def acquire_lock(self, lock_name: str, timeout: int = 10) -> bool:
        """
        Захватывает распределённую блокировку через SET NX.

        :param lock_name: Название блокировки.
        :param timeout: TTL блокировки в секундах (по умолчанию 10).
        :return: True если блокировка захвачена, False иначе.
        """
        try:
            result = bool(
                self.redis_client.set(lock_name, "locked", nx=True, ex=timeout)
            )
            if result:
                self.logger.debug(f"Блокировка захвачена: {lock_name}")
            return result
        except Exception as e:
            self.logger.error(f"Ошибка захвата блокировки: {e}")
            return False

    def release_lock(self, lock_name: str) -> None:
        """
        Освобождает распределённую блокировку.

        :param lock_name: Название блокировки.
        """
        try:
            self.redis_client.delete(lock_name)
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
