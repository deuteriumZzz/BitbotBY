import json
import logging
import os
import pickle
from typing import Any, Dict, Optional

import pandas as pd
import redis


class RedisClient:
    def __init__(self, host=None, port=6379, password=None):
        # Используем переменные окружения или значения по умолчанию
        redis_host = host or os.getenv("REDIS_HOST", "redis")  # имя сервиса в Docker
        redis_port = port or int(os.getenv("REDIS_PORT", 6379))
        redis_password = password or os.getenv("REDIS_PASSWORD")
        
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            password=redis_password, 
            decode_responses=False
        )
        self.logger = logging.getLogger(__name__)
        
        # Проверяем подключение при инициализации
        self._test_connection()

    def _test_connection(self):
        """Test Redis connection"""
        try:
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise

    def save_market_data(self, key: str, data: pd.DataFrame):
        """Save market data to Redis"""
        try:
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(key, 300, serialized_data)  # 5 minutes TTL
            self.logger.debug(f"Saved market data with key: {key}")
        except Exception as e:
            self.logger.error(f"Error saving market data: {e}")

    def load_market_data(self, key: str) -> Optional[pd.DataFrame]:
        """Load market data from Redis"""
        try:
            data = self.redis_client.get(key)
            if data:
                self.logger.debug(f"Loaded market data with key: {key}")
                return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
        return None

    def save_trading_state(self, key: str, state: Dict[str, Any]):
        """Save trading state to Redis"""
        try:
            serialized_state = json.dumps(state)
            self.redis_client.setex(key, 86400, serialized_state)  # 24 hours TTL
            self.logger.debug(f"Saved trading state with key: {key}")
        except Exception as e:
            self.logger.error(f"Error saving trading state: {e}")

    def load_trading_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Load trading state from Redis"""
        try:
            state = self.redis_client.get(key)
            if state:
                self.logger.debug(f"Loaded trading state with key: {key}")
                return json.loads(state)
        except Exception as e:
            self.logger.error(f"Error loading trading state: {e}")
        return None

    def save_model(self, strategy_name: str, model_data: Dict[str, Any]):
        """Save model to Redis"""
        try:
            serialized_model = pickle.dumps(model_data)
            self.redis_client.setex(
                f"model:{strategy_name}", 604800, serialized_model
            )  # 7 days TTL
            self.logger.debug(f"Saved model for strategy: {strategy_name}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    def load_model(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Load model from Redis"""
        try:
            model_data = self.redis_client.get(f"model:{strategy_name}")
            if model_data:
                self.logger.debug(f"Loaded model for strategy: {strategy_name}")
                return pickle.loads(model_data)
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
        return None

    def acquire_lock(self, lock_name: str, timeout: int = 10) -> bool:
        """Acquire distributed lock"""
        try:
            result = bool(self.redis_client.set(lock_name, "locked", nx=True, ex=timeout))
            if result:
                self.logger.debug(f"Acquired lock: {lock_name}")
            return result
        except Exception as e:
            self.logger.error(f"Error acquiring lock: {e}")
            return False

    def release_lock(self, lock_name: str):
        """Release distributed lock"""
        try:
            self.redis_client.delete(lock_name)
            self.logger.debug(f"Released lock: {lock_name}")
        except Exception as e:
            self.logger.error(f"Error releasing lock: {e}")

    def publish_signal(self, signal_data: Dict[str, Any]):
        """Publish signal via Redis Pub/Sub"""
        try:
            self.redis_client.publish("trading_signals", json.dumps(signal_data))
            self.logger.debug("Published trading signal")
        except Exception as e:
            self.logger.error(f"Error publishing signal: {e}")

    def update_performance_stats(self, stats: Dict[str, Any]):
        """Обновляет статистику производительности в Redis."""
        try:
            key = "performance_stats"
            # Сериализуем весь dict как JSON для поддержки вложенных структур
            self.redis_client.set(key, json.dumps(stats), ex=86400)  # TTL 24 часа
            self.logger.debug(f"Updated performance stats: {stats}")
        except Exception as e:
            self.logger.error(f"Error updating performance stats: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Получает статистику производительности из Redis."""
        try:
            key = "performance_stats"
            data = self.redis_client.get(key)
            if data:
                # Декодируем из bytes и парсим JSON
                return json.loads(data.decode('utf-8'))
            return {}
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {}

