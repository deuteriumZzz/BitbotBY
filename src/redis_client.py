import json
import logging
import pickle
from typing import Any, Dict, Optional

import pandas as pd
import redis


class RedisClient:
    def __init__(self, host="localhost", port=6379, password=None):
        self.redis_client = redis.Redis(
            host=host, port=port, password=password, decode_responses=False
        )
        self.logger = logging.getLogger(__name__)

    def save_market_data(self, key: str, data: pd.DataFrame):
        """Save market data to Redis"""
        try:
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(key, 300, serialized_data)  # 5 minutes TTL
        except Exception as e:
            self.logger.error(f"Error saving market data: {e}")

    def load_market_data(self, key: str) -> Optional[pd.DataFrame]:
        """Load market data from Redis"""
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
        return None

    def save_trading_state(self, key: str, state: Dict[str, Any]):
        """Save trading state to Redis"""
        try:
            serialized_state = json.dumps(state)
            self.redis_client.setex(key, 86400, serialized_state)  # 24 hours TTL
        except Exception as e:
            self.logger.error(f"Error saving trading state: {e}")

    def load_trading_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Load trading state from Redis"""
        try:
            state = self.redis_client.get(key)
            if state:
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
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    def load_model(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Load model from Redis"""
        try:
            model_data = self.redis_client.get(f"model:{strategy_name}")
            if model_data:
                return pickle.loads(model_data)
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
        return None

    def acquire_lock(self, lock_name: str, timeout: int = 10) -> bool:
        """Acquire distributed lock"""
        try:
            return bool(self.redis_client.set(lock_name, "locked", nx=True, ex=timeout))
        except Exception as e:
            self.logger.error(f"Error acquiring lock: {e}")
            return False

    def release_lock(self, lock_name: str):
        """Release distributed lock"""
        try:
            self.redis_client.delete(lock_name)
        except Exception as e:
            self.logger.error(f"Error releasing lock: {e}")

    def publish_signal(self, signal_data: Dict[str, Any]):
        """Publish signal via Redis Pub/Sub"""
        try:
            self.redis_client.publish("trading_signals", json.dumps(signal_data))
        except Exception as e:
            self.logger.error(f"Error publishing signal: {e}")
