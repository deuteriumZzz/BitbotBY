import logging
from typing import Any, Dict

import pandas as pd

from .redis_client import RedisClient


class TradingStrategy:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.redis = RedisClient()
        self.model = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize strategy"""
        await self.load_model()

    async def load_model(self):
        """Load model from Redis or train new one"""
        model_data = self.redis.load_model(self.strategy_name)

        if model_data:
            self.model = model_data
            self.logger.info(f"Loaded model for {self.strategy_name} from Redis")
            return True

        self.logger.info(f"No model found in Redis for {self.strategy_name}")
        return False

    async def train_model(self, data: pd.DataFrame):
        """Train model and save to Redis"""
        try:
            # Simple model training example
            trained_model = {
                "parameters": {"ema_short": 12, "ema_long": 26},
                "trained_at": pd.Timestamp.now().isoformat(),
                "performance": {"accuracy": 0.85},
            }

            # Save model to Redis
            self.redis.save_model(self.strategy_name, trained_model)
            self.model = trained_model

            self.logger.info(f"Model trained and saved for {self.strategy_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return False

    def _generate_signal(
        self, data: pd.DataFrame, current_state: Dict
    ) -> Dict[str, Any]:
        """Generate trading signal based on strategy"""
        if self.strategy_name == "ema_crossover":
            return self._ema_crossover_strategy(data, current_state)
        elif self.strategy_name == "rsi_momentum":
            return self._rsi_momentum_strategy(data, current_state)
        else:
            return {"action": "hold", "confidence": 0.0}

    def _ema_crossover_strategy(
        self, data: pd.DataFrame, current_state: Dict
    ) -> Dict[str, Any]:
        """EMA Crossover strategy"""
        last_row = data.iloc[-1]

        if last_row["ema_short"] > last_row["ema_long"]:
            return {"action": "buy", "confidence": 0.8, "price": last_row["close"]}
        elif last_row["ema_short"] < last_row["ema_long"]:
            return {"action": "sell", "confidence": 0.7, "price": last_row["close"]}
        else:
            return {"action": "hold", "confidence": 0.5}

    def _rsi_momentum_strategy(
        self, data: pd.DataFrame, current_state: Dict
    ) -> Dict[str, Any]:
        """RSI Momentum strategy"""
        last_row = data.iloc[-1]

        if last_row["rsi"] < 30:
            return {"action": "buy", "confidence": 0.75, "price": last_row["close"]}
        elif last_row["rsi"] > 70:
            return {"action": "sell", "confidence": 0.75, "price": last_row["close"]}
        else:
            return {"action": "hold", "confidence": 0.6}

    async def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal with state management"""
        current_state = self.redis.load_trading_state(self.strategy_name) or {}

        # Generate signal
        signal = self._generate_signal(data, current_state)

        # Update state
        new_state = {
            "last_signal": signal,
            "timestamp": pd.Timestamp.now().isoformat(),
            "market_conditions": {
                "price": data["close"].iloc[-1],
                "volume": data["volume"].iloc[-1],
                "volatility": data["close"].std(),
            },
        }

        # Save state to Redis
        self.redis.save_trading_state(self.strategy_name, new_state)

        # Publish signal via Redis Pub/Sub
        self.redis.publish_signal(
            {
                "strategy": self.strategy_name,
                "signal": signal,
                "timestamp": new_state["timestamp"],
            }
        )

        return signal

    async def update_strategy_params(self, new_params: Dict[str, Any]):
        """Update strategy parameters"""
        if self.model:
            self.model["parameters"].update(new_params)
            self.redis.save_model(self.strategy_name, self.model)
            self.logger.info(f"Updated parameters for {self.strategy_name}")
