import logging
from typing import Any, Dict

import pandas as pd

from .redis_client import RedisClient


class TradingStrategy:
    """
    Класс для реализации торговых стратегий.
    
    Предоставляет методы для инициализации, загрузки и обучения модели, генерации торговых сигналов
    на основе различных стратегий (например, EMA crossover, RSI momentum). Использует Redis для хранения
    модели и состояния, а также для публикации сигналов. Включает логирование для отслеживания операций.
    """
    
    def __init__(self, strategy_name: str):
        """
        Инициализирует торговую стратегию.
        
        Устанавливает имя стратегии, клиент Redis, модель (изначально None) и логгер.
        
        :param strategy_name: Название стратегии (str).
        """
        self.strategy_name = strategy_name
        self.redis = RedisClient()
        self.model = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """
        Инициализирует стратегию.
        
        Вызывает метод загрузки модели из Redis. Если модель загружена, она готова к использованию.
        """
        await self.load_model()

    async def load_model(self):
        """
        Загружает модель из Redis или отмечает отсутствие модели.
        
        Пытается загрузить модель по имени стратегии. Если модель найдена, присваивает её self.model
        и логирует успех. Если не найдена, логирует отсутствие и возвращает False.
        
        :return: True, если модель загружена, иначе False.
        """
        model_data = self.redis.load_model(self.strategy_name)

        if model_data:
            self.model = model_data
            self.logger.info(f"Loaded model for {self.strategy_name} from Redis")
            return True

        self.logger.info(f"No model found in Redis for {self.strategy_name}")
        return False

    async def train_model(self, data: pd.DataFrame):
        """
        Обучает модель на предоставленных данных и сохраняет в Redis.
        
        Создает простую модель с параметрами (EMA short и long), временем обучения и производительностью.
        Сохраняет модель в Redis под именем стратегии, присваивает self.model и логирует успех.
        В случае ошибки логирует исключение и возвращает False.
        
        :param data: DataFrame с данными для обучения (pd.DataFrame).
        :return: True, если обучение и сохранение успешны, иначе False.
        """
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
        """
        Генерирует торговый сигнал на основе выбранной стратегии.
        
        В зависимости от имени стратегии вызывает соответствующий метод (_ema_crossover_strategy или _rsi_momentum_strategy).
        Если стратегия не распознана, возвращает сигнал "hold" с нулевой уверенностью.
        
        :param data: DataFrame с рыночными данными (pd.DataFrame).
        :param current_state: Текущее состояние стратегии (Dict).
        :return: Словарь с сигналом, содержащий ключи "action", "confidence" и опционально другие (Dict[str, Any]).
        """
        if self.strategy_name == "ema_crossover":
            return self._ema_crossover_strategy(data, current_state)
        elif self.strategy_name == "rsi_momentum":
            return self._rsi_momentum_strategy(data, current_state)
        else:
            return {"action": "hold", "confidence": 0.0}

    def _ema_crossover_strategy(
        self, data: pd.DataFrame, current_state: Dict
    ) -> Dict[str, Any]:
        """
        Реализует стратегию EMA crossover.
        
        Сравнивает короткую EMA с длинной на последней строке данных. Если короткая выше, генерирует сигнал "buy".
        Если ниже, "sell". Иначе "hold". Возвращает сигнал с уверенностью и ценой закрытия.
        
        :param data: DataFrame с рыночными данными, включая колонки "ema_short", "ema_long", "close" (pd.DataFrame).
        :param current_state: Текущее состояние стратегии (Dict, не используется в данной реализации).
        :return: Словарь с сигналом: "action" ("buy", "sell" или "hold"), "confidence" (float) и "price" (float).
        """
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
        """
        Реализует стратегию RSI momentum.
        
        Проверяет значение RSI на последней строке данных. Если RSI < 30, генерирует сигнал "buy".
        Если RSI > 70, "sell". Иначе "hold". Возвращает сигнал с уверенностью и ценой закрытия.
        
        :param data: DataFrame с рыночными данными, включая колонки "rsi", "close" (pd.DataFrame).
        :param current_state: Текущее состояние стратегии (Dict, не используется в данной реализации).
        :return: Словарь с сигналом: "action" ("buy", "sell" или "hold"), "confidence" (float) и "price" (float).
        """
        last_row = data.iloc[-1]

        if last_row["rsi"] < 30:
            return {"action": "buy", "confidence": 0.75, "price": last_row["close"]}
        elif last_row["rsi"] > 70:
            return {"action": "sell", "confidence": 0.75, "price": last_row["close"]}
        else:
            return {"action": "hold", "confidence": 0.6}

    async def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует торговый сигнал с управлением состоянием.
        
        Загружает текущее состояние из Redis, генерирует сигнал, обновляет состояние с новыми данными
        (последний сигнал, timestamp, рыночные условия), сохраняет в Redis и публикует сигнал через Pub/Sub.
        
        :param data: DataFrame с рыночными данными (pd.DataFrame).
        :return: Словарь с сигналом (Dict[str, Any]).
        """
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
        """
        Обновляет параметры стратегии.
        
        Если модель загружена, обновляет её параметры новыми значениями, сохраняет в Redis и логирует обновление.
        
        :param new_params: Словарь с новыми параметрами (Dict[str, Any]).
        """
        if self.model:
            self.model["parameters"].update(new_params)
            self.redis.save_model(self.strategy_name, self.model)
            self.logger.info(f"Updated parameters for {self.strategy_name}")
