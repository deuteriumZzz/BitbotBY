import redis
import json
import pickle
import logging
from datetime import datetime, timedelta
from config import Config

class RedisClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        try:
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                password=Config.REDIS_PASSWORD,
                db=0,
                decode_responses=False
            )
            # Test connection
            self.redis_client.ping()
            logging.info("Redis connection established successfully")
        except Exception as e:
            logging.error(f"Failed to connect to Redis: {e}")
            raise
    
    def save_trading_state(self, symbol, state_data):
        """Save trading state for symbol"""
        key = f"trading_state:{symbol}"
        self.redis_client.setex(key, Config.STATE_CACHE_TTL, pickle.dumps(state_data))
    
    def load_trading_state(self, symbol):
        """Load trading state for symbol"""
        key = f"trading_state:{symbol}"
        data = self.redis_client.get(key)
        return pickle.loads(data) if data else None
    
    def save_market_data(self, symbol, timeframe, data):
        """Cache market data"""
        key = f"market_data:{symbol}:{timeframe}"
        self.redis_client.setex(key, Config.MARKET_DATA_CACHE_TTL, pickle.dumps(data))
    
    def load_market_data(self, symbol, timeframe):
        """Load cached market data"""
        key = f"market_data:{symbol}:{timeframe}"
        data = self.redis_client.get(key)
        return pickle.loads(data) if data else None
    
    def save_model(self, strategy_name, model_data):
        """Save model to Redis"""
        key = f"model:{strategy_name}"
        self.redis_client.setex(key, Config.MODEL_CACHE_TTL, pickle.dumps(model_data))
    
    def load_model(self, strategy_name):
        """Load model from Redis"""
        key = f"model:{strategy_name}"
        data = self.redis_client.get(key)
        return pickle.loads(data) if data else None
    
    def acquire_lock(self, lock_name, timeout=10):
        """Acquire distributed lock"""
        lock = self.redis_client.lock(f"lock:{lock_name}", timeout=timeout)
        return lock.acquire(blocking=False)
    
    def release_lock(self, lock_name):
        """Release distributed lock"""
        lock = self.redis_client.lock(f"lock:{lock_name}")
        lock.release()
    
    def publish_signal(self, signal_data):
        """Publish trading signal to Redis Pub/Sub"""
        self.redis_client.publish("trading_signals", json.dumps(signal_data))
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = self.redis_client.get("performance_stats")
        return json.loads(stats) if stats else {}
    
    def update_performance_stats(self, stats):
        """Update performance statistics"""
        self.redis_client.setex("performance_stats", Config.STATS_UPDATE_INTERVAL, json.dumps(stats))
    
    def save_backtest_result(self, strategy_name, result):
        """Save backtest result"""
        key = f"backtest:{strategy_name}"
        self.redis_client.setex(key, 86400, pickle.dumps(result))
    
    def load_backtest_result(self, strategy_name):
        """Load backtest result"""
        key = f"backtest:{strategy_name}"
        data = self.redis_client.get(key)
        return pickle.loads(data) if data else None
