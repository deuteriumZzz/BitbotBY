import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Bybit API settings
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
    BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')
    
    # Redis settings
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
    
    # Trading settings
    SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')
    TIMEFRAME = os.getenv('TIMEFRAME', '1h')
    INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 1000))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))
    
    # Strategy settings
    DEFAULT_STRATEGY = os.getenv('DEFAULT_STRATEGY', 'ema_crossover')
    
    # Cache settings
    MARKET_DATA_CACHE_TTL = 300  # 5 minutes
    MODEL_CACHE_TTL = 604800     # 7 days
    STATE_CACHE_TTL = 86400      # 24 hours
    
    # Performance monitoring
    STATS_UPDATE_INTERVAL = 3600  # 1 hour
