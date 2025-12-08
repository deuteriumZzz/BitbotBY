import asyncio
import logging
from src.data_loader import DataLoader
from src.strategies import TradingStrategy
from src.redis_client import RedisClient
from config import Config

logger = logging.getLogger(__name__)

async def train_strategy_model(strategy_name: str):
    """Train model for specific strategy"""
    try:
        data_loader = DataLoader()
        strategy = TradingStrategy(strategy_name)
        redis = RedisClient()
        
        await data_loader.initialize(Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET)
        
        # Load training data
        data = await data_loader.get_historical_data(
            Config.SYMBOL, 
            Config.TIMEFRAME, 
            days=180  # 6 months of data
        )
        
        # Calculate indicators
        data = data_loader.calculate_technical_indicators(data)
        
        # Train model
        success = await strategy.train_model(data)
        
        if success:
            logger.info(f"Successfully trained model for {strategy_name}")
            # Verify model was saved to Redis
            model_data = redis.load_model(strategy_name)
            if model_data:
                logger.info(f"Model verified in Redis: {model_data['trained_at']}")
            else:
                logger.warning("Model not found in Redis after training")
        else:
            logger.error(f"Failed to train model for {strategy_name}")
        
        await data_loader.close()
        
    except Exception as e:
        logger.error(f"Error training model: {e}")

async def main():
    """Main training function"""
    strategies_to_train = ['ema_crossover', 'rsi_momentum']
    
    for strategy in strategies_to_train:
        logger.info(f"Training model for {strategy}")
        await train_strategy_model(strategy)
    
    logger.info("All models trained successfully")

if __name__ == "__main__":
    asyncio.run(main())
