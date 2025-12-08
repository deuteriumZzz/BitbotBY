#!/usr/bin/env python3
"""
Main entry point for the Crypto Trading Bot
"""

import asyncio
import signal
import sys
import os
import argparse
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, config
from utils.logger import logger, setup_error_handler, LoggerAdapter
from src.data_fetcher import DataFetcher
from src.trading_engine import TradingEngine
from src.portfolio_manager import PortfolioManager
from src.rl_agent import RLAgent
from src.risk_manager import RiskManager

class CryptoTradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.is_running = False
        self.shutdown_requested = False
        
        # Initialize logger with context
        self.logger = LoggerAdapter(logger, {
            'component': 'main',
            'mode': config.TRADING_MODE
        })
        
        # Initialize components
        self.data_fetcher = DataFetcher(config)
        self.portfolio_manager = PortfolioManager(config.INITIAL_BALANCE)
        self.risk_manager = RiskManager(config.INITIAL_BALANCE, config.RISK_PER_TRADE)
        self.rl_agent = RLAgent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE
        )
        self.trading_engine = TradingEngine(
            config=config,
            data_fetcher=self.data_fetcher,
            portfolio_manager=self.portfolio_manager,
            risk_manager=self.risk_manager,
            rl_agent=self.rl_agent
        )
        
        # Load model if exists
        self._load_model()
        
        self.logger.info("Crypto Trading Bot initialized")

    def _load_model(self):
        """Load trained RL model if available"""
        model_path = os.path.join(self.config.MODELS_DIR, "rl_model.pth")
        if os.path.exists(model_path):
            try:
                self.rl_agent.load_model(model_path)
                self.logger.info(f"Loaded trained model from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
        else:
            self.logger.info("No trained model found, starting fresh")

    async def run(self):
        """Main trading loop"""
        self.is_running = True
        self.logger.info("Starting trading bot...")
        
        try:
            # Initial data fetch
            await self._initial_setup()
            
            # Main trading loop
            while not self.shutdown_requested:
                await self._trading_iteration()
                await asyncio.sleep(self.config.TRADING_INTERVAL)
                
        except Exception as e:
            self.logger.critical(f"Trading bot crashed: {e}", exc_info=True)
        finally:
            self.is_running = False
            await self.shutdown()

    async def _initial_setup(self):
        """Perform initial setup and data loading"""
        self.logger.info("Performing initial setup...")
        
        # Fetch initial market data
        for symbol in self.config.TRADING_SYMBOLS:
            data = await self.data_fetcher.fetch_market_data(symbol)
            if data:
                self.logger.info(f"Initial data for {symbol}: {data['close']:.2f}")
        
        self.logger.info("Initial setup completed")

    async def _trading_iteration(self):
        """Execute one trading iteration"""
        try:
            # Fetch latest market data
            market_data = {}
            for symbol in self.config.TRADING_SYMBOLS:
                data = await self.data_fetcher.fetch_market_data(symbol)
                if data:
                    market_data[symbol] = data
            
            if not market_data:
                self.logger.warning("No market data available")
                return
            
            # Get portfolio state
            portfolio_state = await self.portfolio_manager.get_portfolio_state()
            
            # Make trading decisions
            trading_decisions = await self.trading_engine.make_trading_decisions(
                market_data, portfolio_state
            )
            
            # Execute trades
            if trading_decisions:
                await self.trading_engine.execute_trades(trading_decisions)
            
            # Save state periodically
            if datetime.now().minute % 30 == 0:  # Every 30 minutes
                await self._save_state()
                
        except Exception as e:
            self.logger.error(f"Error in trading iteration: {e}", exc_info=True)

    async def _save_state(self):
        """Save bot state"""
        try:
            # Save RL model
            model_path = os.path.join(self.config.MODELS_DIR, "rl_model.pth")
            self.rl_agent.save_model(model_path)
            
            self.logger.debug("Bot state saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down trading bot...")
        
        try:
            # Save final state
            await self._save_state()
            
            # Close connections
            await self.data_fetcher.close()
            
            self.logger.info("Trading bot shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def handle_signal(self, signal_name):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal: {signal_name}")
        self.shutdown_requested = True

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config file')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'],
                       default='paper', help='Trading mode')
    parser.add_argument('--symbols', nargs='+', 
                       default=['BTC/USDT', 'ETH/USDT'],
                       help='Trading symbols')
    parser.add_argument('--interval', type=int, default=60,
                       help='Trading interval in seconds')
    
    args = parser.parse_args()
    
    # Use global config instance
    config_instance = config
    
    # Override config with command line arguments
    if args.config != 'config.json':
        # Если нужна загрузка из файла, нужно реализовать метод load_config
        pass
    
    config_instance.TRADING_MODE = args.mode
    config_instance.TRADING_SYMBOLS = args.symbols
    config_instance.TRADING_INTERVAL = args.interval
    
    # Setup global error handling (уже сделано в logger.py)
    
    logger.info(f"Starting Crypto Trading Bot in {config_instance.TRADING_MODE} mode")
    logger.info(f"Trading symbols: {config_instance.TRADING_SYMBOLS}")
    logger.info(f"Trading interval: {config_instance.TRADING_INTERVAL}s")
    
    # Create and run bot
    bot = CryptoTradingBot(config_instance)
    
    # Setup signal handlers
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda s, f: bot.handle_signal(signal.Signals(s).name))
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        bot.handle_signal("SIGINT")
    finally:
        if bot.is_running:
            await bot.shutdown()

if __name__ == "__main__":
    # Run the bot
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
