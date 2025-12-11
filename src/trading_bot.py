import asyncio
import logging
from typing import Dict, List, Optional

from config import config
from src.bybit_api import BybitAPI
from src.data_fetcher import DataFetcher
from src.news_analyzer import NewsAnalyzer
from src.portfolio_manager import PortfolioManager
from src.risk_management import RiskManager
from src.strategies import TradingStrategy


class TradingBot:
    """Main trading bot that orchestrates all components"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False

        # Initialize components
        self.api = BybitAPI()
        self.data_fetcher = DataFetcher(config)
        self.portfolio_manager = PortfolioManager(config.INITIAL_BALANCE)
        self.risk_manager = RiskManager(config.INITIAL_BALANCE, config.RISK_PER_TRADE)
        self.news_analyzer = NewsAnalyzer()

        # Strategies
        self.strategies: Dict[str, TradingStrategy] = {}

    async def initialize(self):
        """Initialize all bot components"""
        try:
            # Initialize API
            await self.api.initialize(config.BYBIT_API_KEY, config.BYBIT_API_SECRET)

            # Initialize data fetcher
            await self.data_fetcher.initialize()

            # Initialize strategies
            strategy_names = ["ema_crossover", "rsi_momentum"]
            for name in strategy_names:
                strategy = TradingStrategy(name)
                await strategy.initialize()
                self.strategies[name] = strategy

            self.logger.info("Trading bot initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize trading bot: {e}")
            return False

    async def fetch_market_data(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        """Fetch market data for all symbols"""
        market_data = {}

        for symbol in symbols:
            data = await self.data_fetcher.fetch_market_data(symbol)
            market_data[symbol] = data

        return market_data

    async def analyze_market(self, market_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Analyze market and generate trading signals"""
        signals = {}

        for symbol, data in market_data.items():
            if data is None:
                continue

            # Convert data to DataFrame for strategy analysis
            df = await self._prepare_data_for_analysis(data)

            # Get signals from all strategies
            strategy_signals = {}
            for strategy_name, strategy in self.strategies.items():
                signal = await strategy.get_signal(df)
                strategy_signals[strategy_name] = signal

            signals[symbol] = strategy_signals

        return signals

    async def _prepare_data_for_analysis(self, data: Dict) -> "pd.DataFrame":
        """Prepare data for analysis"""
        # Convert your data to DataFrame format expected by strategies
        import pandas as pd

        # This is an example - adapt to your actual data format
        df = pd.DataFrame([data])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        return df

    async def execute_trading_decisions(
        self, signals: Dict[str, Dict], market_data: Dict[str, Dict]
    ):
        """Execute trading decisions based on signals"""
        for symbol, strategy_signals in signals.items():
            if symbol not in market_data:
                continue

            current_price = market_data[symbol]["price"]

            # Aggregate signals from all strategies
            final_signal = await self._aggregate_signals(strategy_signals)

            if final_signal["action"] == "hold":
                continue

            # Risk validation
            if not await self.risk_manager.validate_signal(
                final_signal, market_data[symbol]
            ):
                self.logger.warning(f"Signal for {symbol} failed risk validation")
                continue

            # Calculate position size
            stop_loss = await self.risk_manager.calculate_stop_loss(
                current_price, final_signal
            )
            position_size = await self.risk_manager.calculate_position_size(
                self.portfolio_manager.current_balance, current_price, stop_loss
            )

            if position_size <= 0:
                continue

            # Execute order
            success = await self.portfolio_manager.update_portfolio(
                symbol, final_signal["action"], position_size, current_price
            )

            if success:
                self.logger.info(
                    f"Executed {final_signal['action']} order for {symbol}, size: {position_size}"
                )
            else:
                self.logger.warning(f"Failed to execute order for {symbol}")

    async def _aggregate_signals(self, strategy_signals: Dict[str, Dict]) -> Dict:
        """Aggregate signals from different strategies"""
        # Simple aggregation logic - can be made more sophisticated
        buy_signals = 0
        sell_signals = 0
        total_confidence = 0

        for signal in strategy_signals.values():
            if signal["action"] == "buy":
                buy_signals += 1
                total_confidence += signal["confidence"]
            elif signal["action"] == "sell":
                sell_signals += 1
                total_confidence += signal["confidence"]

        if buy_signals > sell_signals and buy_signals >= 1:
            return {
                "action": "buy",
                "confidence": total_confidence / len(strategy_signals),
            }
        elif sell_signals > buy_signals and sell_signals >= 1:
            return {
                "action": "sell",
                "confidence": total_confidence / len(strategy_signals),
            }
        else:
            return {"action": "hold", "confidence": 0.5}

    async def monitor_portfolio(self):
        """Monitor and rebalance portfolio"""
        try:
            # Get current prices
            symbols = list(self.portfolio_manager.positions.keys())
            if symbols:
                market_data = await self.fetch_market_data(symbols)
                current_prices = {
                    symbol: data["price"]
                    for symbol, data in market_data.items()
                    if data
                }

                # Calculate portfolio value
                portfolio_value = await self.portfolio_manager.get_portfolio_value(
                    current_prices
                )
                self.logger.info(f"Current portfolio value: {portfolio_value:.2f} USDT")

                # Add rebalancing logic here

        except Exception as e:
            self.logger.error(f"Error in portfolio monitoring: {e}")

    async def run(
        self, symbols: List[str] = ["BTCUSDT", "ETHUSDT"], interval: int = 300
    ):
        """Main trading loop"""
        self.is_running = True

        try:
            while self.is_running:
                self.logger.info("Starting trading cycle...")

                # 1. Fetch market data
                market_data = await self.fetch_market_data(symbols)

                # 2. Analyze market and generate signals
                signals = await self.analyze_market(market_data)

                # 3. Execute trading decisions
                await self.execute_trading_decisions(signals, market_data)

                # 4. Monitor portfolio
                await self.monitor_portfolio()

                # 5. Analyze news (asynchronously)
                news_sentiment = await self.news_analyzer.analyze_news_async()
                if abs(news_sentiment) > 0.3:  # Strong sentiment
                    self.logger.info(
                        f"Strong news sentiment detected: {news_sentiment:.2f}"
                    )

                self.logger.info(
                    f"Trading cycle completed. Waiting {interval} seconds..."
                )
                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown"""
        self.is_running = False
        await self.api.close()
        await self.data_fetcher.close()
        self.logger.info("Trading bot shut down successfully")


async def main():
    """Main entry point"""
    from config import config

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    bot = TradingBot()

    if await bot.initialize():
        try:
            await bot.run(
                symbols=config.TRADING_SYMBOLS, interval=config.TRADING_INTERVAL
            )
        except Exception as e:
            logging.error(f"Bot crashed: {e}")
    else:
        logging.error("Failed to initialize trading bot")


if __name__ == "__main__":
    asyncio.run(main())
