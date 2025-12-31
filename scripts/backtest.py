import asyncio
import logging
from datetime import datetime

from config import Config
from src.data_loader import DataLoader
from src.redis_client import RedisClient
from src.risk_management import RiskManager
from src.strategies import TradingStrategy

logger = logging.getLogger(__name__)


class Backtester:
    """
    Класс для проведения бэктестинга торговых стратегий на исторических данных.
    Включает загрузку данных, выполнение стратегии, расчет рисков и сохранение результатов.
    """

    def __init__(self):
        """
        Инициализирует объект бэктестера с необходимыми компонентами.
        """
        self.redis = RedisClient()
        self.data_loader = DataLoader()
        self.strategy = None
        self.risk_manager = RiskManager(Config.INITIAL_BALANCE)

    async def initialize(self):
        """
        Инициализирует бэктестер: настраивает загрузчик данных и стратегию.
        """
        await self.data_loader.initialize(Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET)
        self.strategy = TradingStrategy(Config.DEFAULT_STRATEGY)

    async def run_backtest(self, symbol: str, timeframe: str, days: int = 30):
        """
        Запускает бэктестинг для заданного символа, таймфрейма и периода.

        :param symbol: Торговая пара (например, "BTCUSDT").
        :param timeframe: Таймфрейм данных (например, "1h").
        :param days: Количество дней для бэктестинга (по умолчанию 30).
        :return: Словарь с результатами бэктестинга, включая метрики производительности и сделки.
        """
        # Check if results are cached
        backtest_key = f"backtest:{symbol}:{timeframe}:{days}"
        cached_result = self.redis.load_backtest_result(backtest_key)

        if cached_result:
            logger.info("Using cached backtest results")
            return cached_result

        try:
            # Load historical data
            data = await self.data_loader.get_historical_data(symbol, timeframe, days)
            data = self.data_loader.calculate_technical_indicators(data)

            # Initialize variables
            balance = Config.INITIAL_BALANCE
            position = 0
            trades = []
            equity_curve = []

            # Run backtest
            for i in range(26, len(data)):  # Start after enough data for indicators
                current_data = data.iloc[: i + 1]
                current_price = data["close"].iloc[i]

                # Get signal
                signal = await self.strategy.get_signal(current_data)

                # Execute trade logic
                if signal["action"] == "buy" and position == 0:
                    # Buy logic
                    position_size = balance / current_price
                    position = position_size
                    balance = 0
                    trades.append(
                        {
                            "type": "buy",
                            "price": current_price,
                            "size": position_size,
                            "timestamp": data.index[i],
                        }
                    )

                elif signal["action"] == "sell" and position > 0:
                    # Sell logic
                    balance = position * current_price
                    trades.append(
                        {
                            "type": "sell",
                            "price": current_price,
                            "size": position,
                            "timestamp": data.index[i],
                        }
                    )
                    position = 0

                # Update equity curve
                current_equity = balance + (
                    position * current_price if position > 0 else 0
                )
                equity_curve.append(current_equity)

            # Calculate performance metrics
            result = self._calculate_performance(equity_curve, trades)

            # Save results to Redis
            self.redis.save_backtest_result(backtest_key, result)

            return result

        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            raise

    def _calculate_performance(self, equity_curve, trades):
        """
        Рассчитывает метрики производительности на основе кривой капитала и сделок.

        :param equity_curve: Список значений капитала на каждом шаге.
        :param trades: Список выполненных сделок.
        :return: Словарь с метриками производительности.
        """
        final_balance = equity_curve[-1] if equity_curve else Config.INITIAL_BALANCE
        total_return = (
            (final_balance - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE * 100
        )

        return {
            "initial_balance": Config.INITIAL_BALANCE,
            "final_balance": final_balance,
            "total_return": total_return,
            "total_trades": len(trades),
            "win_rate": self._calculate_win_rate(trades),
            "max_drawdown": self._calculate_max_drawdown(equity_curve),
            "sharpe_ratio": 0,  # Would be calculated
            "trades": trades,
            "equity_curve": equity_curve,
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_win_rate(self, trades):
        """
        Рассчитывает процент выигрышных сделок.

        :param trades: Список сделок.
        :return: Процент выигрышных сделок (от 0 до 1).
        """
        if len(trades) < 2:
            return 0
        return 0.5  # Placeholder

    def _calculate_max_drawdown(self, equity_curve):
        """
        Рассчитывает максимальную просадку капитала.

        :param equity_curve: Список значений капитала.
        :return: Максимальная просадка в процентах.
        """
        if not equity_curve:
            return 0
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd


async def main():
    """
    Основная функция для запуска бэктестинга и вывода результатов.
    """
    backtester = Backtester()
    await backtester.initialize()

    result = await backtester.run_backtest(Config.SYMBOL, Config.TIMEFRAME, 90)

    print("Backtest Results:")
    print(f"Initial Balance: ${result['initial_balance']:,.2f}")
    print(f"Final Balance: ${result['final_balance']:,.2f}")
    print(f"Total Return: {result['total_return']:.2f}%")
    print(f"Total Trades: {result['total_trades']}")
    print(f"Win Rate: {result['win_rate']:.2f}")
    print(f"Max Drawdown: {result['max_drawdown']:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
