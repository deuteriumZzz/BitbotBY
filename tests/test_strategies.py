import numpy as np
import pytest
from rl_agent import RLAgent
from trading_engine import TradingEngine


class TestStrategies:
    """Тесты торговых стратегий и алгоритмов"""

    @pytest.fixture
    def trading_engine(self):
        """Фикстура для создания торгового движка"""
        config = {
            "TRADING_MODE": "paper",
            "TRADING_STRATEGY": "mean_reversion",
            "STRATEGY_PARAMS": {
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "take_profit": 0.05,
                "stop_loss": 0.03,
            },
        }
        return TradingEngine(config)

    @pytest.fixture
    def rl_agent(self):
        """Фикстура для создания RL агента"""
        config = {
            "MODEL_PATH": "models/test_model.h5",
            "STATE_SIZE": 10,
            "ACTION_SIZE": 4,
            "LEARNING_RATE": 0.001,
        }
        return RLAgent(config)

    def test_mean_reversion_strategy(self, trading_engine):
        """Тест стратегии возврата к среднему"""
        test_cases = [
            # (rsi, expected_action, expected_signal)
            (25, "BUY", "strong_buy"),  # Сильный перепроданность
            (35, "BUY", "weak_buy"),  # Слабый перепроданность
            (55, "HOLD", "neutral"),  # Нейтральная зона
            (65, "SELL", "weak_sell"),  # Слабый перекупленность
            (75, "SELL", "strong_sell"),  # Сильный перекупленность
        ]

        for rsi, expected_action, expected_signal in test_cases:
            market_data = {
                "symbol": "BTC/USDT",
                "price": 50000.0,
                "indicators": {"rsi": rsi},
            }

            decision = trading_engine._apply_mean_reversion_strategy(market_data)

            assert decision["action"] == expected_action
            assert decision["signal_strength"] == expected_signal

    def test_breakout_strategy(self, trading_engine):
        """Тест стратегии пробоя"""
        market_data = {
            "symbol": "BTC/USDT",
            "price": 50000.0,
            "high_24h": 52000.0,
            "low_24h": 48000.0,
            "volume": 1500.0,
            "avg_volume": 1000.0,
        }

        # Тест пробоя сопротивления
        market_data["price"] = 52100.0  # Выше high_24h
        decision = trading_engine._apply_breakout_strategy(market_data)
        assert decision["action"] == "BUY"

        # Тест пробоя поддержки
        market_data["price"] = 47900.0  # Ниже low_24h
        decision = trading_engine._apply_breakout_strategy(market_data)
        assert decision["action"] == "SELL"

        # Тест отсутствия пробоя
        market_data["price"] = 51000.0  # В пределах диапазона
        decision = trading_engine._apply_breakout_strategy(market_data)
        assert decision["action"] == "HOLD"

    def test_trend_following_strategy(self, trading_engine):
        """Тест стратегии следования за трендом"""
        market_data = {
            "symbol": "BTC/USDT",
            "price": 50000.0,
            "ma_fast": 50500.0,
            "ma_slow": 49500.0,
            "macd": 0.02,
            "macd_signal": 0.01,
        }

        # Бычий тренд (быстрая MA выше медленной)
        decision = trading_engine._apply_trend_following_strategy(market_data)
        assert decision["action"] == "BUY"

        # Медвежий тренд (быстрая MA ниже медленной)
        market_data["ma_fast"] = 48500.0
        market_data["ma_slow"] = 49500.0
        decision = trading_engine._apply_trend_following_strategy(market_data)
        assert decision["action"] == "SELL"

        # Нет четкого тренда
        market_data["ma_fast"] = 49500.0
        market_data["ma_slow"] = 49500.0
        decision = trading_engine._apply_trend_following_strategy(market_data)
        assert decision["action"] == "HOLD"

    @pytest.mark.asyncio
    async def test_rl_agent_strategy(self, rl_agent):
        """Тест стратегии на основе RL"""
        # Тестовое состояние рынка + портфеля
        state = np.array(
            [
                50000.0,  # price
                1000.0,  # volume
                60.0,  # rsi
                0.02,  # macd
                10000.0,  # balance
                0.0,  # current position
                0.0,  # pnl
                0.5,  # volatility
                1.2,  # volume ratio
                0.0,  # market sentiment
            ]
        )

        # Получаем действие от агента
        action = await rl_agent.choose_action(state)

        # Проверяем что действие валидно
        assert action in [0, 1, 2, 3]  # BUY, SELL, HOLD, CLOSE

        # Тестируем обучение на примере
        next_state = np.array(
            [
                51000.0,  # price increased
                1200.0,  # volume increased
                65.0,  # rsi increased
                0.03,  # macd increased
                9000.0,  # balance decreased (after buy)
                0.1,  # position opened
                1000.0,  # positive pnl
                0.6,  # volatility increased
                1.5,  # volume ratio increased
                0.2,  # market sentiment improved
            ]
        )

        # Сохраняем опыт и обучаем
        await rl_agent.remember(state, action, 1.0, next_state, False)
        loss = await rl_agent.replay(32)

        # Проверяем что обучение прошло
        assert isinstance(loss, float)

    def test_risk_adjusted_strategy(self, trading_engine):
        """Тест стратегии с учетом рисков"""
        market_data = {
            "symbol": "BTC/USDT",
            "price": 50000.0,
            "volatility": 0.02,  # Низкая волатильность
            "liquidity": 1000.0,
        }

        portfolio_state = {"balance": 10000.0, "positions": {}, "total_risk": 0.0}

        # Низкий риск - можно торговать
        decision = trading_engine._apply_risk_adjusted_strategy(
            market_data, portfolio_state
        )
        assert decision["action"] != "HOLD"
        assert decision["position_size"] <= 0.02  # 2% риска на сделку

        # Высокая волатильность - уменьшаем размер позиции
        market_data["volatility"] = 0.08  # Высокая волатильность
        decision = trading_engine._apply_risk_adjusted_strategy(
            market_data, portfolio_state
        )
        assert decision["position_size"] <= 0.005  # Меньший размер позиции

    def test_strategy_selection(self, trading_engine):
        """Тест выбора стратегии based on market conditions"""
        market_conditions = [
            # (volatility, trend_strength, expected_strategy)
            (0.01, 0.8, "trend_following"),  # Сильный тренд, низкая волатильность
            (0.05, 0.2, "mean_reversion"),  # Слабая волатильность, нет тренда
            (0.10, 0.6, "breakout"),  # Высокая волатильность, умеренный тренд
            (0.15, 0.3, "risk_adjusted"),  # Очень высокая волатильность
        ]

        for volatility, trend_strength, expected_strategy in market_conditions:
            market_data = {"volatility": volatility, "trend_strength": trend_strength}

            selected_strategy = trading_engine._select_strategy(market_data)
            assert selected_strategy == expected_strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
