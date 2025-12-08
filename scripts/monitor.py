import asyncio
import json
import logging

from src.redis_client import RedisClient

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(self):
        self.redis = RedisClient()
        self.is_monitoring = False

    async def start_monitoring(self):
        """Start performance monitoring"""
        self.is_monitoring = True
        logger.info("Starting performance monitoring")

        # Create Redis pubsub connection for signals
        pubsub = self.redis.redis_client.pubsub()
        await pubsub.subscribe("trading_signals")

        while self.is_monitoring:
            try:
                # Get performance stats
                stats = self.redis.get_performance_stats()

                if stats:
                    print("\n" + "=" * 50)
                    print("PERFORMANCE MONITOR")
                    print("=" * 50)
                    print(f"Timestamp: {stats.get('timestamp', 'N/A')}")
                    print(f"Current Balance: ${stats.get('current_balance', 0):,.2f}")
                    print(f"P/L: {stats.get('profit_loss', 0):.2f}%")
                    print(f"Total Trades: {stats.get('total_trades', 0)}")
                    print(f"Win Rate: {stats.get('win_rate', 0):.2f}")
                    print("=" * 50)

                # Check for new signals
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message:
                    signal_data = json.loads(message["data"])
                    print(f"\nNEW SIGNAL: {signal_data['strategy']}")
                    print(f"Action: {signal_data['signal']['action']}")
                    print(f"Confidence: {signal_data['signal']['confidence']:.2f}")

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                await asyncio.sleep(10)

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        logger.info("Performance monitoring stopped")


async def main():
    """Main monitoring function"""
    monitor = PerformanceMonitor()

    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted")
    finally:
        await monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
