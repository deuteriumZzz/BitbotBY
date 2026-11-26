import asyncio
import json

from bitbot.api import live_prices


def _tick(sym: str, price: str) -> str:
    return json.dumps(
        {"topic": f"tickers.{sym}", "data": {"symbol": sym, "lastPrice": price}}
    )


def test_parse_ticker_reads_last_price_and_skips_the_rest():
    assert live_prices.parse_ticker(_tick("BNBUSDT", "552.1")) == ("BNBUSDT", 552.1)
    assert live_prices.parse_ticker('{"op":"pong"}') is None
    assert (
        live_prices.parse_ticker(
            json.dumps({"topic": "tickers.BNBUSDT", "data": {"volume24h": "1"}})
        )
        is None
    )
    assert live_prices.parse_ticker("not json") is None
    assert live_prices.to_bybit("BNB/USDT") == "BNBUSDT"


def test_run_subscribes_open_positions_and_throttles_per_coin():
    class FakeWS:
        def __init__(self):
            self.sent = []
            self.inbox = [
                _tick("BNBUSDT", "552.1"),
                _tick("BNBUSDT", "552.3"),
                _tick("ETHUSDT", "3000"),
            ]

        async def send(self, msg):
            self.sent.append(msg)

        async def recv(self):
            if self.inbox:
                return self.inbox.pop(0)
            await asyncio.sleep(10)
            return ""

    ws = FakeWS()
    published = []

    async def connect():
        return ws

    async def scenario():
        task = asyncio.create_task(
            live_prices.run(
                connect,
                lambda: {"BNBUSDT": "BNB/USDT"},
                lambda s, p: published.append((s, p)),
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(scenario())
    assert {"op": "subscribe", "args": ["tickers.BNBUSDT"]} in [
        json.loads(m) for m in ws.sent
    ]
    # Второй тик BNB — в ту же секунду (троттлинг), ETH — не наша позиция.
    assert published == [("BNB/USDT", 552.1)]
