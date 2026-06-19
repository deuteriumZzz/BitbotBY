"""
Fixtures for integration tests against Bybit testnet.

Required env vars (either pair works):
  BYBIT_TESTNET_API_KEY / BYBIT_TESTNET_API_SECRET   <- preferred
  BYBIT_API_KEY / BYBIT_API_SECRET + TESTNET=true     <- fallback

All tests in this package are skipped automatically when credentials are absent.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

# Ensure redis module is mocked before any src imports (no live Redis needed).
_redis_mock = MagicMock()
_redis_mock.StrictRedis = MagicMock
_redis_mock.Redis = MagicMock
_redis_mock.ConnectionError = ConnectionError
sys.modules.setdefault("redis", _redis_mock)


def _make_redis_client_mock() -> MagicMock:
    """Mock RedisClient so BybitAPI never talks to a real Redis."""
    m = MagicMock()
    m.load_market_data.return_value = None  # always cache-miss -> hit real API
    m.acquire_lock.return_value = True
    return m


# ── Credentials ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def testnet_creds() -> dict:
    key = os.getenv("BYBIT_TESTNET_API_KEY") or os.getenv("BYBIT_API_KEY", "")
    secret = os.getenv("BYBIT_TESTNET_API_SECRET") or os.getenv("BYBIT_API_SECRET", "")
    if not key or not secret:
        pytest.skip(
            "Integration tests skipped -- set BYBIT_TESTNET_API_KEY and "
            "BYBIT_TESTNET_API_SECRET to run against Bybit testnet."
        )
    return {"key": key, "secret": secret}


# ── BybitAPI fixture (one instance per module, avoids repeated load_markets) ──


@pytest_asyncio.fixture(scope="module")
async def bybit_api(testnet_creds):
    from unittest.mock import patch

    from src.bybit_api import BybitAPI

    redis_mock = _make_redis_client_mock()
    with patch("src.bybit_api.RedisClient", return_value=redis_mock):
        api = BybitAPI()
        await api.initialize(
            api_key=testnet_creds["key"],
            api_secret=testnet_creds["secret"],
            testnet=True,
        )
        yield api
        await api.close()
