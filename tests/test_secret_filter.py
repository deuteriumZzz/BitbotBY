"""Tests for _SecretFilter logging filter (src/trading_bot.py)."""

import logging

import pytest

from src.logger import _SecretFilter


@pytest.fixture(autouse=True)
def reset_secrets():
    original = list(_SecretFilter._secrets)
    yield
    _SecretFilter._secrets = original


def _filtered_msg(message: str, *args) -> str:
    """Run a LogRecord through _SecretFilter and return formatted message."""
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=message,
        args=args,
        exc_info=None,
    )
    _SecretFilter().filter(record)
    return record.getMessage()


class TestSecretFilter:
    def test_masks_registered_secret(self):
        secret = "supersecretkey1234"
        _SecretFilter.register(secret)
        result = _filtered_msg(f"Connecting with key={secret}")
        assert secret not in result
        assert "***" in result

    def test_no_masking_without_registered_secrets(self):
        _SecretFilter._secrets = []
        result = _filtered_msg("Normal log message abc123")
        assert result == "Normal log message abc123"

    def test_multiple_secrets_all_masked(self):
        s1, s2 = "apikey_aaa111bbb222", "secret_xyz789abcdef"
        _SecretFilter.register(s1, s2)
        result = _filtered_msg(f"key={s1} secret={s2}")
        assert s1 not in result
        assert s2 not in result
        assert result.count("***") == 2

    def test_short_values_skipped(self):
        _SecretFilter.register("short")
        assert "short" not in _SecretFilter._secrets

    def test_filter_always_returns_true(self):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        assert _SecretFilter().filter(record) is True

    def test_secret_in_format_args(self):
        secret = "mysecrettoken9999x"
        _SecretFilter.register(secret)
        result = _filtered_msg("token=%s", secret)
        assert secret not in result
        assert "***" in result
