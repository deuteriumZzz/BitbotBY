"""Tests for AIAnalyzer._parse_response JSON validation."""

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def analyzer():
    with patch("src.ai_analyzer.Config") as cfg:
        cfg.AI_PROVIDER = "none"
        cfg.ANTHROPIC_API_KEY = ""
        cfg.DEEPSEEK_API_KEY = ""
        cfg.GROQ_API_KEY = ""
        cfg.OPENAI_API_KEY = ""
        cfg.GEMINI_API_KEY = ""
        cfg.MIN_SIGNAL_CONFIDENCE = 0.65
        cfg.AI_MODEL = "claude-3"
        cfg.DEEPSEEK_MODEL = "deepseek-chat"
        cfg.GROQ_MODEL = "llama-3.3-70b-versatile"
        cfg.OPENAI_MODEL = "gpt-4"
        with patch("src.ai_analyzer.get_all_strategies", return_value=[]):
            from src.ai_analyzer import AIAnalyzer

            return AIAnalyzer()


def make_rec(**kwargs):
    base = {
        "symbol": "BTC/USDT",
        "action": "buy",
        "strategy": "ema_crossover",
        "confidence": 0.8,
    }
    base.update(kwargs)
    return base


class TestParseResponse:
    def test_valid_record_passes(self, analyzer):
        raw = json.dumps([make_rec()])
        result = analyzer._parse_response(raw)
        assert len(result) == 1
        assert result[0]["action"] == "buy"

    def test_confidence_string_rejected(self, analyzer):
        raw = json.dumps([make_rec(confidence="high")])
        assert analyzer._parse_response(raw) == []

    def test_confidence_above_one_rejected(self, analyzer):
        raw = json.dumps([make_rec(confidence=1.5)])
        assert analyzer._parse_response(raw) == []

    def test_negative_confidence_rejected(self, analyzer):
        raw = json.dumps([make_rec(confidence=-0.1)])
        assert analyzer._parse_response(raw) == []

    def test_below_min_confidence_rejected(self, analyzer):
        raw = json.dumps([make_rec(confidence=0.3)])
        assert analyzer._parse_response(raw) == []

    def test_invalid_action_rejected(self, analyzer):
        raw = json.dumps([make_rec(action="strong_buy")])
        assert analyzer._parse_response(raw) == []

    def test_missing_required_field_rejected(self, analyzer):
        rec = make_rec()
        del rec["strategy"]
        assert analyzer._parse_response(json.dumps([rec])) == []

    def test_non_dict_entry_skipped(self, analyzer):
        raw = json.dumps(["not_a_dict", make_rec()])
        assert len(analyzer._parse_response(raw)) == 1

    def test_confidence_coerced_to_float(self, analyzer):
        raw = json.dumps([make_rec(confidence=0.8)])
        result = analyzer._parse_response(raw)
        assert isinstance(result[0]["confidence"], float)

    def test_markdown_wrapped_json_parsed(self, analyzer):
        inner = json.dumps([make_rec()])
        raw = f"```json\n{inner}\n```"
        assert len(analyzer._parse_response(raw)) == 1

    def test_hold_action_passes(self, analyzer):
        raw = json.dumps([make_rec(action="hold")])
        assert len(analyzer._parse_response(raw)) == 1

    def test_mixed_valid_and_invalid(self, analyzer):
        recs = [
            make_rec(confidence=0.9),
            make_rec(confidence="bad"),
            make_rec(action="unknown"),
            make_rec(confidence=0.7, symbol="ETH/USDT"),
        ]
        result = analyzer._parse_response(json.dumps(recs))
        assert len(result) == 2
