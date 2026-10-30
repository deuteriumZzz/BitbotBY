"""
Буфер реального торгового опыта для дообучения SAC.

Каждая закрытая сделка сохраняется в data/experiences.jsonl.
Trainer читает этот файл при следующем обучении и дообучает модель
на реальных примерах, а не только на исторических данных.

Формат одной строки JSONL:
{
  "ts": "2026-06-19T13:00:00Z",
  "symbol": "BTC/USDT",
  "action": "buy",
  "entry_price": 64000.0,
  "exit_price": 65000.0,
  "pnl_pct": 0.0156,
  "indicators": { "rsi": 55.2, "macd": 12.3, ... },
  "volume_ratio": 1.4,
  "price": 64000.0,
  "source": "ai"
}
Поле source: "ai" — сигнал от AI провайдера, "local" — локальные стратегии.
SAC обучается только на source="ai" через load_ai_only().
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.getenv("EXPERIENCES_PATH", "data/experiences.jsonl")
_MAX_LINES = int(os.getenv("EXPERIENCES_MAX", "10000"))
_lock = threading.Lock()


# Сигнал сгенерирован локальной стратегией, без AI. Старые записи в истории
# опыта помечены английским «Local analysis» — их тоже считаем локальными.
LOCAL_REASONING_PREFIX = "Локальная стратегия"


def is_local_reasoning(reasoning: str) -> bool:
    return LOCAL_REASONING_PREFIX in reasoning or "Local analysis" in reasoning


def save(
    snap: Dict[str, Any],
    action: str,
    entry_price: float,
    exit_price: float,
    path: str = _DEFAULT_PATH,
    market_type: str = "",
) -> None:
    """
    Добавляет закрытую сделку в файл опыта.

    :param snap: Снэпшот рынка на момент открытия позиции.
    :param action: "buy" или "sell".
    :param entry_price: Цена входа.
    :param exit_price: Цена выхода.
    :param path: Путь к файлу JSONL.
    :param market_type: "spot" | "linear" на момент входа, если известен —
        механика сделки разная (плечо/шорт только на linear), нужно для
        раздельной статистики при переобучении SAC на смешанной истории.
    """
    if not snap:
        return

    if action == "buy":
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price

    reasoning = str(snap.get("reasoning", ""))
    source = "local" if is_local_reasoning(reasoning) else "ai"

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": snap.get("symbol", ""),
        "action": action,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 6),
        "indicators": snap.get("indicators", {}),
        "volume_ratio": snap.get("volume_ratio", 1.0),
        "price": snap.get("price", entry_price),
        "source": source,
    }
    if market_type:
        record["market_type"] = market_type

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            _trim(path)
        logger.debug(
            "Experience saved: %s %s pnl=%.2f%%",
            action,
            snap.get("symbol", ""),
            pnl_pct * 100,
        )
    except OSError as e:
        logger.warning("experience_buffer.save failed: %s", e)


def load(path: str = _DEFAULT_PATH) -> list:
    """
    Читает все сохранённые опыты из файла.

    :param path: Путь к файлу JSONL.
    :return: Список словарей.
    """
    if not os.path.exists(path):
        return []
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError as e:
        logger.warning("experience_buffer.load failed: %s", e)
    return records


def filter_by_market_type(records: list, market_type: str) -> list:
    """Оставляет только опыты текущего типа рынка перед обучением SAC.

    Механика сделки принципиально разная между spot и linear (плечо, шорт,
    funding, риск ликвидации есть только на linear) — обучать модель на
    смеси означает учить её действиям, часть из которых на текущем рынке
    физически невозможна. Записи без поля market_type (до появления этого
    тега) пропускаются как есть — не отбрасываются, чтобы не терять
    накопленную историю целиком при включении фичи.

    :param records: Список опытов, например из load_ai_only().
    :param market_type: Текущий тип рынка ("spot" | "linear").
    :return: Отфильтрованный список — market_type отсутствует ИЛИ совпадает.
    """
    return [r for r in records if r.get("market_type", market_type) == market_type]


def load_ai_only(path: str = _DEFAULT_PATH) -> list:
    """Возвращает только experiences с source="ai" для обучения SAC.

    Старые записи без поля source считаются "ai" для обратной совместимости.
    """
    records = load(path)
    return [r for r in records if r.get("source", "ai") == "ai"]


def count(path: str = _DEFAULT_PATH) -> int:
    """
    Возвращает количество сохранённых опытов.

    :param path: Путь к файлу JSONL.
    :return: Количество непустых строк в файле.
    """
    if not os.path.exists(path):
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _trim(path: str) -> None:
    """Оставляет только последние _MAX_LINES строк в файле."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > _MAX_LINES:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines[-_MAX_LINES:])
    except OSError:
        pass
