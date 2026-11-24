"""Данные одного торгового цикла, которые шаги передают друг другу.

Неизменяемы: шаг возвращает новую копию (`dataclasses.replace`) — по коду шага
видно, что именно он добавил в цикл.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class CycleContext:
    cycle: int  # номер цикла с запуска
    t0: float  # loop.time() начала цикла: от него считается пауза до следующего
    profile: str = ""  # профиль монет: bluechip | altcoin | meme
    market_data: Dict[str, Any] = field(default_factory=dict)  # символ → свечи
    snapshots: List[Dict[str, Any]] = field(default_factory=list)  # снимки скана
    balance: float = 0.0  # свободный баланс USDT на начало цикла
    open_positions: int = 0
    max_positions: int = 0
    recs: List[Dict[str, Any]] = field(default_factory=list)  # рекомендации комбайнера
