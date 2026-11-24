# Поток данных

```
Bybit REST ──► MarketScanner ──► snapshots (+ snapshot_id, статус качества, candle_time)
Bybit WS (shadow, флаг) ──► MarketStream ──► метрики / сверка свечей с REST   [решений не принимает]
                                   │
      DataQuality: VALID / STALE / INCOMPLETE / INVALID / UNAVAILABLE (опц. gate DATA_QUALITY_GATE)
                                   ▼
MarketContext / News / Sentiment ─► SignalCombiner ─► контекстные фильтры ─► OrderExecutor ─► биржа
                                   │                                              │
                                   ▼                                              ▼
                            Decision Ledger (JSONL) ◄── signals, execution, rejections, shadow_decisions,
                                   │                     risk_gate_shadow, ai_call, data_quality, replay_input
                                   │                                              ▼
                                   │                    PositionMonitor ─► закрытие ─► TradeHistory (SQLite)
                                   │                                              │
                                   └──────────────  outcome (по decision_id) ◄────┘
```

## Хранилища
| Что | Где | Назначение |
|---|---|---|
| Сделки | SQLite `data/trades_{paper,live}.db` (WAL) | источник истины по сделкам; проверка и копии — `backend/bitbot/storage/db_health.py` |
| Состояние/настройки | Redis | позиции, пауза, режим, пульс, `bot:mode_readiness`, `bot:last_cycle` |
| События | Redis pub/sub `bot:events` (+ Streams `bot:stream:events` за флагом) | приложение и потребители |
| Журнал решений | `data/decision_ledger.jsonl` (ротация 50 МБ) | аудит, replay, отчёты, RAG |
| Датасеты SAC | Parquet + ZSTD + манифест | воспроизводимое обучение (ADR-001) |
| Модели | `models/*.zip`, `models/registry.json` | реестр и стадии допуска |

## Принципы
Данные до решения фиксируются (`snapshot_id`, `replay_input`) — решение воспроизводимо (`python -m bitbot.research.replay`). REST — источник решений и эталон сверки; WS — параллельный наблюдатель. Любая запись в журнал и шину не имеет права ломать торговый цикл.
