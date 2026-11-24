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

## Кто прав, если источники расходятся
| Данные | Источник истины | Рабочие копии | Как сводятся |
|---|---|---|---|
| Открытые позиции, live (фьючерсы) | биржа | память бота (`_monitored`), Redis `monitored_positions_live` | на старте сверка добавляет/убирает позиции; потом сверка раз в `RECONCILE_INTERVAL_S` — только тревога, исправление за флагом `RECONCILE_AUTOFIX` |
| Открытые позиции, live (спот) | Redis + SQLite (у спота нет API позиций) | память бота | сверка с биржей не выполняется |
| Открытые позиции, paper | SQLite (`status='open'`) | память бота, Redis `monitored_positions_paper` | на старте `_reconcile_paper_positions` приводит память к базе |
| Сделки и их итог | SQLite `trades_{paper,live}.db` | — | единственная запись; отчёты и приложение читают её |
| Баланс | live: биржа; paper: Redis `portfolio_state_paper` | кэш баланса в памяти | live — синхронизация на старте и по циклу |
| Решения бота | журнал `decision_ledger.jsonl` (только дописывается) | Redis Stream (компактные отметки) | копия в шине может теряться, журнал — нет |
| Настройки | значения по умолчанию в `Config` → env → правки из приложения в Redis (`bot:*`, `bot:hot:*`) | — | последняя правка из приложения главнее env |
| Ключи API | только `data/secrets.enc.json` (шифрование, мастер-ключ вне репозитория) | — | `.env` ключи не поставляет |
| Блокировки ордеров | Redis | — | без Redis бот не стартует (проверка в `initialize`) |
| Аварийный уровень | `data/emergency.json` | — | меняется только с подтверждением |
| Допуск моделей | `models/registry.json` | — | SAC торгует только после допуска |

Redis в обоих compose запущен с AOF (`--appendonly yes`): позиции и баланс paper
переживают рестарт Redis. Память процесса — копия на время работы; после
каждого изменения позиции бот пишет её в Redis.

## Принципы
Данные до решения фиксируются (`snapshot_id`, `replay_input`) — решение воспроизводимо (`python -m bitbot.research.replay`). REST — источник решений и эталон сверки; WS — параллельный наблюдатель. Любая запись в журнал и шину не имеет права ломать торговый цикл.
