# BitbotBY — Implementation Plan

Опирается на [MASTER_SPEC_v2.md](../architecture/MASTER_SPEC_v2.md) и [AUDIT_REPORT.md](../architecture/AUDIT_REPORT.md). Статус на 2026-11-23.

## 1. Статус этапов

| Этап | Содержание | Статус |
|---|---|---|
| A. Audit | три прохода, отчёт | ✅ (пробелы — AUDIT_REPORT §10b) |
| Багфиксы | C-9, C-8, M-5, N-1/N-2, H-10, C-6 (лог) | ✅ коммит `f24b042` |
| Защиты (блок 2) | дневной лимит при рестарте, SAC без модели, автоподмена SAC, H-7 | ✅ `efa521c` |
| Наблюдаемость (блок 1) | Decision Ledger v0, provenance, conf_trail, data_status, candle_time | ✅ `9387e71` |
| **B/C. Контракты и режимы (Phase 1)** | `ModelOutput`/`TradingDecision`, `TradingMode`, запись в ledger | ✅ (коммит — см. git log) |
| 2. Data Quality | статусы свежести, snapshot-id, спецификации контракта, опциональный gate | ✅ (gate выключен; WS-часть → 4b) |
| 3. Signal Intelligence | группы независимости, избыточные каналы, конфликты, отчёт по журналу | ✅ (описание/измерение); дедупликация confidence — ⏳ за воротами §5 |
| 4/4b. Data Engineering, WS | Parquet/ClickHouse (ADR), WS-ingestion | ⏳ |
| 5. Execution reliability | Order State Machine, периодическая сверка, DMS | ⏳ |
| 6. Decision Engine / Hybrid | независимые оценки AI и SAC, 4 стратегии объединения | ⏳ |
| 7. SAC pipeline | Dataset Builder, среда с шортом/funding/slippage, lifecycle | ⏳ |
| 8–11 | Forecast/Meta-Model/Drift, Risk, AI infra, Observability | ⏳ |

## 2. Phase 1 — состав и файлы

| Элемент | Файлы | Поведение |
|---|---|---|
| Контракты | `src/core/contracts.py` (новый) | адаптер `decision_from_rec`; значения не меняются |
| Режимы | `src/core/trading_modes.py` (новый) | таблица режимов + `readiness`; combiner-guard использует её |
| Ledger | `src/core/trading_bot.py`, `decision_ledger.py` | в «signals» пишутся `TradingDecision`, в «execution» — `decision_id` |
| Состояние для приложения | `trading_bot._get_bot_state` | + `mode_readiness` (аддитивно) |
| Тесты | `tests/core/test_contracts.py` (контракты, режимы, запись в ledger, состояние бота) | 12 кейсов |

Не входит в Phase 1 (отложено намеренно): новый Decision Engine, Portfolio Context, Strategy Registry (уже есть `STRATEGY_REGISTRY`), Decision Router (роль выполняет `SignalCombiner.combine` по `mode`).

## 3. Риски совместимости

- Приложение/Telegram читают события `radar.stage`, `trace.cycle`, ключ Redis `bot:last_cycle`, `bot_state` — **форматы сохраняются**, новые ключи только добавляются.
- Рекомендации — словари; новые ключи (`conf_trail`, `ai_provider`, `decision_id`, …) безопасны для потребителей (проверено полным прогоном тестов).
- Ledger растёт на каждый AI-вызов (в режиме «слоты заняты» — каждый цикл): следить за размером (ротация 50 МБ).
- Режимы `dqn`/`hybrid` без модели SAC не торгуют (блок 2) — пользователь должен видеть причину (readiness в состоянии бота; UI — позже).

## 4. Критерии приёмки Phase 1

- [x] Все источники дают рекомендации, из которых строится единый `TradingDecision` (тесты: AI, SAC, hybrid, context, local).
- [x] `decision_id` детерминирован и связывает записи «signals» и «execution».
- [x] Каждый `ModelOutput` явно помечает природу confidence (`confidence_kind`).
- [x] `readiness()` — единый источник «готов ли режим»: используется `requires_sac` в combiner-guard и `mode_readiness` в состоянии бота; поведение блока 2 сохранено.
- [x] Поведение торговли не изменилось: полный набор тестов зелёный, ключи рекомендаций только добавлены (`decision_id`).
- [x] Ledger содержит `trace_id`, `decision_id`, происхождение AI-решения.
- [ ] Не сделано в Phase 1: показ `mode_readiness` в UI приложения (бэкенд отдаёт ключ; фронтенд — отдельно); политика перехода между режимами при открытых позициях.

## 5. Дальнейшие входные ворота

- Перед Phase 3 (дедупликация): ≥ несколько сотен записей `conf_trail` в ledger → количественный отчёт, какие факторы стекаются; изменение confidence — только после A/B на истории и согласования.
- Перед SAC-фазой: решение владельца по среде (шорт, funding, slippage), Dataset Builder, Shadow→Paper→Canary.
- Любая новая зависимость — через ADR.

## 6. Implementation Log

Журнал изменений ведётся в [AUDIT_REPORT.md §10c](../architecture/AUDIT_REPORT.md).
