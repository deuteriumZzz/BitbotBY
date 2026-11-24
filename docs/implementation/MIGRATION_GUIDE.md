# Руководство по переходу

Все изменения обратно совместимы: новое либо выключено флагом, либо только добавляет данные. Откат — вернуть коммит; новые файлы данных безопасно удалить.

## Что меняется при обновлении
| Что | Влияние |
|---|---|
| БД сделок: колонка `funding REAL DEFAULT 0` | миграция `ALTER TABLE` при первом запуске; старые строки получают 0 |
| `SAC_REQUIRE_APPROVAL=true` | модель в `models/` не торгует, пока не внесена в реестр и не продвинута (раньше торговала сразу) — режимы SAC без модели и так не торговали |
| Ежедневная копия БД (`DB_BACKUP_ENABLED`, по умолчанию вкл.) | файлы `data/backups/*.db`, хранятся 7 последних |
| Теневой Risk Gate (`RISK_GATE_SHADOW`, по умолчанию вкл.) | только записи в журнал |
| Исходы сделок | запись `outcome` в журнал при закрытии |
| Закрытия | `reduceOnly` на фьючерсах |
| Новые файлы | `data/emergency.json`, `models/registry.json`, `data/backups/` |

## Новые флаги (все выключены по умолчанию)
`WS_SHADOW_ENABLED`, `AI_SANITIZE_PROMPT`, `AI_RAG_CONTEXT`, `AI_STRICT_VALIDATION`, `FORECASTER=timesfm`, `FUNDING_IN_PNL`, `SAC_V2_ENABLED`, `EVENT_STREAM_ENABLED`. Лимиты контейнера: `BOT_CPUS`, `BOT_MEM_LIMIT`, `TRAINER_*` (0 = без лимита).

## Зависимости
- Обязательные добавлены: `pyarrow`, `zstandard` (датасеты SAC).
- Необязательные — `requirements-optional.txt`: `onnx`, `onnxruntime` (экспорт SAC), `orjson` (быстрый JSON шины).
- TimesFM — отдельное окружение `.venv-timesfm` на Python ≥ 3.10 (`pip install "timesfm[torch]" "numpy<2"`); в основной `requirements.txt` не входит.

## Порядок первого запуска
1. Обновить код, `pip install -r requirements.txt`.
2. Запустить бота в **paper**; убедиться, что в `/status` режим готов (`mode_readiness`).
3. Проверить `/dbcheck` и `/digest`; через сутки — наличие копии в `data/backups/`.
4. Через неделю: `python -m src.research.signal_intelligence_report` (теневой гейт, исходы, отказы).
5. Включать флаги по одному, по отчёту; каждое включение — отдельным решением.

## Откат
Выключить флаг и перезапустить. Колонку `funding` и новые файлы можно оставить: на работу прежнего кода они не влияют.
