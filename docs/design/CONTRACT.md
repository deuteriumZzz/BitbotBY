# BitbotBY Terminal — паритет функций и контракт API

Фаза 1 из `UI_PROMPT.md`. Источник правды по функциям — `src/telegram_commander.py` (21 команда, ~70 обработчиков кнопок, 2 текстовых ввода) и `src/telegram_notifier.py` (подтверждение сделки). Прототип — `docs/design/prototype.html`.

Статус в прототипе: ✓ есть · ◐ частично · ✗ нет. «Новое» — в UI есть, в боте нет: нужна доработка бэкенда.

## 1. Как устроено сейчас (важно для контракта)

- **Бот и дашборд — разные процессы.** `TradingBot` (внутри него `TelegramCommander`) и FastAPI-дашборд `src/dashboard` общаются только через Redis и SQLite.
- **Настройки** живут в Redis под ключами `bot:*` (`src/runtime_config.py`, ~40 ключей). Дашборд может менять их напрямую через тот же `RuntimeConfig`, бот подхватит на следующем цикле.
- **Действия** (закрыть всё, Paper↔Live, перезапуск, обучение) Telegram выполняет внутри процесса бота. Из дашборда их нужно доставлять **очередью команд** в процесс бота.
- **Состояние портфеля** — Redis `portfolio_state_{paper|live}`; **сделки** — SQLite `data/trades_{paper|live}.db` (WAL, дашборд читает параллельно).
- **Событий для живого UI нет.** Есть только `redis.publish("trading_signals")` в `src/redis_client.py`. Радар, тики, открытия и закрытия придётся публиковать.

## 2. Паритет Telegram → UI → API

### Главное и статус
| Telegram | Что делает | UI | API | Прототип |
|---|---|---|---|---|
| `/start`, `/help`, `main`, `help:*` (7 страниц справки) | Панель и справка | Подсказки `[?]` у каждой опции + раздел «Справка» в Настройках | статично во фронтенде | ✓ |
| `/status`, `status` | Режим, баланс, слоты, пауза | Шапка + карточка баланса | `GET /api/state`, WS `state` | ✓ |
| `/pnl`, `pnl` | P&L за день и всего | Карточка баланса, период 24ч/7д/30д/всё | `GET /api/metrics?period=` | ✓ |
| `/pos`, `pos` | Открытые позиции | Список позиций + радар | `GET /api/positions`, WS `position.*` | ✓ |
| `/closeall`, `closeall` → `closeall_confirm` | Закрыть все позиции | PANIC CLOSE (удержание 1 с) | `POST /api/bot/panic` | ✓ |
| `/pause`, `/resume`, `pause`, `resume` | Пауза торговли | Master Switch в шапке | `POST /api/bot/pause`, `/resume` | ✓ |
| `restart_bot` | Перезапуск процесса | «Перезапустить бота» (после настроек + в «Основное») | `POST /api/bot/restart` | ✓ |
| `noop`, `pos_noop`, `dd_confirm_noop` | Заглушки кнопок | — | — | — |

### Торговля и риск
| Telegram | Что делает | UI | API | Прототип |
|---|---|---|---|---|
| `/mode`, `mode_menu`, `mode:*` | Local / AI / SAC / Hybrid | Боты → «Кто принимает решение» | `PATCH /api/config {MODE}` | ✓ |
| `toggle_chronos`, `chronos_on/off` | Модификатор Chronos | Боты → переключатель Chronos | `PATCH /api/config {CHRONOS}` | ✓ |
| `/scan`, `scan_menu`, `scan:N` | Сколько монет сканировать | Все параметры → `SCAN_TOP_N` | `PATCH /api/config` | ✓ |
| `market_profile_menu`, `market_profile:*` | Bluechip / Altcoin / Meme (пакет настроек) | Боты → «Профиль рынка» | `POST /api/profile {name}` → `apply_market_profile` | ✓ |
| `toggle_season_mode` | Сезон: спросить / переключить сам | Боты → «Менять профиль по сезону» | `PATCH /api/config {SEASON_SWITCH_MODE}` | ✓ |
| `season_dismiss` (+ уведомление о смене сезона) | Оставить профиль | Баннер «Сменился сезон: переключить / оставить» | WS `season.changed`, `POST /api/profile` / `POST /api/season/dismiss` | ✓ |
| `market_type_menu`, `market_type:*` | Спот / фьючерсы | Боты → «Рынок» | `PATCH /api/config {MARKET_TYPE}` | ✓ |
| `toggle_auto_exec` | Автоисполнение | Боты → «Без моего подтверждения» | `PATCH /api/config {AUTO_EXECUTE}` | ✓ |
| `timeout_menu`, `timeout:*` | Таймаут подтверждения | Боты → «Сколько ждать ответа» | `PATCH /api/config {TELEGRAM_CONFIRM_TIMEOUT}` | ✓ |
| notifier `confirm` / `reject` | Подтвердить сделку | Окно подтверждения с таймером | WS `signal.pending`, `POST /api/signals/{id}/confirm`, `/reject` | ✓ |
| `/risk`, `risk_menu`, `risk:*` | Пресеты риска | Боты → карточки режимов | `POST /api/presets/risk {name}` | ✓ |
| `pos_more`, `pos_less`, `pos_paper_max` | Слоты (в т.ч. максимум для Paper) | Боты → «Сделок одновременно» | `PATCH /api/config {MAX_POSITIONS}` | ✓ |
| `toggle_drawdown` | Масштабирование при просадке | Боты → «Защита при просадке» | `PATCH /api/config {DRAWDOWN_SCALE_ENABLED}` | ✓ |
| `dd_confirm_more/less` | Подтверждений до остановки | Все параметры → `DRAWDOWN_CONFIRM_CYCLES` | `PATCH /api/config` | ✓ |
| `conf_paper_up/down`, `conf_live_*` | Порог уверенности ±0.05 | Все параметры → `MIN_SIGNAL_CONFIDENCE*` | `PATCH /api/config` | ✓ |
| `/lev`, `lev_menu`, `lev:mode:*` | Режим плеча | Боты → «Плечо» | `PATCH /api/config {LEVERAGE_MODE}` | ✓ |
| `/lev target N`, `lev:target:*` | Риск на ATR | Все параметры → `LEVERAGE_TARGET_RISK` | `PATCH /api/config` | ✓ |
| `/lev drawdown N`, `lev:drawdown:*`, `lev:manual` | Порог просадки для плеча | Все параметры → `LEVERAGE_DD_THRESHOLD` (после ветки `claude/sharp-noyce-ce1646`) | `PATCH /api/config` | ✓ |
| `/hours`, `hours_info`, `hours:*` | Торговые часы UTC | Настройки → «Часы торговли» | `PATCH /api/config {TRADING_HOURS}` | ✓ |
| `/add SYM` | Всегда включать в скан | Настройки → «Всегда сканировать» | `PUT /api/symbols {forced}` | ✓ |
| `/remove SYM` | Исключить из скана | Настройки → «Исключить» | `PUT /api/symbols {excluded}` | ✓ |
| `/symbols` | Показать оба списка | Те же два списка | `GET /api/symbols` | ✓ |
| `/strategies`, `strategies`, `strat:NAME`, `strat_reset` | Вкл/выкл стратегий | Боты → «Стратегии»: переключатель у каждой + «Включить все» | `PATCH /api/strategies/{name} {enabled}`, `POST /api/strategies/reset` | ✓ |

### AI, обучение, проверки
| Telegram | Что делает | UI | API | Прототип |
|---|---|---|---|---|
| `/provider`, `provider_menu`, `provider:*` | Выбор AI-провайдера | Настройки → порядок провайдеров + Все параметры → `AI_PROVIDER` | `PATCH /api/config`, `PUT /api/ai-providers/order` | ✓ порядок цепочки — новое (сейчас зашит в `config.py`) |
| `/trainn N` | Монет для обучения SAC | Все параметры → `TRAIN_TOP_N` | `PATCH /api/config` | ✓ |
| `train_sac_now/skip` | Обучить SAC (~60 мин) | Боты → «Дообучить» | `POST /api/jobs {type:"train"}` | ✓ |
| `tune_sac_menu/now/skip` | Тюнинг (~3 ч) | Боты → «Тюнинг» | `POST /api/jobs {type:"tune"}` | ✓ |
| `train_progress`, `backtest_progress` | Прогресс | Прогресс-бар в карточке + лента | `GET /api/jobs/{id}`, WS `job.progress` | ✓ |
| `rollback_sac` | Откат модели | Боты → «Откатить модель» | `POST /api/models/sac/rollback` | ✓ |
| `backtest_menu`, `backtest_now` | Бэктест стратегий | Боты → «Бэктест» | `POST /api/jobs {type:"backtest"}` | ✓ |
| `alpha_report_now` | Значимость альфы | Боты → «Значимость альфы» | `POST /api/jobs {type:"alpha"}` | ✓ |
| `live_divergence_now` | Live против бэктеста | Боты → «Live vs бэктест» | `POST /api/jobs {type:"divergence"}` | ✓ |
| `ai_isolation_menu/now/skip` | Изолированный бэктест AI (~150 вызовов) | Боты → «Проверка AI» с предупреждением о расходе | `POST /api/jobs {type:"ai_isolation"}` | ✓ |

### Уведомления, счёт, сброс
| Telegram | Что делает | UI | API | Прототип |
|---|---|---|---|---|
| `/alerts`, `alerts_menu`, `alert_toggle:*` | Вкл/выкл 7 видов: `atr_spike`, `drawdown`, `daily_limit`, `bot_stalled`, `new_signal`, `position_open`, `position_close` | Настройки → «Уведомления» — ровно эти 7 | `GET /api/alerts`, `PATCH /api/alerts/{kind} {enabled}` | ✓ |
| `alert_cooldown:*` + текстовый ввод `_awaiting_alert_cooldown` | Пауза между повторами, мин | Поле «не чаще раза в N мин» у каждого вида | `PATCH /api/alerts/{kind} {cooldown_min}` | ✓ |
| `switch_trading_mode` → `confirm_mode_switch` + текстовый ввод PIN `_awaiting_mode_pin` | Paper ↔ Live | Настройки → «Счёт»: экран подтверждения + PIN | `POST /api/account/mode {paper, pin}` | ✓ |
| `reset_defaults` → `reset_defaults_confirm` | Сброс настроек | Настройки → «Сбросить» с подтверждением | `POST /api/settings/reset` | ✓ |

### Новое — есть в UI, нет в боте
| UI | Что нужно в бэкенде | API |
|---|---|---|
| Закрыть одну позицию / 50% | Частичное и полное закрытие по id в `order_executor` | `POST /api/positions/{id}/close {fraction}` |
| Изменить SL/TP | Перевыставить SL/TP на бирже + валидация сторон | `PATCH /api/positions/{id} {sl, tp}` |
| Трейлинг для одной позиции | Флаг трейлинга на позицию в `position_monitor` | `PATCH /api/positions/{id} {trailing}` |
| Радар: стадии скана | Публикация событий из `cycle.py` / `order_executor.py` | WS `radar.stage` |
| Задержка, проскальзывание | Писать `expected_price`, `latency_ms` в таблицу `trades` (миграция) | `GET /api/metrics/execution` |
| P&L-календарь, Sharpe, PF, R:R, просадка | Агрегаты по SQLite | `GET /api/metrics`, `GET /api/pnl/daily` |
| Ключи API | Зашифрованное хранилище + проверка ключа | `/api/secrets/*` |
| Все параметры | Реестр настроек + слой переопределений | `/api/config*` |
| Порядок AI-провайдеров | Настраиваемая цепочка вместо зашитой в `config.py` | `PUT /api/ai-providers/order` |
| Audit-лог | Таблица `audit` в SQLite | `GET /api/audit` |

Прототип догнан по всем пунктам (версия 3 Artifact): паритет с Telegram-ботом полный.

## 3. Авторизация

- **Telegram MiniApp:** `POST /api/auth/telegram {initData}` — HMAC-SHA256 проверка подписи токеном бота, `auth_date` не старше 24 ч, `user.id` = `TELEGRAM_CHAT_ID`. Ответ — короткоживущий токен (Bearer, держится в памяти, не в `localStorage`).
- **Desktop (Tauri):** без паролей. `POST /api/auth/desktop/request` — бот присылает владельцу в Telegram 6-значный код; `POST /api/auth/desktop/verify {code}` → токен. Токен хранится в системной связке ключей ОС.
- Все ответы `401` при просроченном токене; фронтенд переавторизуется молча.
- Опасные команды (`panic`, `account/mode`, `secrets`, `restart`) пишутся в `audit` с источником (`miniapp` / `desktop` / `telegram`).

## 4. REST

Все ответы — JSON. Ошибки: `{error: {code, message}}`, `message` — по-русски, для показа пользователю.

### Чтение
| Метод | Что отдаёт |
|---|---|
| `GET /api/state` | `{bot:{alive, paused, cycle, guard, paper, mode, profile, market_type, latency_ms}, balance, peak_balance, slots:{used, max}, positions[]}` |
| `GET /api/positions` | `[{id, symbol, side, leverage, entry, price, qty, sl, tp, trailing, strategy, confidence, flags[], opened_at, pnl, pnl_pct}]` |
| `GET /api/trades?from&to&limit` | Закрытые сделки из SQLite |
| `GET /api/metrics?period=24h\|7d\|30d\|all` | `{realized, fees, net, n, win_rate, profit_factor, avg_win, avg_loss, rr, max_drawdown, sharpe\|null}`; `sharpe = null` при n < 50 |
| `GET /api/metrics/execution` | `{latency_ms_avg, slippage_plan, slippage_fact, fees_30d}` |
| `GET /api/pnl/daily?days=35` | `[{date, net, gross, fees, trades}]` |
| `GET /api/setup` | `{keys_in_app, bybit, telegram}` — для мастера первого запуска: ключ есть в приложении или в `.env` (без значений) |
| `GET /api/journal?period=24h\|7d\|30d\|all` | `{period, limit, trades: [{id, symbol, side, strategy, confidence, entry, exit, qty, pnl, pnl_pct, fees, opened, closed, reason, reasoning}]}` — свежие сверху, не больше `limit` |
| `GET /api/candles?symbol&tf&limit` | Свечи из кэша `DataLoader` |
| `GET /api/strategies` | `[{name, enabled, kill_switch, backtest:{win_rate, trades, sharpe}, live:{win_rate, ev, trades}}]` |
| `GET /api/jobs/{id}` | `{type, status, pct, text, result?}` |
| `GET /api/config/schema` | Реестр настроек: `[{key, group, label, hint, type, default, min, max, step, unit, options, hot, danger}]` |
| `GET /api/config` | Текущие значения + источник каждого (`ui` / `env` / `default`) |
| `GET /api/secrets` | `[{service, fields:[{env, set, last4}], model?, status, message, checked_at}]` — значения ключей не отдаются никогда |
| `GET /api/alerts` | `[{kind, label, enabled, cooldown_min}]` |
| `GET /api/symbols` | `{forced[], excluded[]}` |
| `GET /api/audit?limit` | Журнал действий |

### Изменение
| Метод | Тело → результат |
|---|---|
| `PATCH /api/config` | `{KEY: value, …}` → валидация всего пакета как `Config.validate()`; ответ `{applied_hot[], pending_restart[]}`; невалидный пакет отклоняется целиком |
| `POST /api/presets/risk` | `{name}` |
| `POST /api/profile` | `{name}` — `apply_market_profile` |
| `PUT /api/symbols` | `{forced[], excluded[]}` |
| `PATCH /api/strategies/{name}` · `POST /api/strategies/reset` | `{enabled}` |
| `PATCH /api/alerts/{kind}` | `{enabled?, cooldown_min?}` |
| `PUT /api/ai-providers/order` | `["groq", "gemini", …]` |
| `PUT /api/secrets/{service}` | `{fields:{ENV: value}, model?}` → проверочный вызов → `{status, message, last4}`; Bybit с правом вывода — `422` |
| `DELETE /api/secrets/{service}` | — |
| `POST /api/settings/reset` | — |

### Команды боту
Идут через очередь (раздел 6). У каждой обязательный `request_id` (UUID от клиента) — повтор с тем же id не выполняется второй раз.

| Метод | Тело |
|---|---|
| `POST /api/bot/pause` · `/resume` | — |
| `POST /api/bot/panic` | — · закрыть всё по рынку, отменить ордера, пауза |
| `POST /api/bot/restart` | — |
| `POST /api/positions/{id}/close` | `{fraction: 1 \| 0.5}` |
| `PATCH /api/positions/{id}` | `{sl?, tp?, trailing?}` |
| `POST /api/signals/{id}/confirm` · `/reject` | — · первый ответ (UI или Telegram) выигрывает |
| `POST /api/account/mode` | `{paper: bool, pin?}` |
| `POST /api/jobs` | `{type: train \| tune \| backtest \| alpha \| divergence \| ai_isolation}` |
| `POST /api/models/sac/rollback` | — |
| `POST /api/season/dismiss` | — |

## 5. WebSocket `/ws`

Первое сообщение клиента — `{type:"auth", token}` (токен не в URL, чтобы не попадал в логи nginx). Дальше сервер шлёт события `{type, ts, data}`:

| Событие | Данные | Источник в боте |
|---|---|---|
| `state` | Полный снимок как `GET /api/state` — при подключении и после переподключения | дашборд |
| `tick` | `{symbol, price}`, не чаще 1/с на монету: открытые позиции + выбранная на графике | `position_monitor` |
| `position.opened` / `.updated` / `.closed` | позиция; у `opened` — `strategy, confidence, reasoning, opened_at`; у `closed` — `{reason, price}`, reason: `tp\|sl\|trail\|manual_app\|telegram_closeall\|emergency\|hard_drawdown\|atr_spike` или текст динамического выхода (`signal_reversal …`) | `order_executor`, `position_monitor` |
| `radar.stage` | `{symbol, stage: scan\|prescore\|signal\|context\|guards\|open\|rejected, reason?, strategy?, side?, confidence?}` | `cycle.py`, `order_executor` |
| `cycle` | `{state: scan\|monitor\|paused\|hours\|guard\|offline, detail}` | `trading_bot` |
| `guard` | `{kind: drawdown_halt\|atr_spike\|macro\|circuit_breaker\|daily_limit, active, detail}` | `trading_bot`, `risk_management` |
| `signal.pending` / `.resolved` | `{id, symbol, side, confidence, strategy, reason, expires_at, backtest, live}` | `telegram_notifier` → общий pending в Redis |
| `season.changed` | `{from, to, index}` | `season_detector` |
| `latency` | `{exchange_ms}` | `bybit_api` |
| `job.progress` | `{id, type, pct, text}` | обучение, бэктесты |
| `config.changed` · `secret.changed` | `{keys[]}` · `{service, status}` | API |
| `alert` | То же, что уходит в Telegram | `telegram_notifier` |

`offline` выставляет сам дашборд, если healthcheck бота старше 120 с: UI блокирует опасные кнопки.

## 6. Транспорт между дашбордом и ботом

- **События:** бот публикует в Redis pub/sub `bot:events`; дашборд подписан и раздаёт по WS. Один хелпер `emit(type, data)` в `src/redis_client.py`, вызовы — в местах из таблицы раздела 5.
- **Команды:** дашборд делает `LPUSH bot:commands {request_id, type, args, source}`; в боте отдельная asyncio-задача `BRPOP` → выполняет тем же кодом, что и Telegram-кнопки → `SET bot:cmdres:{request_id}` (TTL 10 мин) + событие в `bot:events`. Идемпотентность — `SET NX bot:cmd:{request_id}`.
- **Общий сервисный слой:** логику кнопок `telegram_commander.py` вынести в `src/bot_service.py` (pause, panic, apply_profile, switch_mode, toggle_strategy…), чтобы её вызывали и Telegram, и очередь команд. Telegram остаётся полноценным пультом.
- **Ожидающие подтверждения сделки** переезжают из памяти `TelegramNotifier` в Redis `bot:pending_signal:{id}` — тогда ответить можно и из Telegram, и из UI, первый ответ выигрывает.

## 7. Порядок реализации (фаза 2)

**Статус (2026-10-16):**
- ✓ 1–2. Отдельный `bot_service.py` не понадобился: `TelegramCommander` уже работает через колбэки бота (`_handle_close_all`, `_handle_restart`, `_handle_mode_switch`) и `RuntimeConfig`. Команды приложения вызывают те же методы — `src/bot_commands.py`. Шина — `src/bot_bus.py` (`emit`, `enqueue`, `CommandConsumer`), потребитель запущен в `TradingBot.trading_loop`. Переключение Paper/Live вынесено в `RuntimeConfig.apply_paper_trading` и используется и Telegram, и приложением.
- ✓ 3. `src/dashboard/routers/control.py`: `/api/state`, `/api/positions`, `/api/metrics`, `/api/pnl/daily`, `/api/strategies`, команды `pause` · `resume` · `panic` · `restart` · `positions/{trade_id}/close` · `account/mode`, `/api/commands/{id}`, WebSocket `/ws` (события + heartbeat). Команды работают только при заданном `DASHBOARD_API_KEY`.
- ✓ Исправлено попутно: дашборд читал `portfolio_state`, а бот пишет `portfolio_state_{paper|live}` — `/api/status` и `/metrics` всегда показывали стартовый баланс.
- События: все из раздела 5, кроме `job.progress` и `season.changed` (нет API фоновых задач и события сезона).
- Ограничения: PANIC в Paper очищает позиции без записи результата в историю — это существующее поведение `/closeall`.
- ✓ 4. Реестр настроек `src/settings_registry.py` — 79 параметров с подписями, типами, границами и способом применения (≈30 «сразу» через `RuntimeConfig`, остальные — в Redis-хэше `bot:config_overrides`, бот применяет их при старте). `GET /api/config/schema`, `GET /api/config`, `PATCH /api/config` (проверка пакета целиком через `Config.validate()`; `null` возвращает значение из .env).
- ✓ 5. Хранилище ключей `src/secrets_store.py`: 17 сервисов, Fernet-шифрование, файл `data/secrets.enc.json` (права 600), мастер-ключ `BITBOT_SECRETS_KEY` в env. Проверка у сервиса перед сохранением: неверный ключ (401/403) не сохраняется, ключ Bybit с правом вывода средств отклоняется. `GET /api/secrets` (без значений), `PUT`/`DELETE /api/secrets/{service}`. Все секретные поля маскируются в логах.
- ✓ 6. Вход: `POST /api/auth/telegram` (initData), `POST /api/auth/desktop/request` + `/verify` (6-значный код в Telegram), `POST /api/auth/logout`. Сессия — Bearer-токен на 12 ч, в Redis только его SHA-256. Команды, настройки и ключи принимают сессию или `X-API-Key`; без настроенного входа управление выключено.
- Найдено попутно: `Config` — dataclass, и `Config().validate()` проверял значения из .env на момент импорта, не видя ключей и настроек, подставленных позже (и Paper/Live из Telegram). Бот и `supervisor.py` теперь проверяют действующие значения (`effective_config`).
- Найдено попутно: `DQN_WEIGHT`, `AI_WEIGHT`, `DQN_SOLO_CONFIDENCE`, `AI_STRATEGY_SELECTION` из `.env.example` код не читает (веса Hybrid зашиты по режиму рынка в `signal_combiner.py`) — в реестр не включены, в прототипе их нужно убрать.
- Ограничения: новый ключ начинает работать после перезапуска бота (ответ API: `restart_required: true`); горячая перезагрузка клиентов — отдельная задача. Мастер-ключ и зашифрованный файл лежат на одном сервере: это защита от утечки папки `data/` (бэкап, git), а не от взлома сервера.
- ✓ События для живого UI:
  - `radar.stage` — пачкой `{stage: "scan" | "prescore", symbols: [...]}` (скан и отбор), по монете `{symbol, stage: "signal", side, confidence, strategy}`, `{symbol, stage: "guards"}`, `{symbol, stage: "rejected", reason}` (20 причин по-русски из `OrderExecutor` + «не хватает баланса»), `{symbol, stage: "open"}`;
  - `tick` — каждые 5 с по открытой позиции: `{symbol, trade_id, price, side, entry, qty, sl, tp, trailing}` (SL виден и после подтяжки трейлингом);
  - `cycle` — `{state: "scan" | "monitor" | "paused", used, max}`;
  - `guard` — `{kind: "drawdown_halt" | "daily_limit" | "atr_spike" | "macro" | "circuit_breaker", active, detail, until?}`;
  - `signal.pending` / `signal.resolved` — предложение сделки и ответ `{id, confirmed, by: "telegram" | "app" | "timeout"}`; ожидающие хранятся в Redis `bot:pending_signal:{id}` и отдаются `GET /api/signals/pending` после переподключения;
  - `position.updated` — после частичного закрытия, смены SL/TP или трейлинга.
- ✓ Действия с одной позицией: `POST /api/positions/{trade_id}/close {fraction: 1 | 0.1…0.9}`, `PATCH /api/positions/{trade_id} {sl?, tp?, trailing?}`, `POST /api/signals/{id}/confirm | /reject`. В Live новые SL/TP ставятся до отмены старых — позиция не остаётся без защиты; если биржа их не приняла, старые остаются. Частичное закрытие идёт тем же путём, что и автоматическая частичная фиксация (`PositionMonitor.close_fraction`).
- ✓ Исправлено попутно: `PositionMonitor` вызывал `BybitAPI.cancel_order(символ, id)` при сигнатуре `(id, символ)` — во всех 7 местах. В Live отмена биржевых SL/TP при закрытии молча не срабатывала, ордера оставались висеть.
- Для фронтенда (прототип сейчас показывает иначе):
  - по таймауту подтверждения бот сделку **исполняет**, если не включён ручной режим (таймаут 0); поле `on_timeout_execute` в `signal.pending`;
  - трейлинг включён для всех позиций, когда `TRAILING_STOP_ATR_MULT > 0`; переключатель в карточке — отключение для конкретной позиции, по умолчанию «вкл»;
  - убрать из «Все параметры» `DQN_WEIGHT`, `AI_WEIGHT`, `DQN_SOLO_CONFIDENCE`, `AI_STRATEGY_SELECTION`.
- Известные проблемы, вне этой работы: частичное закрытие (и автоматическое, и ручное) не отражается в истории сделок и бумажном балансе — `record_close` считает P&L по исходному объёму; в восстановлении после дневного лимита `_recovery_eq` перезаписывается `_live_equity_cache` (в Paper досрочное возобновление, вероятно, не срабатывает).
- ✓ Фаза 3 — фронтенд `webapp/` (Vite 5 + React 18 + TypeScript strict + zustand, lightweight-charts; CSS-токены темы v0 из прототипа). Один код для MiniApp, Tauri и браузера (`src/lib/platform.ts`). Терминал: баланс с кривой equity, метрики, радар с FLIP-переездом монет между этапами, живые позиции (закрыть / 50% / SL-TP / трейлинг), график со свечами из кэша и живой ценой, лента, календарь P&L. Боты: режимы риска, тонкая настройка, подтверждение, профиль/рынок/режим/Chronos, стратегии. Настройки: часы, AI-провайдер, монеты, уведомления, подключения, Paper↔Live с PIN, перезапуск, справка, ключи API, все 80 параметров со схемы. Вход: initData в MiniApp, код из Telegram на десктопе. Демо без бота: `?mock=1`.
  - Сборка: `cd webapp && npm ci && npm run build`; дашборд отдаёт `webapp/dist` по `/app/` (без ключа — это статика; `/api` и `/ws` по-прежнему с ключом или сессией).
  - Запуск на сервере, чтобы приложение работало целиком:
    1. `.env`: `DASHBOARD_API_KEY` (без него чтение `/api` открыто), `BITBOT_SECRETS_KEY` (ключи из приложения), `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` (вход), `WEBAPP_URL=https://<домен>/app/` (кнопка «Открыть терминал» в панели бота и у поля ввода).
    2. HTTPS перед nginx (certbot, Caddy или Cloudflare Tunnel): Telegram открывает MiniApp только по https.
    3. `docker compose up -d --build`: фронтенд собирается в стадии Dockerfile, nginx пропускает `/app/`, `/api/`, `/ws` без basic auth (доступ проверяет дашборд), старый дашборд на `/` остаётся за basic auth.
    4. Десктоп (`desktop/`): адрес сервера — тот же домен; CORS для `tauri://localhost` и `http(s)://tauri.localhost` включён в дашборде.
  - ✓ AI-провайдеры: каталог `src/ai_providers.py` — 16 провайдеров (13 с бесплатным доступом), цепочка с режимом `free` (по умолчанию) | `auto` | `paid`, своим порядком, моделями и провайдерами без ключа (LLM7, своя Ollama — только если включены). API: `GET /api/ai`, `PUT /api/ai/chain {mode?, order?, disabled?, keyless?, models?}`, `GET /api/ai/providers/{id}/models` (живой список по сохранённому ключу); событие `ai.status {provider, model, status, until}`; `bot.ai` в `/api/state`. В приложении — «Настройки → AI»; настройки разнесены на «Основное · AI · Подключения · Параметры», в «Параметрах» сначала 25 основных. `/provider` в Telegram строится из того же каталога.
  - WebSocket закрывается с кодом 4401, если сессия не принята — приложение не переподключается, а показывает вход.
  - Пока только в Telegram (нет API): обучение/тюнинг/откат SAC, бэктест, значимость альфы, live vs бэктест, проверка AI; баннер смены сезона; задержка и проскальзывание исполнения. В приложении на их месте — честная пометка.

1. `src/bot_service.py` + перевод `telegram_commander.py` на него (поведение не меняется; существующие тесты — зелёные).
2. Очередь команд + `emit()` событий.
3. Чтение: `/api/state`, `/positions`, `/metrics`, `/pnl/daily`, `/strategies`, WS.
4. Реестр настроек + `/api/config*` + валидация.
5. Хранилище ключей + `/api/secrets*`.
6. Авторизация MiniApp и Desktop.
7. Новое: действия с одной позицией, `expected_price`/`latency_ms`, audit, порядок AI-провайдеров.

Каждый пункт — с тестами: ключ никогда не возвращается в ответе и не попадает в лог; повтор `request_id` не исполняет команду дважды; `PATCH /api/config` с невалидным пакетом ничего не меняет.
