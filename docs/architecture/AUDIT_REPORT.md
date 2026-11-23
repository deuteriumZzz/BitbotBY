# BitbotBY — Audit Report (PHASE 0)

Дата: 2026-11-23. Ветка: `frontend/design-system`. Режим: **только чтение кода**; торговая логика, веса, confidence, параметры SAC и Risk **не менялись**.

## 0. Охват и честные границы

**Прочитано полностью:** `src/core/signal_combiner.py`, `src/core/cycle.py`, `src/core/risk_management.py`, `src/ai/ai_analyzer.py`, `src/ai/ai_providers.py`, `src/ai/dqn_signal.py`, `src/ai/experience_buffer.py`, `reinforcement_learning/rl_env.py`.
**Прочитано частично:** `src/ai/online_learner.py` (стр. 1–407), `reinforcement_learning/train_sac.py` (стр. 224–473 + grep остального), `src/core/trading_bot.py` (стр. 1330–1545 + список функций), `src/core/order_executor.py` (стр. 199–370 + grep проверок), `src/core/trusted_trader.py` и `decision_trace.py` (только заголовки), `data/*.json`, `data/experiences.jsonl`.
**НЕ проверено (требует отдельного прохода):** `trading_loop` целиком (trading_bot.py:1608–2181), `position_monitor.py` (выходы), `market_context.py`, `data_loader.py`, `market_scanner.py`, `regime_detector.py`, `chronos_analyzer.py`, `portfolio_optimizer.py` (внутренности CVaR), `bybit_api.py` и реконсиляция (`_reconcile_positions`), все стратегии (`src/strategies/*`), `src/research/*` кроме JSON-результатов, Redis-деградация, Telegram-путь, dashboard. Всё, что в отчёте помечено «не проверено», — не вывод, а пробел.

Масштаб кода: ~25 тыс. строк в `src/`+`reinforcement_learning/`, 1670 тестовых функций в `tests/`.

---

## 1. Реальный путь решения (по коду)

```
Scanner / data_loader → prescore Top-N (trading_bot._prescore_snapshots:1450) →
CycleRunner.collect_snapshots (+news sentiment) →
SignalCombiner.combine (signal_combiner.py:75):
   mode=local  → [] → локальный турнир стратегий (trading_bot._generate_signals:1372)
   mode=ai     → AIAnalyzer.analyze (один batch-запрос на все символы)
   mode=dqn    → _sac_only
   mode=hybrid → _hybrid (AI ∩ SAC, веса по режиму, +Chronos soft-modifier)
 → _drop_invalid_for_market (spot: убрать sell)
 → _apply_market_context_filter (13 контекстных факторов, потолок штрафа 0.25)
 → _apply_sentiment_filter
 → _generate_context_signals (прямые buy/sell из тех же контекстных факторов)
 → _merge_with_context (+0.05 к совпадающему сигналу)
→ CycleRunner.optimize_allocation (CVaR-доля, если ≥2 buy)
→ _composite_score (ранжирование) → OrderExecutor.execute:
   торговые часы, слоты, дубликат, correlation guard, kill-switch EV,
   подтверждение, ликвидность/спред, SL/TP-валидация, лимит нагрузки портфеля,
   Almgren-Chriss/market impact, Kelly (Half-Kelly, кап 20%), dynamic leverage
→ позиция → PositionMonitor (выходы) → trade_history / experience_buffer / OnlineLearner
```
Источник: `signal_combiner.py:75–135`, `trading_bot.py:1330–1538`, `order_executor.py:199–578`.

**Режим по умолчанию** `MODE=ai` (`config.py:189`); `LOCAL_FALLBACK=false` (`config.py:209`).

## 2. Реальные точки принятия решений

| Точка | Файл | Что решает |
|---|---|---|
| Выбор источника по MODE | `signal_combiner.py:99–117` | кто генерирует сигнал |
| Hybrid-согласие | `signal_combiner.py:232–352` | вход только при совпадении AI и SAC |
| Контекстный фильтр/сигналы | `signal_combiner.py:354–789` | подстройка confidence и **независимые** входы |
| Локальный fallback | `trading_bot.py:1367–1441` | если решающая модель недоступна **и** `LOCAL_FALLBACK=true` |
| Composite score | `trading_bot.py:1500` | приоритет сделок |
| OrderExecutor.execute | `order_executor.py:199` | финальные guard-проверки и размер |
| Trusted Telegram | `order_executor.py:222–226,312` | **обход** большинства проверок |

---

## 3. Найденные дефекты и риски (с доказательствами)

Серьёзность: **C** — влияет на корректность торговых решений/деньги; **H** — искажает статистику/обучение; **M** — улучшение/долг.

### C-1. Один и тот же рыночный фактор учитывается до 3–4 раз
`signal_combiner.py`: для **funding, liquidation, orderbook imbalance, PCR, Fear&Greed, basis, ETF** одни и те же поля `market_context` используются:
1. как **входы SAC** — obs[14..20] (`dqn_signal.py:94–130`, `rl_env.py:96–111`);
2. как **штраф/бонус** к confidence AI/SAC-сигнала (`_apply_market_context_filter:381–473`);
3. как **самостоятельные buy/sell-сигналы** с вручную заданным confidence (`_generate_context_signals:638–789`, уже после фильтров — `combine:126–133`);
4. как **+0.05** к совпавшему сигналу (`_merge_with_context:811–818`).
Контекстные сигналы создаются **после** фильтров, поэтому на них фильтры не действуют. Сумма штрафов ограничена потолком 0.25 (`_MAX_CONTEXT_PENALTY:43,475`) — это частичная защита только для стороны «штраф», не для «бонус/сигнал/вход SAC».

### C-2. Двойной учёт новостного сентимента
AI-промпт получает `news_sentiment`/`social_sentiment` и правило «>0.3 boosts buy» (`ai_analyzer.py:239–243,277–281`), затем `_apply_sentiment_filter` снова меняет confidence по тому же сентименту (`signal_combiner.py:575–636`, вызов `combine:123–124`).

### C-3. SAC в инференсе не знает о позиции; семантика «sell» не совпадает с обучением
- `SignalCombiner` вызывает `self.sac.get_signal(snap, balance)` без `position`/`entry_price` (`signal_combiner.py:212,264`) → `in_position=0` **всегда** (`dqn_signal.py:113,233`).
- В обучающей среде `sell` лишь **уменьшает** лонг (`rl_env.py:214–224`), шорта нет. При `position=0` отрицательное действие в среде — no-op, награды за него не было → поведение на этой ветке **не обучено**, но `dqn_signal.py:244–250` и combiner трактуют его как сигнал **short-входа** (futures).
- Среда long-only, без funding, без slippage, комиссия 0.1% (`rl_env.py:19,204–222`), а бот торгует с плечом и шортами.

### C-4. SAC-модели нет; Hybrid без модели не торгует через AI+SAC
`models/` пуст (проверено `ls`), `SACSignal.loaded=False` → `get_signal` всегда `hold` (`dqn_signal.py:229–230`). В `_hybrid` при `d_action == "hold"` — `continue` (`signal_combiner.py:270–271`): ни одного hybrid-сигнала. При этом контекстные сигналы (C-1) всё равно добавляются (`combine:126–133`). `decider_unavailable("hybrid")` возвращает True (`signal_combiner.py:147–149`), но локальный fallback выключен по умолчанию → Hybrid в текущем виде = «только context_signal».

### C-5. Автоподмена production-SAC без достаточных ворот
`OnlineLearner._run_full_retrain` принимает новую модель при `SAC ≥ Buy&Hold − 10 п.п.` (`online_learner.py:256–315`): модель **хуже buy&hold на 10 п.п. всё равно промоутится**; оценка — на одном 80/20 split (`train_sac.py:396`), без walk-forward в этом пути, без Shadow/Paper/Canary, горячая замена файла (`os.replace`, `:321`). `ONLINE_LEARNING_MODE` по умолчанию `periodic` (`config.py:384`) — триггер каждые 50 сделок. Режим `online` вообще делает градиентные шаги и перезаписывает модель без какой-либо оценки (`:230–254`).

### C-6. Дообучение SAC на реальном опыте молча не работает и содержит утечку
`train_sac._finetune_on_experiences`:
- `ind.get("macd", 0.0) / p` (`train_sac.py:296`) — в `experiences.jsonl` `macd` это **строка** `"bullish"/"bearish"` → `TypeError`, перехватываемый `except` с логом уровня debug (`:338–339`): **каждая запись молча пропускается**. (Сейчас записей всего 24 < 50, поэтому до этого не доходит, но при 50+ дообучение останется пустым.)
- Даже если бы отрабатывало: `reward = pnl_pct` (доля, ~0.1) против среды, где награда — лог-доходность ×100 за шаг и клип ±5 (`rl_env.py:229–247`) — разный масштаб; `done=False` всегда; в `next_obs` портфельные слоты [12],[13] заполняются **итоговым PnL** (`:302–303`) — утечка исхода в состояние; obs[0..2]=0; действие = «±1.0» по направлению сделки (behavior cloning знака, а не RL-переход).

### C-7. Опыт (`experience_buffer`) непригоден как RL-датасет
Запись: `ts, symbol, action, entry/exit, pnl_pct, indicators, volume_ratio, price, source` (`experience_buffer.py:81–92`). Нет: reward-компонентов, комиссий/funding/slippage, следующего состояния, `terminated/truncated`, snapshot-id, контекстных признаков (funding и т.д. в `indicators` нет), размера позиции, режима. `source` определяется по русской подстроке в `reasoning` (`:46–47,79`) — хрупкая эвристика. «AI»-опыт включает всё, что не «Локальная стратегия», т.е. и `context_signal`. Файл переписывается целиком на каждом `save` (`_trim:178`). Накоплено **24 записи** (`data/experiences.jsonl`).

### H-1. Confidence используется как вероятность и смешивается линейно
`combined = d_conf*w_sac + a_conf*w_ai` (`signal_combiner.py:297`). `a_conf` — самооценка LLM (не откалибрована), `d_conf = |a|` — **величина действия SAC** (`dqn_signal.py:241,248`), не вероятность. Веса по режиму `_REGIME_WEIGHTS` — константы без обоснования (`signal_combiner.py:22–28`). Контекстные сигналы имеют вручную заданные confidence 0.55–0.88 (`:698–768`), сопоставимые с порогом 0.65.

### H-2. Провайдер и сырой ответ AI не сохраняются на решение
Цепочка с fallback реализована хорошо (`ai_providers.py:370–408`, `ai_analyzer.py:439–530`), но: в рекомендацию **не пишется** provider/model/version фактически ответившего звена; `set_last_ai_provider` (`ai_analyzer.py:389`) — одно «последнее» значение в Redis, перезаписываемое каждый вызов; сырой ответ и промпт не сохраняются; `temperature`/`seed` не задаются (`_call:349–378`) → решения невоспроизводимы; один batch-запрос на все символы (ответ одного провайдера на весь цикл). Следствие: оценка качества AI по провайдерам и «воспроизвести решение после сделки» **невозможны** (пункт ТЗ 1.3).

### H-3. Валидация ответа LLM минимальна
`_parse_response` (`ai_analyzer.py:296–335`) проверяет наличие 4 ключей, enum действия и диапазон confidence. **Не проверяет** `entry/stop_loss/take_profit` против snapshot (цены в JSON доверяются), неизвестные поля не отклоняются, `strategy` не сверяется с реестром, нет схемы/Pydantic. Парсинг ломается при markdown-блоках нестандартного вида (`split("```")[1]`).

### H-4. Ranking и prescore: фиксированный Top-N и повторное использование confidence
`_prescore_snapshots` (`trading_bot.py:1450–1498`): веса 0.35/0.30/0.20/0.15 и RSI-экстремум — захардкожены; режет пул до `limit` **до** AI, т.е. перспективные активы без RSI/объёмного триггера (OI-spike, ликвидации, funding) не доходят до анализа. `_composite_score:1500–1538` умножает confidence на `(1+tp_bonus)(1+volume_bonus)·regime_mult` — объём и режим уже участвуют в prescore и в весах Hybrid.

### H-5. Нет Decision Ledger
`decision_trace.py` — слушатель событий для UI (`bot:last_cycle`, шаги к открытой сделке), он «ничего не решает». Не сохраняются: snapshot входов, сырые ответы моделей, исключённые/отклонённые сигналы с последующим исходом, версия модели/промпта. Отклонённые возможности (`last_reject` в `order_executor.py:169,187`) — только последняя причина в памяти, без counterfactual.

### H-6. Исключения из «единого пути» и ослабление защит
- **Trusted Telegram** обходит сканер, AI, пороги, фильтры, корреляцию, kill-switch, лимит нагрузки, подтверждение и Kelly (`order_executor.py:222–226,312`; `trusted_trader.py:1–17`). Оставлены SL, свой риск/слоты, дневной лимит, спред/ликвидность. Это осознанная фича, но противоречит принципу ТЗ «единый путь» — **решение владельца**.
- `paper_guards_off()` отключает часть guard-проверок в Paper (`order_executor.py:312`): статистика paper не равна live-пути.

### M-1. Risk: нет портфельного CVaR/BTC-беты в решении о размере
`risk_management.py` — Half-Kelly (кап 20%) и дневной лимит (UTC-сброс). CVaR есть только в `PortfolioOptimizer.allocate` и применяется лишь при ≥2 buy-рекомендациях, как `alloc_fraction`, ограниченный `RISK_PER_TRADE*3` (`cycle.py:107–137`). Корреляция — через `correlation_filter` (guard), общая BTC-бета и стресс-корреляция не моделируются. Источник `win_rate` для Kelly (`order_executor.py:331–343`) — история **по названию стратегии**, а для AI-сигналов название — «ярлык» (см. сам промпт `ai_analyzer.py:283–285`): не проверено, как это сводится на практике.

### M-2. Часы, стакан, спецификации, funding-interval — не проверены
Не прочитаны `bybit_api.py`, `data_loader.py`, `market_context.py`; нельзя утверждать наличие/отсутствие монотонных часов, проверки `u/seq`, динамических `tickSize/qtyStep`, `fundingInterval`, TTL данных, `adlAlert`. → **первоочередной пункт следующего прохода**.

### M-3. Документация/код расходятся
`rl_env.py` docstring: `Box shape=(21,)`, константа `OBS_DIM=22` (`:21,33`); `dqn_signal.py` docstring «OBS_DIM=21» при 22 элементах (`:5,108–132`); `signal_combiner.py` docstring «ai (только Claude API)» при цепочке из 16 провайдеров.

---

## 4. Что уже реализовано (аналоги компонентов ТЗ)

| Компонент ТЗ | Есть в коде | Оценка |
|---|---|---|
| Fallback-цепочка LLM, cooldown по статусам | `ai_providers.py`, `ai_analyzer.py` | реализовано, зрелое; нет provenance на решение |
| Режимы Local/AI/SAC/Hybrid + Chronos | `signal_combiner.py` | реализовано; Hybrid = согласие, не независимые оценки |
| Regime detector (Gaussian HMM) | `market/regime_detector.py` | **не аудировано** |
| Chronos | `ai/chronos_analyzer.py` (84 строки), soft-modifier ±0.05/0.15 | реализовано как модификатор; сравнения с TimesFM нет |
| Контекст-штраф с потолком | `_MAX_CONTEXT_PENALTY` | частичная защита от стекинга |
| Kelly + дневной лимит | `risk_management.py` | реализовано |
| CVaR/Markowitz-аллокация | `research/portfolio_optimizer.py`, `cycle.py` | реализовано частично |
| Correlation guard, market impact, liquidity guard, kill-switch по live EV | `order_executor.py`, `market/*` | реализовано |
| Decision trace (UI) | `core/decision_trace.py` | не ledger (H-5) |
| Experience buffer / online learning | `ai/experience_buffer.py`, `ai/online_learner.py` | реализовано, но с дефектами C-5…C-7 |
| Walk-forward SAC | `train_sac.train_walk_forward:468` | есть, не в пути промоушена |
| AI-isolation / live-divergence / paper-alpha отчёты | `src/research/*` | есть, не прочитаны подробно |
| Telegram-«доверенные каналы» | `trusted_trader.py` | реализовано, обходит единый путь |
| Reconciliation | `trading_bot._reconcile_positions:401`, `_reconcile_paper_positions:521` | **не аудировано** |

## 5. Результаты прежних исследований (как факты)

- `data/backtest_results.json` (2026-07-27, BTC/USDT 15m, 6 мес.): 4 из показанных стратегий — 0 сделок; ненулевые (`mean_reversion` 65 сделок WR 38.5%, `rsi_momentum` 76 / 34.2%, `bollinger_bands` 97 / 32.0%) имеют **отрицательное** EV и отрицательный Sharpe. Согласуется с зафиксированным ранее выводом «0/23 значимы».
- `data/ai_isolation_backtest_results.json`: **3 сделки**, итог +0.034% — статистически бессмысленно.
- `data/historical_btc.csv` — заглушка из 3 строк (в репозитории); откуда `train_sac` берёт реальные свечи, колонки контекста (`funding_rate`, `ob_imbalance`, `pcr`, …) в обучающих данных **отсутствуют → в обучении константы-нейтрали**, а на инференсе — живые значения (дрейф train/serve; **требует проверки по реальному датасету**).

Вывод: преимущество (edge) ни одной стратегии/модели **не доказано**; SAC не обучена; AI-слой невозможно оценить по провайдерам из-за H-2.

## 6. Архитектурные риски (сводка)

1. Двойной учёт факторов (C-1, C-2) + непрозрачные константы confidence → «уверенность» не интерпретируема.
2. SAC: несоответствие среды обучения и боевого использования (C-3), отсутствие модели (C-4), небезопасное обновление (C-5), нерабочий онлайн-опыт (C-6, C-7).
3. Нет воспроизводимости и провенанса AI-решений (H-2, H-3, H-5).
4. Исключения из единого риск-пути (H-6).
5. Пробелы аудита по бирже/данным (M-2) — до их закрытия нельзя оценивать надёжность исполнения.

## 7. Отсутствующие компоненты (относительно ТЗ v2.1)

Decision Ledger со snapshot и counterfactual; Data Quality Engine с TTL/статусами; Signal Intelligence (дедупликация/группы независимости); Model Registry (provider/model/prompt version на решение; версии SAC); Meta-Model + калибровка; сравнение Chronos↔TimesFM; Dataset Builder для SAC; Shadow/Paper/Canary для моделей; Order State Machine / idempotent `orderLinkId` и Dead Man's Switch (**не проверено**, возможно частично есть); Drift/OOD; RAG.

## 8. Рекомендации по интеграции (для PHASE 1, не выполнены)

Порядок — по зависимостям и без изменения торговой логики до согласования:
1. **Decision Ledger** (расширить `decision_trace.py`): snapshot-id, provider/model/version, сырой ответ, исключённые сигналы, причина отказа; `trace_id`. Чисто аддитивно.
2. **Контракт `ModelOutput`/`TradingDecision`** с `provider/model/version/latency/error_status`; обернуть существующие выходы AI/SAC/Chronos/context_signal без смены логики.
3. **Атрибуция факторов** (аудит-режим, без изменения confidence): логировать, какие контекстные факторы вошли через obs SAC / фильтр / context-signal / merge — основа для Signal Intelligence (C-1, C-2).
4. **Закрыть пробелы аудита M-2** (биржа/данные/реконсиляция) до проектирования Execution-фаз.
5. **SAC:** до любой доработки — заморозить автопромоушен (решение владельца, C-5), описать Dataset Builder и согласованную среду (шорт, funding, slippage, `position` в инференсе) — **все изменения только после согласования**.

## 9. Затрагиваемые файлы (план, ничего не изменено)

`src/core/decision_trace.py`, `src/core/signal_combiner.py` (только логирование атрибуции), `src/ai/ai_analyzer.py` (provenance), `src/core/types.py` (контракты), новые модули под `src/` (ledger, contracts) и тесты в `tests/core`, `tests/ai`. Совместимость: события `radar.stage`/`trace.cycle` и ключ Redis `bot:last_cycle` используются приложением — сохранить формат.

## 10. Второй проход (биржа, цикл, данные, режим, выходы)

Дополнительно прочитано: `exchange/bybit_api.py` (ключевые части), `trading_bot.py` (`trading_loop` 1608–2180, `_reconcile_positions`, `_is_atr_spike`, `_detect_regimes`, `_fit_regime_detector`), `market/regime_detector.py`, `ai/chronos_analyzer.py`, `research/portfolio_optimizer.py`, `position_monitor.py` (`_check_dynamic_exit`, `_update_circuit_breaker`), grep по `market_context.py`. **Всё ещё не прочитано:** `data_loader.py`, `market_scanner.py`, внутренности `position_monitor.run/_check_and_close/_check_partial_tp`, `strategies/*`, `src/research/*` кроме JSON, `telegram_bot/*`, `dashboard/*`.

### Поправка к допущениям ТЗ (важно)
**Бот не использует WebSocket для рыночных данных.** Это ccxt REST-опрос (`bybit_api.py`, `get_ohlcv`, `fetch_positions`) с циклом `TRADING_INTERVAL` (`trading_bot.py:1608–2171`). WebSocket в коде только у `pumpfun_tracker` (PumpPortal) и `bot_bus` (события приложению). Следовательно, идеи «13+ WS-потоков», «реконструкция стакана `u/seq`», «lag тика > 500 мс», cryptofeed, uvloop для потоков **к текущей архитектуре не применимы**, пока не будет принято решение добавить WS-ingestion (это отдельная большая фаза, не «доработка»).

### Что на бирже уже сделано хорошо
- **Идемпотентность ордеров:** один `clientOrderId` (`bb-<uuid>`) на все повторы; перед повтором после сетевой ошибки ищется ордер по `orderLinkId` среди открытых и закрытых; обработка «duplicate» (`bybit_api.py:41–47,226–425`). Это реальная защита от задвоения позиции.
- Redis-блокировка на символ (`:289–291`), `reduceOnly` для SL/TP на фьючерсах (`:553–555`), округление объёма через `amount_to_precision` + проверка минимума (`:614–645`), биржевые SL/TP как страховка при падении бота (`:527–586`).
- TTL по источникам в `MarketContext`: 30 с (стакан) … 4 ч (ETF) (`market_context.py:22–29`).
- Hard-drawdown halt с подтверждением, дневной лимит, ATR-spike breaker, серия убытков → остановка (`trading_bot.py:1771–2011`, `position_monitor.py:1161`).

### Новые дефекты

**C-8. Ветки защит без `sleep` → «подтверждение N циклов» не время, а число итераций; риск busy-loop.**
`trading_bot.py:1866–1867` (`else: continue` при просадке до порога подтверждения) и `:1983–2011` (ATR-spike: `continue`) возвращают в начало цикла **без паузы**. Каждая итерация делает полный REST-скан (`scanner`, `balance`, `data`), так что подтверждение просадки «N циклов подряд» набирается за N скан-времён, а не N×интервал; при устойчивом ATR-спайке цикл крутится на лимите API. Действие ATR-spike — `_close_all_open_positions` (закрыть **все** позиции) — выполняется раз в кулдаун алертов, но `continue` без паузы продолжается.

**H-7. Сверка с биржей — только при старте и при смене режима.**
`_reconcile_positions` вызывается из `initialize` (`trading_bot.py:344`) и `_handle_mode_switch` (`:967`), **не периодически**. Дрейф между рестартами (потерянный ответ, ручное закрытие на бирже, частичное исполнение) обнаружится только позже. Дополнительно: позиции, «закрытые на бирже», просто удаляются из `_monitored` (`:439–445`) — без записи сделки/PnL в `trade_history`; восстановленные позиции получают `atr=0`, `exchange_sl_id=None` (`:497–500`) и запасной SL по процентам.

**H-8. Fail-neutral в `MarketContext` скрывает недоступность данных.**
При любой ошибке источника возвращается нейтраль (`return 0.0, "neutral"`, `"oi_neutral"`, `(50,"neutral")` — `market_context.py:232–265,342–344,390–392,400–422`); `get_context_for_symbols` подменяет весь результат `_neutral_context()` (`:198`). Для фильтров и SAC (obs [14..20]) «источник упал» неотличим от «рынок нейтрален»: нет статуса VALID/STALE/UNAVAILABLE, нет age. Это ровно та «тихая» деградация, о которой предупреждал Data-Quality-слой ТЗ.

**H-9. Режим рынка: HMM обучен один раз на одном символе и применён ко всем.**
`_fit_regime_detector` обучает `GaussianHMM` на 2000 свечах `Config.SYMBOL` при старте (`trading_bot.py:676–686`); `predict` для альтов/мемов использует ту же модель (`:1306`) — шкала доходностей/волатильности другая. Нет переобучения, нет вероятностей состояний и матрицы переходов (`predict` возвращает жёсткую метку, `regime_detector.py:105–124`). Метки по среднему лог-доходности (`:92–97`) — «ranging» это просто среднее состояние, а не доказанный флэт; режима «высокая/низкая волатильность» нет. Веса SAC/AI по режиму дублируются в `RegimeDetector.regime_weights` и `_REGIME_WEIGHTS` (две копии констант).

**H-10. Chronos: блокирующий вызов в event loop и слабая постановка.**
`predict_direction` — синхронный `ChronosPipeline.predict` внутри `async _hybrid` (`signal_combiner.py:315`, `chronos_analyzer.py:62`) → блокирует loop на каждый символ; модель `chronos-t5-small` на CPU в `bfloat16`; прогноз на **1 шаг** по сырым ценам (не доходностям); контекст 64 свечей; при Hybrid без SAC (C-4) Chronos **не вызывается вообще**. Сравнение с TimesFM/baseline отсутствует, оценок качества прогноза нет.

**M-4. Контекстные факторы — ещё и в выходах.**
`PositionMonitor._check_dynamic_exit` (`position_monitor.py:270–325`) закрывает/подтягивает SL по тем же funding / Fear&Greed / liquidation / regime и по «развороту сигнала» с confidence ≥ 0.80 — пятое использование тех же факторов (вход SAC, фильтр, context-signal, merge, выход). На споте reversal-ветка недостижима: `_drop_invalid_for_market` убирает sell до этого (`signal_combiner.py:159–177`).

**M-5. `_is_atr_spike` ищет символ `"BTCUSDT"`** (`trading_bot.py:997`), тогда как ключи `market_data` везде вида `"BTC/USDT"`; фактически работает `Config.SYMBOL` и первые попавшиеся символы — детектор спайка «по рынку» может оценивать не BTC.

**M-6. Нет Dead Man's Switch, `fundingInterval`, `adlAlert`, синхронизации часов, округления цены по `tickSize`.**
grep по `src/` не находит `cancelAllAfter`, `fundingInterval`, `adlAlert`, `price_to_precision`, `adjustForTimeDifference`/`recvWindow`; `time.monotonic()` используется только для кэшей/TTL. Цены SL/TP в `place_exchange_sl_tp` передаются как есть (`bybit_api.py:559–578`); при ошибке постановки SL/TP — лишь `logger.error` (`:571–584`), позиция остаётся без биржевой защиты (ответственность вызывающего — **не проверено**). Времена в сохранении состояния — `datetime.now()` (локальное, наивное, `:314`).

**M-7. Оптимизатор CVaR: шумная оценка и малое применение.**
LP Рокафеллара–Урясева на последних `pct_change` ≥30 точек (`cycle.py:118`, `portfolio_optimizer.py:107`), границы весов 5–60 %, без shrinkage; результат только `alloc_fraction`, дополнительно режется `RISK_PER_TRADE*3` (`cycle.py:132–134`). Не влияет на Kelly-размер и на допуск входа.

**M-8. Мониторинговый режим при заполненных слотах всё равно вызывает AI.**
При `_slots <= 0` цикл вызывает `_generate_signals` по открытым монетам + BTC/ETH (`trading_bot.py:2068–2074`) → `combine` → `ai.analyze` каждый цикл. Это расходует бесплатные лимиты провайдеров и, вероятно, вносит вклад в наблюдавшиеся отказы провайдеров (связь **не доказана**, нужно сопоставить с `ai.status` логами).

### Обновлённое резюме рисков
Прибавляются к §6: (6) защиты цикла имеют логические дыры (C-8, M-5); (7) нет периодической сверки и нет статуса свежести данных (H-7, H-8); (8) режим рынка статичен и не откалиброван по активам (H-9).

### Обновлённые рекомендации (PHASE 1, без изменения торговой логики до согласования)
1. **Исправления «чистых багов»** — отдельным согласованным пакетом, каждый с регрессионным тестом: C-8 (пауза в ветках `continue`), M-5 (формат символа), C-6 (silent `TypeError` в fine-tune), H-7 (запись сделки при удалении устаревшей позиции).
2. **Data Quality слой:** вместо fail-neutral — результат `{value, status, age, source}`; фильтры/SAC решают, что делать при STALE/UNAVAILABLE (H-8).
3. **Периодическая сверка** (аддитивно, раз в 30–60 с) + алерт при расхождении (H-7).
4. **Решение владельца принято (2026-11-23): WebSocket-ingestion добавляется** — Phase 4b в `MASTER_SPEC_v2.md`; REST остаётся fallback, включение по воротам Shadow → Paper → live. Идеи про стакан/lag/cryptofeed/uvloop теперь применимы.
5. Остальное — как в §8.

## 10b. Третий проход (данные, сканер, выходы, бэктест-реализм, стратегии)

Прочитано: `market/data_loader.py` (полностью), `market/market_scanner.py` (полностью), `exchange/bybit_api.get_ohlcv`, `core/position_monitor.py` (`run`, `_check_and_close`, `_check_dynamic_exit`, `_update_circuit_breaker`), `infra/redis_client.py` (кэш, lock), `research/backtest.py` и `*_report.py` (grep по методологии), `strategies/__init__.py` и `base.py` (реестр, метаданные). **Тела стратегий (`strategies/__init__.py`, `ported*.py`, ~3 700 строк) и `_check_partial_tp`/`_apply_trailing_stop` не читались** — оценка стратегий ниже только по реестру и результатам бэктеста.

### Новые дефекты

**C-9. Программное закрытие позиции без `reduceOnly` (futures).**
`_check_and_close` закрывает позицию `create_order(sym, "market", close_side, qty, lock_suffix="close")` (`position_monitor.py:1047`), а `BybitAPI.create_order` передаёт в ccxt только `clientOrderId` (`bybit_api.py:298–305`); `reduceOnly` проставляется **только** в `place_exchange_sl_tp` (`:551–555`). Если между опросом цены (раз в 5 с) и закрытием биржевой SL/TP успел исполниться (код сам называет это race condition, `:1022`), рыночный ордер на фьючерсах **откроет встречную позицию**. Отмена биржевых SL/TP перед закрытием снижает окно, но не исключает его; при неудаче закрытия защита восстанавливается только после 3 попыток (`:1051–1067`), а между отменой и повторной постановкой позиция без биржевого стопа.

**H-11. Индикаторы и сигналы считаются на незакрытой свече (repainting).**
`_process_ohlcv` не отбрасывает последнюю (формирующуюся) свечу (`bybit_api.py:161–174`); стратегии читают `iloc[-1]`, снэпшот — `df.iloc[-1]` (`market_scanner.py:259`), цикл 30 с на 15-минутных барах (`config.py:123,149`). Бэктест же генерирует сигнал «на закрытии бара» (`backtest.py:558`). Живое поведение ≠ бэктест, сигналы внутри бара могут появляться/исчезать. Данные для SAC-obs и экспериментов — тоже с незакрытой свечой.

**H-12. Бэктест: комиссии есть, slippage/funding/задержки — нет.**
`backtest.py` учитывает taker/maker комиссии (`:499–640`), walk-forward без look-ahead по сигналу, hold-out 20 % и rolling walk-forward (`:741–880`); grep не находит slippage/funding/latency ни в `backtest.py`, ни в `src/research/*` для моделирования исполнения. Sharpe — на сделку, не годовой (`:460`). Значимость: bootstrap Sharpe + Wilcoxon + Bonferroni (`backtest.py:1368–1396`, `paper_alpha_report.py`) — **хорошая база**, но нет Deflated Sharpe, PBO, CPCV/purged CV+embargo; универсум — текущий топ по объёму (`market_scanner.py:130–137`) → survivorship bias для любых мульти-символьных исследований.

**H-13. Снэпшот без метки данных; время локальное.**
`build_snapshot` ставит `datetime.now().isoformat()` (локальное наивное, `market_scanner.py:344`), но не время последней свечи и не возраст данных; `get_paginated_history` использует локальное `datetime.now()` для `since_ms` (`data_loader.py:283–286`). Свежесть данных проверить нельзя; воспроизведение по snapshot невозможно (нет snapshot-id).

**H-14. «7d»-изменение в снэпшоте может быть ложным.**
`_pct(close, df, 672)` при `len(df) <= 672` берёт `df.iloc[0]` (`market_scanner.py:230–234`) — подпись «7d» тогда означает «за весь доступный период». `get_historical_data(days=30)` запрашивает 2880 свечей одним `fetch_ohlcv(limit=2880)` (`data_loader.py:181–201`, `bybit_api.py:200–202`) — Bybit отдаёт ограниченную порцию на запрос (порядка сотен–1000); **фактический объём возвращаемых строк не проверялся** (нужен прогон), а `_FETCH_LIMIT=200` для пагинации указывает, что в проекте исходят из лимита 200. Значение `REDIS_TTL_MARKET_DATA` в проверенных файлах не найдено.

**M-9. Тихая потеря символов и источников.**
`_fetch_one` глотает любую ошибку на уровне debug (`market_scanner.py:179–181`); при падении `fetch_tickers` возвращается `Config.SYMBOLS[:n]` с одной error-записью (`:147–149`). Redis-обёртка после **одного** сбоя навсегда выставляет `_available=False` (`redis_client.py:112,131`) → кэш рынка отключается до рестарта; при этом `acquire_lock` намеренно «fail-open» (`:239–245`) — разумно, но при Redis-сбое остаётся только внутрипроцессная блокировка.

**M-10. Мониторинг позиций: сроки и деградация.**
Цены позиций опрашиваются REST раз в 5 с (`position_monitor.py:252`), SL/TP по цене опроса (не по high/low свечи); после `MONITOR_MAX_ERRORS` сетевых ошибок позиция **удаляется из мониторинга** с сообщением «будет сверена при рестарте» (`:205–222`) — а периодической сверки нет (H-7). Опыт SAC и `trade_history` получают PnL без funding; комиссия входа в `pnl_pct` не вычитается (`:1084–1088`).

**M-11. Стратегии: реестр большой, преимущество не доказано.**
В реестре >25 классов (`strategies/__init__.py`) плюс `ported*.py`; есть флаг `experimental` и механизм «по умолчанию выключены» (`base.py:28–33`) — хорошая практика. Уверенность каждая стратегия считает внутри себя (`generate_signal`); локальный «турнир» берёт максимум confidence (`ai_analyzer.recommend_strategy_local:657–684`) — тот самый риск «ловушки confidence от порога», описанный в вашей памяти для `volume_spike`. Результаты бэктеста (§5) — отрицательное EV у всех ненулевых.

### Что хорошо (подтверждено кодом)
Валидация символа/таймфрейма и защита от path traversal в `_csv_cache_path` (`data_loader.py:205–221`); retry только для временных сетевых ошибок; режим «фильтр meme/junk/stablecoin» в сканере; значимость с поправкой на множественное тестирование; изоляция «paper vs live»-статистики по `market_type`; `experimental`-флаг.

### Итог по аудиту (три прохода)
Ядро (комбинатор, AI-цепочка, SAC, Risk, исполнение, цикл, данные, выходы) прочитано. **Не прочитаны:** тела стратегий, `_check_partial_tp`/`_apply_trailing_stop`/`_tighten_sl`, `trade_history.py` (расчёт win rate/EV для Kelly и kill-switch), `portfolio_manager.py`, `src/sentiment/*`, `telegram_bot/*`, `dashboard/*`, `settings/*`, `launcher/*`, `desktop/*`, `webapp/*`. Для Phase 1 (контракты, Decision Ledger, Data Quality) критичны из них `trade_history.py` и `portfolio_manager.py` — их стоит прочитать в начале Phase 1.

### Обновлённый перечень дефектов (по приоритету разбора, не по порядку номеров)
C-9 (reduceOnly), C-8 (защиты без паузы), C-1/C-2 (двойной учёт), C-3/C-4 (SAC), C-5/C-6/C-7 (обучение SAC), H-11 (repainting), H-8 (fail-neutral), H-7 (сверка), H-2/H-3 (AI provenance/валидация), H-12 (реализм бэктеста), остальное — M.

## 10c. Implementation Log — исправления «чистых багов» (2026-11-23)

По решению владельца («сначала чистые баги») исправлены дефекты, не требующие смены торговой логики. Веса, confidence, параметры SAC и Risk **не менялись**.

| Дефект | Что сделано | Файлы |
|---|---|---|
| **C-9** закрытие без `reduceOnly` | `BybitAPI.create_order(..., reduce_only=False)`; на фьючерсах добавляет `reduceOnly`, на споте игнорируется. Передаётся во всех 5 закрывающих вызовах (`close_all_positions`, `close_losers_tighten_winners`, `_check_partial_tp`, `_force_close_at_market`, `_check_and_close`). Открывающий ордер не затронут. | `src/exchange/bybit_api.py`, `src/core/position_monitor.py` |
| **C-8** защиты без паузы | новый `TradingBot._sleep_rest_of_cycle(t0)`; вызывается перед `continue` в ветках «просадка до подтверждения» и «ATR-спайк». Побочный эффект (намеренный): «подтверждение N циклов» теперь занимает N×`TRADING_INTERVAL`, а не N скан-времён. | `src/core/trading_bot.py` |
| **M-5** символ BTC | `"BTCUSDT"` → `"BTC/USDT"` в `_is_atr_spike`: BTC теперь действительно проверяется первым. | `src/core/trading_bot.py` |
| **C-6** (частично) тихий пропуск | `_finetune_on_experiences` теперь пишет **warning** с числом пропущенных записей. **Намеренно не исправлено:** разбор `macd`-строки, шкала reward, утечка PnL в состояние — иначе включилось бы дефектное дообучение; это работа Phase SAC. | `reinforcement_learning/train_sac.py` |

**Второй пакет (тот же день, «ещё баги»):**

| Дефект | Что сделано | Файлы |
|---|---|---|
| **N-1** (найден при втором чтении исполнителя) потеря позиции при сбое `record_open` | В live ордер уже на бирже, в paper залог уже списан; исключение из `record_open` улетало в `finally`, слот освобождался, позиция оставалась без мониторинга SL/TP до рестарта. Теперь: `logger.critical`, позиция ведётся с `trade_id=None` (закрытие это уже поддерживает). | `src/core/order_executor.py` |
| **N-2** молчаливый отказ биржевого SL | Если в live `place_exchange_sl_tp` не вернул `sl_id`, в Telegram-сообщении об открытии добавляется предупреждение «защищает только мониторинг бота». Поведение не меняется, только видимость. | `src/core/order_executor.py` |
| **H-10** Chronos блокирует event loop | `predict_direction` вызывается через `asyncio.to_thread`. Результат тот же. | `src/core/signal_combiner.py` |
| **M-3** расхождение docstring и кода | `OBS_DIM=22` в `rl_env.py` и `dqn_signal.py`, добавлен `atr_norm` в перечень. Только документация. | `reinforcement_learning/rl_env.py`, `src/ai/dqn_signal.py` |

Новые тесты второго пакета: `tests/core/test_executor_tracking.py` (4 кейса). Существующий набор после правок: 2019 passed, 13 skipped.

**Не исправлялось сознательно (логика/решение владельца):** C-1/C-2 (двойной учёт — меняет confidence), C-3/C-4/C-5/C-7 (SAC), H-2/H-3 (AI provenance и валидация — новая функциональность), H-7 (учёт PnL при удалении устаревшей позиции), H-8 (fail-neutral → статусы данных), H-11 (repainting: отбросить незакрытую свечу — меняет сигналы), H-14, потеря дневного лимита при рестарте (`RiskManager._day_start_balance` не сохраняется — кандидат на отдельное решение).

**Третий пакет — «блок 2» (защитные решения, согласовано владельцем):**

| Пункт | Что сделано | Файлы |
|---|---|---|
| Дневной лимит при рестарте | `RiskManager` принимает `redis_client`, сохраняет баланс начала дня в `bot:risk:daily_ref:<paper\|live>` (`{"date": UTC, "balance"}`, TTL 3 сут.), при старте восстанавливает его за те же сутки UTC. Live-синхронизация баланса с биржей при старте больше не затирает восстановленную ссылку (`set_day_reference` только если не восстановлена). Сбой Redis не ломает риск-менеджер. | `src/core/risk_management.py`, `src/core/trading_bot.py` |
| **C-4** SAC/Hybrid без модели | `SignalCombiner.combine` в режимах `dqn`/`hybrid` без загруженной SAC возвращает `[]` (раз пишет warning), AI не вызывается впустую, `context_signal` больше не подменяет Hybrid. Режимы `ai`/`local` не затронуты; запасной локальный путь работает через `decider_unavailable` как раньше. | `src/core/signal_combiner.py` |
| **C-5** автоподмена SAC | `ONLINE_LEARNING_MODE` по умолчанию `disabled` (config + реестр настроек); допуск `_RETRAIN_REJECT_DELTA` 10 → 0 п.п. (модель хуже Buy&Hold не подменяет текущую). Переопределения пользователя в приложении сохраняются. | `config.py`, `src/settings/settings_registry.py`, `src/ai/online_learner.py` |
| **H-7** устаревшая позиция при сверке | позиции, исчезнувшие с биржи, теперь помечают сделку `orphaned` через существующий `mark_orphaned` (вне статистики, без выдуманного PnL); сбой БД не роняет сверку. | `src/core/trading_bot.py` |

Тесты: `tests/core/test_block2_safety.py` (9 кейсов: рестарт в те же сутки, чужая дата, без Redis, сбой Redis, SAC-режимы без модели, `ai` без SAC, сверка с `orphaned`, сбой `mark_orphaned`).

**Известный побочный эффект:** пользователь, у которого `MODE=hybrid`/`dqn` сохранён в приложении, но модели нет, раньше получал (скрытые) `context_signal`-сделки, теперь не получает ничего до обучения и допуска SAC; сообщение об этом — warning в логе. Явного статуса «SAC не допущен» в UI пока нет (кандидат на Phase 1).

**Четвёртый пакет — «блок 1» (аддитивные, решений не меняют):**

| Пункт | Что сделано | Файлы |
|---|---|---|
| **H-2 / H-5** Decision Ledger v0 | Новый append-only JSONL `data/decision_ledger.jsonl` (`ts` UTC, `kind`, `trace_id`, поля; сырой ответ ≤ 8000 симв.; ротация при 50 МБ; запись не бросает исключений; отключается `DECISION_LEDGER_ENABLED=0`). Пишет: `ai_call` (провайдер, модель, latency, `prompt_sha256`, цепочка попыток с причинами отказа, сырой ответ, либо `exhausted`), `signals` (рекомендации цикла со стадиями), `execution` (открыта ли позиция, причина отказа исполнителя). `trace_id` на цикл. | `src/core/decision_ledger.py` (новый), `src/ai/ai_analyzer.py`, `src/core/trading_bot.py` |
| **H-2** provenance | Каждая AI-рекомендация несёт `ai_provider` и `ai_model` того звена fallback-цепочки, которое **фактически** ответило. | `src/ai/ai_analyzer.py` |
| **C-1 / C-2** измерение двойного учёта | Рекомендации получают `conf_trail` — список `[стадия, confidence]` (`source:<mode>` / `ai`+`sac`+`hybrid` → `context_filter` → `sentiment_filter` → `context_merge`), а также `ctx_penalty` / `ctx_penalty_raw` (с потолком и без) и `context_boost`. **Confidence не изменён.** Накопив журнал, можно количественно оценить, где один фактор стекается, и проверить фикс на истории. | `src/core/signal_combiner.py` |
| **H-8** статусы данных | `MarketContext` помечает сбой каждого из 10 источников (`_source_errors`), контекст получает `data_status.failed_sources` (источники, падавшие за последние 5 мин; `["all"]` если контекст символа не получен). Значения-«нейтрали» отдаются как раньше; фильтры и SAC новый ключ не читают. | `src/market/market_context.py` |
| **H-13** метка свежести | Снэпшот получает `candle_time` (время последней свечи, UTC) и `snapshot_utc`. В промпт AI и SAC-obs эти ключи не попадают. | `src/market/market_scanner.py` |

Тесты: `tests/core/test_decision_ledger.py` (12 кейсов; найдена и исправлена ошибка `numpy.int64` в `_candle_time`); автофикстура в `tests/conftest.py` отключает журнал в тестах, чтобы они не писали в реальный `data/`. Полный прогон: **2056 passed, 13 skipped**.

**Что ещё НЕ покрыто блоком 1 (ограничения):** полный snapshot входов не сохраняется (в журнале только хэш промпта — для воспроизведения нужен отдельный snapshot-store, Phase 1/2); `temperature`/`seed` LLM не заданы; валидация `entry/SL/TP` ответа по snapshot (H-3) не добавлена — это изменение поведения; журнал растёт на каждый AI-вызов (в режиме «слоты заняты» это каждый цикл — см. M-8), при 50 МБ + одной ротации хватит на часы–сутки, а не недели.

**Phase 1 — контракты и режимы (поведение торговли не менялось):**

| Элемент | Что сделано | Файлы |
|---|---|---|
| Контракты | `ModelOutput` (предложение политики) и `TradingDecision` (итог по символу: статус LONG/SHORT/NO_TRADE/INSUFFICIENT_DATA, `outputs`, `conf_trail`, `reason_codes`, детерминированный `decision_id`). Каждый confidence помечен `confidence_kind`: `llm_self_report`, `sac_action_magnitude`, `heuristic_constant`, `strategy_score`, `combined_heuristic`, `direction_only` — закрывает смысловую часть H-1 (разные числа больше не выглядят сравнимыми). Строятся адаптером `decision_from_rec` над существующими рекомендациями, ничего в них не меняя. | `src/core/contracts.py` (новый) |
| Trading Mode Manager | Таблица режимов `MODE_SPECS` (какие политики активны, нужны ли SAC/AI), `readiness(mode)` → (готов, причина), `requires_sac`. Combiner-guard (блок 2) использует её. Состояние бота отдаёт `mode_readiness` (аддитивный ключ) — приложение сможет показать «SAC не допущен». | `src/core/trading_modes.py` (новый), `src/core/signal_combiner.py`, `src/core/trading_bot.py` |
| Ledger | «signals» теперь содержит `TradingDecision` (вместо урезанных словарей), рекомендации получают `decision_id`, «execution» его пишет — сигнал и исполнение связаны. | `src/core/trading_bot.py` |
| Документы | `TARGET_ARCHITECTURE.md` и `IMPLEMENTATION_PLAN.md` (критерии приёмки Phase 1, риски совместимости, ворота следующих фаз) — обязательные результаты Phase 0 по ТЗ. | `docs/architecture/`, `docs/implementation/` |

Тесты: `tests/core/test_contracts.py` (12). Полный прогон: см. ниже в коммите.

**Не вошло в Phase 1 намеренно:** новый Decision Engine (сейчас роль у `SignalCombiner`), Portfolio Context, Strategy Registry (есть `STRATEGY_REGISTRY`), Decision Router (по `mode` внутри `combine`), показ `mode_readiness` в UI, политика перехода между режимами при открытых позициях.

**Тесты первого пакета:** новые `tests/exchange/test_bybit_reduce_only.py` (3), `tests/core/test_guard_fixes.py` (4); обновлены ожидаемые аргументы закрывающего вызова в `tests/core/test_position_monitor_new.py` (2 места) и `tests/core/test_monitor_positions.py` (1) — проверка осталась строгой, добавлен `reduce_only=True`. Полный прогон: **2019 passed, 13 skipped**.

**Оставшиеся известные ограничения:** ветки с `continue` проверены юнит-тестом хелпера, а не прогоном `trading_loop`; гонка SL/TP и поведение `reduceOnly` на реальной бирже не проверялись (нужен testnet); H-7 (запись сделки при удалении устаревшей позиции в reconcile) не исправлялся — затрагивает учёт PnL и требует решения о формате записи.

## 11. Критерии приёмки Phase 0

- [x] Реальный путь решения прослежен по коду (§1–2).
- [x] Дефекты с файлами и строками (§3, §10).
- [x] Явно отмечено непроверенное (§0, §10).
- [x] Изменений торговой логики нет.
- [x] Второй проход по бирже/циклу/данным/режиму выполнен (§10).
- [x] Третий проход выполнен (§10b): данные, сканер, выходы, реализм бэктеста, реестр стратегий.
- [ ] Принятые пробелы (читать в начале Phase 1): `trade_history.py`, `portfolio_manager.py`, тела стратегий, `_check_partial_tp`/трейлинг, sentiment/telegram/dashboard/settings.
- [ ] Эмпирические проверки, которые код сам не подтверждает (нужны прогоны, не чтение): сколько свечей реально возвращает `fetch_ohlcv(limit=2880)`; значение `REDIS_TTL_MARKET_DATA`; поведение SL/TP-гонки на testnet.
