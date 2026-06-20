# BitbotBY — AI-торговый бот для Bybit

Гибридный крипто-трейдинг бот на Python: **Claude / DeepSeek / OpenAI** + SAC нейросеть + рыночные сигналы реального времени + кванты (CVaR, Almgren-Chriss, Kelly) + 9 стратегий.  
Автоматически сканирует топ-20 монет по объёму, торгует на perpetual фьючерсах (long + short), присылает сигналы в Telegram.

---

## Как это работает

Каждые 30 секунд бот проходит полный цикл:

```
1. Сканирует топ-20 монет по объёму на Bybit
2. Загружает OHLCV (15m), считает индикаторы
3. Параллельно получает рыночный контекст (funding, OI, F&G, orderbook, ...)
4. Определяет режим рынка per-символ (trending_up / ranging / trending_down)
5. Генерирует торговые сигналы (9 стратегий + AI + SAC + контекстные сигналы)
6. Комбинирует и фильтрует сигналы с учётом режима, контекста и риска
7. Выбирает топ-1 рекомендацию по уверенности
8. Запрашивает подтверждение в Telegram (или исполняет автоматически)
9. Открывает позицию с динамическим плечом, ставит SL/TP на бирже
10. Мониторит позиции: trailing stop + динамические выходы по условиям
```

---

## Возможности

### Рыночный Edge — сигналы реального времени

Бот собирает 10+ внешних источников и генерирует из них **прямые BUY/SELL сигналы**, а не только фильтры:

| Источник | Логика | Пример сигнала |
|---|---|---|
| **Funding Rate** (Bybit) | >0.1% = лонги перегреты → SHORT contrarian | SELL ETH conf=0.82 |
| **Open Interest + цена** | OI упал + цена падает → ликвидация лонгов | SELL BTC conf=0.72 |
| **Fear & Greed** (alternative.me) | ≤10 = паника → BUY; ≥92 = эйфория → SELL | BUY SOL conf=0.68 |
| **Orderbook Imbalance** (Bybit) | ask_volume / bid_volume > 1.4 → SELL | SELL XRP conf=0.71 |
| **Basis (futures-spot)** | >2% премия = жадность → SELL | SELL BTC conf=0.69 |
| **Deribit PCR + IV Skew** | PCR <0.5 = all-in calls → SELL | SELL ETH conf=0.65 |
| **ETF Flows** (farside.co.uk) | Outflow <-$100M → SELL; Inflow >$150M → BUY | BUY BTC conf=0.65 |
| **Google Trends** | >75 = retail FOMO пик → SELL | SELL BTC conf=0.67 |
| **Reddit** (r/CryptoCurrency) | sentiment < -0.3 → SELL | опционально |
| **Twitter/X** | VADER по топ-10 монетам | опционально |
| **Stablecoin Supply** (CoinGecko) | USDT market cap упал → риск оттока | фильтр |

Все источники кэшированы (5 мин — 4 часа), работают независимо и gracefully degradируют при недоступности.

### Торговые стратегии

- **9 технических стратегий** — EMA crossover, RSI, MACD, Bollinger Bands, Scalping, Swing, Breakout, Mean Reversion, Trend Following
- **Автопереключение стратегии** по режиму рынка (`REGIME_STRATEGY_SWITCH=true`):
  - `trending_up` → Trend Following
  - `trending_down` → EMA Crossover
  - `ranging` → Mean Reversion
- **AI-анализ** — Claude / DeepSeek / OpenAI: анализирует OHLCV + контекст, выбирает стратегию
- **SAC нейросеть** (Stable-Baselines3) — RL-агент, observation space 21 фича (14 рыночных + 7 контекстных)

### Кванты

- **CVaR / Markowitz** — оптимальная аллокация капитала на каждую монету (scipy, 95% CVaR)
- **Almgren-Chriss (2001)** — оценка рыночного импакта, корректирует размер позиции
- **Half-Kelly** — ×0.5 от формулы Келли, кэп 20%, fallback на `RISK_PER_TRADE` при <10 сделках
- **Динамическое плечо** — бот сам рассчитывает плечо для каждой сделки:

```
Base от confidence:  ≥0.85 → max;  ≥0.75 → 75%;  ≥0.65 → 50%
Штрафы-множители:
  ATR > 5% цены        → × 0.50  (очень волатильно)
  ATR > 3% цены        → × 0.70
  Fear&Greed ≤15/≥85   → × 0.70  (экстремальный sentiment)
  Ликвидационный каскад → × 0.50
  Боковой рынок        → × 0.80
  Funding против сделки → × 0.70
Итог: clip(1, LEVERAGE)  ← LEVERAGE — максимум из .env
```

### Управление позициями

- **Фьючерсный рынок (linear)** — торгует perpetual фьючерсами, умеет открывать LONG и SHORT
- **Лимитные ордера** — вход 0.05% лучше рынка, таймаут 30с → автофоллбек на рыночный
- **SL/TP на бирже** — сразу при открытии ставятся `stop_market` + `limit` ордера на Bybit
- **Trailing stop** — SL подтягивается за ценой на основе ATR
- **Динамические выходы** — позиция закрывается досрочно при смене условий, не дожидаясь SL:

| Условие | Логика |
|---|---|
| Сигнал развернулся (conf ≥ 0.72) | Открыт LONG + пришёл SELL → выход |
| Смена режима | SHORT + рынок перешёл в `trending_up` → выход |
| Funding squeeze | SHORT + `short_overheated` → выход до сквиза |
| Extreme F&G | LONG + F&G ≥ 92 → фиксируем прибыль |
| Liquidation cascade | LONG + `long_liquidation` → выход до каскада |
| Short squeeze | SHORT + `short_squeeze` → выход |

### Риск-менеджмент

- **Circuit breaker** — автостоп после N подряд убыточных сделок
- **Дневной лимит потерь** — 5% по умолчанию, проверяется каждый цикл
- **Корреляционный фильтр** — блокирует одновременные позиции с |корреляцией| выше порога
- **Стейблкоин-фильтр** — USDC, BUSD, DAI и ещё 17 стейблкоинов никогда не попадают в сканирование
- **AI бюджет-гард** — лимит вызовов AI в сутки (UTC), при исчерпании — VADER fallback
- **Funding Arbitrage Detector** — алертит в Telegram когда |funding| >0.05% (HIGH) или >0.10% (EXTREME)

### Инфраструктура

- **Telegram** — сигналы по топ-20, кнопки Trade/Skip, алерт при тишине >6ч, funding arb алерты
- **Position reconciliation** — при рестарте бот сверяет позиции с биржей
- **Redis persistence** — circuit breaker, корреляции, кэш переживают рестарт
- **Secret filter** — API-ключи маскируются в логах (`***`)
- **Health server** — `GET /health` (JSON) и `GET /metrics` (Prometheus)
- **Alertmanager** — 6 правил алертинга → Telegram webhook
- **SIGTERM / SIGINT** — чистое завершение, `docker stop` работает корректно
- **Веб-дашборд** — баланс, позиции, история сделок (http://localhost:8080)
- **Grafana** — метрики в реальном времени (http://localhost:3000)

---

## Быстрый старт

### Требования

- Python 3.11+ или [Docker Desktop](https://docs.docker.com/get-docker/)
- Redis (локально или Docker)
- API ключи: Bybit + один из AI-провайдеров + Telegram Bot

### 1. Клонировать и настроить

```bash
git clone <repo-url>
cd BitbotBY
cp .env.example .env   # заполните ключи в .env
```

### 2. Запустить через Docker (рекомендуется)

```bash
make up
```

Поднимает: бот + дашборд + Redis + Prometheus + Grafana.

| Сервис | URL |
|--------|-----|
| Веб-дашборд | http://localhost:8080 |
| Grafana | http://localhost:3000 (admin / `GRAFANA_PASSWORD` из .env) |
| Prometheus | http://localhost:9090 |
| Alertmanager | http://localhost:9093 |

### 3. Или запустить локально

```bash
pip install -r requirements.txt

make paper      # paper trading (MODE=local, без реальных ордеров)
make paper-ai   # paper trading с AI
```

---

## Обучение SAC-модели

SAC-нейросеть нужна только для режимов `dqn` и `hybrid`. Для `MODE=ai` — не нужна.

```bash
make train       # 500k шагов, ~60 мин на CPU
make train-long  # 1M шагов, ~2 ч на CPU
```

Observation space: **21 фича** — 14 рыночных (OHLCV, индикаторы) + 7 контекстных:
`funding_rate`, `orderbook_imbalance`, `put_call_ratio`, `fear_greed`, `iv_skew`, `basis_pct`, `google_trends`

После обучения скрипт оценивает модель на out-of-sample тесте (последние 20%):

```
IN-SAMPLE  (13600 candles) — Sharpe: 1.84
OUT-SAMPLE (3400 candles)  — Sharpe: 1.21  Overfit ratio: 1.52x  ✓
```

Overfit ratio > 2x = переобучение, нужно переобучить с меньшим числом шагов.  
Модель сохраняется в `models/sac_model.zip`.

---

## Бэктест

```bash
BT_MONTHS=2 BT_TIMEFRAME=15m BT_TOP_N=20 python3 backtest.py
BT_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT python3 backtest.py
make backtest
```

Walk-forward с holdout 20% — показывает IN-SAMPLE и OUT-SAMPLE Sharpe отдельно.  
Результаты в `data/backtest_results.json`.

---

## Makefile команды

| Команда | Описание |
|---------|----------|
| `make train` | Обучить SAC-модель (500k шагов) |
| `make train-long` | Обучить SAC-модель (1M шагов) |
| `make tune` | Optuna поиск гиперпараметров (30 trials) |
| `make retrain` | Walk-forward retraining |
| `make backtest` | Walk-forward бэктест (топ-20 монет) |
| `make paper` | Paper trading (MODE=local) |
| `make paper-ai` | Paper trading с AI |
| `make live` | Живая торговля |
| `make test` | Тесты с покрытием |
| `make lint` | flake8 + mypy |
| `make fmt` | black + isort |
| `make up` | `docker compose up -d --build` |
| `make down` | `docker compose down` |
| `make logs` | Логи бота в реальном времени |

---

## Конфигурация (.env)

### Bybit и рынок

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `BYBIT_API_KEY` | — | **Обязателен** при `PAPER_TRADING=false` |
| `BYBIT_API_SECRET` | — | **Обязателен** при `PAPER_TRADING=false` |
| `MARKET_TYPE` | `linear` | `linear` — perpetual фьючерсы (long+short); `spot` — только long |
| `LEVERAGE` | `3` | Максимальное плечо (бот рассчитывает динамически ≤ этого значения) |
| `TESTNET` | `false` | Bybit testnet |
| `PAPER_TRADING` | `false` | Симуляция без реальных ордеров |

### AI провайдеры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `AI_PROVIDER` | `auto` | `auto` / `anthropic` / `deepseek` / `openai` |
| `ANTHROPIC_API_KEY` | — | Claude — [console.anthropic.com](https://console.anthropic.com) |
| `AI_MODEL` | `claude-sonnet-4-6` | Модель Claude |
| `DEEPSEEK_API_KEY` | — | DeepSeek (~10× дешевле Claude) |
| `OPENAI_API_KEY` | — | ChatGPT |
| `AI_DAILY_BUDGET` | `200` | Лимит AI-вызовов в сутки (UTC); сверх → VADER |

При `AI_PROVIDER=auto` при billing/quota ошибке бот переключается: Claude → DeepSeek → OpenAI → локальные стратегии.

### Торговля

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `MODE` | `ai` | `local` / `ai` / `dqn` / `hybrid` |
| `AUTO_EXECUTE` | `false` | Авто-исполнение топ-1 рекомендации |
| `REGIME_STRATEGY_SWITCH` | `true` | Автопереключение стратегии по режиму рынка |
| `SCAN_TOP_N` | `20` | Количество монет для сканирования |
| `RISK_PER_TRADE` | `0.02` | Риск на сделку (2% баланса) |
| `MAX_POSITIONS` | `3` | Максимум одновременных позиций |
| `MIN_SIGNAL_CONFIDENCE` | `0.65` | Минимальный порог уверенности сигнала |
| `TRAILING_STOP_ATR_MULT` | `1.0` | Trailing stop (× ATR) |
| `SAC_MODEL_PATH` | `models/sac_model.zip` | Путь к модели (нужна для `dqn` / `hybrid`) |
| `BACKTEST_HOLDOUT_RATIO` | `0.2` | Доля out-of-sample в бэктесте |

### Риск-менеджмент

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `DAILY_LOSS_LIMIT` | `0.05` | Лимит дневного убытка (5%) |
| `CIRCUIT_BREAKER_LOSSES` | `3` | Стоп после N подряд убытков (0 = выключен) |
| `MAX_CORRELATION` | `0.7` | Блокировать если \|корр.\| с открытой ≥ порога (0 = выключен) |
| `CORRELATION_WINDOW` | `50` | Окно корреляции в барах |

### Telegram

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `TELEGRAM_BOT_TOKEN` | — | Токен от [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | — | Chat ID ([@userinfobot](https://t.me/userinfobot)) |
| `TELEGRAM_CONFIRM_TIMEOUT` | `60` | Секунд ждать Trade/Skip |
| `SILENT_DEATH_HOURS` | `6` | Алерт если нет сделок N часов |

### Внешние сигналы (опционально)

| Параметр | Описание |
|----------|----------|
| `TWITTER_BEARER_TOKEN` | Twitter API v2 — sentiment по топ-10 монетам; без ключа — пропускается |
| `REDDIT_CLIENT_ID` | Reddit API — sentiment r/CryptoCurrency |
| `REDDIT_CLIENT_SECRET` | Reddit API |
| `REDDIT_USER_AGENT` | Reddit API user agent |
| `GLASSNODE_API_KEY` | Glassnode — on-chain данные |
| `NEWS_API_KEY` | NewsAPI — новостной sentiment |

---

## Режимы работы

| Режим | Описание | Требования |
|-------|----------|------------|
| `local` | 9 стратегий + рыночный контекст | Bybit API |
| `ai` | AI + контекстные сигналы | Bybit + AI-ключ |
| `dqn` | SAC нейросеть (21 фича) | Bybit + обученная модель |
| `hybrid` | SAC + AI, веса по режиму рынка | Bybit + AI-ключ + модель |

В режиме `hybrid` веса адаптируются:

| Режим рынка | Вес SAC | Вес AI |
|-------------|---------|--------|
| `trending_up` | 50% | 50% |
| `ranging` | 40% | 60% |
| `trending_down` | 30% | 70% |

---

## Архитектура

```
supervisor.py
  └── TradingBot (src/trading_bot.py)
        ├── MarketScanner       — топ-N монет, фильтр стейблкоинов
        ├── DataLoader          — OHLCV ccxt async, кэш CSV 24ч
        ├── RegimeDetector      — GaussianHMM per-symbol, TTL-кэш 5 мин
        ├── MarketContext       — 10+ внешних источников (TTL-кэш 5–240 мин)
        │     funding/OI/liquidation · Fear&Greed · Orderbook · Basis
        │     Deribit PCR+IV · Google Trends · ETF Flows · Reddit · Twitter
        ├── PortfolioOptimizer  — CVaR / Markowitz (scipy)
        ├── CorrelationFilter   — блокировка коррелированных позиций
        ├── NewsAnalyzer        — NewsAPI + RSS (CoinDesk/CoinTelegraph) + VADER
        ├── TwitterAnalyzer     — Twitter API v2 VADER, топ-10 монет
        ├── FundingArbDetector  — delta-neutral арбитраж алерты
        ├── SignalCombiner      — SAC + AI + контекстные сигналы → топ-1
        ├── OrderExecutor       — CVaR→Kelly→AC + лимитный ордер + dynamic leverage
        ├── PositionMonitor     — SL/TP/trailing + dynamic condition-based exits
        ├── BybitAPI            — ccxt async, linear+spot, exchange SL/TP
        ├── TelegramNotifier    — сигналы, Trade/Skip, silent death, funding arb
        ├── TradeHistory        — SQLite: win rate, EV
        └── HealthServer        — /health + /metrics (Prometheus)

reinforcement_learning/
  ├── rl_env.py      — TradingEnv (Gymnasium), OBS_DIM=21, exec по open[i+1]
  ├── train_sac.py   — SAC: 80/20 split, Optuna, walk-forward
  └── tune_sac.py    — Optuna hyperparameter search

backtest.py          — walk-forward с holdout 20%, overfit ratio
dashboard.py         — FastAPI :8080 (баланс, позиции, история)
```

---

## Тесты

```bash
make test
# pytest tests/ -v --cov=src --cov-fail-under=50
```

**593 unit-теста** — индикаторы, стратегии, риск-менеджмент, CVaR, Kelly, Almgren-Chriss, корреляция, SAC-инференс, dynamic leverage, dynamic exits, position monitor, e2e торговый цикл.  
Coverage: **93%**.

---

## Перед реальной торговлей

1. Заполнить `.env` — `BYBIT_API_KEY`, `BYBIT_API_SECRET`, AI-ключ, `TELEGRAM_BOT_TOKEN`
2. Запустить integration-тесты: `BYBIT_TESTNET_API_KEY=xxx ... pytest tests/integration/ -m integration`
3. Установить `PAPER_TRADING=true`, `MARKET_TYPE=linear`, `LEVERAGE=3`
4. Запустить `make paper-ai` минимум **1–2 недели** — проверить сигналы, логи, dynamic exits
5. Запустить `make backtest` — изучить win rate и EV, проверить overfit ratio (<2x)
6. Если нужен `hybrid` — обучить модель: `make train`
7. Прочитать [RUNBOOK.md](RUNBOOK.md)
8. Начать с `RISK_PER_TRADE=0.01` (1%) и `MAX_POSITIONS=2`
9. Включить `PAPER_TRADING=false` только после уверенности в сигналах

---

## Отказ от ответственности

Торговля криптовалютой сопряжена с высоким риском потери средств.  
Этот бот не является финансовым советником. Используйте на свой риск.  
Всегда начинайте с бумажной торговли (`PAPER_TRADING=true`).
