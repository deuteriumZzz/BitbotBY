# BitbotBY — AI-торговый бот для Bybit

Гибридный крипто-трейдинг бот на Python: **Claude / DeepSeek / OpenAI** + SAC нейросеть + кванты (CVaR, Almgren-Chriss, Kelly) + 9 стратегий.  
Автоматически сканирует топ-20 монет по объёму, присылает сигналы в Telegram.

---

## Возможности

### Сигналы и анализ
- **4 режима** — `local`, `ai`, `dqn`, `hybrid`
- **9 технических стратегий** — EMA, RSI, MACD, Bollinger Bands, Scalping, Swing, Breakout, Mean Reversion, Trend Following
- **AI-анализ рынка** — Claude / DeepSeek / OpenAI с **автоматическим переключением** при ошибках billing/quota (HTTP 402/429)
- **SAC нейросеть** (Stable-Baselines3) — RL-агент с reward = log-return + rolling Sharpe − drawdown penalty
- **Market regime detection** — GaussianHMM с `covariance_type="diag"` определяет `trending_up` / `ranging` / `trending_down` **для каждой монеты** отдельно; **TTL-кэш 5 минут** на символ

### Кванты
- **CVaR / Markowitz** — аллокация портфеля через scipy (95% CVaR, вес каждой монеты)
- **Almgren-Chriss (2001)** — адаптивная оценка рыночного импакта, зависит от таймфрейма
- **Kelly criterion** — Half-Kelly (×0.5), кэп 20%, fallback на `RISK_PER_TRADE` для первых 10 сделок

### Риск-менеджмент
- **SL/TP на бирже** — при открытии позиции сразу ставятся `stop_market` + `limit` ордера на Bybit; позиция защищена даже при падении бота
- **Circuit breaker** — бот автоматически останавливается после N подряд убыточных сделок
- **Дневной лимит потерь** — 5% по умолчанию, проверяется каждый цикл
- **Trailing stop-loss** на основе ATR
- **Стейблкоин-фильтр** — USDC, BUSD, DAI и ещё 17 вариантов никогда не попадают в список монет
- **Max позиций** — защита от одновременного открытия слишком многих сделок
- **AI бюджет-гард** — лимит вызовов AI в сутки (`AI_DAILY_BUDGET`), при исчерпании — VADER fallback

### Инфраструктура
- **Telegram сигналы** — уведомления о новых buy/sell сигналах по топ-20 монетам (без спама: только изменения)
- **Telegram подтверждения** — кнопки Trade/Skip перед исполнением (при `AUTO_EXECUTE=true`)
- **Silent death alert** — уведомление в Telegram если нет сделок более `SILENT_DEATH_HOURS` часов (по умолчанию 6ч)
- **Correlation filter** — блокирует одновременные позиции с |корреляцией| выше порога; состояние сохраняется в Redis между рестартами
- **Position reconciliation** — при рестарте бот сверяет `_monitored` с реальными позициями на бирже (включая нормализацию `BTC/USDT:USDT` → `BTC/USDT`)
- **Secret filter** — API-ключи автоматически маскируются в логах (`***`) на уровне `logging.Filter`
- **Health server** — `GET /health` (JSON) и `GET /metrics` (Prometheus) встроены в процесс бота
- **Alertmanager** — 6 правил алертинга (BotDown, HighConsecutiveLosses, LargePnLLoss и др.) → Telegram webhook
- **Redis graceful degradation** — если Redis недоступен, бот работает без персистентности (не падает)
- **SIGTERM / SIGINT** — супервизор ловит оба сигнала и завершает работу чисто (`docker stop` работает корректно)
- **Paper trading** — полный тест без реальных денег
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

SAC-нейросеть нужна только для режимов `dqn` и `hybrid`. Для `MODE=ai` не нужна.

```bash
make train       # 500k шагов, ~60 мин на CPU
make train-long  # 1M шагов, ~2 ч на CPU
```

После обучения скрипт оценивает модель на out-of-sample тесте (последние 20%):

```
TEST SET (3400 candles) — SAC: $10832 (+8.3%) | Buy&Hold: $10650 (+6.5%) | Commissions: $47
```

Модель сохраняется в `models/sac_model.zip`.

---

## Бэктест

```bash
# Автоматически берёт топ-20 монет с биржи
BT_MONTHS=2 BT_TIMEFRAME=15m BT_TOP_N=20 python3 backtest.py

# Конкретные монеты
BT_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT python3 backtest.py

# Через make
make backtest
```

Результаты сохраняются в `data/backtest_results.json`.

---

## Makefile команды

| Команда | Описание |
|---------|----------|
| `make train` | Обучить SAC-модель (500k шагов) |
| `make train-long` | Обучить SAC-модель (1M шагов) |
| `make tune` | Optuna поиск гиперпараметров (30 trials) |
| `make retrain` | Walk-forward retraining (4м окно, 1м шаг) |
| `make backtest` | Walk-forward бэктест (топ-20 монет) |
| `make paper` | Paper trading (MODE=local, без ордеров) |
| `make paper-ai` | Paper trading с AI |
| `make live` | Живая торговля (нужны API-ключи) |
| `make test` | Запустить тесты с покрытием |
| `make lint` | flake8 + mypy |
| `make fmt` | black + isort |
| `make up` | `docker compose up -d --build` |
| `make down` | `docker compose down` |
| `make logs` | Логи бота в реальном времени |

---

## Конфигурация (.env)

### AI провайдеры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `AI_PROVIDER` | `auto` | Провайдер: `auto` / `anthropic` / `deepseek` / `openai` |
| `ANTHROPIC_API_KEY` | — | Claude — [console.anthropic.com](https://console.anthropic.com) |
| `AI_MODEL` | `claude-sonnet-4-6` | Модель Claude |
| `DEEPSEEK_API_KEY` | — | DeepSeek — [platform.deepseek.com](https://platform.deepseek.com) (~10× дешевле Claude) |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Модель DeepSeek |
| `OPENAI_API_KEY` | — | ChatGPT — [platform.openai.com](https://platform.openai.com) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Модель OpenAI |
| `AI_DAILY_BUDGET` | `200` | Лимит AI-вызовов в сутки (UTC); сверх → VADER-фолбек |

При `AI_PROVIDER=auto` все провайдеры с ключами инициализируются при старте. При billing/quota ошибке (HTTP 402/429) бот автоматически переключается на следующий: Claude → DeepSeek → OpenAI → локальные стратегии.

### Основные параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `MODE` | `ai` | Режим: `local` / `ai` / `dqn` / `hybrid` |
| `PAPER_TRADING` | `true` | Paper trading (без реальных ордеров) |
| `MARKET_TYPE` | `spot` | Тип рынка: `spot` или `linear` (фьючерсы) |
| `TESTNET` | `false` | Bybit testnet |
| `BYBIT_API_KEY` | — | **Обязателен** при `PAPER_TRADING=false` |
| `BYBIT_API_SECRET` | — | **Обязателен** при `PAPER_TRADING=false` |
| `SAC_MODEL_PATH` | `models/sac_model.zip` | **Обязателен** при `MODE=dqn` или `hybrid` |
| `TELEGRAM_BOT_TOKEN` | — | Опционально, для уведомлений |
| `TELEGRAM_CHAT_ID` | — | Опционально, для уведомлений |
| `TRADING_SYMBOL` | `BTC/USDT` | Основная монета |
| `TIMEFRAME` | `15m` | Таймфрейм |
| `SCAN_TOP_N` | `20` | Топ монет для сканирования (по объёму 24ч) |
| `RISK_PER_TRADE` | `0.02` | Риск на сделку (2% баланса) |
| `MAX_POSITIONS` | `3` | Максимум одновременных позиций |
| `MAX_CORRELATION` | `0.7` | Блокировать позицию если \|корр.\| с открытой ≥ порога (0 = выключен) |
| `CORRELATION_WINDOW` | `50` | Окно расчёта корреляции в барах |
| `HEALTH_PORT` | `8081` | Порт health server бота (0 = выключен; в docker явно задан 8080) |
| `DAILY_LOSS_LIMIT` | `0.05` | Лимит дневного убытка (5%) |
| `CIRCUIT_BREAKER_LOSSES` | `3` | Стоп после N подряд убытков (0 = выключен) |
| `MIN_SIGNAL_CONFIDENCE` | `0.65` | Минимальный уровень уверенности |
| `TRAILING_STOP_ATR_MULT` | `1.0` | Trailing stop (× ATR) |
| `AUTO_EXECUTE` | `false` | Авто-исполнение топ-1 рекомендации |
| `SILENT_DEATH_HOURS` | `6` | Алерт если нет сделок N часов (только в не-local режиме) |
| `REGIME_CACHE_TTL` | `300` | TTL кэша режимов рынка в секундах |

При старте бот **автоматически проверяет** обязательные переменные и падает с понятной ошибкой если что-то не задано. Стейблкоины в `TRADING_SYMBOL` тоже отклоняются.

---

## Режимы работы

| Режим | Описание | Требования |
|-------|----------|------------|
| `local` | Только 9 технических стратегий | Bybit API |
| `ai` | AI-провайдер анализирует рынок и выбирает стратегию | Bybit + любой AI-ключ |
| `dqn` | Только SAC-нейросеть | Bybit API + обученная модель |
| `hybrid` | SAC × вес + AI × вес (адаптируется к режиму рынка) | Bybit + любой AI-ключ + модель |

В режиме `hybrid` веса автоматически адаптируются под режим рынка:

| Режим рынка | Вес SAC | Вес AI |
|-------------|---------|--------|
| `trending_up` | 50% | 50% |
| `ranging` | 40% | 60% |
| `trending_down` | 30% | 70% |

---

## Telegram

Создайте бота через [@BotFather](https://t.me/BotFather), получите токен и chat_id через [@userinfobot](https://t.me/userinfobot).

### Сигналы по топ-20 (всегда, при любом AUTO_EXECUTE)

Бот присылает уведомление когда появляется **новый** buy/sell сигнал (повторные не дублируются):

```
📊 Цикл #42 | 14:30:05 | Баланс: $10,234.50

🟢 BTC/USDT — BUY
   Conf: 78% | hybrid(ai+sac) | Режим: trending_up
   Entry: $67234.00 | SL: $65100.00 | TP: $71500.00
   "Пробой сопротивления, объём растёт..."

🟢 SOL/USDT — BUY
   Conf: 71% | ai | Режим: ranging
   Entry: $142.50 | SL: $138.20 | TP: $150.80
```

### Подтверждение сделки (только при AUTO_EXECUTE=true)

```
BTC/USDT  —  BUY
Strategy: hybrid(ai+sac)
Entry: $67,420   SL: $65,100   TP: $71,500
Confidence: 78%
Backtest: 63% win  (180 сделок)  EV: +1.24%
Live:     71% win  (12 сделок)   EV: +0.98%

Auto-execute in 60s
```

---

## Архитектура

```
supervisor.py                          ← единственная точка входа
  └── TradingBot (src/trading_bot.py)  ← оркестратор
        │
        ├── MarketScanner       — топ-N монет по объёму, фильтр стейблкоинов
        ├── DataLoader          — OHLCV через ccxt async, кэш CSV 24ч
        ├── RegimeDetector      — GaussianHMM (diag): режим per-symbol + TTL-кэш 5 мин
        ├── PortfolioOptimizer  — CVaR / Markowitz аллокация (scipy)
        ├── CorrelationFilter   — блокировка коррелированных позиций + Redis persistence
        ├── NewsAnalyzer        — NewsAPI + AI sentiment (VADER fallback), Redis 15 мин
        │
        ├── SignalCombiner      — финальный взвешенный сигнал по режиму рынка
        │     ├── SACSignal     — inference SAC-модели (Stable-Baselines3)
        │     └── AIAnalyzer    — Claude/DeepSeek/OpenAI + auto-fallback при billing
        │
        ├── OrderExecutor       — размер позиции (CVaR → Half-Kelly → AC) + ордер
        ├── PositionMonitor     — фоновый контроль SL/TP/trailing/circuit breaker
        ├── RiskManager         — Kelly, daily loss limit, валидация
        ├── BybitAPI            — ордера (ccxt async), exchange SL/TP, Redis lock
        ├── TelegramNotifier    — новые сигналы, Trade/Skip, silent death alert
        ├── TradeHistory        — SQLite: win rate, EV, история
        ├── PortfolioManager    — баланс, позиции, комиссии
        └── HealthServer        — GET /health (JSON) + GET /metrics (Prometheus)

dashboard.py → src/dashboard/ (FastAPI :8080)
  ├── /api/status, /api/trades, /api/backtest  — данные из Redis/SQLite
  ├── /metrics                                  — Prometheus для Grafana
  ├── /health                                   — healthcheck для docker
  └── /webhook/alerts                           — Alertmanager → Telegram

reinforcement_learning/
  ├── rl_env.py      — TradingEnv (Gymnasium)
  ├── train_sac.py   — обучение SAC: 80/20 split, Optuna, walk-forward
  └── tune_sac.py    — Optuna поиск гиперпараметров

backtest.py          — мультисимвольный walk-forward бэктест
bitbot.service       — systemd unit (авто-перезапуск на Linux)
```

---

## Структура файлов

```
BitbotBY/
├── supervisor.py              — точка входа (SIGTERM, retry, health server)
├── config.py                  — конфигурация (dataclass + env + validate)
├── dashboard.py               — запуск FastAPI дашборда
├── backtest.py                — walk-forward бэктест
├── Makefile                   — 14 команд
├── Dockerfile
├── docker-compose.yml         — bot + dashboard + trainer + prometheus + grafana + alertmanager
│
├── src/
│   ├── trading_bot.py         — главный оркестратор (цикл, инициализация)
│   ├── cycle.py               — вспомогательные задачи одной итерации
│   ├── strategies.py          — 9 торговых стратегий
│   ├── signal_combiner.py     — комбинирование SAC + AI сигналов
│   ├── ai_analyzer.py         — Claude / DeepSeek / OpenAI + auto-fallback
│   ├── dqn_signal.py          — inference SAC (SB3) + drift detection
│   ├── risk_management.py     — Kelly criterion, SL/TP, daily limit
│   ├── portfolio_optimizer.py — CVaR (LP, Rockafellar-Uryasev) / Markowitz
│   ├── portfolio_manager.py   — баланс, позиции, trailing stop
│   ├── regime_detector.py     — GaussianHMM (covariance_type=diag)
│   ├── correlation_filter.py  — фильтр корреляции + Redis persistence
│   ├── order_executor.py      — размер позиции + размещение ордера
│   ├── position_monitor.py    — фоновый монитор SL/TP/circuit breaker
│   ├── market_scanner.py      — параллельный сканер топ-N монет
│   ├── bybit_api.py           — CCXT async (spot + linear) + Redis lock
│   ├── data_loader.py         — OHLCV + кэш + технические индикаторы
│   ├── indicators.py          — EMA, RSI, MACD, BB, SMA, ATR, momentum
│   ├── news_analyzer.py       — NewsAPI + AI/VADER sentiment, Redis 15 мин
│   ├── trade_history.py       — SQLite логирование, win rate, EV
│   ├── telegram_notifier.py   — Telegram уведомления и подтверждения
│   ├── redis_client.py        — Redis обёртка (graceful degradation)
│   ├── health_server.py       — /health + /metrics (встроен в процесс бота)
│   ├── market_impact.py       — Almgren-Chriss рыночный импакт
│   ├── alpha_tester.py        — статистический анализ alpha стратегий
│   ├── experience_buffer.py   — replay buffer для SAC
│   ├── constants.py           — глобальные константы
│   ├── types.py               — TypedDict и Enum
│   └── logger.py              — JSON/text logging + _SecretFilter
│
│   └── dashboard/
│       ├── data.py            — слой доступа к Redis/SQLite
│       ├── index.html         — фронтенд
│       └── routers/
│           ├── api.py         — /api/* маршруты
│           ├── ops.py         — /metrics, /health, /webhook/alerts
│           └── frontend.py    — статика
│
├── reinforcement_learning/
│   ├── rl_env.py              — торговая среда (Gymnasium)
│   ├── train_sac.py           — обучение SAC-агента
│   └── tune_sac.py            — Optuna hyperparameter search
│
├── tests/                     — 593 тест, coverage 93%
│   ├── conftest.py
│   └── test_*.py
│
├── monitoring/
│   ├── prometheus.yml
│   ├── alertmanager.yml
│   ├── rules/bitbot.yml       — 6 правил алертинга
│   └── grafana/
│
├── RUNBOOK.md                 — что делать при каждом алерте
└── data/
    ├── backtest_results.json
    ├── cache/                 — CSV кэш OHLCV (TTL 24ч)
    └── trades.db              — SQLite история сделок
```

---

## Тесты

```bash
make test
# или: pytest tests/ --ignore=tests/integration -v --cov=src --cov-fail-under=50

# Integration-тесты против Bybit testnet (нужны ключи)
BYBIT_TESTNET_API_KEY=xxx BYBIT_TESTNET_API_SECRET=yyy \
  pytest tests/integration/ -m integration -v
```

**593 unit-теста** — индикаторы, стратегии, риск-менеджмент, портфель, CVaR, Kelly, Almgren-Chriss, корреляция, SAC-инференс, история сделок, конфиг, secret filter, AI парсинг, regime cache, reconciliation, e2e торговый цикл, OrderExecutor, PositionMonitor, BybitAPI, дашборд.  
**12 integration-тестов** — подключение к Bybit testnet, OHLCV, баланс, create/cancel ордер, round_quantity.  
Coverage: **93%**. CI падает при coverage < 50% (live-сервисы исключены из измерения).

---

## Перед реальной торговлей

1. Заполнить `.env` — `BYBIT_API_KEY`, `BYBIT_API_SECRET`, любой AI-ключ (`ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`), `TELEGRAM_BOT_TOKEN`
2. Запустить integration-тесты против testnet: `BYBIT_TESTNET_API_KEY=xxx ... pytest tests/integration/ -m integration`
3. Запустить `make paper-ai` на Bybit testnet минимум **1-2 недели** — проверить стабильность, логи, алерты
4. Запустить `make backtest` — изучить win rate и EV по монетам, выбрать `ACTIVE_STRATEGY`
5. Если нужен `hybrid` — обучить модель: `make train`, проверить тест-метрики
6. Прочитать [RUNBOOK.md](RUNBOOK.md) — знать что делать при каждом алерте
7. Выставить `DAILY_LOSS_LIMIT` и `CIRCUIT_BREAKER_LOSSES` по своему риск-аппетиту
8. Начать с минимального баланса и `RISK_PER_TRADE=0.01` (1%)
9. Включить `PAPER_TRADING=false` только после уверенности в сигналах

---

## Отказ от ответственности

Торговля криптовалютой сопряжена с высоким риском потери средств.  
Этот бот не является финансовым советником. Используйте на свой риск.  
Всегда начинайте с бумажной торговли (`PAPER_TRADING=true`).
