# BitbotBY — AI-торговый бот для Bybit

Гибридный крипто-трейдинг бот на Python: Claude AI + SAC нейросеть (Stable-Baselines3) + 9 стратегий.  
Работает в Docker, управляется через веб-интерфейс и Telegram. Мониторинг через Grafana.

---

## Возможности

- **4 режима торговли** — `local`, `ai`, `dqn`, `hybrid`
- **9 технических стратегий** — EMA, RSI, MACD, Bollinger Bands, Scalping, Swing, Breakout, Mean Reversion, Trend Following
- **Claude AI** анализирует рынок и выбирает стратегию/монету для каждой сделки
- **SAC нейросеть** (Stable-Baselines3) — RL-агент с reward на основе log-return − штраф за просадку
- **Train/test split 80/20** — оценка модели на out-of-sample данных после обучения
- **Анализ новостей** — NewsAPI + VADER sentiment, кэш 15 мин
- **Telegram-подтверждения** — кнопки Trade/Skip, авто-исполнение через 60 сек
- **Backtest** — walk-forward тест всех 9 стратегий на 6 месяцах истории
- **Win Rate и EV** — бэктест и live статистика отображаются раздельно
- **Trailing stop-loss** на основе ATR
- **Paper trading** — полный тест без реальных денег
- **Secrets validation** — проверка всех обязательных API-ключей при старте
- **Веб-дашборд** — баланс, позиции, история сделок (http://localhost:8080)
- **Grafana** — метрики в реальном времени (http://localhost:3000)
- **Supervisor** — автоматический рестарт при сбое + Telegram алерт

---

## Быстрый старт

### Требования

- Python 3.11+ или [Docker Desktop](https://docs.docker.com/get-docker/)
- API ключи: Bybit + Anthropic Claude + Telegram Bot (опционально)

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

### 3. Или запустить локально

```bash
pip install -r requirements.txt

make paper      # paper trading (MODE=local, без реальных ордеров)
make paper-ai   # paper trading с Claude AI
```

---

## Обучение SAC-модели

SAC-нейросеть нужна только для режимов `dqn` и `hybrid`. Если файл модели отсутствует — бот выведет ошибку при старте.

```bash
make train       # 500k шагов, ~60 мин на CPU
make train-long  # 1M шагов, ~2 ч на CPU
```

После обучения скрипт автоматически оценивает модель на out-of-sample тесте (последние 20% данных):

```
TEST SET (3400 candles) — SAC: $10832 (+8.3%) | Buy&Hold: $10650 (+6.5%) | Commissions: $47
```

Модель сохраняется в `models/sac_model.zip`. Обучение использует 6 месяцев BTC/USDT 15m с Bybit.

---

## Бэктест

```bash
make backtest          # BTC/USDT (по умолчанию)
make backtest-eth      # ETH/USDT
```

Результаты сохраняются в `data/backtest_results.json` и отображаются в веб-дашборде.

---

## Makefile команды

| Команда | Описание |
|---------|----------|
| `make train` | Обучить SAC-модель (500k шагов) |
| `make train-long` | Обучить SAC-модель (1M шагов) |
| `make backtest` | Walk-forward бэктест на BTC/USDT |
| `make paper` | Paper trading (MODE=local, без ордеров) |
| `make paper-ai` | Paper trading с Claude AI |
| `make live` | Живая торговля (нужны API-ключи) |
| `make test` | Запустить тесты с покрытием |
| `make lint` | flake8 + mypy |
| `make fmt` | black + isort |
| `make up` | `docker compose up -d --build` |
| `make down` | `docker compose down` |
| `make logs` | Логи бота в реальном времени |

---

## Конфигурация (.env)

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `MODE` | `ai` | Режим: `local` / `ai` / `dqn` / `hybrid` |
| `PAPER_TRADING` | `false` | Paper trading (без реальных ордеров) |
| `BYBIT_API_KEY` | — | **Обязателен** при `PAPER_TRADING=false` |
| `BYBIT_API_SECRET` | — | **Обязателен** при `PAPER_TRADING=false` |
| `ANTHROPIC_API_KEY` | — | **Обязателен** при `MODE=ai` или `hybrid` |
| `SAC_MODEL_PATH` | `models/sac_model.zip` | **Обязателен** при `MODE=dqn` или `hybrid` |
| `TELEGRAM_BOT_TOKEN` | — | Опционально, для уведомлений |
| `TELEGRAM_CHAT_ID` | — | Опционально, для уведомлений |
| `TRADING_SYMBOL` | `BTC/USDT` | Основная монета |
| `TIMEFRAME` | `15m` | Таймфрейм |
| `RISK_PER_TRADE` | `0.02` | Риск на сделку (2% баланса) |
| `MAX_POSITIONS` | `3` | Максимум одновременных позиций |
| `MIN_SIGNAL_CONFIDENCE` | `0.65` | Минимальный уровень уверенности |
| `DAILY_LOSS_LIMIT` | `0.05` | Лимит дневного убытка (5%) |
| `TRAILING_STOP_ATR_MULT` | `1.0` | Trailing stop (× ATR) |
| `AI_STRATEGY_SELECTION` | `false` | AI автовыбор стратегии |
| `SCAN_TOP_N` | `20` | Топ монет для сканирования |
| `AUTO_EXECUTE` | `false` | Авто-исполнение без Telegram |
| `TESTNET` | `false` | Bybit testnet |
| `GRAFANA_PASSWORD` | `bitbot` | Пароль Grafana (admin) |

При старте бот **автоматически проверяет** обязательные переменные и падает с понятной ошибкой, если что-то не задано.

---

## Режимы работы

| Режим | Описание | Требования |
|-------|----------|------------|
| `local` | Только 9 технических стратегий | Только Bybit API |
| `ai` | Claude AI анализирует рынок и выбирает стратегию | Bybit + Anthropic API |
| `dqn` | Только SAC-нейросеть | Bybit API + обученная модель |
| `hybrid` | SAC × 0.4 + AI × 0.6 | Bybit + Anthropic API + модель |

---

## Telegram

Создайте бота через [@BotFather](https://t.me/BotFather), получите токен и chat_id через [@userinfobot](https://t.me/userinfobot).

Перед каждой сделкой бот присылает сообщение:

```
BTC/USDT  --  BUY

Strategy: ema_crossover
Entry: $67,420.0000
SL: $66,200.0000   TP: $69,100.0000

AI confidence: 84%

Backtest:  63%  (180 trades)  EV: +1.24%
Live:      71%  (12 trades)   EV: +0.98%

Auto-execute in 60s
```

Нажмите **Trade** или **Skip**. Без ответа — сделка исполнится через 60 сек.

---

## Архитектура

```
supervisor.py / run_bot.py
  └── TradingBot (src/trading_bot.py)
        ├── MarketScanner      — топ монет по объёму + OHLCV снэпшот
        ├── DataLoader         — OHLCV через ccxt, кэш CSV 24ч
        ├── indicators.py      — RSI, MACD, BB, ATR, EMA, SMA
        ├── 9 стратегий        — сигналы buy/sell/hold + confidence
        ├── DQNSignal          — SAC-инференс (SB3), вес 40% в hybrid
        ├── Claude AI          — анализ рынка, вес 60% в hybrid
        ├── NewsAnalyzer       — VADER sentiment, Redis кэш 15 мин
        ├── SignalCombiner     — финальный взвешенный сигнал
        ├── RiskManager        — размер позиции, SL/TP, daily limit
        ├── TelegramNotifier   — Trade/Skip кнопки, 60с таймаут
        ├── BybitAPI           — ордера (ccxt async) + Redis lock
        └── TradeHistory       — SQLite, win rate, EV

dashboard.py (FastAPI :8080)
  ├── /           — веб-интерфейс
  └── /metrics    — Prometheus метрики

docker-compose.yml
  ├── bitbot_bot        — торговый бот
  ├── bitbot_dashboard  — веб-интерфейс + /metrics
  ├── bitbot_redis      — Redis (кэш, pub/sub, locks)
  ├── bitbot_prometheus — сбор метрик (:9090)
  └── bitbot_grafana    — дашборды (:3000)

reinforcement_learning/
  ├── rl_env.py      — TradingEnv (Gymnasium): log-return reward − drawdown penalty
  └── train_sac.py   — обучение SAC: 80/20 train/test split, eval на out-of-sample
```

---

## Структура файлов

```
BitbotBY/
├── src/
│   ├── trading_bot.py        — главный цикл и оркестратор
│   ├── strategies.py         — 9 стратегий
│   ├── signal_combiner.py    — объединение сигналов SAC + AI
│   ├── indicators.py         — технические индикаторы
│   ├── risk_management.py    — управление рисками
│   ├── portfolio_manager.py  — портфель и trailing stop
│   ├── ai_analyzer.py        — Claude API интеграция
│   ├── dqn_signal.py         — SAC-инференс (SB3)
│   ├── market_scanner.py     — сканер монет + OHLCV снэпшот
│   ├── news_analyzer.py      — новости и sentiment
│   ├── bybit_api.py          — биржевой адаптер (ccxt async)
│   ├── data_loader.py        — загрузка OHLCV
│   ├── redis_client.py       — Redis клиент
│   ├── telegram_notifier.py  — Telegram уведомления
│   └── trade_history.py      — история сделок (SQLite)
├── reinforcement_learning/
│   ├── rl_env.py             — торговая среда (Gymnasium)
│   └── train_sac.py          — обучение SAC-агента
├── tests/                    — 75 тестов (pytest)
├── monitoring/
│   ├── prometheus.yml        — конфиг Prometheus
│   └── grafana/              — provisioning + дашборд BitbotBY
├── config.py                 — конфигурация (dataclass + validate)
├── backtest.py               — walk-forward бэктест
├── dashboard.py              — FastAPI дашборд + /metrics
├── supervisor.py             — менеджер процессов
├── Makefile                  — make train / backtest / paper / up
├── mypy.ini                  — настройки type checker
├── setup.cfg                 — flake8 + isort
├── Dockerfile
└── docker-compose.yml
```

---

## Тесты

```bash
make test
# или: python3 -m pytest tests/ -v --tb=short --cov=src
```

75 тестов: индикаторы, стратегии, риск-менеджмент, портфель, SAC-инференс, история сделок.  
CI (GitHub Actions): black + isort + flake8 + mypy + pytest с coverage.

---

## Перед реальной торговлей

1. Запустить `make paper` минимум 2-3 дня — убедиться что логика верна
2. Запустить `make backtest` — изучить win rate и EV по каждой стратегии
3. Если нужен `dqn`/`hybrid` — обучить модель: `make train`, проверить тест-метрики
4. Проверить Telegram-уведомления (Trade/Skip работают)
5. Выставить `DAILY_LOSS_LIMIT` по своему риск-аппетиту
6. Включить `AUTO_EXECUTE=true` только после уверенности в боте
7. Начать с минимальным балансом и `RISK_PER_TRADE=0.01` (1%)

---

## Отказ от ответственности

Торговля криптовалютой сопряжена с высоким риском потери средств.  
Этот бот не является финансовым советником. Используйте на свой риск.  
Всегда начинайте с бумажной торговли (`PAPER_TRADING=true`).
