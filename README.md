# BitbotBY — AI-торговый бот для Bybit

Гибридный крипто-трейдинг бот на Python: Claude AI + DQN нейросеть + 9 стратегий.  
Работает в Docker, управляется через веб-интерфейс и Telegram.

---

## Возможности

- **4 режима торговли** — `ai`, `hybrid`, `dqn`, `local`
- **9 технических стратегий** — EMA, RSI, MACD, Bollinger Bands, Scalping, Swing, Breakout, Mean Reversion, Trend Following
- **Claude AI** выбирает лучшую стратегию и монету для каждой сделки
- **DQN нейросеть** (PyTorch) — второй независимый сигнал
- **Анализ новостей** — NewsAPI + VADER sentiment, кэш 15 мин
- **Telegram-подтверждения** — кнопки ✅/❌, авто-исполнение через 60 сек
- **Backtest** — walk-forward тест всех стратегий на 6 месяцах истории
- **Win Rate и EV** — бэктест и live статистика отображаются раздельно
- **Trailing stop-loss** на основе ATR
- **Бумажная торговля** — тест без реальных денег
- **Веб-дашборд** — баланс, позиции, история сделок в браузере
- **Supervisor** — автоматический рестарт при сбое + Telegram алерт

---

## Быстрый старт

### Требования

- [Docker Desktop](https://docs.docker.com/get-docker/) (Windows / Mac / Linux)
- Ключи API: Bybit + Anthropic Claude + Telegram Bot

### 1. Клонировать репозиторий

```bash
git clone <repo-url>
cd BitbotBY
```

### 2. Первичная настройка (один раз)

```bash
./setup.sh       # Mac / Linux
setup.bat        # Windows
```

Wizard запросит ваши ключи API, запишет `.env` и соберёт Docker образ (~3-5 минут).

### 3. Запуск

```bash
./start.sh       # Mac / Linux
start.bat        # Windows
```

Бот стартует. Веб-интерфейс откроется автоматически: **http://localhost:8080**

### Управление

```bash
./stop.sh        # остановить
./logs.sh        # логи бота в реальном времени
```

---

## Веб-дашборд

После запуска открыть в браузере: `http://localhost:8080`

- Текущий баланс и PnL
- Win Rate и количество сделок
- Открытые позиции
- История сделок
- Результаты бэктеста по всем 9 стратегиям
- Статус бота (работает / остановлен)
- Обновляется каждые 5 секунд

---

## Telegram

Создайте бота через [@BotFather](https://t.me/BotFather), получите токен и свой chat_id через [@userinfobot](https://t.me/userinfobot).

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

Нажмите **Trade** или **Skip**. Если не ответить — сделка исполнится автоматически через 60 сек.

---

## Конфигурация

Все параметры в файле `.env`. Основные:

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `MODE` | `ai` | Режим: `ai` / `hybrid` / `dqn` / `local` |
| `PAPER_TRADING` | `true` | Бумажная торговля (без реальных ордеров) |
| `AUTO_EXECUTE` | `false` | Исполнять сделки автоматически |
| `TRADING_SYMBOL` | `BTC/USDT` | Основная монета (если AI не выбирает) |
| `TIMEFRAME` | `15m` | Таймфрейм |
| `RISK_PER_TRADE` | `0.02` | Риск на сделку (2% баланса) |
| `MAX_POSITIONS` | `3` | Максимум одновременных позиций |
| `MIN_SIGNAL_CONFIDENCE` | `0.65` | Минимальный уровень уверенности сигнала |
| `DAILY_LOSS_LIMIT` | `0.05` | Лимит дневного убытка (5% → стоп) |
| `TRAILING_STOP_ATR_MULT` | `1.0` | Trailing stop (× ATR, 0 = выключен) |
| `AI_STRATEGY_SELECTION` | `true` | AI сам выбирает стратегию |
| `SCAN_TOP_N` | `20` | Топ монет для сканирования |
| `TESTNET` | `false` | Bybit testnet |

---

## Режимы работы

| Режим | Описание |
|-------|----------|
| `local` | Только 9 технических стратегий, без внешних API |
| `dqn` | Только DQN нейросеть (нужна обученная модель) |
| `ai` | Claude AI анализирует рынок и выбирает стратегию |
| `hybrid` | DQN × 0.4 + AI × 0.6 — взвешенный сигнал |

---

## Бэктест

Запустить тест всех 9 стратегий на исторических данных:

```bash
# Внутри Docker контейнера
docker compose exec bot python backtest.py

# Без Docker
python backtest.py
```

Результаты сохранятся в `data/backtest_results.json` и отобразятся в веб-дашборде.  
Win rate из бэктеста и из live-торговли всегда отображаются **отдельно**.

---

## Обучение DQN (опционально)

Нужно только для режимов `dqn` и `hybrid`:

```bash
python reinforcement_learning/train_dqn.py
```

Модель сохранится в `models/dqn_model.pth`. Укажите путь в `.env`:

```
DQN_MODEL_PATH=models/dqn_model.pth
```

---

## Архитектура

```
supervisor.py / run_bot.py
  └── TradingBot (src/trading_bot.py)
        ├── MarketScanner      — топ монет по объёму
        ├── DataLoader         — OHLCV через ccxt, кэш CSV 24ч
        ├── indicators.py      — RSI, MACD, BB, ATR, EMA, SMA
        ├── 9 стратегий        — сигналы buy/sell/hold + confidence
        ├── DQN (PyTorch)      — нейросеть, вес 40% в hybrid
        ├── Claude AI          — batch анализ, вес 60% в hybrid
        ├── NewsAnalyzer       — VADER sentiment, Redis кэш 15 мин
        ├── SignalCombiner     — финальный взвешенный сигнал
        ├── RiskManager        — размер позиции, SL/TP, daily limit
        ├── TelegramNotifier   — кнопки ✅/❌, 60с таймаут
        ├── BybitAPI           — исполнение ордеров (ccxt async)
        └── TradeHistory       — SQLite, win rate, EV

dashboard.py (FastAPI)
  └── http://localhost:8080   — веб-интерфейс

docker-compose.yml
  ├── bitbot_bot              — торговый бот
  ├── bitbot_dashboard        — веб-интерфейс
  └── bitbot_redis            — Redis (кэш, pub/sub, locks)
```

---

## Структура файлов

```
BitbotBY/
├── src/
│   ├── trading_bot.py        — главный цикл и оркестратор
│   ├── strategies.py         — 9 стратегий
│   ├── signal_combiner.py    — объединение сигналов DQN + AI
│   ├── indicators.py         — технические индикаторы
│   ├── risk_management.py    — управление рисками
│   ├── portfolio_manager.py  — портфель и trailing stop
│   ├── ai_analyzer.py        — Claude API интеграция
│   ├── dqn_signal.py         — DQN inference
│   ├── news_analyzer.py      — новости и sentiment
│   ├── bybit_api.py          — биржевой адаптер
│   ├── data_loader.py        — загрузка OHLCV
│   ├── market_scanner.py     — сканер монет
│   ├── redis_client.py       — Redis клиент
│   ├── telegram_notifier.py  — Telegram уведомления
│   └── trade_history.py      — история сделок (SQLite)
├── reinforcement_learning/
│   ├── rl_agent.py           — DQN агент
│   ├── rl_env.py             — торговая среда (Gym)
│   └── train_dqn.py          — скрипт обучения
├── tests/                    — 46 тестов (pytest)
├── config.py                 — Pydantic BaseSettings
├── backtest.py               — walk-forward бэктест
├── dashboard.py              — веб-интерфейс FastAPI
├── supervisor.py             — менеджер процессов
├── Dockerfile
├── docker-compose.yml
├── setup.sh / setup.bat      — первичная настройка
├── start.sh / start.bat      — запуск
├── stop.sh  / stop.bat       — остановка
└── logs.sh                   — просмотр логов
```

---

## Тесты

```bash
python -m pytest tests/ -v
```

46 тестов: индикаторы, риск-менеджмент, стратегии, история сделок.

---

## Важно перед реальной торговлей

1. Запустить с `PAPER_TRADING=true` минимум 3-5 дней
2. Проверить Telegram-уведомления (кнопки работают)
3. Запустить `backtest.py` — изучить результаты по стратегиям
4. Убедиться что `data/healthcheck.txt` обновляется каждые 30 сек
5. Установить `DAILY_LOSS_LIMIT` по своему риск-аппетиту
6. Включить `AUTO_EXECUTE=true` только после уверенности в боте

---

## Отказ от ответственности

Торговля криптовалютой сопряжена с высоким риском потери средств.  
Этот бот не является финансовым советником. Используйте на свой риск.  
Всегда начинайте с бумажной торговли (`PAPER_TRADING=true`).
