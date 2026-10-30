# Конфигурация и справочник переменных

> Обычно ничего из этого руками задавать не нужно: настройки меняются в приложении (Настройки и вкладка «Бот») и в Telegram,
> а ключи хранятся только зашифрованными (Настройки → Подключения). Переменные окружения нужны для
> разработки без Docker и для старого запуска через `deploy/legacy/` (он читает `.env`).
> Значения по умолчанию — в [`config.py`](../config.py) и [`.env.example`](../.env.example).

Полный список переменных с описанием — в [`.env.example`](../.env.example) (комментарии на русском ко
всем секциям: Bybit API, AI-провайдер, режим торговли, Telegram, торговые параметры, управление
позициями, риск-менеджмент, macro blackout, онлайн-обучение SAC, внешние сигналы, мониторинг).

| Категория | Ключевые переменные |
|---|---|
| Биржа | `BYBIT_API_KEY/SECRET`, `MARKET_TYPE`, `BYBIT_TESTNET` |
| AI | `AI_PROVIDER`, `<PROVIDER>_API_KEY`, `AI_DAILY_BUDGET`, `AI_MAX_SYMBOLS` |
| Режим | `MODE` (`local`/`ai`/`dqn`/`hybrid`), `PAPER_TRADING`, `AUTO_EXECUTE` |
| Профиль | `MARKET_PROFILE` (`bluechip`/`altcoin`/`meme`) |
| Позиции | `MAX_POSITIONS` (1–20), `RISK_PER_TRADE`, `LEVERAGE`, `LEVERAGE_MODE` |
| Риск | `DAILY_LOSS_LIMIT`, `DRAWDOWN_HALT_HOURS`, `MAX_CORRELATION`, `MAX_SPREAD_PCT` |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_CONFIRM_TIMEOUT` |
| Инфра | `REDIS_HOST/PASSWORD`, `DASHBOARD_USER/PASSWORD` |

Большинство параметров (режим, стратегии, риск-профиль, часы торговли, число позиций) меняются
**прямо из приложения и Telegram без перезапуска** — хранятся в Redis с AOF-персистентностью.

---

## Все переменные

Построчное описание каждой переменной. Разворачивай нужную категорию.

<details>
<summary><b>🔑 Bybit API</b></summary>

```env
BYBIT_API_KEY=           # Получить: bybit.com → Account → API Management
BYBIT_API_SECRET=        # ⚠ Никогда не публиковать, не коммитить в git

TESTNET=false            # true → торговать на testnet.bybit.com (без реальных денег)
                         # Ключи testnet создаются отдельно от production!

MARKET_TYPE=linear       # linear → perpetual futures (РЕКОМЕНДУЕТСЯ, дефолт при старте)
                         #   + Умеет открывать SHORT позиции
                         #   + Есть funding rate (дополнительный сигнал)
                         #   + Плечо до 100x (мы ставим max=3)
                         # spot → только LONG, нет шорта, нет плеча
                         # Это только стартовое значение при первом запуске —
                         # дальше режим переключается на лету через Telegram
                         # (💱 Рынок в Настройках), без правки .env и рестарта.

LEVERAGE=3               # МАКСИМАЛЬНОЕ плечо. Бот рассчитывает динамически от 1 до LEVERAGE.
                         # Рекомендуется: 2–3 для старта. Не ставить >5 пока нет статистики.
                         # LEVERAGE=1 = торговля без плеча (как спот, но на фьючерсах)
```

> **Права API ключей:** для futures — `Contract — Orders` и `Contract — Positions`. Если планируете
> когда-либо переключаться на spot через Telegram — включите также `Spot — Trade` сразу, оба
> права спокойно живут на одном ключе. Право на вывод средств — **НЕ включать**.

</details>

<details>
<summary><b>🧠 AI-провайдер</b></summary>

```env
AI_PROVIDER=auto         # auto      → пробует Claude → OpenAI → DeepSeek → Groq → Gemini → локальные стратегии
                         # anthropic → только Claude
                         # deepseek  → только DeepSeek
                         # groq      → только Groq (Llama 3.3 70B, самый быстрый)
                         # openai    → только ChatGPT
                         # gemini    → только Google Gemini Flash

# ── Groq — БЕСПЛАТНЫЙ тир, 100k токенов/день, Llama 3.3 70B ──
# console.groq.com → API Keys → Create
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# ── Google Gemini Flash — БЕСПЛАТНЫЙ тир, 1500 запросов/день ──
# aistudio.google.com → Get API Key (бесплатно, без карты)
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash-lite

# ── DeepSeek — САМЫЙ ДЕШЁВЫЙ (~$0.002 за запрос) ──
# platform.deepseek.com → API Keys → Create
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat

# ── Claude (Anthropic) — лучшее качество анализа ──
# console.anthropic.com → API Keys → Create Key
ANTHROPIC_API_KEY=sk-ant-...
AI_MODEL=claude-sonnet-4-6

# ── ChatGPT (OpenAI) ──
# platform.openai.com → API Keys
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

AI_DAILY_BUDGET=500      # Лимит AI-вызовов в сутки (UTC). Сверх лимита → локальные стратегии.
                         # Bluechip: 96 запросов/день. Altcoin: 288 запросов/день.
```

> **Совет для старта (бесплатно):** добавь Groq + Gemini — хватает на весь день для обоих профилей.
> Для лучшего качества дополнительно пополни DeepSeek на $5 (~9–26 дней торговли).

**Цепочка надёжности — бот не останавливается никогда:**

```
Claude (Anthropic) — лучшее качество, платный
    ↓ нет баланса → уведомление в Telegram
OpenAI GPT-4o-mini — хорошее качество, платный
    ↓ нет баланса → уведомление в Telegram
DeepSeek — хорошее качество, самый дешёвый, ~$5 на 9–26 дней
    ↓ нет баланса → уведомление в Telegram
Groq + Gemini — бесплатные, хватает на весь день
    ↓ исчерпались к концу дня → уведомление с обратным отсчётом до сброса
Локальные стратегии — 10 базовых + до 4 профильных стратегий без AI, всегда работают
```

Настроены только те провайдеры, у которых есть ключ в `.env`. Реальный порядок и статус каждого —
в Telegram: ⚙️ Настройки → 🤖 AI-провайдер.

| Провайдер | Стоимость | Качество | Лимит |
|---|---|---|---|
| DeepSeek | ~$5 на 9–26 дней | Хорошее | по балансу |
| Groq (Llama 3.3 70B) | бесплатно | Хорошее | 100k токенов/день |
| Gemini 2.0 Flash | бесплатно | Хорошее | 1500 запросов/день |
| Claude Sonnet | ~$0.01/запрос | Отличное | по балансу |
| OpenAI GPT-4o-mini | ~$0.003/запрос | Хорошее | по балансу |

</details>

<details>
<summary><b>🔀 Режим торговли</b></summary>

```env
MODE=ai                  # ai     → AI анализирует каждую монету и генерирует сигнал
                         # local  → только базовые стратегии (10+профильные), без AI (бесплатно)
                         # dqn    → только SAC нейросеть (нужна обученная модель)
                         # hybrid → SAC + AI должны согласиться (самый консервативный)

AUTO_EXECUTE=false       # false → бот присылает сигнал с кнопками и таймером.
                         #         Авто-исполнение по истечении таймаута.
                         #         Нажми Skip чтобы отменить конкретную сделку.
                         # true  → бот торгует сразу, без диалога. Приходит уведомление о факте.
                         # Рекомендуется: начать с false (2 недели), потом включить true

PAPER_TRADING=true       # true  → симуляция, реальных ордеров НЕТ (для старта)
                         # false → реальная торговля (только после проверки сигналов!)
                         # Переключение в реальном времени через Telegram:
                         # ⚙️ Настройки → 📄 Режим: PAPER → Live (требует кодовое слово)

TRADING_MODE_PIN=        # Кодовое слово для переключения Paper ↔ Live через Telegram.
                         # Оставить пустым — кнопка переключения не показывается.
                         # Пример: TRADING_MODE_PIN=mysecret2024
                         # После ввода слова бот отправляет уведомление 🚨 ВАЖНО!

SCAN_TOP_N=20            # Сколько монет сканировать (топ по объёму за 24ч).
                         # Больше = больше возможностей, но больше API-запросов.

MAX_POSITIONS=3          # Начальное значение (1–20). Меняется в реальном времени
                         # через Telegram: ⚖️ Риск-профиль → ➖ / ➕.
                         # В paper режиме рекомендуется 10–15 для быстрого обучения SAC.
```

</details>

<details>
<summary><b>💬 Telegram</b></summary>

```env
TELEGRAM_BOT_TOKEN=      # Шаг 1: написать @BotFather → /newbot → скопировать токен
                         # Формат: 1234567890:ABCdefGHI...

TELEGRAM_CHAT_ID=        # Шаг 2: написать @userinfobot → скопировать "Your ID"
                         # ⚠ ВАЖНО: это ВАШ числовой ID, не ID бота.
                         # Формат: 123456789 (только цифры, без @)

TELEGRAM_CONFIRM_TIMEOUT=60   # Секунд ждать ответа на кнопки Trade/Skip (при AUTO_EXECUTE=false).
                               # По истечении → авто-исполняет сделку.
                               # Менять без перезапуска: ⚙️ Настройки → ⏱ Таймаут подтверждения

SILENT_DEATH_HOURS=6.0   # Алерт в Telegram если нет ни одной сделки за N часов.
                         # Помогает вовремя заметить зависание бота. 0 = выключен.
```

</details>

<details>
<summary><b>📈 Торговые параметры</b></summary>

```env
TIMEFRAME=15m            # Таймфрейм свечей для стратегий.
                         # 15m — хороший баланс. Варианты: 1m (шумно), 5m, 15m, 1h, 4h.

INITIAL_BALANCE=10000.0  # Начальный баланс для paper trading (для расчётов и отображения).
                         # В реальной торговле бот берёт баланс с биржи автоматически.

RISK_PER_TRADE=0.02      # Риск на одну сделку как доля баланса.
                         # 0.02 = 2% баланса. При $10k → рискуем $200 на сделку.
                         # Рекомендуется: 0.01 для старта, потом 0.02.

MIN_SIGNAL_CONFIDENCE=0.75        # Минимальная уверенность сигнала в live режиме (0–1).
                                  # Повысить до 0.80–0.85 для более консервативной торговли.
MIN_SIGNAL_CONFIDENCE_PAPER=0.60  # Порог для paper режима — ниже, чтобы накопить больше
                                  # сделок для обучения SAC. Меняется через Telegram без
                                  # перезапуска: ⚖️ Риск-профиль → 🎯 Paper ▼/▲.

TRADING_INTERVAL=60      # Fallback-интервал цикла в секундах (без профиля).
                         # При активном профиле используется интервал из профиля:
                         # bluechip → 900с (15 мин), altcoin → 300с (5 мин).
```

</details>

<details>
<summary><b>🎯 Управление позициями</b></summary>

```env
# ── Частичная фиксация прибыли ──────────────────────────────────────────
PARTIAL_TP_ENABLED=true  # При достижении 60% пути до TP:
                         # 1. Закрывает 50% позиции (фиксирует часть прибыли)
                         # 2. Переносит SL на breakeven (остаток торгуется без риска)
PARTIAL_TP_TRIGGER=0.6   # Доля пути entry→TP для срабатывания. 0.6 = 60%.
PARTIAL_TP_FRACTION=0.5  # Какую долю позиции закрыть. 0.5 = 50%.

# ── Trailing stop ────────────────────────────────────────────────────────
TRAILING_STOP_ATR_MULT=1.0   # SL подтягивается за ценой с шагом 1×ATR.
                              # 0.5 = тесный стоп (больше срабатываний).
                              # 1.5 = широкий стоп (меньше срабатываний, больше просадка).

# ── Drawdown scaling ─────────────────────────────────────────────────────
DRAWDOWN_SCALE_ENABLED=true       # При просадке от пика — уменьшить размер позиций.
DRAWDOWN_SCALE_THRESHOLD=0.10     # Порог: 0.10 = при просадке >10% от пика баланса.
DRAWDOWN_SCALE_FACTOR=0.5         # Множитель размера при просадке: 0.5 = торговать вдвое меньше.
                                  # Пример: баланс был $10k, упал до $9k → позиции ×0.5.

# ── Фильтр ликвидности ───────────────────────────────────────────────────
MIN_VOLUME_USDT=1000000  # Минимальный объём монеты за 24ч в USDT.
                         # $1M = только ликвидные монеты. Ниже → риск манипуляций.
MAX_SPREAD_PCT=0.3       # Максимальный спред bid/ask в %.
                         # 0.3% = на входе и выходе теряем не более 0.3%.

# ── Cooldown (защита от оверtrading) ─────────────────────────────────────
SYMBOL_COOLDOWN_HOURS=3  # Не переоткрывать один и тот же символ N часов после закрытия.
                         # Защищает от ситуации когда бот многократно заходит в один токен
                         # после быстрых SL (например 7 раз в TAC/USDT за один день).
                         # 0 = выключен. Рекомендуется: 2–4 часа.

# ── Торговые часы ────────────────────────────────────────────────────────
TRADING_HOURS=           # Пусто = торговать круглосуточно (РЕКОМЕНДУЕТСЯ для крипты).
                         # "8-22" = только с 8:00 до 22:00 UTC.
                         # Не рекомендуется: крипта активна 24/7, ночные движения реальные.
```

</details>

<details>
<summary><b>🛡 Риск-менеджмент</b></summary>

```env
DAILY_LOSS_LIMIT=0.05    # Дневной лимит убытка: 0.05 = 5% баланса.
                         # При превышении → все позиции закрываются, торговля до UTC-дня.
                         # Рекомендуется: 0.03 для старта, потом 0.05.

# ── Black swan / аварийные стоп-краны ────────────────────────────────────
MAX_DRAWDOWN_PERCENT=0.20     # Hard drawdown halt: просадка equity от пика ≥ 20%.
                               # Equity = баланс + unrealized PnL (точнее чем free USDT).
                               # При срабатывании: все позиции закрыть → пауза N часов.

DRAWDOWN_HALT_HOURS=4          # Пауза в часах после срабатывания hard drawdown halt.

DRAWDOWN_CONFIRM_CYCLES=3      # Flash-защита: halt только после N подтверждений подряд.
                               # 1 цикл ≈ 30с. 3 = сработает через ~90с устойчивой просадки.
                               # Flash crash (восстановился за секунды) не триггернёт.
                               # Меняется без перезапуска: Telegram → ⚖️ Риск-профиль → 🛡 Flash-защита ±

ATR_SPIKE_MULT=3.0             # ATR spike: текущий ATR в N× выше среднего → закрыть позиции.
                               # 3.0 = только экстремальная волатильность (война, крах).

ATR_SPIKE_LOOKBACK=20          # Окно для расчёта среднего ATR (в свечах).

CIRCUIT_BREAKER_LOSSES=3 # Автостоп после N убыточных сделок подряд.
                         # 3 = три стопа подряд → бот останавливается до ручного перезапуска.
                         # 0 = выключен. Рекомендуется: 3–5.

MAX_CORRELATION=0.7      # Блокировать сигнал если |корреляция| с открытой позицией ≥ порога.
                         # 0.7 = не откроем ETH long если уже держим BTC long.
                         # 0.0 = фильтр выключен. Рекомендуется: 0.7.
MAX_CORRELATION_PAPER=0.85  # Порог корреляции для paper режима.
                         # В боковом рынке крипто коррелирует на 0.70–0.77 —
                         # порог 0.7 блокировал бы почти все входы.
                         # 0.85 позволяет накопить больше сделок для SAC.
CORRELATION_WINDOW=50    # Окно расчёта корреляции в барах (50 × 15m = 12.5 часов).
```

</details>

<details>
<summary><b>📅 Macro blackout</b></summary>

```env
FINNHUB_API_KEY=         # Ключ Finnhub (бесплатный тир): finnhub.io → Dashboard → API Key
                         # Если не задан — macro blackout отключается автоматически.

MACRO_BLACKOUT_ENABLED=true  # Блокировать новые позиции за 30 мин до/после:
                             # FOMC (ставка ФРС), CPI (инфляция), NFP (занятость), GDP, PCE.
                             # Эти события вызывают резкие движения — лучше переждать.
```

</details>

<details>
<summary><b>🎓 Онлайн-обучение SAC</b></summary>

```env
ONLINE_LEARNING_MODE=periodic  # disabled → выключено (SAC не обучается автоматически)
                                # online   → gradient steps после КАЖДОЙ сделки
                                #            ⚠ РИСКОВАННО: серия убытков может испортить модель
                                # periodic → переобучение каждые N сделок в фоне [РЕКОМЕНДУЕТСЯ]
                                # hybrid   → periodic + динамические веса стратегий в реальном времени

ONLINE_LEARNING_TRIGGER=50      # Переобучение каждые 50 закрытых сделок (для periodic/hybrid).
                                # 3 позиции × 10 сделок/день ≈ переобучение каждые ~17 часов.

ONLINE_LEARNING_GRADIENT_STEPS=50  # Шагов градиента за один online-апдейт.
                                    # Только для режима online. Рекомендуется ≤ 50.

TRAIN_TOP_N=20           # Сколько топ-монет по объёму брать для обучения SAC.
                         # Менять не перезапуская — команда Telegram: /trainn 15
                         # При активном профиле переопределяется автоматически:
                         #   bluechip → 20 монет (15m),  altcoin → 50 монет (5m)
TRAIN_MIN_CANDLES=2880   # Минимум свечей для символа при обучении SAC.
                         # 2880 = ~1 месяц 15m-данных. Символы с меньшим количеством
                         # пропускаются — слишком мало истории для обучения.
TOTAL_TIMESTEPS=500000   # Шагов обучения SAC (переопределяется env при make train).
                         # 500k ≈ 60 мин на CPU. 1M = более качественная модель (~2ч).
SAC_MODEL_PATH=models/sac_model.zip  # Путь к файлу SAC-модели.
                                     # Переопределяется профилем:
                                     #   bluechip → models/sac_model.zip
                                     #   altcoin  → models/sac_model_altcoin.zip
```

</details>

<details>
<summary><b>🌐 Внешние сигналы (опционально)</b></summary>

Все ключи опциональны. Без них бот работает, просто без этих источников сигналов.

```env
# Twitter/X sentiment — VADER по твитам монет (активен только при профиле Meme)
# Тариф Basic: $100/мес, ~10 000 твитов/мес. Free-тир READ не поддерживает.
# developer.twitter.com → Projects & Apps → Keys and Tokens → Bearer Token
# Если не задан — TwitterAnalyzer молча возвращает 0.0
TWITTER_BEARER_TOKEN=
TWITTER_MAX_RESULTS=10   # Твитов на запрос (10 = экономия лимита Basic)

# Telegram sentiment — мониторинг публичных памп-каналов (только профиль Meme)
# Бесплатно через Telethon MTProto. Получить: my.telegram.org → API development tools
# Если не задан — TelegramSentiment молча возвращает 0.0
TELEGRAM_SENTIMENT_API_ID=
TELEGRAM_SENTIMENT_API_HASH=
TELEGRAM_SENTIMENT_PHONE=+375291234567   # Формат с кодом страны
TELEGRAM_SENTIMENT_CHANNELS=CryptoMoonAlerts,pumpsignalscrypto,memecoinsnews

# Параметры опросчика (SentimentPoller, активен только при профиле Meme)
SENTIMENT_POLL_INTERVAL=300   # Интервал опроса в секундах (каждые 5 мин)
SENTIMENT_CACHE_TTL=600       # TTL кэша в Redis (10 мин)
SENTIMENT_BUY_THRESHOLD=0.4   # Score ≥0.4 → buy-сигнал
SENTIMENT_SELL_THRESHOLD=-0.4 # Score ≤-0.4 → sell-сигнал

# Reddit sentiment — r/CryptoCurrency
# reddit.com/prefs/apps → create app (тип: script)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=BitbotBY/1.0

# Glassnode — on-chain метрики (stablecoin supply и др.)
# glassnode.com → Account → API Access
GLASSNODE_API_KEY=

# NewsAPI — новостной sentiment (CoinDesk, CoinTelegraph)
# newsapi.org → Get API Key (бесплатно)
NEWS_API_KEY=
NEWS_UPDATE_INTERVAL=900   # Обновлять новости каждые 15 минут
```

</details>

<details>
<summary><b>🖥 Мониторинг и инфраструктура</b></summary>

```env
HEALTH_PORT=8080         # Порт HTTP health-сервера:
                         # GET /health  → JSON статус бота
                         # GET /metrics → Prometheus метрики
                         # 0 = выключен

DASHBOARD_API_KEY=       # Опциональный ключ для веб-дашборда.
                         # Если задан → все запросы требуют заголовок: X-API-Key: значение

GRAFANA_PASSWORD=admin   # Пароль Grafana (http://localhost:3000, логин: admin)

REDIS_HOST=localhost     # В Docker автоматически переопределяется на "redis".
REDIS_PORT=6379
REDIS_PASSWORD=          # Рекомендуется задать в production.
```

</details>

---

