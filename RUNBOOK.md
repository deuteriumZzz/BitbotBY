# BitbotBY — Runbook

Что делать когда что-то пошло не так. Все команды выполняются из корня проекта.

---

## BotProcessDown

**Что:** Prometheus не получает метрики с `bot:8080` более 2 минут.

**Причины (по вероятности):**
1. Контейнер `bitbot_bot` упал или завис
2. Health server не запустился (HEALTH_PORT не задан или занят)
3. Нехватка памяти — OOM killer убил процесс

**Действия:**
```bash
# 1. Проверить статус
docker compose ps

# 2. Посмотреть последние логи
docker compose logs --tail=100 bot

# 3. Если контейнер не запущен — перезапустить
docker compose restart bot

# 4. Если падает в loop — найти причину
docker compose logs --tail=50 bot | grep -E "ERROR|CRITICAL|Traceback"

# 5. Проверить память хоста
free -h
```

**Ожидаемое время восстановления:** 1–3 минуты после `restart`.

---

## DashboardDown

**Что:** Prometheus не получает метрики с `dashboard:8080` более 2 минут.

**Действия:**
```bash
docker compose logs --tail=50 dashboard
docker compose restart dashboard
```

Не критично для торговли — бот работает независимо от dashboard.

---

## BotStalled

**Что:** Бот жив (отвечает на `/health`), но `bot_cycles_total` не растёт 10+ минут.

**Причины:**
1. Trading loop завис на `await` (зависший запрос к бирже)
2. Deadlock в `_monitored_lock`
3. ccxt rate limit — бот ждёт слишком долго

**Действия:**
```bash
# 1. Посмотреть последние строки
docker compose logs --tail=50 bot

# 2. Если видны "Waiting for rate limit" или зависшие await
docker compose restart bot

# 3. Если повторяется — проверить доступность биржи
curl -s https://api.bybit.com/v5/market/time | python3 -m json.tool
```

---

## HighConsecutiveLosses

**Что:** `bot_consecutive_losses >= 3` — серия убытков подряд, близко к circuit breaker.

**Это НЕ баг** — нормальная торговая ситуация. Может означать:
- Плохие рыночные условия (высокая волатильность, резкий тренд против стратегии)
- Неподходящая стратегия для текущего рыночного режима
- Проблема с ATR-расчётом SL/TP

**Действия:**
```bash
# 1. Посмотреть последние сделки в dashboard
open http://localhost:8080

# 2. Если circuit breaker сработал — бот остановил торговлю сам.
# Подождите или перезапустите после анализа:
docker compose restart bot

# 3. Рассмотреть смену стратегии через .env
# ACTIVE_STRATEGY=mean_reversion
docker compose restart bot
```

---

## LargePnLLoss

**Что:** Накопленный PnL упал ниже -500 USDT.

**СТОП. Сначала разберитесь, потом перезапускайте.**

**Действия:**
```bash
# 1. Остановить бота
docker compose stop bot

# 2. Проверить открытые позиции в Grafana или dashboard
open http://localhost:8080

# 3. При необходимости — закрыть позиции вручную на Bybit

# 4. Разобраться в причине по логам
docker compose logs --tail=200 bot | grep -E "TRADE|POSITION|ERROR"

# 5. Перезапустить только после анализа
docker compose start bot
```

---

## LowBalance

**Что:** `bot_paper_balance_usdt < 5000` — баланс упал ниже 50% от стартового.

**Действия:**
```bash
# Проверить баланс через dashboard
open http://localhost:8080

# Снизить риск на сделку в .env
# RISK_PER_TRADE=0.01  (с 2% до 1%)
docker compose restart bot
```

---

## LowWinRate

**Что:** Win rate ниже 35% при более чем 100 циклах подряд.

**Сигнал к пересмотру стратегии**, не к паническому перезапуску.

```bash
# Запустить свежий бэктест для выбора лучшей стратегии
docker compose exec bot python backtest.py

# После бэктеста — сменить стратегию в .env
# ACTIVE_STRATEGY=<лучшая из бэктеста>
docker compose restart bot
```

---

## Общие команды

```bash
# Статус всех сервисов
docker compose ps

# Логи (live)
docker compose logs -f bot
docker compose logs -f dashboard

# Перезапуск всего стека
docker compose down && docker compose up -d

# Health endpoint
curl http://localhost:8080/health      # dashboard
curl http://localhost:8082/health      # bot (если проброшен порт)
curl http://localhost:8082/metrics | grep bot_

# UI
open http://localhost:8080   # Dashboard
open http://localhost:9090   # Prometheus
open http://localhost:3000   # Grafana (admin / bitbot)
open http://localhost:9093   # Alertmanager
```

---

## Эскалация

Если не удаётся восстановить за 15 минут:

1. `docker compose stop bot` — остановить бота
2. Закрыть открытые позиции вручную на Bybit
3. Разобраться в причине без временного давления
4. Перезапустить только после понимания причины
