#!/usr/bin/env bash
set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         BitbotBY Setup Wizard        ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
  echo "ERROR: Docker не установлен."
  echo "   Скачайте: https://docs.docker.com/get-docker/"
  exit 1
fi

if ! docker compose version &> /dev/null; then
  echo "ERROR: Docker Compose не найден."
  exit 1
fi

echo "OK: Docker найден"

# Copy .env if missing
if [ ! -f .env ]; then
  cp .env.example .env
  echo "OK: Создан .env из .env.example"
fi

# Ask for keys
echo ""
echo "Введите ваши ключи (Enter = пропустить):"
echo ""

read -p "ANTHROPIC_API_KEY (Claude AI): " ANTHROPIC_KEY
read -p "BYBIT_API_KEY: " BYBIT_KEY
read -p "BYBIT_API_SECRET: " BYBIT_SECRET
read -p "TELEGRAM_BOT_TOKEN: " TG_TOKEN
read -p "TELEGRAM_CHAT_ID: " TG_CHAT
read -p "NEWS_API_KEY (опционально): " NEWS_KEY

# Write to .env
update_env() {
  local key=$1 val=$2
  if [ -n "$val" ]; then
    if grep -q "^${key}=" .env; then
      sed -i.bak "s|^${key}=.*|${key}=${val}|" .env && rm -f .env.bak
    else
      echo "${key}=${val}" >> .env
    fi
  fi
}

update_env "ANTHROPIC_API_KEY" "$ANTHROPIC_KEY"
update_env "BYBIT_API_KEY" "$BYBIT_KEY"
update_env "BYBIT_API_SECRET" "$BYBIT_SECRET"
update_env "TELEGRAM_BOT_TOKEN" "$TG_TOKEN"
update_env "TELEGRAM_CHAT_ID" "$TG_CHAT"
update_env "NEWS_API_KEY" "$NEWS_KEY"

echo ""
echo "OK: .env настроен"

# Create local dirs
mkdir -p data models logs

# Build
echo ""
echo "Сборка Docker образа (первый раз ~2-5 минут)..."
docker compose build

echo ""
echo "╔══════════════════════════════════════╗"
echo "║          Настройка завершена!        ║"
echo "║                                      ║"
echo "║  Запустить:  ./start.sh              ║"
echo "║  Дашборд:    http://localhost:8080   ║"
echo "╚══════════════════════════════════════╝"
