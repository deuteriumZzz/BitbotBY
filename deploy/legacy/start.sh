#!/usr/bin/env bash
set -e

echo ""
echo "Запуск BitbotBY..."
docker compose up -d

echo ""
echo "╔══════════════════════════════════════╗"
echo "║           BitbotBY запущен!          ║"
echo "║                                      ║"
echo "║  Дашборд:  http://localhost:8080     ║"
echo "║  Логи:     ./logs.sh                 ║"
echo "║  Стоп:     ./stop.sh                 ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Open browser
sleep 2
if command -v open &>/dev/null; then
  open http://localhost:8080
elif command -v xdg-open &>/dev/null; then
  xdg-open http://localhost:8080
fi
