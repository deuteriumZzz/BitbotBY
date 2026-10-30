#!/bin/bash
# Двойной щелчок: останавливает BitbotBY (данные и ключи сохраняются).
# Ярлык лежит в launcher/ — работаем из корня проекта.
cd "$(dirname "$0")/.." || exit 1
python3 launcher/launch.py stop
echo
read -n 1 -s -r -p "Готово. Нажмите любую клавишу…"
