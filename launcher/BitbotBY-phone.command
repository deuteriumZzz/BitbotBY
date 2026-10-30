#!/bin/bash
# Двойной щелчок: включает MiniApp в Telegram (адрес до этого компьютера через туннель).
# Ярлык лежит в launcher/ — работаем из корня проекта.
cd "$(dirname "$0")/.." || exit 1
python3 launcher/launch.py phone on
echo
read -n 1 -s -r -p "Готово. Нажмите любую клавишу…"
