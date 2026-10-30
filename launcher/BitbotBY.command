#!/bin/bash
# Двойной щелчок в Finder: запускает BitbotBY на этом компьютере и открывает приложение.
# Ярлык лежит в launcher/ — работаем из корня проекта.
cd "$(dirname "$0")/.." || exit 1
if command -v python3 >/dev/null 2>&1; then
  python3 launcher/launch.py "$@"
  code=$?
else
  echo "Не найден Python 3. Установите: https://www.python.org/downloads/"
  code=1
fi
echo
read -n 1 -s -r -p "Нажмите любую клавишу, чтобы закрыть окно…"
exit $code
