#!/usr/bin/env bash
# Linux/любая оболочка: ./bitbotby.sh [start|stop|status|logs|url]
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  exec python3 launcher/launch.py "$@"
fi
echo "Не найден Python 3. Установите python3 и повторите." >&2
exit 1
