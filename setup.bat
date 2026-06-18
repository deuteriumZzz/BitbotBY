@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════╗
echo ║         BitbotBY Setup Wizard        ║
echo ╚══════════════════════════════════════╝
echo.

where docker >nul 2>&1
if %errorlevel% neq 0 (
  echo [ERROR] Docker не установлен.
  echo Скачайте: https://docs.docker.com/get-docker/
  pause
  exit /b 1
)
echo [OK] Docker найден

if not exist .env (
  copy .env.example .env
  echo [OK] Создан .env
)

echo.
echo Введите ваши ключи (Enter = пропустить):
echo.

set /p ANTHROPIC_KEY="ANTHROPIC_API_KEY (Claude AI): "
set /p BYBIT_KEY="BYBIT_API_KEY: "
set /p BYBIT_SECRET="BYBIT_API_SECRET: "
set /p TG_TOKEN="TELEGRAM_BOT_TOKEN: "
set /p TG_CHAT="TELEGRAM_CHAT_ID: "

powershell -Command "(gc .env) -replace 'ANTHROPIC_API_KEY=.*', 'ANTHROPIC_API_KEY=%ANTHROPIC_KEY%' | sc .env"
powershell -Command "(gc .env) -replace 'BYBIT_API_KEY=.*', 'BYBIT_API_KEY=%BYBIT_KEY%' | sc .env"
powershell -Command "(gc .env) -replace 'BYBIT_API_SECRET=.*', 'BYBIT_API_SECRET=%BYBIT_SECRET%' | sc .env"
powershell -Command "(gc .env) -replace 'TELEGRAM_BOT_TOKEN=.*', 'TELEGRAM_BOT_TOKEN=%TG_TOKEN%' | sc .env"
powershell -Command "(gc .env) -replace 'TELEGRAM_CHAT_ID=.*', 'TELEGRAM_CHAT_ID=%TG_CHAT%' | sc .env"

if not exist data mkdir data
if not exist models mkdir models
if not exist logs mkdir logs

echo.
echo [Build] Сборка Docker образа...
docker compose build

echo.
echo ╔══════════════════════════════════════╗
echo ║          Настройка завершена!        ║
echo ║                                      ║
echo ║  Запустить:  start.bat               ║
echo ║  Дашборд:    http://localhost:8080   ║
echo ╚══════════════════════════════════════╝
pause
