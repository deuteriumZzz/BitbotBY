@echo off
chcp 65001 >nul
echo.
echo Запуск BitbotBY...
docker compose up -d

echo.
echo ╔══════════════════════════════════════╗
echo ║         BitbotBY запущен!            ║
echo ║                                      ║
echo ║  Дашборд:  http://localhost:8080     ║
echo ╚══════════════════════════════════════╝
echo.

timeout /t 3 /nobreak >nul
start http://localhost:8080
pause
