@echo off
rem Двойной щелчок: останавливает BitbotBY (данные и ключи сохраняются).
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 launcher\launch.py stop) else (python launcher\launch.py stop)
echo.
pause
