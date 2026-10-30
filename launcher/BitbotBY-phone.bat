@echo off
rem Двойной щелчок: включает MiniApp в Telegram (адрес до этого компьютера через туннель).
chcp 65001 >nul
rem Ярлык лежит в launcher\ — работаем из корня проекта.
cd /d "%~dp0.."
where py >nul 2>nul
if %errorlevel%==0 (py -3 launcher\launch.py phone on) else (python launcher\launch.py phone on)
echo.
pause
