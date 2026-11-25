@echo off
rem Двойной щелчок: запускает BitbotBY на этом компьютере и открывает приложение.
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 bitbot start %*
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    python bitbot start %*
  ) else (
    echo Не найден Python 3. Установите: https://www.python.org/downloads/
  )
)
echo.
pause
