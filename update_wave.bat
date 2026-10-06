
@echo off
title F2 Windy Forecast Updater

REM 切換到 BAT 所在資料夾
cd /d "%~dp0"

echo ==============================
echo F2 Windy Forecast Updater
echo ==============================
echo.

REM 使用 Anaconda Python
"C:\Users\ali.chang\AppData\Local\anaconda3\python.exe" update_wave.py

if errorlevel 1 (
    echo.
    echo [FAILED] Windy update failed.
) else (
    echo.
    echo [SUCCESS] Windy update completed.
)

echo.
pause