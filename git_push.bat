
@echo off
setlocal
title Offshore Dashboard - Code Sync

cd /d "%~dp0"
if errorlevel 1 goto error

echo ==============================
echo Offshore Dashboard - Code Sync
echo ==============================
echo.

REM Step 1: Check local changes
echo [1/5] Checking local changes...
git status --short
if errorlevel 1 goto error

echo.
REM Step 2: Stage changes
echo [2/5] Staging changes...
git add -A
if errorlevel 1 goto error

REM Safety check: prevent local data and secrets from being committed
git diff --cached --name-only --diff-filter=ACMR | findstr /I /R /C:"^\.env$" /C:"^progress\.xlsx$" /C:"^hourly\.xlsx$" /C:"^progress\.pkl$" /C:"^hourly\.pkl$" /C:"^metadata\.pkl$" >nul

if not errorlevel 1 (
    echo [BLOCKED] Local data or secrets are staged!
    echo Please review git status.
    goto error
)

echo.
REM Step 3: Commit
echo [3/5] Committing changes...
git diff --cached --quiet
if errorlevel 2 goto error
if errorlevel 1 goto commit
echo No local changes to commit.
goto pull

:commit
git commit -m "update dashboard"
if errorlevel 1 goto error

:pull
echo.
REM Step 4: Pull remote changes
echo [4/5] Pulling from GitHub...
git pull --rebase origin main
if errorlevel 1 goto error

echo.
REM Step 5: Push
echo [5/5] Pushing to GitHub...
git push origin main
if errorlevel 1 goto error

echo.
echo ==============================
echo [SUCCESS] Git sync completed!
echo ==============================
echo.
git status
goto end

:error
echo.
echo ==============================
echo [FAILED] Git sync failed!
echo Check the message above.
echo ==============================

:end
echo.
pause
endlocal


