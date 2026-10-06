
@echo off
setlocal
title Offshore Dashboard - Git Sync

REM Use the folder containing this BAT file
cd /d "%~dp0"
if errorlevel 1 goto error

echo ==============================
echo Offshore Dashboard - Git Sync
echo ==============================
echo.

REM Step 1: Check Git status
echo [1/4] Checking local changes...
git status --short
if errorlevel 1 goto error

echo.
REM Step 2: Stage and commit
echo [2/4] Committing local changes...
git add .
if errorlevel 1 goto error

git diff --cached --quiet
if errorlevel 2 goto error
if errorlevel 1 goto commit
echo No local changes to commit.
goto pull

:commit
git commit -m "update data"
if errorlevel 1 goto error

:pull
echo.
REM Step 3: Pull remote changes
echo [3/4] Pulling from GitHub...
git pull --rebase origin main
if errorlevel 1 goto error

echo.
REM Step 4: Push local changes
echo [4/4] Pushing to GitHub...
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
echo Check the error message above.
echo ==============================

:end
echo.
pause
endlocal


