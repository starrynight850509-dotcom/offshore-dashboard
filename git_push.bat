
@echo off
title Offshore Dashboard - Git Sync

cd /d "C:\Users\ali.chang\Desktop\Ali正力\專案\S2603BEX50\offshore gantt"
if errorlevel 1 goto error

echo ==============================
echo 1. Stage and commit local changes
echo ==============================
git add .
if errorlevel 1 goto error

git diff --cached --quiet
if errorlevel 1 (
    git commit -m "update data"
    if errorlevel 1 goto error
) else (
    echo No local changes to commit.
)

echo.
echo ==============================
echo 2. Pull latest changes
echo ==============================
git pull --rebase origin main
if errorlevel 1 goto error

echo.
echo ==============================
echo 3. Push to GitHub
echo ==============================
git push origin main
if errorlevel 1 goto error

echo.
echo [SUCCESS] Git sync completed!
goto end

:error
echo.
echo [FAILED] Git operation failed.
echo Please check the error above.

:end
pause

