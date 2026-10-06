
@echo off
title Offshore Dashboard - Git Sync

cd /d "C:\Users\ali.chang\Desktop\Ali正力\專案\S2603BEX50\offshore gantt"
if errorlevel 1 goto error

echo ==============================
echo 1. Check local changes
echo ==============================
git status --short
if errorlevel 1 goto error

echo.
echo ==============================
echo 2. Stage and commit
echo ==============================
git add .
if errorlevel 1 goto error

echo Files staged for commit:
git diff --cached --name-only

git diff --cached --quiet
if errorlevel 1 (
    git commit -m "update data"
    if errorlevel 1 goto error
) else (
    echo No local changes to commit.
)

echo.
echo ==============================
echo 3. Pull latest changes
echo ==============================
git pull --rebase origin main
if errorlevel 1 goto error

echo.
echo ==============================
echo 4. Push to GitHub
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


