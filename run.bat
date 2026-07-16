@echo off
chcp 65001 > nul
title Simple Transcriber
cd /d "%~dp0"

echo =========================================
echo  Simple Transcriber - Starting...
echo =========================================

set "PY=venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [INFO] First run: creating virtual environment...
    echo This may take a few minutes.
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Is Python installed?
        pause
        exit /b 1
    )
)

echo [INFO] Checking dependencies...
"%PY%" -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

set KMP_DUPLICATE_LIB_OK=TRUE

echo.
echo [INFO] Ready. Launching transcription tool...
echo =========================================
echo  Browser will open automatically.
echo  Close this window to stop the server.
echo =========================================
"%PY%" app.py

pause
