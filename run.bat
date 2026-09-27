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

rem --- PyTorch: use the CUDA build when an NVIDIA GPU is available ---
where nvidia-smi > nul 2>&1
if not errorlevel 1 (
    "%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" > nul 2>&1
    if errorlevel 1 (
        echo [INFO] Installing PyTorch with CUDA support. This is a large download, please wait...
        "%PY%" -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu130
        if errorlevel 1 (
            echo [WARN] CUDA PyTorch install failed. Speaker identification will run on CPU.
        )
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
