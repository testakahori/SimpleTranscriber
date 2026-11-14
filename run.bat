@echo off
chcp 65001 > nul
title Simple Transcriber
cd /d "%~dp0"

echo =========================================
echo  Simple Transcriber - Starting...
echo =========================================

set "PY=venv\Scripts\python.exe"
rem A copy of requirements.txt is stored here after a successful install.
rem While requirements.txt is unchanged, the slow dependency check is skipped.
set "STAMP=venv\requirements.installed"

rem "run.bat --check" (or deleting venv\requirements.installed) forces the full dependency check.
set "FORCE_CHECK="
if /i "%~1"=="--check" set "FORCE_CHECK=1"
if /i "%~1"=="--update" set "FORCE_CHECK=1"

if not exist "%PY%" (
    echo [INFO] First run: creating virtual environment...
    echo This may take a few minutes.
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Is Python installed?
        pause
        exit /b 1
    )
    set "FORCE_CHECK=1"
)

if defined FORCE_CHECK goto :check_deps
if not exist "%STAMP%" goto :check_deps
fc /b "requirements.txt" "%STAMP%" > nul 2>&1
if errorlevel 1 goto :check_deps
echo [INFO] Dependencies unchanged. Skipping the check ^(run "run.bat --check" to force it^).
goto :launch

:check_deps
rem --- PyTorch: use the CUDA build when an NVIDIA GPU is available ---
rem The stamp is only written when this check passes, so a failed CUDA install is retried next time.
set "TORCH_OK=1"
where nvidia-smi > nul 2>&1
if not errorlevel 1 (
    "%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" > nul 2>&1
    if errorlevel 1 (
        echo [INFO] Installing PyTorch with CUDA support. This is a large download, please wait...
        "%PY%" -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu130
        if errorlevel 1 (
            echo [WARN] CUDA PyTorch install failed. Speaker identification will run on CPU.
            set "TORCH_OK="
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
if defined TORCH_OK (
    copy /y "requirements.txt" "%STAMP%" > nul
) else (
    echo [INFO] The dependency check will run again next time because the CUDA install failed.
)

:launch
set KMP_DUPLICATE_LIB_OK=TRUE

echo.
echo [INFO] Ready. Launching transcription tool...
echo =========================================
echo  Browser will open automatically.
echo  Close this window to stop the server.
echo =========================================
"%PY%" app.py

pause
