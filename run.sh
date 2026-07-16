#!/bin/bash
# SimpleTranscriber launcher for macOS / Linux
set -e
cd "$(dirname "$0")"

echo "========================================="
echo " Simple Transcriber - Starting..."
echo "========================================="

PY="venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "[INFO] First run: creating virtual environment..."
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[ERROR] python3 not found. Please install Python 3.10+."
        exit 1
    fi
    python3 -m venv venv
fi

echo "[INFO] Checking dependencies..."
"$PY" -m pip install -r requirements.txt --quiet

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[WARN] ffmpeg not found. Install it with: brew install ffmpeg"
    echo "       (or: $PY -m pip install imageio-ffmpeg)"
fi

export KMP_DUPLICATE_LIB_OK=TRUE

echo "[INFO] Ready. Launching transcription tool..."
"$PY" app.py
