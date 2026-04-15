#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if [ ! -x ".venv/bin/python" ]; then
    echo "[setup] Creating venv..."
    "$PY" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt sqlalchemy
fi

if [ ! -f "data/nepse_market_data.db" ]; then
    echo "[setup] Downloading market database..."
    .venv/bin/python setup_data.py
fi

exec .venv/bin/python -m apps.tui.dashboard_tui
