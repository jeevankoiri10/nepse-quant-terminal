@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating venv with Python 3.13...
    py -3.13 -m venv .venv || (echo Python 3.13 not found. Install with: winget install Python.Python.3.13 & exit /b 1)
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt sqlalchemy
)

if not exist "data\nepse_market_data.db" (
    echo [setup] Downloading market database...
    .venv\Scripts\python.exe setup_data.py
)

.venv\Scripts\python.exe -m apps.tui.dashboard_tui
