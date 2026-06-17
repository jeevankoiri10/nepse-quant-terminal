"""Behaviour-pinning tests for generate_hydro_rainfall_signals_at_date: the
satellite rainfall -> hydropower signal generator only emits bullish signals
when a basin's rainfall anomaly clears the threshold (and the ticker has price
history); missing weather data or below-threshold rainfall emits nothing. Temp
weather_data DB. No change to satellite_data.py."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from backend.quant_pro.alpha_practical import SignalType
from backend.quant_pro.satellite_data import (
    HYDRO_BASINS,
    MONTHLY_BASELINE_MM,
    generate_hydro_rainfall_signals_at_date,
)

AS_OF = datetime(2026, 7, 30)  # July (monsoon)
_BASIN = "koshi"
_TICKER = HYDRO_BASINS[_BASIN]["tickers"][0]
_DAILY_BASE = MONTHLY_BASELINE_MM.get(7, 30.0) / 30.0  # daily baseline rainfall for July


def _weather_db(path, daily_rain: float | None) -> str:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE weather_data (basin TEXT, date TEXT, rainfall_mm REAL, temperature_c REAL)"
    )
    if daily_rain is not None:
        rows = [(_BASIN, f"2026-07-{d:02d}", daily_rain, 25.0) for d in range(1, 31)]
        conn.executemany("INSERT INTO weather_data VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return str(path)


def _prices(ticker: str, n_rows: int) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-07-30", periods=n_rows).strftime("%Y-%m-%d")
    return pd.DataFrame({"symbol": ticker, "date": dates, "close": [100.0] * n_rows})


def test_empty_weather_table_emits_nothing(tmp_path):
    # weather_data exists but has no rows -> _get_basin_rainfall returns None
    # for every basin -> no signals (the graceful no-data guard).
    db = _weather_db(tmp_path / "m.db", None)
    assert generate_hydro_rainfall_signals_at_date(_prices(_TICKER, 25), AS_OF, db_path=db) == []


def test_below_threshold_anomaly_emits_nothing(tmp_path):
    db = _weather_db(tmp_path / "m.db", _DAILY_BASE)  # at baseline -> anomaly ~0 < 0.3
    assert generate_hydro_rainfall_signals_at_date(_prices(_TICKER, 25), AS_OF, db_path=db) == []


def test_strong_anomaly_emits_hydro_buy(tmp_path):
    db = _weather_db(tmp_path / "m.db", 2 * _DAILY_BASE)  # +100% anomaly > 0.3
    sigs = generate_hydro_rainfall_signals_at_date(_prices(_TICKER, 25), AS_OF, db_path=db)
    assert len(sigs) >= 1
    s = sigs[0]
    assert s.symbol == _TICKER
    assert s.signal_type == SignalType.SATELLITE_HYDRO
    assert s.direction == 1
    assert 0.0 < s.strength <= 1.0


def test_strong_anomaly_skips_thin_price_history(tmp_path):
    db = _weather_db(tmp_path / "m.db", 2 * _DAILY_BASE)
    # only 10 price rows (< 20 required) -> no signal even with strong rainfall
    assert generate_hydro_rainfall_signals_at_date(_prices(_TICKER, 10), AS_OF, db_path=db) == []
