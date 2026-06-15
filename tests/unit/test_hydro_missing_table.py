"""Regression test for the fix: a MISSING weather_data table must degrade to no
signals (return None per basin), not crash the hydro signal generator. Before
the fix, _get_basin_rainfall caught only sqlite3.Error while pandas raises
DatabaseError for a missing table, so this path raised instead of returning []."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from backend.quant_pro.satellite_data import (
    _get_basin_rainfall,
    generate_hydro_rainfall_signals_at_date,
)


def _db_without_weather_table(path) -> str:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE unrelated (x INTEGER)")  # weather_data intentionally absent
    conn.commit()
    conn.close()
    return str(path)


def test_get_basin_rainfall_returns_none_when_table_missing(tmp_path):
    db = _db_without_weather_table(tmp_path / "m.db")
    assert _get_basin_rainfall(db, "koshi", "2026-07-30", lookback_days=30) is None


def test_generator_emits_nothing_when_weather_table_missing(tmp_path):
    db = _db_without_weather_table(tmp_path / "m.db")
    prices = pd.DataFrame({"symbol": ["UPPER"], "date": ["2026-07-30"], "close": [100.0]})
    # must not raise; degrades to an empty signal list
    assert generate_hydro_rainfall_signals_at_date(prices, datetime(2026, 7, 30), db_path=db) == []
