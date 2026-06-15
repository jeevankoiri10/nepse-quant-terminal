"""Behaviour-pinning tests for the remittance macro regime (NRB remittance YoY
growth -> strong/normal/weak). Uses a temp SQLite macro_indicators table and a
fixed as_of_date so results are deterministic. Thresholds: STRONG >= 15%,
WEAK < 5%. No change to macro_signals.py."""

from __future__ import annotations

import sqlite3

from backend.quant_pro.macro_signals import get_remittance_regime


def _make_db(path, rows: list[tuple[str, str, float]]) -> str:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE macro_indicators (date TEXT, indicator_name TEXT, value REAL)")
    conn.executemany(
        "INSERT INTO macro_indicators (date, indicator_name, value) VALUES (?, ?, ?)", rows
    )
    conn.commit()
    conn.close()
    return str(path)


def test_no_data_when_table_empty(tmp_path):
    db = _make_db(tmp_path / "m.db", [])
    out = get_remittance_regime(db)
    assert out["regime"] == "no_data"
    assert out["multiplier"] == 1.0


def test_strong_regime_with_value_and_age(tmp_path):
    db = _make_db(
        tmp_path / "m.db",
        [
            ("2026-01-01", "remittance_yoy_growth_pct", 18.0),
            ("2026-01-01", "remittance_usd_millions", 1200.0),
        ],
    )
    out = get_remittance_regime(db, as_of_date="2026-02-01")
    assert out["regime"] == "strong"
    assert out["multiplier"] == 1.05
    assert out["yoy_growth"] == 18.0
    assert out["latest_value_usd_m"] == 1200.0
    assert out["latest_date"] == "2026-01-01"
    assert out["data_age_days"] == 31  # 2026-02-01 minus 2026-01-01


def test_weak_regime(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-01", "remittance_yoy_growth_pct", 2.0)])
    out = get_remittance_regime(db, as_of_date="2026-01-15")
    assert out["regime"] == "weak"
    assert out["multiplier"] == 0.95


def test_normal_regime(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-01", "remittance_yoy_growth_pct", 10.0)])
    out = get_remittance_regime(db, as_of_date="2026-01-15")
    assert out["regime"] == "normal"
    assert out["multiplier"] == 1.00


def test_strong_threshold_is_inclusive_at_15(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-01", "remittance_yoy_growth_pct", 15.0)])
    assert get_remittance_regime(db, as_of_date="2026-01-02")["regime"] == "strong"


def test_weak_threshold_is_exclusive_at_5(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-01", "remittance_yoy_growth_pct", 5.0)])
    # exactly 5.0 is not < 5 -> normal
    assert get_remittance_regime(db, as_of_date="2026-01-02")["regime"] == "normal"


def test_as_of_date_uses_latest_point_on_or_before(tmp_path):
    db = _make_db(
        tmp_path / "m.db",
        [
            ("2026-01-01", "remittance_yoy_growth_pct", 2.0),    # weak
            ("2026-02-01", "remittance_yoy_growth_pct", 20.0),   # strong, but after as_of
        ],
    )
    out = get_remittance_regime(db, as_of_date="2026-01-15")
    assert out["regime"] == "weak"            # only the on-or-before point is visible
    assert out["latest_date"] == "2026-01-01"
