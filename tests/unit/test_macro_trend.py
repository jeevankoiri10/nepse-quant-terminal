"""Behaviour-pinning tests for get_remittance_trend (is remittance YoY growth
accelerating/stable/decelerating over the last N months). Temp DB + fixed
as_of_date for determinism. Split rule: recent-half mean vs earlier-half mean,
threshold +/-2.0. No change to macro_signals.py."""

from __future__ import annotations

import sqlite3

from backend.quant_pro.macro_signals import get_remittance_trend


def _make_db(path, growth_rows: list[tuple[str, float]]) -> str:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE macro_indicators (date TEXT, indicator_name TEXT, value REAL)")
    conn.executemany(
        "INSERT INTO macro_indicators (date, indicator_name, value) "
        "VALUES (?, 'remittance_yoy_growth_pct', ?)",
        growth_rows,
    )
    conn.commit()
    conn.close()
    return str(path)


def test_no_data_on_empty_table(tmp_path):
    out = get_remittance_trend(_make_db(tmp_path / "m.db", []))
    assert out["trend"] == "no_data"


def test_no_data_on_single_point(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-01", 10.0)])  # < 2 rows
    assert get_remittance_trend(db, as_of_date="2026-02-01")["trend"] == "no_data"


def test_accelerating_recent_half_higher(tmp_path):
    rows = [
        ("2025-08-01", 5.0), ("2025-09-01", 5.0), ("2025-10-01", 5.0),
        ("2025-11-01", 10.0), ("2025-12-01", 10.0), ("2026-01-01", 10.0),
    ]
    db = _make_db(tmp_path / "m.db", rows)
    out = get_remittance_trend(db, n_months=6, as_of_date="2026-02-01")
    assert out["trend"] == "accelerating"   # recent 10 vs earlier 5, diff +5 > 2
    assert out["latest_growth"] == 10.0
    assert out["avg_growth"] == 7.5
    assert len(out["growth_history"]) == 6
    assert out["growth_history"][0][0] == "2025-08-01"   # chronological order
    assert out["growth_history"][-1][0] == "2026-01-01"


def test_decelerating_recent_half_lower(tmp_path):
    rows = [
        ("2025-08-01", 10.0), ("2025-09-01", 10.0), ("2025-10-01", 10.0),
        ("2025-11-01", 5.0), ("2025-12-01", 5.0), ("2026-01-01", 5.0),
    ]
    db = _make_db(tmp_path / "m.db", rows)
    out = get_remittance_trend(db, n_months=6, as_of_date="2026-02-01")
    assert out["trend"] == "decelerating"
    assert out["latest_growth"] == 5.0


def test_stable_when_flat(tmp_path):
    rows = [(f"2025-0{m}-01", 8.0) for m in range(4, 10)]  # 6 flat months
    db = _make_db(tmp_path / "m.db", rows)
    out = get_remittance_trend(db, n_months=6, as_of_date="2026-02-01")
    assert out["trend"] == "stable"
    assert out["avg_growth"] == 8.0


def test_too_few_points_for_split_is_stable(tmp_path):
    rows = [("2025-12-01", 5.0), ("2026-01-01", 9.0)]  # 2 rows: >=2 but <4 -> stable
    db = _make_db(tmp_path / "m.db", rows)
    out = get_remittance_trend(db, as_of_date="2026-02-01")
    assert out["trend"] == "stable"
    assert out["latest_growth"] == 9.0   # chronologically last
    assert out["avg_growth"] == 7.0
