"""windows_preflight data-freshness check: catches an empty stock_prices table
(passes the table-existence check but can't drive signals) and reports the
latest bar date + age when populated."""

from __future__ import annotations

import sqlite3

from scripts.ops import windows_preflight as wp


def _make_db(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE stock_prices (symbol TEXT, date TEXT, open REAL, high REAL,"
        " low REAL, close REAL, volume REAL)"
    )
    conn.executemany(
        "INSERT INTO stock_prices (symbol, date, close) VALUES (?, ?, ?)", rows
    )
    conn.commit()
    conn.close()


def test_freshness_missing_db_is_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(wp, "get_db_path", lambda: tmp_path / "nope.db")
    ok, line = wp._check_data_freshness()
    assert ok is True
    assert "no database" in line


def test_freshness_empty_table_fails(monkeypatch, tmp_path):
    db = tmp_path / "market.db"
    _make_db(db, [])  # table exists, zero rows
    monkeypatch.setattr(wp, "get_db_path", lambda: db)
    ok, line = wp._check_data_freshness()
    assert ok is False
    assert "empty" in line
    assert "setup_data.py" in line


def test_freshness_populated_reports_date_and_symbol_count(monkeypatch, tmp_path):
    db = tmp_path / "market.db"
    _make_db(db, [("NEPSE", "2026-06-10", 2000.0), ("NABIL", "2026-06-12", 500.0)])
    monkeypatch.setattr(wp, "get_db_path", lambda: db)
    ok, line = wp._check_data_freshness(max_stale_days=3650)  # huge window -> never STALE
    assert ok is True
    assert "2026-06-12" in line          # the latest date
    assert "2 symbols" in line
    assert "STALE" not in line


def test_freshness_flags_stale_data(monkeypatch, tmp_path):
    db = tmp_path / "market.db"
    _make_db(db, [("NEPSE", "2000-01-01", 100.0)])  # ancient -> stale under any sane window
    monkeypatch.setattr(wp, "get_db_path", lambda: db)
    ok, line = wp._check_data_freshness(max_stale_days=7)
    assert ok is True  # stale data warns in-line but doesn't hard-fail the launch
    assert "STALE" in line


def test_freshness_is_wired_into_main_checks(monkeypatch, tmp_path):
    # the new check must actually run as part of the preflight
    db = tmp_path / "market.db"
    _make_db(db, [("NEPSE", "2026-06-12", 2000.0)])
    monkeypatch.setattr(wp, "get_db_path", lambda: db)
    captured = {}
    real = wp._check_data_freshness

    def _spy(*a, **k):
        captured["ran"] = True
        return real(*a, **k)

    monkeypatch.setattr(wp, "_check_data_freshness", _spy)
    wp.main()
    assert captured.get("ran") is True
