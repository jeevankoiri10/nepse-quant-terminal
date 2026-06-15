"""Behaviour-pinning tests for run_health_check: DB existence, data freshness
(staleness vs today), symbol count (excluding SECTOR:: rows), and the overall
status rollup. Uses a temp SQLite DB; the fresh case stamps today's date so it
stays deterministic relative to the run. No change to monitoring.py."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from backend.quant_pro.monitoring import run_health_check


def _make_db(path, price_rows: list[tuple[str, str, float]]):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE stock_prices (symbol TEXT, date TEXT, close REAL)")
    conn.executemany(
        "INSERT INTO stock_prices (symbol, date, close) VALUES (?, ?, ?)", price_rows
    )
    conn.commit()
    conn.close()


def _today() -> str:
    return datetime.now().date().isoformat()


def test_missing_db_is_critical(tmp_path):
    report = run_health_check(tmp_path / "nope.db")
    assert report["status"] == "CRITICAL"
    assert report["checks"]["db_exists"]["ok"] is False


def test_fresh_data_is_ok(tmp_path):
    db = tmp_path / "m.db"
    _make_db(db, [("AAA", _today(), 100.0), ("BBB", _today(), 50.0)])
    report = run_health_check(db)
    assert report["status"] == "OK"
    assert report["checks"]["data_freshness"]["ok"] is True
    assert report["checks"]["data_freshness"]["staleness_days"] == 0
    assert report["checks"]["symbol_count"]["value"] == 2


def test_stale_data_is_critical(tmp_path):
    db = tmp_path / "m.db"
    _make_db(db, [("AAA", "2000-01-01", 100.0)])  # ancient -> > 30 days stale
    report = run_health_check(db)
    assert report["checks"]["data_freshness"]["ok"] is False
    assert report["status"] == "CRITICAL"


def test_empty_table_is_critical(tmp_path):
    db = tmp_path / "m.db"
    _make_db(db, [])
    report = run_health_check(db)
    assert report["status"] == "CRITICAL"
    assert report["checks"]["data_freshness"]["ok"] is False


def test_symbol_count_excludes_sector_rows(tmp_path):
    db = tmp_path / "m.db"
    _make_db(db, [("AAA", _today(), 100.0), ("SECTOR::BANKING", _today(), 1000.0)])
    report = run_health_check(db)
    assert report["checks"]["symbol_count"]["value"] == 1  # SECTOR:: excluded
