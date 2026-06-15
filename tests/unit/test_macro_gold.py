"""Behaviour-pinning tests for get_gold_macro_regime (gold 20d momentum ->
risk_off / neutral / risk_on equity-confidence multiplier). Temp DB of
gold_usd_per_oz points + fixed as_of_date. Rules: >+3% risk_off (x0.85),
[-2%,+3%] neutral (x1.0), <-2% risk_on (x1.05). No change to the module."""

from __future__ import annotations

import sqlite3

import pytest

from backend.quant_pro.macro_signals import get_gold_macro_regime


def _make_db(path, gold_rows: list[tuple[str, float]]) -> str:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE macro_indicators (date TEXT, indicator_name TEXT, value REAL)")
    conn.executemany(
        "INSERT INTO macro_indicators (date, indicator_name, value) "
        "VALUES (?, 'gold_usd_per_oz', ?)",
        gold_rows,
    )
    conn.commit()
    conn.close()
    return str(path)


def test_no_data_on_empty_table(tmp_path):
    out = get_gold_macro_regime(_make_db(tmp_path / "m.db", []), as_of_date="2026-02-01")
    assert out["regime"] == "no_data"
    assert out["multiplier"] == 1.0


def test_risk_off_when_gold_spikes(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-10", 2000.0), ("2026-02-01", 2100.0)])  # +5%
    out = get_gold_macro_regime(db, as_of_date="2026-02-01")
    assert out["regime"] == "risk_off"
    assert out["multiplier"] == 0.85
    assert out["momentum_20d"] == pytest.approx(0.05)
    assert out["gold_price_usd"] == 2100.0


def test_risk_on_when_gold_falls(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-10", 2100.0), ("2026-02-01", 2000.0)])  # -4.76%
    out = get_gold_macro_regime(db, as_of_date="2026-02-01")
    assert out["regime"] == "risk_on"
    assert out["multiplier"] == 1.05
    assert out["momentum_20d"] < -0.02


def test_neutral_on_small_move(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-10", 2000.0), ("2026-02-01", 2010.0)])  # +0.5%
    out = get_gold_macro_regime(db, as_of_date="2026-02-01")
    assert out["regime"] == "neutral"
    assert out["multiplier"] == 1.0
