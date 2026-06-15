"""Behaviour-pinning tests for get_nrb_policy_regime (NRB policy-rate cycle:
hiking/cutting/hold from the basis-point change between the last two data
points). Temp DB + fixed as_of_date. Thresholds: >=+25bps hike (x0.85),
<=-25bps cut (x1.08), else hold. No change to macro_signals.py."""

from __future__ import annotations

import sqlite3

from backend.quant_pro.macro_signals import (
    NRB_CUTTING_MULTIPLIER,
    NRB_HIKING_MULTIPLIER,
    get_nrb_policy_regime,
)


def _make_db(path, rate_rows: list[tuple[str, float]]) -> str:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE macro_indicators (date TEXT, indicator_name TEXT, value REAL)")
    conn.executemany(
        "INSERT INTO macro_indicators (date, indicator_name, value) "
        "VALUES (?, 'nrb_policy_rate_pct', ?)",
        rate_rows,
    )
    conn.commit()
    conn.close()
    return str(path)


def test_no_data_on_empty_table(tmp_path):
    out = get_nrb_policy_regime(_make_db(tmp_path / "m.db", []))
    assert out["cycle"] == "no_data"
    assert out["multiplier"] == 1.0


def test_single_point_is_hold(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-01", 5.0)])
    out = get_nrb_policy_regime(db, as_of_date="2026-02-01")
    assert out["cycle"] == "hold"
    assert out["latest_rate_pct"] == 5.0
    assert out["rate_change_bps"] == 0.0


def test_hiking_cycle(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-01", 5.0), ("2026-02-01", 5.5)])
    out = get_nrb_policy_regime(db, as_of_date="2026-03-01")
    assert out["cycle"] == "hiking"          # +50 bps >= 25
    assert out["latest_rate_pct"] == 5.5
    assert out["rate_change_bps"] == 50.0
    assert out["multiplier"] == NRB_HIKING_MULTIPLIER
    assert out["sector_adjustments"]         # non-empty on a hike


def test_cutting_cycle(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-01", 5.5), ("2026-02-01", 5.0)])
    out = get_nrb_policy_regime(db, as_of_date="2026-03-01")
    assert out["cycle"] == "cutting"         # -50 bps <= -25
    assert out["rate_change_bps"] == -50.0
    assert out["multiplier"] == NRB_CUTTING_MULTIPLIER


def test_small_change_is_hold(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-01", 5.0), ("2026-02-01", 5.10)])
    out = get_nrb_policy_regime(db, as_of_date="2026-03-01")
    assert out["cycle"] == "hold"            # +10 bps, inside the band
    assert out["multiplier"] == 1.0
    assert out["sector_adjustments"] == {}


def test_hike_threshold_is_inclusive_at_25bps(tmp_path):
    db = _make_db(tmp_path / "m.db", [("2026-01-01", 5.0), ("2026-02-01", 5.25)])
    out = get_nrb_policy_regime(db, as_of_date="2026-03-01")
    assert out["cycle"] == "hiking"          # exactly +25 bps -> hike (>=)
    assert out["rate_change_bps"] == 25.0
