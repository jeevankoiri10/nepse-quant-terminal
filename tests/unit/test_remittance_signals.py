"""Behaviour-pinning tests for generate_remittance_signals_at_date: the macro
remittance signal generator only emits banking-sector buy signals in a STRONG,
fresh regime (and skips tickers without enough price history); every other
regime, no-data, or stale-data case emits nothing. Temp DB + fixed date. No
change to macro_signals.py."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from backend.quant_pro.alpha_practical import SignalType
from backend.quant_pro.macro_signals import (
    REMITTANCE_BENEFICIARY_TICKERS,
    generate_remittance_signals_at_date,
)

AS_OF = datetime(2026, 5, 15)
_TICKER = REMITTANCE_BENEFICIARY_TICKERS[0]


def _macro_db(path, yoy: float | None, growth_date: str = "2026-05-01") -> str:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE macro_indicators (date TEXT, indicator_name TEXT, value REAL)")
    if yoy is not None:
        conn.execute(
            "INSERT INTO macro_indicators VALUES (?, 'remittance_yoy_growth_pct', ?)",
            (growth_date, yoy),
        )
    conn.commit()
    conn.close()
    return str(path)


def _prices(ticker: str, n_rows: int, end: str = "2026-05-15") -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=n_rows).strftime("%Y-%m-%d")
    return pd.DataFrame({"symbol": ticker, "date": dates, "close": [100.0] * n_rows})


def test_no_data_emits_nothing(tmp_path):
    db = _macro_db(tmp_path / "m.db", None)
    assert generate_remittance_signals_at_date(pd.DataFrame(), AS_OF, db_path=db) == []


def test_weak_regime_emits_nothing(tmp_path):
    db = _macro_db(tmp_path / "m.db", 2.0)  # < 5% -> weak
    assert generate_remittance_signals_at_date(pd.DataFrame(), AS_OF, db_path=db) == []


def test_normal_regime_emits_nothing(tmp_path):
    db = _macro_db(tmp_path / "m.db", 10.0)  # 5-15% -> normal
    assert generate_remittance_signals_at_date(pd.DataFrame(), AS_OF, db_path=db) == []


def test_stale_strong_data_emits_nothing(tmp_path):
    db = _macro_db(tmp_path / "m.db", 20.0, growth_date="2026-01-01")  # strong but >90d old
    assert generate_remittance_signals_at_date(_prices(_TICKER, 25), AS_OF, db_path=db) == []


def test_strong_fresh_regime_emits_banking_buy(tmp_path):
    db = _macro_db(tmp_path / "m.db", 20.0, growth_date="2026-05-01")  # strong + fresh
    sigs = generate_remittance_signals_at_date(_prices(_TICKER, 25), AS_OF, db_path=db)
    assert len(sigs) >= 1
    s = sigs[0]
    assert s.symbol == _TICKER
    assert s.signal_type == SignalType.MACRO_REMITTANCE
    assert s.direction == 1
    assert 0.0 < s.strength <= 1.0


def test_strong_regime_skips_ticker_with_thin_history(tmp_path):
    db = _macro_db(tmp_path / "m.db", 20.0, growth_date="2026-05-01")
    # only 10 price rows (< 20 required) -> no signal even in a strong regime
    assert generate_remittance_signals_at_date(_prices(_TICKER, 10), AS_OF, db_path=db) == []
