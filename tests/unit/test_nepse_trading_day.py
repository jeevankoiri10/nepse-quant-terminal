"""Behaviour-pinning tests for _is_nepse_trading_day: the Nepal trading calendar
(Sun-Thu weekmask) with a benchmark-index override. load_from_db is monkeypatched
so the fallback path is deterministic. No change to data_io.py.

Reference weekdays (2024): Jan 1 is a Monday, so Jan 5 = Fri, Jan 6 = Sat,
Jan 7 = Sun, Jan 8 = Mon ... Jan 11 = Thu."""

from __future__ import annotations

import pandas as pd

from backend.quant_pro.data_io import _is_nepse_trading_day

_LOAD = "backend.quant_pro.data_io.load_from_db"


def test_fallback_sun_through_thu_are_trading_days(monkeypatch):
    monkeypatch.setattr(_LOAD, lambda *a, **k: None)  # no benchmark -> weekmask fallback
    for d in ("2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"):  # Sun..Thu
        assert _is_nepse_trading_day(d) is True


def test_fallback_friday_and_saturday_are_not(monkeypatch):
    monkeypatch.setattr(_LOAD, lambda *a, **k: None)
    assert _is_nepse_trading_day("2024-01-05") is False  # Friday
    assert _is_nepse_trading_day("2024-01-06") is False  # Saturday


def test_invalid_input_returns_false():
    assert _is_nepse_trading_day("not-a-date") is False


def test_benchmark_index_overrides_weekmask(monkeypatch):
    # the benchmark has a Friday -> that Friday counts as a trading day...
    bench = pd.DataFrame({"close": [100.0]}, index=pd.DatetimeIndex(["2024-01-05"]))
    monkeypatch.setattr(_LOAD, lambda *a, **k: bench)
    assert _is_nepse_trading_day("2024-01-05") is True   # Friday, but present in benchmark
    # ...and a Sunday absent from the benchmark is NOT (benchmark path is authoritative)
    assert _is_nepse_trading_day("2024-01-07") is False
