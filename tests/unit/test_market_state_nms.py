"""Behaviour-pinning tests for the NEPSE Momentum Score (NMS) sub-signal of the
market-state detector: the 60-day median cross-sectional return mapped to
trend / neutral / choppy with a normalised 0-1 contribution. Thresholds:
TREND > 8%, CHOPPY < 3%. No change to market_state_detector.py."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.quant_pro.market_state_detector import _compute_nms


def _grid(n_symbols: int, n_dates: int, last_close: float) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Flat at 100 on every date except the last (= last_close), so each symbol's
    start->end return over the window is exactly last_close/100 - 1."""
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    rows = []
    for s in range(n_symbols):
        sym = f"S{s:02d}"
        for i, d in enumerate(dates):
            rows.append({"symbol": sym, "date": d, "close": last_close if i == n_dates - 1 else 100.0})
    return pd.DataFrame(rows), dates


def test_nms_trending_above_8pct():
    df, dates = _grid(25, 65, 110.0)  # +10% > 8%
    nms, r = _compute_nms(df, dates[-1], lookback=60)
    assert nms == pytest.approx(0.10)
    assert r.trend_signal is True
    assert r.choppy_signal is False
    assert r.label == "trend"
    assert r.norm == pytest.approx(1.0)  # clipped at the top of the band


def test_nms_choppy_below_3pct():
    df, dates = _grid(25, 65, 101.0)  # +1% < 3%
    nms, r = _compute_nms(df, dates[-1], lookback=60)
    assert nms == pytest.approx(0.01)
    assert r.choppy_signal is True
    assert r.trend_signal is False
    assert r.label == "choppy"
    assert r.norm == pytest.approx(0.0)  # clipped at the bottom of the band


def test_nms_neutral_between_thresholds():
    df, dates = _grid(25, 65, 105.0)  # +5%, between 3% and 8%
    nms, r = _compute_nms(df, dates[-1], lookback=60)
    assert nms == pytest.approx(0.05)
    assert r.trend_signal is False
    assert r.choppy_signal is False
    assert r.label == "neutral"
    assert r.norm == pytest.approx((0.05 - 0.03) / (0.08 - 0.03))  # 0.4


def test_nms_insufficient_history_returns_neutral_zero():
    df, dates = _grid(25, 40, 110.0)  # < 60 trading days
    nms, r = _compute_nms(df, dates[-1], lookback=60)
    assert nms == 0.0
    assert r.label == "neutral"


def test_nms_insufficient_breadth_returns_neutral_zero():
    df, dates = _grid(10, 65, 110.0)  # < 20 common symbols
    nms, r = _compute_nms(df, dates[-1], lookback=60)
    assert nms == 0.0
    assert r.label == "neutral"
