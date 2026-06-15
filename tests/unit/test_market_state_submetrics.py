"""Behaviour-pinning tests for the remaining market-state sub-signals:
Rolling Breadth (RB), Volatility Regime (VR), and Momentum Persistence (MP).
Each is exercised at a controlled extreme plus its insufficient-history
fallback. No change to market_state_detector.py."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import pytest

from backend.quant_pro.market_state_detector import (
    _compute_mp,
    _compute_rb,
    _compute_vr,
)


def _build(n_symbols: int, dates: pd.DatetimeIndex, close_fn: Callable[[int, int], float]) -> pd.DataFrame:
    rows = []
    for i in range(n_symbols):
        sym = f"S{i:02d}"
        for t, d in enumerate(dates):
            rows.append({"symbol": sym, "date": d, "close": close_fn(i, t), "volume": 1000.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Rolling Breadth (RB) : % of liquid stocks above their 50d MA
# --------------------------------------------------------------------------- #

def test_rb_all_above_ma_is_full_breadth_trend():
    dates = pd.bdate_range("2023-01-01", periods=60)
    df = _build(25, dates, lambda i, t: 100.0 + 0.5 * t)  # rising -> every stock above its 50d MA
    rb, r = _compute_rb(df, dates[-1])
    assert rb == pytest.approx(1.0)
    assert r.trend_signal is True
    assert r.label == "trend"


def test_rb_insufficient_history_neutral_half():
    dates = pd.bdate_range("2023-01-01", periods=40)  # < ma_window(50) + 5
    df = _build(25, dates, lambda i, t: 100.0 + t)
    rb, r = _compute_rb(df, dates[-1])
    assert rb == 0.5
    assert r.label == "neutral"


# --------------------------------------------------------------------------- #
# Volatility Regime (VR) : annualised vol of the cross-sectional median return
# --------------------------------------------------------------------------- #

def test_vr_flat_market_is_zero_vol_trend():
    dates = pd.bdate_range("2023-01-01", periods=25)
    df = _build(15, dates, lambda i, t: 100.0)  # flat -> every daily median return is 0
    vr, r = _compute_vr(df, dates[-1])
    assert vr == pytest.approx(0.0)
    assert r.trend_signal is True
    assert r.label == "trend"


def test_vr_insufficient_history_neutral_default():
    dates = pd.bdate_range("2023-01-01", periods=15)  # < lookback(20) + 2
    df = _build(15, dates, lambda i, t: 100.0 + t)
    vr, r = _compute_vr(df, dates[-1])
    assert vr == 0.20
    assert r.label == "neutral"


# --------------------------------------------------------------------------- #
# Momentum Persistence (MP) : top-quintile overlap now vs 5 days ago
# --------------------------------------------------------------------------- #

def test_mp_stable_leaders_full_persistence_trend():
    dates = pd.bdate_range("2023-01-01", periods=35)
    # distinct, monotone growth per symbol -> identical momentum ranking now and lagged
    df = _build(25, dates, lambda i, t: 100.0 * (1.0 + i * 0.002) ** t)
    mp, r = _compute_mp(df, dates[-1])
    assert mp == pytest.approx(1.0)
    assert r.trend_signal is True
    assert r.label == "trend"


def test_mp_insufficient_history_neutral_default():
    dates = pd.bdate_range("2023-01-01", periods=20)  # < window(20) + lag(5) + 2
    df = _build(25, dates, lambda i, t: 100.0 + i + t)
    mp, r = _compute_mp(df, dates[-1])
    assert mp == 0.575
    assert r.label == "neutral"
