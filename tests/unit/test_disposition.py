"""Behaviour-pinning tests for the Capital Gains Overhang (CGO) disposition
signal (Grinblatt & Han 2005). Covers the pure VWAP/CGO math and the
no-lookahead signal generator. These pin existing behaviour - they do not
change it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.quant_pro.alpha_practical import SignalType
from backend.quant_pro.disposition import (
    CGO_LOOKBACK,
    _compute_cgo,
    generate_cgo_signals_at_date,
)


# --------------------------------------------------------------------------- #
# _compute_cgo - pure function
# --------------------------------------------------------------------------- #

def test_compute_cgo_insufficient_history_returns_none():
    assert _compute_cgo(np.array([10.0, 11.0]), np.array([1.0, 1.0]), lookback=3) is None


def test_compute_cgo_zero_total_volume_returns_none():
    close = np.array([10.0, 10.0, 20.0])
    vol = np.array([0.0, 0.0, 0.0])
    assert _compute_cgo(close, vol, lookback=3) is None


def test_compute_cgo_known_value():
    # vwap = (10*1 + 10*1 + 20*2) / 4 = 15 ; cgo = (20 - 15) / 20 = 0.25
    close = np.array([10.0, 10.0, 20.0])
    vol = np.array([1.0, 1.0, 2.0])
    assert _compute_cgo(close, vol, lookback=3) == pytest.approx(0.25)


def test_compute_cgo_negative_when_price_below_reference():
    close = np.array([20.0, 20.0, 10.0])
    vol = np.array([1.0, 1.0, 1.0])
    assert _compute_cgo(close, vol, lookback=3) < 0


def test_compute_cgo_nonpositive_current_price_returns_none():
    close = np.array([10.0, 10.0, 0.0])
    vol = np.array([1.0, 1.0, 1.0])
    assert _compute_cgo(close, vol, lookback=3) is None


def test_compute_cgo_uses_only_the_lookback_window():
    # an ancient outlier outside the window must not affect the result
    close = np.array([9999.0, 10.0, 10.0, 20.0])
    vol = np.array([1.0, 1.0, 1.0, 2.0])
    assert _compute_cgo(close, vol, lookback=3) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# generate_cgo_signals_at_date - signal generator
# --------------------------------------------------------------------------- #

def _prices(symbol: str, closes: list[float], volumes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range(start="2024-01-01", periods=len(closes))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
        }
    )


_N = CGO_LOOKBACK + 10  # enough history to clear the length guard


def test_fires_on_high_cgo_plus_volume_spike():
    closes = [100.0] * (_N - 1) + [200.0]          # price well above the VWAP reference
    volumes = [100_000.0] * (_N - 1) + [500_000.0]  # 5x volume breakout on the last bar
    df = _prices("WIN", closes, volumes)

    signals = generate_cgo_signals_at_date(df, df["date"].max())

    assert len(signals) == 1
    s = signals[0]
    assert s.symbol == "WIN"
    assert s.signal_type == SignalType.DISPOSITION
    assert s.direction == 1
    assert 0.0 < s.strength <= 0.75
    assert s.confidence > 0.4


def test_no_lookahead_future_spike_is_ignored():
    closes = [100.0] * (_N - 1) + [200.0]
    volumes = [100_000.0] * (_N - 1) + [500_000.0]
    df = _prices("WIN", closes, volumes)

    # evaluate the day BEFORE the spike: only flat history is visible -> no signal
    earlier = df["date"].iloc[-2]
    assert generate_cgo_signals_at_date(df, earlier) == []


def test_flat_price_below_threshold_no_signal():
    closes = [100.0] * _N                             # CGO ~ 0, below the 0.15 threshold
    volumes = [100_000.0] * (_N - 1) + [500_000.0]    # spike present, but CGO gates it out
    df = _prices("FLAT", closes, volumes)
    assert generate_cgo_signals_at_date(df, df["date"].max()) == []


def test_without_volume_spike_no_signal():
    closes = [100.0] * (_N - 1) + [200.0]             # high CGO...
    volumes = [100_000.0] * _N                        # ...but flat volume (no breakout)
    df = _prices("NOBREAK", closes, volumes)
    assert generate_cgo_signals_at_date(df, df["date"].max()) == []


def test_illiquid_symbol_is_filtered_out():
    closes = [100.0] * (_N - 1) + [200.0]
    volumes = [1_000.0] * (_N - 1) + [5_000.0]        # 20d avg far below MIN_VOLUME
    df = _prices("ILLIQ", closes, volumes)
    assert generate_cgo_signals_at_date(df, df["date"].max()) == []


def test_results_sorted_by_strength_descending():
    strong = _prices("STRONG", [100.0] * (_N - 1) + [200.0], [100_000.0] * (_N - 1) + [500_000.0])
    weak = _prices("WEAK", [100.0] * (_N - 1) + [130.0], [100_000.0] * (_N - 1) + [500_000.0])
    df = pd.concat([weak, strong], ignore_index=True)

    signals = generate_cgo_signals_at_date(df, df["date"].max())

    assert [s.symbol for s in signals] == ["STRONG", "WEAK"]
    assert signals[0].strength >= signals[1].strength


def test_liquid_symbols_filter_restricts_universe():
    closes = [100.0] * (_N - 1) + [200.0]
    volumes = [100_000.0] * (_N - 1) + [500_000.0]
    df = pd.concat([_prices("A", closes, volumes), _prices("B", closes, volumes)], ignore_index=True)

    only_a = generate_cgo_signals_at_date(df, df["date"].max(), liquid_symbols=["A"])
    assert [s.symbol for s in only_a] == ["A"]
