"""Behaviour-pinning tests for _compute_rainfall_anomaly: fractional rainfall
anomaly vs monthly climatology, (actual - baseline) / baseline. The baseline is
derived from MONTHLY_BASELINE_MM so the expected anomaly is exact. No change to
satellite_data.py."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.quant_pro.satellite_data import MONTHLY_BASELINE_MM, _compute_rainfall_anomaly

_MONTH = 7  # any month; .get(month, 30.0) keeps it consistent either way
_DAILY_BASE = MONTHLY_BASELINE_MM.get(_MONTH, 30.0) / 30.0  # daily baseline mm


def _rain_df(daily_mm: float, n_days: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": f"2024-{_MONTH:02d}-{d:02d}", "rainfall_mm": daily_mm} for d in range(1, n_days + 1)]
    )


def test_empty_frame_returns_zero():
    assert _compute_rainfall_anomaly(pd.DataFrame()) == 0.0


def test_at_baseline_is_zero_anomaly():
    # 30 days at exactly the daily baseline -> actual == baseline -> 0
    assert _compute_rainfall_anomaly(_rain_df(_DAILY_BASE, 30)) == pytest.approx(0.0, abs=1e-9)


def test_double_baseline_is_plus_one():
    assert _compute_rainfall_anomaly(_rain_df(2 * _DAILY_BASE, 30)) == pytest.approx(1.0)


def test_half_baseline_is_minus_half():
    assert _compute_rainfall_anomaly(_rain_df(0.5 * _DAILY_BASE, 30)) == pytest.approx(-0.5)


def test_zero_rainfall_is_minus_one():
    assert _compute_rainfall_anomaly(_rain_df(0.0, 30)) == pytest.approx(-1.0)
