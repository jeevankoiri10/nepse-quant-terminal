"""Behaviour-pinning tests for conformal Value-at-Risk (distribution-free VaR via
split conformal prediction). Pins constructor validation, the documented
short-input fallback, volatility monotonicity, the position-scale formula, and
coverage_test violation counting. No change to conformal_risk.py."""

from __future__ import annotations

import numpy as np
import pytest

from backend.quant_pro.conformal_risk import (
    ConformalVaR,
    compute_conformal_position_scale,
    compute_conformal_var,
)


# --------------------------------------------------------------------------- #
# constructor validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError):
        ConformalVaR(alpha=alpha)


def test_window_below_minimum_raises():
    with pytest.raises(ValueError):
        ConformalVaR(window=10)


@pytest.mark.parametrize("cal_ratio", [0.05, 0.6])
def test_cal_ratio_out_of_range_raises(cal_ratio):
    with pytest.raises(ValueError):
        ConformalVaR(cal_ratio=cal_ratio)


def test_valid_construction_keeps_params():
    m = ConformalVaR(alpha=0.01, window=100, cal_ratio=0.25)
    assert (m.alpha, m.window, m.cal_ratio) == (0.01, 100, 0.25)


# --------------------------------------------------------------------------- #
# fit_predict / compute_conformal_var
# --------------------------------------------------------------------------- #

def test_short_input_falls_back_to_empirical_quantile():
    r = np.linspace(-0.1, 0.1, 20)  # < 30 -> documented fallback
    assert ConformalVaR(alpha=0.05).fit_predict(r) == pytest.approx(float(np.quantile(r, 0.05)))


def test_compute_conformal_var_matches_class_method():
    r = np.linspace(-0.1, 0.1, 25)
    assert compute_conformal_var(r, alpha=0.05) == pytest.approx(
        ConformalVaR(alpha=0.05).fit_predict(r)
    )


def test_more_volatile_series_gives_deeper_var():
    rng = np.random.default_rng(42)
    base = rng.normal(0.0, 0.02, 300)
    m = ConformalVaR(alpha=0.05, window=252)
    v_low = m.fit_predict(base)
    v_high = m.fit_predict(base * 3.0)
    assert v_low < 0
    assert v_high < v_low  # 3x volatility -> more negative (deeper) VaR


# --------------------------------------------------------------------------- #
# compute_conformal_position_scale
# --------------------------------------------------------------------------- #

def test_position_scale_is_full_when_no_expected_loss():
    r = np.linspace(0.01, 0.05, 20)  # all-positive -> VaR >= 0 -> full size
    assert compute_conformal_position_scale(r, alpha=0.05) == 1.0


def test_position_scale_caps_at_one_for_tiny_var():
    r = np.linspace(-0.001, 0.001, 20)  # max_loss/|VaR| > 1 -> capped
    assert compute_conformal_position_scale(r, alpha=0.05, max_loss_pct=0.02) == 1.0


def test_position_scale_follows_max_loss_over_abs_var():
    r = np.linspace(-0.10, 0.10, 20)
    cvar = float(np.quantile(r, 0.05))  # short-input fallback path
    expected = min(0.02 / abs(cvar), 1.0)
    got = compute_conformal_position_scale(r, alpha=0.05, max_loss_pct=0.02)
    assert got == pytest.approx(expected)
    assert got < 1.0


# --------------------------------------------------------------------------- #
# coverage_test
# --------------------------------------------------------------------------- #

def test_coverage_test_counts_violations_and_coverage():
    returns = np.array([0.01, -0.20, 0.02, -0.30, 0.00])
    var_est = np.full(5, -0.10)
    out = ConformalVaR(alpha=0.05).coverage_test(returns, var_est)
    assert out["violations"] == 2          # -0.20 and -0.30 breach -0.10
    assert out["total"] == 5
    assert out["empirical_coverage"] == pytest.approx(1 - 2 / 5)
    assert out["expected_coverage"] == pytest.approx(0.95)


def test_coverage_test_length_mismatch_raises():
    with pytest.raises(ValueError):
        ConformalVaR().coverage_test(np.array([0.1, 0.2]), np.array([-0.1]))
