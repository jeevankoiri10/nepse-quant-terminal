"""Behaviour-pinning tests for the pure pairs-trading math helpers
(Gatev et al. 2006 / Ornstein-Uhlenbeck). Uses exact deterministic inputs so the
expected values are analytic, not approximate. Pins behaviour - no code change."""

from __future__ import annotations

import numpy as np
import pytest

from backend.quant_pro.pairs_trading import _compute_hedge_ratio, _estimate_ou_halflife


# --------------------------------------------------------------------------- #
# _compute_hedge_ratio - OLS beta of prices_a = alpha + beta * prices_b
# --------------------------------------------------------------------------- #

def test_hedge_ratio_recovers_exact_positive_beta():
    b = np.linspace(10.0, 50.0, 100)
    a = 2.5 * b + 7.0  # exact linear relationship
    assert _compute_hedge_ratio(a, b) == pytest.approx(2.5, rel=1e-6)


def test_hedge_ratio_recovers_exact_negative_beta():
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = -0.8 * b + 3.0
    assert _compute_hedge_ratio(a, b) == pytest.approx(-0.8, rel=1e-6)


def test_hedge_ratio_is_unaffected_by_intercept():
    b = np.linspace(1.0, 20.0, 40)
    # same slope, different intercepts -> same hedge ratio
    assert _compute_hedge_ratio(1.3 * b + 0.0, b) == pytest.approx(
        _compute_hedge_ratio(1.3 * b + 99.0, b), rel=1e-6
    )


# --------------------------------------------------------------------------- #
# _estimate_ou_halflife - AR(1) regression on the spread
# --------------------------------------------------------------------------- #

def test_ou_halflife_too_short_returns_none():
    assert _estimate_ou_halflife(np.arange(10.0)) is None  # < 30 samples


def test_ou_halflife_linear_trend_is_not_mean_reverting():
    # a pure trend has regression phi ~ 0 (>= 0) -> not mean-reverting -> None
    assert _estimate_ou_halflife(np.arange(50.0)) is None


def test_ou_halflife_known_geometric_decay():
    # S_t = 100 * 0.9^t  =>  delta_S = -0.1 * S_{t-1}  =>  phi = -0.1
    # half-life = -ln(2) / ln(1 + phi) = -ln(2) / ln(0.9)
    t = np.arange(60)
    spread = 100.0 * (0.9 ** t)
    expected = -np.log(2) / np.log(0.9)
    hl = _estimate_ou_halflife(spread)
    assert hl is not None
    assert hl == pytest.approx(expected, rel=1e-5)


def test_ou_halflife_faster_reversion_is_shorter_halflife():
    t = np.arange(60)
    slow = _estimate_ou_halflife(100.0 * (0.95 ** t))   # phi = -0.05
    fast = _estimate_ou_halflife(100.0 * (0.80 ** t))   # phi = -0.20
    assert slow is not None and fast is not None
    assert fast < slow  # stronger mean reversion -> shorter half-life
