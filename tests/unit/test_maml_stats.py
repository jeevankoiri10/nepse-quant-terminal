"""Behaviour-pinning tests for the pure stat helpers in maml_regime.py
(population skewness / excess kurtosis) and the prepare_features shape contract.
Uses analytic inputs with known values. No change to maml_regime.py."""

from __future__ import annotations

import numpy as np
import pytest

from backend.quant_pro.maml_regime import _kurtosis, _skewness, prepare_features


# --------------------------------------------------------------------------- #
# _skewness
# --------------------------------------------------------------------------- #

def test_skewness_symmetric_is_zero():
    assert _skewness(np.array([-2.0, -1.0, 0.0, 1.0, 2.0])) == pytest.approx(0.0, abs=1e-9)


def test_skewness_constant_is_zero():
    assert _skewness(np.array([5.0, 5.0, 5.0, 5.0])) == 0.0


def test_skewness_too_short_is_zero():
    assert _skewness(np.array([1.0, 2.0])) == 0.0  # n < 3


def test_skewness_right_tail_is_positive():
    assert _skewness(np.array([0.0, 0.0, 0.0, 0.0, 10.0])) > 0


def test_skewness_left_tail_is_negative():
    assert _skewness(np.array([0.0, 0.0, 0.0, 0.0, -10.0])) < 0


# --------------------------------------------------------------------------- #
# _kurtosis (excess)
# --------------------------------------------------------------------------- #

def test_kurtosis_uniform_ramp_known_value():
    # x = [-2,-1,0,1,2]: mean(z^4) = 1.7, excess = 1.7 - 3 = -1.3
    assert _kurtosis(np.array([-2.0, -1.0, 0.0, 1.0, 2.0])) == pytest.approx(-1.3)


def test_kurtosis_constant_is_zero():
    assert _kurtosis(np.array([3.0, 3.0, 3.0, 3.0, 3.0])) == 0.0


def test_kurtosis_too_short_is_zero():
    assert _kurtosis(np.array([1.0, 2.0, 3.0])) == 0.0  # n < 4


# --------------------------------------------------------------------------- #
# prepare_features - shape contract
# --------------------------------------------------------------------------- #

def test_prepare_features_too_short_is_empty():
    out = prepare_features(np.zeros(5, dtype=float))
    assert out.shape == (0, 10)


def test_prepare_features_returns_clean_10dim_matrix():
    rng = np.random.default_rng(0)
    out = prepare_features(rng.normal(0.0, 0.01, 100))
    assert out.ndim == 2
    assert out.shape[1] == 10
    assert out.shape[0] > 0
    assert not np.isnan(out).any()
    assert out.dtype == np.float32
