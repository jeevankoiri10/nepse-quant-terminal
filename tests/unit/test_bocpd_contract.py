"""Contract tests for run_bocpd_on_returns (the deterministic BOCPD entry point
in regime_detection.py). Pins the output contract — per-step changepoint
probabilities and a boolean changepoint mask, both aligned to the input length —
which is the part a caller can rely on. Asserts only verified, deterministic
properties (no behavioural claims about detection quality). No change to the
module."""

from __future__ import annotations

import numpy as np

from backend.quant_pro.regime_detection import run_bocpd_on_returns


def test_output_shapes_dtype_and_ranges():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, 200)
    cp_probs, changepoints = run_bocpd_on_returns(returns)

    assert cp_probs.shape == returns.shape
    assert changepoints.shape == returns.shape
    assert changepoints.dtype == bool
    assert np.all((cp_probs >= 0.0) & (cp_probs <= 1.0))  # valid probabilities


def test_empty_input_returns_empty_aligned_arrays():
    cp_probs, changepoints = run_bocpd_on_returns(np.array([]))
    assert cp_probs.shape == (0,)
    assert changepoints.shape == (0,)


def test_runs_on_flat_and_short_series():
    for returns in (np.full(150, 0.001), np.zeros(50), np.array([0.01, -0.02, 0.0])):
        cp_probs, changepoints = run_bocpd_on_returns(returns)
        assert len(cp_probs) == len(returns) == len(changepoints)
        assert np.all((cp_probs >= 0.0) & (cp_probs <= 1.0))


def test_is_deterministic_for_same_input():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, 120)
    a_probs, a_cps = run_bocpd_on_returns(returns)
    b_probs, b_cps = run_bocpd_on_returns(returns)
    assert np.array_equal(a_probs, b_probs)
    assert np.array_equal(a_cps, b_cps)
