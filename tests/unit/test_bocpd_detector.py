"""Behaviour-pinning tests for BOCPDDetector (Adams-MacKay online changepoint
detection). Covers the deterministic, verified behaviours: the warmup guard,
the documented P(r=0)=hazard design quirk, run-length growth on a stable
segment, and reset. No change to regime_detection.py."""

from __future__ import annotations

import numpy as np
import pytest

from backend.quant_pro.regime_detection import BOCPDDetector


def test_detect_returns_false_during_warmup():
    # detect() guards on len(run_length_probs) < 5, so it can't fire in the first
    # few steps regardless of threshold.
    d = BOCPDDetector()
    for x in (0.01, -0.01, 0.02):
        d.update(x)
        assert d.detect(0.0) is False
    assert len(d.run_length_probs) < 5


def test_update_changepoint_probability_equals_hazard_by_design():
    # update() returns P(r_t=0), which under per-step normalization collapses to
    # the constant hazard rate. This is INTENTIONAL: detect() deliberately uses
    # the short-run-length mass instead (see detect()'s docstring), so this is a
    # documented property, not a defect.
    d = BOCPDDetector(hazard_lambda=100.0)
    rng = np.random.default_rng(0)
    cp = 0.0
    for x in rng.normal(0.0, 0.01, 50):
        cp = d.update(x)
    assert cp == pytest.approx(1.0 / 100.0)
    assert d.changepoint_probability == pytest.approx(1.0 / 100.0)


def test_expected_run_length_grows_on_a_stable_segment():
    d = BOCPDDetector()
    rng = np.random.default_rng(0)
    e_early = e_late = 0.0
    for i, x in enumerate(rng.normal(0.0, 0.005, 60)):
        d.update(x)
        if i == 10:
            e_early = d.expected_run_length
        e_late = d.expected_run_length
    assert e_late > e_early  # no changepoints -> the current run keeps growing


def test_reset_restores_initial_state():
    d = BOCPDDetector()
    for x in (0.01, 0.02, -0.01, 0.03, 0.0, 0.02):
        d.update(x)
    d.reset()
    assert np.array_equal(d.run_length_probs, np.array([1.0]))
    assert d._t == 0
    assert d.expected_run_length == 0.0
    assert d.changepoint_probability == 0.0
