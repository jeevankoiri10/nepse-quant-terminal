"""Behaviour-pinning tests for the gold-hedge pure math: the minimum-variance
hedge ratio (Ederington 1979) and the equal-weighted equity-return builder.
Uses analytic inputs where the answer is exact. No change to gold_hedge.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.quant_pro.gold_hedge import _compute_equity_returns, _compute_mvhr


# --------------------------------------------------------------------------- #
# _compute_mvhr : h* = Cov(R_p, R_g) / Var(R_g)
# --------------------------------------------------------------------------- #

def test_mvhr_perfect_linear_hedge_recovers_beta_and_full_effectiveness():
    n = 40
    idx = range(n)
    rng = np.random.default_rng(0)
    g = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    p = pd.Series(2.0 * g.values + 0.001, index=idx)  # R_p = 2*R_g + const

    h_star, he = _compute_mvhr(p, g)

    assert h_star == pytest.approx(2.0, rel=1e-9)   # exact OLS beta
    assert he == pytest.approx(1.0, abs=1e-9)        # hedged variance is ~0 -> HE 1


def test_mvhr_independent_series_has_near_zero_ratio_and_effectiveness():
    n = 400
    idx = range(n)
    rng = np.random.default_rng(7)
    p = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    g = pd.Series(rng.normal(0.0, 0.01, n), index=idx)

    h_star, he = _compute_mvhr(p, g)

    assert abs(h_star) < 0.2     # uncorrelated -> hedge ratio near zero
    assert 0.0 <= he < 0.1


def test_mvhr_insufficient_overlap_returns_zeros():
    idx = range(10)  # < 20 common dates
    s = pd.Series(np.arange(10, dtype=float), index=idx)
    assert _compute_mvhr(s, s) == (0.0, 0.0)


def test_mvhr_zero_gold_variance_returns_zeros():
    n = 30
    idx = range(n)
    rng = np.random.default_rng(1)
    p = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    g = pd.Series(np.full(n, 0.005), index=idx)  # constant -> Var(R_g) = 0
    assert _compute_mvhr(p, g) == (0.0, 0.0)


def test_mvhr_effectiveness_is_bounded_unit_interval():
    n = 60
    idx = range(n)
    rng = np.random.default_rng(3)
    g = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    p = pd.Series(0.7 * g.values + rng.normal(0.0, 0.005, n), index=idx)
    _, he = _compute_mvhr(p, g)
    assert 0.0 <= he <= 1.0


# --------------------------------------------------------------------------- #
# _compute_equity_returns
# --------------------------------------------------------------------------- #

def _prices(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes)).astype(str)
    return pd.DataFrame({"symbol": symbol, "date": dates, "close": closes})


def test_equity_returns_none_for_empty_frame():
    assert _compute_equity_returns(pd.DataFrame(), ["A"], "2024-12-31", 60) is None


def test_equity_returns_none_for_single_symbol():
    df = _prices("A", [100.0 + i for i in range(60)])
    # needs at least 2 symbols with data
    assert _compute_equity_returns(df, ["A"], df["date"].max(), 60) is None


def test_equity_returns_equal_weight_two_symbols():
    n = 60
    a = _prices("A", list(np.linspace(100.0, 120.0, n)))
    b = _prices("B", list(np.linspace(50.0, 60.0, n)))
    df = pd.concat([a, b], ignore_index=True)

    out = _compute_equity_returns(df, ["A", "B"], df["date"].max(), lookback=60)

    assert out is not None
    assert len(out) >= 20
    assert (out > 0).all()  # both legs trend up -> equal-weighted log returns positive
