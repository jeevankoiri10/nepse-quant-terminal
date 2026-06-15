"""Behaviour-pinning tests for the pure portfolio-construction helpers:
equal-weight fallback allocation and the aligned return-matrix builder (the
input to the HRP/CVaR optimizers). No change to portfolio_construction.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.quant_pro.portfolio_construction import _equal_weight, _extract_return_matrix


# --------------------------------------------------------------------------- #
# _equal_weight
# --------------------------------------------------------------------------- #

def test_equal_weight_empty_symbols():
    assert _equal_weight([], 1000.0) == {}


def test_equal_weight_splits_capital_evenly():
    out = _equal_weight(["A", "B", "C", "D"], 1000.0)
    assert out == {"A": 250.0, "B": 250.0, "C": 250.0, "D": 250.0}
    assert sum(out.values()) == pytest.approx(1000.0)


def test_equal_weight_handles_indivisible_capital():
    out = _equal_weight(["A", "B", "C"], 100.0)
    assert all(v == pytest.approx(100.0 / 3) for v in out.values())
    assert sum(out.values()) == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# _extract_return_matrix
# --------------------------------------------------------------------------- #

def _df(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes)).strftime("%Y-%m-%d")
    return pd.DataFrame({"symbol": symbol, "date": dates, "close": closes})


def test_return_matrix_valid_shape_and_columns():
    n = 70
    a = _df("A", list(np.linspace(100.0, 140.0, n)))
    b = _df("B", list(np.linspace(50.0, 70.0, n)))
    df = pd.concat([a, b], ignore_index=True)

    out = _extract_return_matrix(df, ["A", "B"], df["date"].max(), lookback=60)

    assert out is not None
    assert out.shape == (60, 2)
    assert list(out.columns) == ["A", "B"]
    assert not out.isnull().any().any()


def test_return_matrix_known_constant_returns():
    # closes = 100 * 1.01^t  =>  every simple return is exactly 0.01
    n = 70
    df = _df("A", [100.0 * (1.01 ** t) for t in range(n)])
    out = _extract_return_matrix(df, ["A"], df["date"].max(), lookback=60)
    assert out is not None
    assert np.allclose(out["A"].values, 0.01)


def test_return_matrix_insufficient_history_returns_none():
    df = _df("A", list(np.linspace(100.0, 110.0, 30)))  # < lookback + 1
    assert _extract_return_matrix(df, ["A"], df["date"].max(), lookback=60) is None


def test_return_matrix_nonpositive_close_returns_none():
    closes = list(np.linspace(100.0, 140.0, 70))
    closes[65] = 0.0  # inside the trailing lookback+1 window -> invalidates the series
    df = _df("A", closes)
    assert _extract_return_matrix(df, ["A"], df["date"].max(), lookback=60) is None


def test_return_matrix_missing_symbol_returns_none():
    df = _df("A", list(np.linspace(100.0, 140.0, 70)))
    # "B" has no rows -> insufficient history -> None
    assert _extract_return_matrix(df, ["A", "B"], df["date"].max(), lookback=60) is None
