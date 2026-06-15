"""Behaviour-pinning tests for the pure data-parsing helpers in data_io.py
(CSV loading, the price-matrix transform, macro-series loading, sector lookup).
These parse user-supplied data, so the transforms are a real correctness
surface. No change to data_io.py."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from backend.quant_pro.data_io import (
    SECTOR_INDEX_SYMBOLS,
    extract_price_matrix,
    get_sector_index_symbol,
    load_csv,
    load_macro_series,
)


# --------------------------------------------------------------------------- #
# load_csv
# --------------------------------------------------------------------------- #

def test_load_csv_from_path(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    df = load_csv(str(p))
    assert df is not None
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_load_csv_from_file_like():
    df = load_csv(io.BytesIO(b"a,b\n1,2\n"))
    assert df is not None
    assert int(df.iloc[0]["a"]) == 1


def test_load_csv_bad_input_returns_none():
    assert load_csv(io.BytesIO(b"")) is None  # EmptyDataError is caught -> None


# --------------------------------------------------------------------------- #
# extract_price_matrix
# --------------------------------------------------------------------------- #

def test_extract_price_matrix_empty_raises():
    with pytest.raises(ValueError):
        extract_price_matrix(pd.DataFrame())


def test_extract_price_matrix_sorts_and_indexes_by_date():
    raw = pd.DataFrame({"Date": ["2024-01-03", "2024-01-01", "2024-01-02"], "AAA": [12.0, 10.0, 11.0]})
    out = extract_price_matrix(raw)
    assert list(out.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert list(out["AAA"]) == [10.0, 11.0, 12.0]


def test_extract_price_matrix_dedups_keep_last():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-01"], "AAA": [10.0, 99.0]})
    out = extract_price_matrix(raw)
    assert len(out) == 1
    assert out["AAA"].iloc[0] == 99.0  # keep="last"


def test_extract_price_matrix_drops_unparseable_dates():
    raw = pd.DataFrame({"Date": ["2024-01-01", "not-a-date", "2024-01-02"], "AAA": [10.0, 50.0, 11.0]})
    out = extract_price_matrix(raw)
    assert len(out) == 2  # the unparseable-date row is dropped


# --------------------------------------------------------------------------- #
# load_macro_series
# --------------------------------------------------------------------------- #

def test_load_macro_series_none_input():
    assert load_macro_series(None) is None


def test_load_macro_series_parses_date_value():
    s = load_macro_series(io.BytesIO(b"date,val\n2024-01-01,100\n2024-01-02,110\n"))
    assert s is not None
    assert s.loc[pd.Timestamp("2024-01-01")] == 100
    assert list(s.values) == [100, 110]


# --------------------------------------------------------------------------- #
# get_sector_index_symbol
# --------------------------------------------------------------------------- #

def test_get_sector_index_symbol_known_and_unknown():
    known_key = next(iter(SECTOR_INDEX_SYMBOLS))
    assert get_sector_index_symbol(known_key) == SECTOR_INDEX_SYMBOLS[known_key]
    assert get_sector_index_symbol("__not_a_sector__") is None
