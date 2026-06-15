"""Behaviour-pinning tests for _aggregate_sentiment: per-symbol and market-wide
sentiment aggregation with a minimum-documents filter. No change to
nepali_sentiment.py."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.quant_pro.nepali_sentiment import _aggregate_sentiment


def _scores(rows: list[dict]) -> pd.DataFrame:
    cols = ["symbol", "score", "n_documents", "confidence", "source"]
    return pd.DataFrame(rows, columns=cols)


def test_empty_returns_empty():
    assert _aggregate_sentiment(_scores([])) == {}


def test_per_symbol_aggregation():
    df = _scores([
        {"symbol": "AAA", "score": 0.5, "n_documents": 1, "confidence": 0.8, "source": "news"},
        {"symbol": "AAA", "score": 0.7, "n_documents": 2, "confidence": 0.9, "source": "twitter"},
    ])
    out = _aggregate_sentiment(df, min_documents=2)
    assert "AAA" in out
    a = out["AAA"]
    assert a["mean_score"] == pytest.approx(0.6)        # (0.5 + 0.7) / 2
    assert a["total_docs"] == 3                          # 1 + 2
    assert a["mean_confidence"] == pytest.approx(0.85)
    assert a["n_sources"] == 2                           # news, twitter


def test_symbol_below_min_documents_excluded():
    df = _scores([
        {"symbol": "BBB", "score": 0.5, "n_documents": 1, "confidence": 0.8, "source": "news"},
    ])
    assert _aggregate_sentiment(df, min_documents=2) == {}  # total_docs 1 < 2


def test_market_wide_stored_under_special_key():
    df = _scores([
        {"symbol": None, "score": 0.2, "n_documents": 2, "confidence": 0.6, "source": "news"},
        {"symbol": None, "score": 0.4, "n_documents": 3, "confidence": 0.8, "source": "news"},
    ])
    out = _aggregate_sentiment(df, min_documents=2)
    assert "__MARKET__" in out
    m = out["__MARKET__"]
    assert m["mean_score"] == pytest.approx(0.3)
    assert m["total_docs"] == 5
    assert m["mean_confidence"] == pytest.approx(0.7)


def test_mix_of_per_symbol_and_market_wide():
    df = _scores([
        {"symbol": "AAA", "score": 0.5, "n_documents": 1, "confidence": 0.8, "source": "news"},
        {"symbol": "AAA", "score": 0.7, "n_documents": 2, "confidence": 0.9, "source": "twitter"},
        {"symbol": None, "score": 0.2, "n_documents": 2, "confidence": 0.6, "source": "news"},
        {"symbol": None, "score": 0.4, "n_documents": 3, "confidence": 0.8, "source": "news"},
    ])
    out = _aggregate_sentiment(df, min_documents=2)
    assert set(out.keys()) == {"AAA", "__MARKET__"}
