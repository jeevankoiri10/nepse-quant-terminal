"""Behaviour-pinning tests for generate_sentiment_signals_at_date: the Nepali
text-sentiment signal generator only emits long-only buys when the blended score
clears the threshold (and the symbol is in the price universe). Missing/empty
data, too few documents, below-threshold, or negative sentiment all emit
nothing. Temp sentiment_scores DB + fixed date. No change to nepali_sentiment.py."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from backend.quant_pro.alpha_practical import SignalType
from backend.quant_pro.nepali_sentiment import generate_sentiment_signals_at_date

AS_OF = datetime(2026, 5, 15)


def _sent_db(path, rows: list[tuple], create: bool = True) -> str:
    conn = sqlite3.connect(str(path))
    if create:
        conn.execute(
            "CREATE TABLE sentiment_scores "
            "(date TEXT, symbol TEXT, source TEXT, model TEXT, score REAL, confidence REAL, n_documents INTEGER)"
        )
        conn.executemany("INSERT INTO sentiment_scores VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return str(path)


def _prices(*symbols: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": list(symbols), "date": ["2026-05-15"] * len(symbols), "close": [100.0] * len(symbols)}
    )


def test_missing_table_emits_nothing(tmp_path):
    # the generator wraps the query in try/except -> a missing table is graceful
    db = _sent_db(tmp_path / "m.db", [], create=False)
    assert generate_sentiment_signals_at_date(_prices("AAA"), AS_OF, db_path=db) == []


def test_empty_table_emits_nothing(tmp_path):
    db = _sent_db(tmp_path / "m.db", [])
    assert generate_sentiment_signals_at_date(_prices("AAA"), AS_OF, db_path=db) == []


def test_below_min_documents_emits_nothing(tmp_path):
    db = _sent_db(tmp_path / "m.db", [("2026-05-15", "AAA", "news", "m", 0.9, 0.8, 1)])  # 1 < 2
    assert generate_sentiment_signals_at_date(_prices("AAA"), AS_OF, db_path=db) == []


def test_below_threshold_emits_nothing(tmp_path):
    rows = [
        ("2026-05-14", "AAA", "news", "m", 0.1, 0.8, 2),
        ("2026-05-15", "AAA", "twitter", "m", 0.1, 0.8, 2),
    ]
    db = _sent_db(tmp_path / "m.db", rows)  # blended ~0.07 < 0.3
    assert generate_sentiment_signals_at_date(_prices("AAA"), AS_OF, db_path=db) == []


def test_negative_sentiment_no_short(tmp_path):
    rows = [
        ("2026-05-14", "AAA", "news", "m", -0.8, 0.8, 2),
        ("2026-05-15", "AAA", "twitter", "m", -0.8, 0.8, 2),
    ]
    db = _sent_db(tmp_path / "m.db", rows)  # strong but negative -> long-only -> nothing
    assert generate_sentiment_signals_at_date(_prices("AAA"), AS_OF, db_path=db) == []


def test_strong_positive_emits_nlp_buy(tmp_path):
    rows = [
        ("2026-05-14", "AAA", "news", "m", 0.8, 0.8, 2),
        ("2026-05-15", "AAA", "twitter", "m", 0.8, 0.9, 2),
    ]
    db = _sent_db(tmp_path / "m.db", rows)
    sigs = generate_sentiment_signals_at_date(_prices("AAA"), AS_OF, db_path=db)
    assert len(sigs) == 1
    s = sigs[0]
    assert s.symbol == "AAA"
    assert s.signal_type == SignalType.NLP_SENTIMENT
    assert s.direction == 1
    assert 0.0 < s.strength <= 1.0
