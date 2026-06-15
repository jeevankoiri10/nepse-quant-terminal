"""Behaviour-pinning tests for the Telegram/TUI message formatters. These pin the
exact user-facing output (emoji, escaping, separators, pnl signs) so it can't
regress silently. The file is UTF-8; expected emoji are kept in named constants
so the assertions stay readable."""

from __future__ import annotations

from backend.quant_pro import message_formatters as mf
from backend.quant_pro.tms_models import ExecutionStatus

GREEN = "\U0001f7e2"
RED = "\U0001f534"
WHITE = "⚪"
INFO = "ℹ️"
CHECK = "✅"
CROSS = "❌"
DOT = "•"  # the " • " separator


def test_action_prefix():
    assert mf._action_prefix("BUY") == f"{GREEN} BUY"
    assert mf._action_prefix("sell") == f"{RED} SELL"  # case-insensitive
    assert mf._action_prefix("") == "TRADE"            # empty -> generic label
    assert mf._action_prefix("foo") == "FOO"           # unknown -> uppercased verbatim


def test_action_emoji():
    assert mf._action_emoji("BUY") == GREEN
    assert mf._action_emoji("CANCEL") == WHITE
    assert mf._action_emoji("zzz") == INFO             # unknown -> info fallback


def test_polarity_emoji():
    assert mf._polarity_emoji(2.5) == GREEN
    assert mf._polarity_emoji(-0.1) == RED
    assert mf._polarity_emoji(0) == WHITE


def test_status_label_known_and_default():
    assert mf._status_label(str(ExecutionStatus.FILLED)) == ("Filled", CHECK)
    assert mf._status_label(str(ExecutionStatus.FILLED).upper()) == ("Filled", CHECK)  # case-insensitive
    assert mf._status_label("not-a-status") == ("Updated", INFO)
    assert mf._status_label("") == ("Updated", INFO)


def test_trade_line_sell_shows_signed_pnl_and_check():
    line = mf.format_trade_activity_line(
        date="2026-06-15", action="SELL", symbol="NABIL", shares=10, price=500.0, pnl=1234.0
    )
    assert line == f"2026-06-15 {DOT} {RED} SELL NABIL {DOT} 10 @ NPR 500.0 {DOT} NPR +1,234 {CHECK}"


def test_trade_line_sell_negative_pnl_shows_cross_and_omits_date():
    line = mf.format_trade_activity_line(
        date=None, action="SELL", symbol="X", shares=1, price=100.0, pnl=-50.0
    )
    assert f"NPR -50 {CROSS}" in line
    assert line.startswith(f"{RED} SELL X")  # date=None -> no leading date


def test_trade_line_buy_ignores_pnl_and_uses_status_text():
    line = mf.format_trade_activity_line(
        date="d", action="BUY", symbol="SBL", shares=5, price=640.5,
        pnl=999.0, status_text="Queued", include_date=False,
    )
    assert line == f"{GREEN} BUY SBL {DOT} 5 @ NPR 640.5 {DOT} Queued"
    assert "999" not in line  # the pnl branch is SELL-only


def test_trade_line_include_date_toggle():
    args = dict(date="2026-06-15", action="BUY", symbol="A", shares=1, price=10.0)
    assert mf.format_trade_activity_line(**args).startswith("2026-06-15")
    assert not mf.format_trade_activity_line(**args, include_date=False).startswith("2026-06-15")


def test_trade_activity_html_escapes_symbol():
    html = mf.format_trade_activity_html(
        date=None, action="BUY", symbol="A<b>", shares=1, price=10.0, include_date=False
    )
    assert "&lt;b&gt;" in html
    assert "<b>" not in html


def test_portfolio_holding_html_single_line():
    out = mf.format_portfolio_holding_html(symbol="A&B", direction_value=1.0, primary_text="+5%")
    assert out == f"{GREEN} <b>A&amp;B</b> {DOT} +5%"  # & escaped, green for positive
    assert "\n" not in out


def test_portfolio_holding_html_second_line_for_metrics_and_flags():
    out = mf.format_portfolio_holding_html(
        symbol="X", direction_value=-1.0, primary_text="p",
        holding_days=3, extra_metrics=["m1"], flags=["FLAG"],
    )
    first, second = out.split("\n")
    assert first.startswith(f"{RED} <b>X</b>")  # red for negative direction
    assert "Day 3" in first
    assert second == f"m1 {DOT} FLAG"
