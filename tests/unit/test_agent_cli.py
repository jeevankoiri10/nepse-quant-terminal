"""agent dispatcher: routes to the operator subcommands with their exit codes,
and is self-documenting on bad/empty/help input."""

from __future__ import annotations

import pytest

from scripts import agent


def test_status_subcommand_routes(monkeypatch, capsys):
    monkeypatch.setattr(
        "backend.agents.runtime_config.load_active_agent_config",
        lambda: {"backend": "ollama", "model": "llama3", "selected_preset": "ollama", "label": "Ollama"},
    )
    monkeypatch.setattr("backend.agents.runtime_config.list_agent_backends", lambda: [])
    rc = agent.main(["status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "agent backend" in out


def test_doctor_subcommand_routes_and_returns_exit_code(monkeypatch, capsys):
    from scripts import agent_doctor

    monkeypatch.setattr(agent_doctor, "_which", lambda c: None)  # claude cli missing -> FAIL
    rc = agent.main(["doctor", "--backend", "claude"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "preflight" in out


def test_unknown_command_is_rejected_with_usage(capsys):
    rc = agent.main(["frobnicate"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "unknown command" in err
    assert "commands:" in err


def test_no_args_prints_usage_nonzero(capsys):
    rc = agent.main([])
    assert rc == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["-h", "--help", "help"])
def test_help_lists_subcommands(flag, capsys):
    rc = agent.main([flag])
    out = capsys.readouterr().out
    assert rc == 0
    assert "status" in out and "doctor" in out
