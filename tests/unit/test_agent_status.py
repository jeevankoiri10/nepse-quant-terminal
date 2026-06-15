"""agent-status: surfaces the active backend + selectable presets, read-only,
no model call."""

from __future__ import annotations

from scripts import agent_status as asx


def _fake(monkeypatch, active_backend="claude_sdk", preset="claude_sdk"):
    monkeypatch.setattr(
        "backend.agents.runtime_config.load_active_agent_config",
        lambda: {
            "backend": active_backend,
            "model": "sonnet",
            "selected_preset": preset,
            "label": "Claude Agent SDK",
            "fallback_backend": "",
        },
    )
    monkeypatch.setattr(
        "backend.agents.runtime_config.list_agent_backends",
        lambda: [
            {"id": "ollama", "label": "Ollama (local)", "description": "default", "backend": "ollama", "model": "llama3"},
            {"id": "claude_sdk", "label": "Claude Agent SDK", "description": "in-process", "backend": "claude_sdk", "model": "sonnet"},
        ],
    )


def test_overview_reports_active_and_presets(monkeypatch):
    _fake(monkeypatch)
    report = asx.build_overview()
    assert report["active"]["backend"] == "claude_sdk"
    assert report["active"]["model"] == "sonnet"
    ids = {p["id"] for p in report["presets"]}
    assert ids == {"ollama", "claude_sdk"}


def test_render_marks_the_active_preset(monkeypatch):
    _fake(monkeypatch, preset="claude_sdk")
    lines = asx.render(asx.build_overview()).splitlines()
    # the active preset row is marked with '*'; the non-active one is not
    active_row = next(line for line in lines if line.lstrip().startswith("* claude_sdk"))
    assert active_row
    ollama_row = next(line for line in lines if line.lstrip().startswith("ollama"))
    assert not ollama_row.lstrip().startswith("*")


def test_render_is_ascii_safe(monkeypatch):
    _fake(monkeypatch)
    asx.render(asx.build_overview()).encode("cp1252")  # must not raise on cp1252


def test_main_json_is_machine_readable(monkeypatch, capsys):
    _fake(monkeypatch)
    rc = asx.main(["--json"])
    assert rc == 0
    import json

    data = json.loads(capsys.readouterr().out)
    assert data["active"]["selected_preset"] == "claude_sdk"
    assert isinstance(data["presets"], list)
