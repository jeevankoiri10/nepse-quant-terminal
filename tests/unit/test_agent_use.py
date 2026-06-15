"""agent-use: switches the active backend via set_active_agent, validates the
preset, and never mutates real config under test (set_active_agent is mocked)."""

from __future__ import annotations

from scripts import agent_use as au


def test_apply_passes_preset_and_model(monkeypatch):
    captured = {}

    def _fake_set(preset, *, model=None):
        captured["preset"] = preset
        captured["model"] = model
        return {"backend": preset, "model": model or "", "selected_preset": preset}

    monkeypatch.setattr("backend.agents.runtime_config.set_active_agent", _fake_set)

    cfg = au.apply("claude_sdk", model="opus")
    assert captured == {"preset": "claude_sdk", "model": "opus"}
    assert cfg["backend"] == "claude_sdk"


def test_main_switches_and_reports(monkeypatch, capsys):
    monkeypatch.setattr(
        "backend.agents.runtime_config.set_active_agent",
        lambda preset, *, model=None: {
            "backend": preset,
            "model": model or "sonnet",
            "selected_preset": preset,
        },
    )
    rc = au.main(["claude_sdk", "--model", "opus"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "claude_sdk" in out
    assert "opus" in out


def test_main_unknown_preset_exits_2_with_valid_list(monkeypatch, capsys):
    def _raise(preset, *, model=None):
        raise ValueError(f"Unknown agent backend or preset: {preset}")

    monkeypatch.setattr("backend.agents.runtime_config.set_active_agent", _raise)

    rc = au.main(["bogus"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "ERROR:" in out
    assert "selectable presets:" in out


def test_main_list_shows_presets_exit_0(monkeypatch, capsys):
    monkeypatch.setattr(au, "_valid_presets", lambda: ["claude", "claude_sdk", "ollama"])
    rc = au.main(["--list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "claude_sdk" in out


def test_main_no_preset_is_nonzero_usage(monkeypatch, capsys):
    monkeypatch.setattr(au, "_valid_presets", lambda: ["ollama"])
    rc = au.main([])
    assert rc == 2
    assert "usage:" in capsys.readouterr().out
