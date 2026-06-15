"""agent-doctor: resolves the active backend and reports per-backend readiness
checks with the right exit code, all without making a model call."""

from __future__ import annotations

import pytest

from scripts import agent_doctor as ad


def test_claude_sdk_checks_cover_deps_and_cli(monkeypatch):
    # all prerequisites present -> all OK, exit 0
    monkeypatch.setattr(ad, "_has_module", lambda m: True)
    monkeypatch.setattr(ad, "_which", lambda c: f"/usr/bin/{c}")

    status = ad.build_agent_status("claude_sdk")
    assert status["backend"] == "claude_sdk"
    names = {c["name"] for c in status["checks"]}
    assert names == {"sdk package", "anyio", "claude cli"}
    assert status["failed"] == []
    assert all(c["status"] == ad.OK for c in status["checks"])


def test_claude_sdk_missing_everything_fails_with_hints(monkeypatch):
    monkeypatch.setattr(ad, "_has_module", lambda m: False)
    monkeypatch.setattr(ad, "_which", lambda c: None)

    status = ad.build_agent_status("claude_sdk")
    assert set(status["failed"]) == {"sdk package", "anyio", "claude cli"}
    blob = " ".join(c["detail"] for c in status["checks"])
    assert "pip install claude-agent-sdk" in blob
    assert "npm install -g @anthropic-ai/claude-code" in blob


def test_claude_cli_backend_needs_only_the_cli(monkeypatch):
    monkeypatch.setattr(ad, "_which", lambda c: "/usr/bin/claude")
    status = ad.build_agent_status("claude")
    assert [c["name"] for c in status["checks"]] == ["claude cli"]
    assert status["failed"] == []


def test_ollama_server_unreachable_is_fatal(monkeypatch):
    monkeypatch.setattr(ad, "_which", lambda c: "/usr/bin/ollama")
    monkeypatch.setattr(ad, "_port_open", lambda host, port, timeout=0.4: False)
    status = ad.build_agent_status("ollama")
    assert "ollama server" in status["failed"]
    # the cli check is non-fatal (you could target a remote host)
    assert "ollama cli" not in status["failed"]


def test_gemma4_mlx_missing_is_warning_not_failure(monkeypatch):
    # MLX is Apple-Silicon only, so absence is a warning, not a hard failure
    monkeypatch.setattr(ad, "_has_module", lambda m: False)
    status = ad.build_agent_status("gemma4_mlx")
    assert status["failed"] == []
    assert "mlx_lm" in status["warned"]


def test_active_backend_resolved_from_config(monkeypatch):
    monkeypatch.setattr(
        ad, "_backend_checks", lambda b: [ad._check("probe", ad.OK, f"backend={b}")]
    )
    monkeypatch.setattr(
        "backend.agents.runtime_config.load_active_agent_config",
        lambda: {"backend": "claude", "model": "", "selected_preset": "claude"},
    )
    status = ad.build_agent_status(None)
    assert status["backend"] == "claude"
    assert status["selected_preset"] == "claude"


def test_main_exit_code_and_render(monkeypatch, capsys):
    monkeypatch.setattr(ad, "_which", lambda c: None)  # claude cli missing -> FAIL
    rc = ad.main(["--backend", "claude"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "agent backend preflight" in out
    assert "NOT READY" in out


def test_probe_adds_live_check_on_success(monkeypatch):
    # static checks pass, and the live call answers -> a passing "live call" check
    monkeypatch.setattr(ad, "_has_module", lambda m: True)
    monkeypatch.setattr(ad, "_which", lambda c: f"/usr/bin/{c}")
    from backend.agents import agent_analyst

    monkeypatch.setattr(agent_analyst, "_call_claude_sdk", lambda *a, **k: "ok")

    status = ad.build_agent_status("claude_sdk", probe=True)
    live = [c for c in status["checks"] if c["name"] == "live call"]
    assert live and live[0]["status"] == ad.OK
    assert status["failed"] == []


def test_probe_surfaces_actionable_error_as_fail(monkeypatch):
    monkeypatch.setattr(ad, "_has_module", lambda m: True)
    monkeypatch.setattr(ad, "_which", lambda c: f"/usr/bin/{c}")
    from backend.agents import agent_analyst

    monkeypatch.setattr(
        agent_analyst, "_call_claude_sdk",
        lambda *a, **k: "ERROR: Claude SDK is not authenticated. Run `claude login` and retry.",
    )

    status = ad.build_agent_status("claude_sdk", probe=True)
    live = [c for c in status["checks"] if c["name"] == "live call"][0]
    assert live["status"] == ad.FAIL
    assert "claude login" in live["detail"]
    assert "live call" in status["failed"]


def test_probe_skipped_when_static_check_already_failed(monkeypatch):
    # deps missing -> static FAIL -> no live call attempted (don't probe a broken setup)
    monkeypatch.setattr(ad, "_has_module", lambda m: False)
    monkeypatch.setattr(ad, "_which", lambda c: None)
    from backend.agents import agent_analyst

    def _must_not_call(*_a, **_k):
        raise AssertionError("probe must not run when a static check already failed")

    monkeypatch.setattr(agent_analyst, "_call_claude_sdk", _must_not_call)

    status = ad.build_agent_status("claude_sdk", probe=True)
    assert "live call" not in {c["name"] for c in status["checks"]}


def test_probe_is_noop_for_non_claude_backends(monkeypatch):
    monkeypatch.setattr(ad, "_which", lambda c: "/usr/bin/ollama")
    monkeypatch.setattr(ad, "_port_open", lambda *a, **k: True)
    status = ad.build_agent_status("ollama", probe=True)
    assert "live call" not in {c["name"] for c in status["checks"]}


@pytest.mark.parametrize("backend", ["claude_sdk", "claude", "ollama", "gemma4_mlx"])
def test_render_is_ascii_safe(monkeypatch, backend):
    monkeypatch.setattr(ad, "_has_module", lambda m: True)
    monkeypatch.setattr(ad, "_which", lambda c: f"/usr/bin/{c}")
    monkeypatch.setattr(ad, "_port_open", lambda *a, **k: True)
    out = ad.render(ad.build_agent_status(backend))
    out.encode("cp1252")  # must not raise on Windows consoles
