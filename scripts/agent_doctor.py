#!/usr/bin/env python3
"""agent-doctor - will my selected AI agent backend actually work?

Read-only preflight for the agent layer. It resolves the *active* backend
(`backend/agents/runtime_config.py`) and checks that backend's prerequisites are
present BEFORE you switch the TUI to it or run an analysis:

  - ollama      : `ollama` on PATH + the host port reachable
  - gemma4_mlx  : the `mlx_lm` package importable (Apple-Silicon only)
  - claude      : the `claude` CLI on PATH
  - claude_sdk  : `claude_agent_sdk` + `anyio` importable AND the `claude` CLI on
                  PATH (the SDK drives the CLI subprocess and reuses its login)

Each check is OK / WARN / FAIL; exit code is non-zero if anything FAILs, so it
drops into a launcher or CI gate. It never makes a paid/model call - it only
inspects imports, PATH, and a localhost socket.

Usage:
    python -m scripts.agent_doctor            # human report for the active backend
    python -m scripts.agent_doctor --json     # machine-readable
    python -m scripts.agent_doctor --backend claude_sdk   # check a specific preset
"""

from __future__ import annotations

import argparse
import json
from typing import Any

OK, WARN, FAIL = "ok", "warn", "fail"


def _check(name: str, status: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail}


def _has_module(mod: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _which(cli: str) -> str | None:
    import shutil

    return shutil.which(cli)


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _module_check(label: str, mod: str, hint: str, *, fatal: bool = True) -> dict[str, Any]:
    if _has_module(mod):
        return _check(label, OK, f"{mod} importable")
    return _check(label, FAIL if fatal else WARN, f"{mod} not importable - {hint}")


def _cli_check(label: str, cli: str, hint: str, *, fatal: bool = True) -> dict[str, Any]:
    path = _which(cli)
    if path:
        return _check(label, OK, f"`{cli}` found at {path}")
    return _check(label, FAIL if fatal else WARN, f"`{cli}` not on PATH - {hint}")


def _ollama_checks() -> list[dict[str, Any]]:
    from backend.agents.runtime_config import DEFAULT_OLLAMA_HOST

    checks = [
        _cli_check(
            "ollama cli", "ollama",
            "install from https://ollama.com (or point at a remote host)",
            fatal=False,
        )
    ]
    host = DEFAULT_OLLAMA_HOST.split("://", 1)[-1]
    hostname, _, port = host.partition(":")
    port_n = int(port or "11434")
    if _port_open(hostname or "localhost", port_n):
        checks.append(_check("ollama server", OK, f"reachable at {DEFAULT_OLLAMA_HOST}"))
    else:
        checks.append(
            _check("ollama server", FAIL, f"not reachable at {DEFAULT_OLLAMA_HOST} - run `ollama serve`")
        )
    return checks


def _backend_checks(backend: str) -> list[dict[str, Any]]:
    backend = (backend or "").strip().lower()
    if backend == "claude_sdk":
        return [
            _module_check("sdk package", "claude_agent_sdk", "pip install claude-agent-sdk"),
            _module_check("anyio", "anyio", "pip install anyio"),
            _cli_check(
                "claude cli", "claude",
                "npm install -g @anthropic-ai/claude-code, then `claude login`",
            ),
        ]
    if backend == "claude":
        return [
            _cli_check(
                "claude cli", "claude",
                "npm install -g @anthropic-ai/claude-code, then `claude login`",
            )
        ]
    if backend == "ollama":
        return _ollama_checks()
    if backend == "gemma4_mlx":
        return [
            _module_check(
                "mlx_lm", "mlx_lm",
                "pip install mlx-lm (Apple Silicon only)",
                fatal=False,
            )
        ]
    return [_check("backend", WARN, f"unknown backend '{backend}' - no readiness checks defined")]


def _probe_check(backend: str) -> dict[str, Any] | None:
    """Live readiness: make one trivial call and confirm the backend answers.

    Returns None for backends with no probe defined (ollama already has a port
    check; gemma is a local model load). For the Claude backends this reuses the
    real call path, so a failure carries the same actionable message the backend
    returns (e.g. "run `claude login`"). Only invoked under --probe.
    """
    backend = (backend or "").strip().lower()
    if backend not in ("claude_sdk", "claude"):
        return None
    from backend.agents import agent_analyst

    fn = agent_analyst._call_claude_sdk if backend == "claude_sdk" else agent_analyst._call_claude
    try:
        resp = fn("Reply with the single word: ok")
    except Exception as exc:  # noqa: BLE001 - surface any call failure as a probe FAIL
        return _check("live call", FAIL, f"probe raised: {exc}")
    text = str(resp).strip()
    if text.startswith("ERROR:"):
        return _check("live call", FAIL, text[len("ERROR:"):].strip())
    return _check("live call", OK, "backend answered a test prompt")


def build_agent_status(backend: str | None = None, *, probe: bool = False) -> dict[str, Any]:
    """Resolve the backend (active config unless overridden) and run its checks.

    With ``probe=True``, and only if no static check FAILed, make one trivial
    live call to confirm the backend actually responds (auth valid, reachable).
    """
    from backend.agents.runtime_config import AGENT_BACKEND_PRESETS, load_active_agent_config

    if backend:
        preset = AGENT_BACKEND_PRESETS.get(backend.strip().lower())
        resolved = str((preset or {}).get("backend") or backend).strip().lower()
        model = str((preset or {}).get("model") or "")
        selected = backend.strip().lower()
    else:
        cfg = load_active_agent_config()
        resolved = str(cfg.get("backend") or "")
        model = str(cfg.get("model") or "")
        selected = str(cfg.get("selected_preset") or resolved)

    checks = _backend_checks(resolved)
    if probe and not any(c["status"] == FAIL for c in checks):
        live = _probe_check(resolved)
        if live is not None:
            checks.append(live)
    return {
        "selected_preset": selected,
        "backend": resolved,
        "model": model,
        "checks": checks,
        "failed": [c["name"] for c in checks if c["status"] == FAIL],
        "warned": [c["name"] for c in checks if c["status"] == WARN],
    }


def render(status: dict[str, Any]) -> str:
    glyph = {OK: "[ok]", WARN: "[~]", FAIL: "[!]"}
    lines = ["NEPSE Quant Terminal - agent backend preflight", "=" * 56]
    lines.append(f"backend : {status['backend']}  (preset: {status['selected_preset']})")
    lines.append(f"model   : {status['model'] or '-'}")
    lines.append("-" * 56)
    for c in status.get("checks", []):
        lines.append(f"{glyph.get(c['status'], '[?]'):5} {c['name']:<14} {c['detail']}")
    lines.append("-" * 56)
    if status.get("failed"):
        lines.append(f"NOT READY - fix: {', '.join(status['failed'])}")
    elif status.get("warned"):
        lines.append(f"READY (with warnings: {', '.join(status['warned'])})")
    else:
        lines.append("READY - the active agent backend looks good to go")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-doctor",
        description="Read-only readiness check for the active AI agent backend.",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument(
        "--backend",
        default=None,
        help="check a specific preset (ollama|gemma4_mlx|claude|claude_sdk) instead of the active one",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="also make ONE live call to confirm the backend answers (Claude backends; uses your quota)",
    )
    args = parser.parse_args(argv)

    status = build_agent_status(args.backend, probe=args.probe)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(render(status))
    return 1 if status["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
