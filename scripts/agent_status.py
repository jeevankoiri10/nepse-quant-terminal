#!/usr/bin/env python3
"""agent-status - what AI agent backend am I running, and what can I switch to?

Read-only one-glance view of the agent layer: the active backend + model (the
one the TUI and analyses use), plus every selectable preset. It answers "what's
configured" - the companion to `agent_doctor`, which answers "will it work".

Never writes config or makes a model call.

Usage:
    python -m scripts.agent_status            # human report
    python -m scripts.agent_status --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def build_overview() -> dict[str, Any]:
    """Active agent config + the list of selectable presets."""
    from backend.agents.runtime_config import list_agent_backends, load_active_agent_config

    cfg = load_active_agent_config()
    active = {
        "selected_preset": str(cfg.get("selected_preset") or cfg.get("backend") or ""),
        "backend": str(cfg.get("backend") or ""),
        "model": str(cfg.get("model") or ""),
        "label": str(cfg.get("label") or ""),
        "fallback_backend": str(cfg.get("fallback_backend") or ""),
    }
    presets = list_agent_backends()
    return {"active": active, "presets": presets}


def render(report: dict[str, Any]) -> str:
    active = report.get("active", {})
    lines = ["NEPSE Quant Terminal - agent backend", "=" * 60]
    lines.append(f"active   : {active.get('selected_preset') or active.get('backend') or '-'}"
                 f"  ({active.get('label') or '-'})")
    lines.append(f"model    : {active.get('model') or '-'}")
    lines.append(f"fallback : {active.get('fallback_backend') or '(none)'}")
    lines.append("-" * 60)
    lines.append("available presets:")
    active_id = active.get("selected_preset") or active.get("backend")
    for p in report.get("presets", []):
        marker = "*" if p.get("id") == active_id else " "
        label = str(p.get("label") or p.get("id") or "")
        desc = str(p.get("description") or "")
        lines.append(f"  {marker} {str(p.get('id') or ''):<20} {label:<24} {desc}")
    lines.append("-" * 60)
    lines.append("switch in the TUI Agents tab: `/agent <id>` then `/model <name>`")
    lines.append("check readiness first:        python -m scripts.agent_doctor")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-status",
        description="Read-only view of the active AI agent backend and available presets.",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = parser.parse_args(argv)

    report = build_overview()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
