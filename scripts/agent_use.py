#!/usr/bin/env python3
"""agent-use - switch the active AI agent backend from the command line.

The TUI Agents tab switches backends with `/agent <id>` and `/model <name>`;
this is the same action for scripts, CI, or a quick terminal flip. It wraps
`backend.agents.runtime_config.set_active_agent`, which persists the choice to
the active-agent config the TUI and analyses read.

Usage:
    python -m scripts.agent_use claude_sdk              # switch backend/preset
    python -m scripts.agent_use claude_sdk --model opus # switch + set model
    python -m scripts.agent_use --list                  # list selectable presets
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _valid_presets() -> list[str]:
    from backend.agents.runtime_config import AGENT_BACKEND_PRESETS

    return sorted(AGENT_BACKEND_PRESETS)


def apply(preset: str, model: str | None = None) -> dict[str, Any]:
    """Persist the active backend; returns the saved config. Raises ValueError
    (from set_active_agent) if the preset/backend is unknown."""
    from backend.agents.runtime_config import set_active_agent

    return set_active_agent(preset, model=model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-use",
        description="Switch the active AI agent backend (persists to the active-agent config).",
    )
    parser.add_argument("preset", nargs="?", help="preset/backend id (see --list)")
    parser.add_argument("--model", default=None, help="model alias or id to set (e.g. sonnet, opus)")
    parser.add_argument("--list", action="store_true", help="list selectable presets and exit")
    parser.add_argument("--json", action="store_true", help="emit the resulting config as JSON")
    args = parser.parse_args(argv)

    if args.list or not args.preset:
        presets = _valid_presets()
        if args.json:
            print(json.dumps(presets, indent=2))
        else:
            print("selectable presets: " + ", ".join(presets))
            if not args.preset and not args.list:
                print("usage: python -m scripts.agent_use <preset> [--model <name>]")
        return 0 if args.list else 2

    try:
        cfg = apply(args.preset, model=args.model)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print("selectable presets: " + ", ".join(_valid_presets()))
        return 2

    if args.json:
        print(json.dumps(cfg, indent=2, sort_keys=True, default=str))
    else:
        backend = cfg.get("backend") or args.preset
        model = cfg.get("model") or "-"
        print(f"active agent backend -> {cfg.get('selected_preset') or backend}  (model: {model})")
        print("verify readiness:  python -m scripts.agent_doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
