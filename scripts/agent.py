#!/usr/bin/env python3
"""agent - one entry point for the AI agent operator CLIs.

Instead of remembering each module path, run everything through one command:

    python -m scripts.agent status   [--json]
    python -m scripts.agent doctor   [--json] [--backend <preset>]
    python -m scripts.agent use <preset> [--model <name>]

  status  what backend/model is active, and what presets you can switch to
  doctor  whether the active backend's prerequisites are actually in place
  use     switch the active backend (persists the choice)

Subcommands delegate to each tool's main() verbatim (same flags, same exit
codes). Read-only: the dispatcher adds no behaviour of its own.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

# name -> (one-line help, lazy loader returning the subcommand's main(argv)->int)
_SUBCOMMANDS: dict[str, tuple[str, Callable[[], Callable[[list[str]], int]]]] = {
    "status": (
        "active backend/model + selectable presets",
        lambda: __import__("scripts.agent_status", fromlist=["main"]).main,
    ),
    "doctor": (
        "readiness check for the active backend",
        lambda: __import__("scripts.agent_doctor", fromlist=["main"]).main,
    ),
    "use": (
        "switch the active backend (persists the choice)",
        lambda: __import__("scripts.agent_use", fromlist=["main"]).main,
    ),
}


def _usage() -> str:
    lines = [
        "agent - NEPSE Quant Terminal AI agent CLIs",
        "",
        "usage: python -m scripts.agent <command> [options]",
        "",
        "commands:",
    ]
    for name, (help_text, _) in _SUBCOMMANDS.items():
        lines.append(f"  {name:<8} {help_text}")
    lines.append("")
    lines.append("run 'python -m scripts.agent <command> --help' for command options")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(_usage(), file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd in ("-h", "--help", "help"):
        print(_usage())
        return 0
    entry = _SUBCOMMANDS.get(cmd)
    if entry is None:
        print(f"agent: unknown command '{cmd}'\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    sub_main = entry[1]()
    return int(sub_main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
