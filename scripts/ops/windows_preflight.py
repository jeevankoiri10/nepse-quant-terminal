"""Native Windows/PowerShell readiness check for NEPSE Quant Terminal.

Run:
    python -m scripts.ops.windows_preflight
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sqlite3
import sys
from pathlib import Path

from backend.quant_pro.database import get_db_path
from backend.quant_pro.paths import ensure_dir, get_runtime_dir


def _status(ok: bool, label: str, detail: str) -> tuple[bool, str]:
    mark = "OK" if ok else "FAIL"
    return ok, f"[{mark}] {label}: {detail}"


def _check_python() -> tuple[bool, str]:
    version = sys.version_info
    ok = (3, 10) <= (version.major, version.minor) <= (3, 13)
    return _status(ok, "Python", platform.python_version())


def _check_git() -> tuple[bool, str]:
    path = shutil.which("git")
    return _status(bool(path), "Git", path or "required for the pinned nepse dependency")


def _check_database() -> tuple[bool, str]:
    db_path = Path(get_db_path())
    if not db_path.exists():
        return _status(False, "Database", f"{db_path} not found; run python setup_data.py")
    try:
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
    except Exception as exc:
        return _status(False, "Database", f"{db_path} exists but cannot be opened: {exc}")
    return _status(bool(tables), "Database", f"{db_path} ({len(tables)} tables)")


def _check_data_freshness(max_stale_days: int = 7) -> tuple[bool, str]:
    """Beyond 'the DB has tables': is stock_prices actually populated, and how
    stale is it? An empty-but-existing DB passes the Database check yet can't
    drive signals - flag it. Reports the latest bar date and its age."""
    db_path = Path(get_db_path())
    if not db_path.exists():
        return _status(True, "Data freshness", "n/a (no database yet)")
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT MAX(date), COUNT(DISTINCT symbol) FROM stock_prices"
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return _status(True, "Data freshness", f"could not query stock_prices: {exc}")

    latest, symbols = (row or (None, 0))
    if not latest:
        return _status(False, "Data freshness", "stock_prices is empty; run python setup_data.py")
    try:
        from datetime import date, datetime

        latest_d = datetime.strptime(str(latest)[:10], "%Y-%m-%d").date()
        age = (date.today() - latest_d).days
    except ValueError:
        return _status(True, "Data freshness", f"latest={latest} ({symbols} symbols)")

    detail = f"latest {latest_d.isoformat()} ({age}d ago), {symbols} symbols"
    if age > max_stale_days:
        detail += f" [STALE > {max_stale_days}d]"
    return _status(True, "Data freshness", detail)


def _check_active_agent() -> tuple[bool, str]:
    """Is the *selected* agent backend ready to run? Reuses agent_doctor so the
    launch preflight surfaces a misconfigured agent (missing dep / CLI / server)
    up front. Informational: the dashboard launches without a working agent, so
    a not-ready backend is reported, not a hard fail. `agent doctor` has detail."""
    try:
        from scripts.agent_doctor import build_agent_status

        status = build_agent_status()
    except Exception as exc:  # noqa: BLE001 - never let this check break the preflight
        return _status(True, "Active agent", f"could not resolve: {exc}")

    backend = status.get("backend") or "?"
    failed = status.get("failed") or []
    if failed:
        return _status(
            True, "Active agent",
            f"{backend}: NOT ready ({', '.join(failed)}) - run python -m scripts.agent doctor",
        )
    warned = status.get("warned") or []
    note = f" (warnings: {', '.join(warned)})" if warned else ""
    return _status(True, "Active agent", f"{backend}: ready{note}")


def _check_runtime_writable() -> tuple[bool, str]:
    runtime = ensure_dir(get_runtime_dir(__file__))
    probe = runtime / ".windows_preflight_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as exc:
        return _status(False, "Runtime path", f"{runtime} is not writable: {exc}")
    return _status(True, "Runtime path", str(runtime))


def _check_timezone() -> tuple[bool, str]:
    tz = os.environ.get("TZ", "")
    system = platform.system()
    detail = f"system={system}"
    if tz:
        detail += f", TZ={tz}"
    return _status(True, "Timezone", detail)


def _check_optional_agents() -> tuple[bool, str]:
    """Report which optional agent backends are available, in sync with the
    selectable presets (ollama / gemma-mlx / claude CLI / claude_sdk). Always
    informational - none is required to launch the dashboard."""

    def _yes_no(present: object) -> str:
        return "yes" if present else "no"

    bits = [
        f"ollama={_yes_no(shutil.which('ollama'))}",
        f"mlx_vlm={_yes_no(importlib.util.find_spec('mlx_vlm'))}",
        f"claude_cli={_yes_no(shutil.which('claude'))}",
        f"claude_sdk={_yes_no(importlib.util.find_spec('claude_agent_sdk'))}",
    ]
    return _status(True, "Optional agents", ", ".join(bits))


def main() -> int:
    checks = [
        _check_python(),
        _check_git(),
        _check_database(),
        _check_data_freshness(),
        _check_runtime_writable(),
        _check_timezone(),
        _check_optional_agents(),
        _check_active_agent(),
    ]
    for _, line in checks:
        print(line)
    failed = [line for ok, line in checks if not ok]
    if failed:
        print("\nPreflight failed. Fix the FAIL items above before launching the dashboard.")
        return 1
    print("\nPreflight passed. Launch with: python -m apps.tui.dashboard_tui")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
