# Agent operator CLIs

Read-only-by-default command-line tools for the AI agent layer — inspect, check,
and switch the backend the TUI and analyses use, without opening the TUI.

One entry point ties them together:

```bash
python -m scripts.agent <command> [options]
```

| Command | What it does | Side effects |
|---------|--------------|--------------|
| `status` | Active backend/model + every selectable preset (active one marked) | none (read-only) |
| `doctor` | Whether the active backend's prerequisites are present | none (read-only) |
| `doctor --probe` | …plus ONE live call to confirm the backend actually answers | one model call |
| `use <preset>` | Switch the active backend (persists the choice) | writes active-agent config |

`make` shortcuts wrap the same commands:

```bash
make agent-status
make agent-doctor                       # ARGS='--probe' for the live check
make agent-use PRESET=claude_sdk MODEL=opus
```

## status — what's configured

```bash
python -m scripts.agent status          # human report
python -m scripts.agent status --json
```

Shows the active backend, model, and fallback, then lists all presets
(`ollama`, `gemma4_mlx`, `claude`, `claude_sdk`, …) with the active one marked
`*`.

## doctor — will it work?

```bash
python -m scripts.agent doctor                  # check the active backend
python -m scripts.agent doctor --backend claude_sdk   # check a specific preset
python -m scripts.agent doctor --json
python -m scripts.agent doctor --probe          # also make one live call
```

Per-backend readiness, each `[ok]` / `[~]` (warn) / `[!]` (fail). Exit code is
non-zero if anything FAILs, so it drops into a launcher or CI gate.

| Backend | Checks |
|---------|--------|
| `ollama` | `ollama` on PATH (warn) + host port reachable |
| `gemma4_mlx` | `mlx_lm` importable (warn; Apple-Silicon only) |
| `claude` | `claude` CLI on PATH |
| `claude_sdk` | `claude_agent_sdk` + `anyio` importable **and** `claude` CLI on PATH |

`--probe` adds the one thing static checks can't: a single trivial live call to
confirm the backend truly responds (login valid, reachable). Defined for the
Claude backends; it reuses the real call path, so a failure shows the exact fix
(e.g. *run `claude login`*). Skipped if a static check already failed.

## use — switch the backend

```bash
python -m scripts.agent use claude_sdk                # switch backend/preset
python -m scripts.agent use claude_sdk --model opus   # switch + set model
python -m scripts.agent_use --list                    # enumerate presets
```

The scriptable/CI equivalent of the TUI Agents tab's `/agent <id>` and
`/model <name>`. Validates the preset (unknown → non-zero exit + the valid
list) and persists the choice to the active-agent config.

## Typical first-run flow for the Claude Agent SDK backend

```bash
npm install -g @anthropic-ai/claude-code     # the SDK drives this CLI
pip install claude-agent-sdk anyio
claude login

python -m scripts.agent doctor --backend claude_sdk   # expect all [ok]
python -m scripts.agent use claude_sdk                # make it active
python -m scripts.agent doctor --probe                # live: "backend answered a test prompt"
```
