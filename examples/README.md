# Examples

Per-backend tutorials for running LIBERO benchmarks against each
`SimEngine` shipped by `strands-robots-sim` and its upstream sibling
`strands-robots`.

## Two flavours

Each backend ships **two** files. Pick the one that matches your situation:

| File suffix | Driver | LLM required | Output | Used by R15 matrix |
|---|---|---|---|---|
| `*.py` | direct Python API on `Simulation` | no | `success_rate=… wall_time=…s` | yes |
| `*_agent.py` | `Agent(tools=[sim])` + natural-language prompt | yes (Bedrock by default) | LLM-generated summary | no |

The direct-API files are the **deterministic, CI-runnable smoke tests**
that R15 ([`libero_backend_matrix.py`](https://github.com/strands-labs/robots-sim/issues/22))
ingests for the side-by-side comparison table — output lines are
intentionally grep-stable.

The agent files are the **headline pedagogical demos** showing why the
Strands integration buys you anything beyond direct method calls. Both
files in a pair use the same upstream `Simulation` AgentTool and the same
LIBERO adapter — only the *driver* changes.

## Backend matrix

Same task, three backends. Numbers fill in as each row's example lands.

| Example | Backend | `n_envs` | Wall-time (`libero-spatial`, 10 episodes, mock policy) | Notes |
|---|---|---:|---|---|
| [`libero_mujoco.py`](libero_mujoco.py) | MuJoCo (in `strands-robots`) | 1 | ~0.8 s* | macOS / CPU OK |
| `libero_isaac.py` | Isaac Sim | 1 | _TBD ([R8 / #15](https://github.com/strands-labs/robots-sim/issues/15))_ | RTX path-traced |
| `libero_newton.py` | Newton / Warp | 1 | _TBD ([R12 / #19](https://github.com/strands-labs/robots-sim/issues/19))_ | CUDA only |
| `libero_newton_fleet.py` | Newton / Warp | 4096 | _TBD ([R12 / #19](https://github.com/strands-labs/robots-sim/issues/19))_ | fleet |

\* Measured on a single-CPU dev machine with `policy_provider="mock"`; with a
real policy the wall-time is dominated by inference, not physics. Re-measure
on reference hardware once the matrix is stable. The exact task picked is
the first registered LIBERO spatial task — currently
`libero-spatial-pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`.

The flagship that runs all four side-by-side and prints the comparison
table from a single command lives in
[R15 / #22](https://github.com/strands-labs/robots-sim/issues/22) (`libero_backend_matrix.py`).

## Running the MuJoCo baseline

```bash
pip install 'strands-robots[sim-mujoco,benchmark-libero]'

# 1. Direct-API, deterministic, no LLM. R15 ingests this output.
python examples/libero_mujoco.py
#   benchmark_name=libero-spatial-<task>
#   success_rate=0.00  wall_time=0.8s

# 2. Strands-Agent + natural language. Requires an LLM provider
#    (Bedrock by default — see https://strandsagents.com/ for setup).
pip install strands-agents
python examples/libero_mujoco_agent.py
#   <LLM-generated summary of the benchmark run>
```

> **Note:** the `[sim-mujoco]` and `[benchmark-libero]` extras are
> currently on `strands-robots` `main` only and will land in the next
> PyPI release (`>= 0.4.0`). Until then, install from git:
> `pip install 'strands-robots[sim-mujoco,benchmark-libero] @ git+https://github.com/strands-labs/robots.git@main'`.

> **Note:** `load_libero_suite(...)` requires upstream
> [strands-labs/robots#147](https://github.com/strands-labs/robots/pull/147)
> (case-insensitive BDDL parsing) to register tasks from real LIBERO
> BDDL files. Without it every task is skipped.

`policy_provider="mock"` is the default in both files — the point is to
exercise the engine + benchmark adapter, not to bench a real policy.
Each file documents how to swap in `groot` / `lerobot` / a custom
`Policy` instance.

## Migration from the legacy 0.1.x API

`strands-robots-sim` 0.1.x shipped `SimEnv` / `SteppedSimEnv` with a
LIBERO env layer baked in. Those code paths moved upstream in 0.2.0 — see
[`MIGRATION.md`](MIGRATION.md) for the old → new mapping.
