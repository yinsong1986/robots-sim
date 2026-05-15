# Examples

Per-backend tutorials for running LIBERO benchmarks against each
`SimEngine` shipped by `strands-robots-sim` and its upstream sibling
`strands-robots`.

## Two execution patterns

Each backend ships **two sibling files** that mirror the two driver
patterns the deleted `SimEnv` API used to cover.

| File suffix | Driver | Replaces | Best for |
|---|---|---|---|
| `<backend>.py` | **Programmatic** — Python script calls `sim.evaluate_benchmark(...)` directly | `SimEnv` | CI / benchmark numbers / R15 matrix table |
| `<backend>_agent.py` | **Strands `Agent` + natural language** — single `agent("Run the LIBERO benchmark …")` call drives setup + eval + cleanup | the natural-language entry point in the deleted `libero_example.py` | Demoing why a Strands integration buys you anything beyond direct calls |

The programmatic files print two grep-stable lines (`benchmark_name=...`
and `policy=... task=... success_rate=... wall_time=...s`) that R15
([`libero_backend_matrix.py`](https://github.com/strands-labs/robots-sim/issues/22))
subprocess-and-parses for the side-by-side comparison table. The agent
files are for human inspection — output is non-deterministic
LLM-generated prose, not matrix-ingested.

> **Iterative supervision** (the deleted `SteppedSimEnv` pattern) is
> deliberately *not* in this directory. With
> `nvidia/GR00T-N1.7-LIBERO/libero_<suite>/` finetuned end-to-end on
> its training distribution the policy executes the canonical task
> without stalling, so System-2 supervision over an in-distribution
> finetuned policy has nothing to actually decide. The OOD-anchored
> iterative demo (cross-suite mismatch / LIBERO-PRO perturbations /
> distractor injection) is filed as
> [R24 / #29](https://github.com/strands-labs/robots-sim/issues/29);
> the canonical pattern doc lives upstream at
> [`strands-labs/robots#136`](https://github.com/strands-labs/robots/issues/136) (U6).

## Two policy choices

Both files in every pair accept `--policy {mock,groot}` and
`--task <benchmark_name>`:

| Flag | Provider | When | Reproducibility |
|---|---|---|---|
| `--policy mock` (default) | random-action stub in `strands_robots.policies.mock` | smoke tests / CI / no-GPU dev boxes / "did the plumbing work" sanity check | deterministic given `--seed` |
| `--policy groot` | NVIDIA GR00T VLA, served via `nvcr.io/nvidia/isaac-gr00t` Docker (or `gr00t_inference` Strands tool) on `--port 8000` against the suite-specific subfolder of `nvidia/GR00T-N1.7-LIBERO` | real LIBERO success-rate measurements | depends on the GR00T checkpoint + service config |

`nvidia/GR00T-N1.7-LIBERO` on HuggingFace is **a tree of four
sub-checkpoints** — `libero_spatial/`, `libero_10/`, `libero_object/`,
`libero_goal/` — each finetuned end-to-end on the matching LIBERO
suite. The `--task <libero-<suite>-<task_stem>>` flag auto-derives
which subfolder to use; service-start commands (Strands tool *and*
bare-Docker fallback) live in `libero_mujoco.py`'s docstring;
`libero_mujoco_agent.py` makes the agent itself run them based on
`--task`'s suite.

The mock invocation's wall-time is a smoke-test reference only; **the
canonical mujoco baseline number for the matrix table is the
`--policy=groot` measurement of `libero_mujoco.py`** against the
`libero_spatial/` sub-checkpoint.

## Backend matrix

Same task — `libero-spatial-pick_up_the_red_cube`, 10 episodes, seed
42 — on every available backend with success rate and wall-time
side-by-side. Numbers come from the *programmatic* file with
`--policy=groot` against the matching `libero_<suite>/` sub-checkpoint
unless a row says otherwise; mock-policy smoke runs are listed below
the table for reference.

| Example | Backend | `n_envs` | Wall-time @ success-rate | Notes |
|---|---|---:|---|---|
| [`libero_mujoco.py`](libero_mujoco.py) | MuJoCo (in `strands-robots`) | 1 | TBD @ TBD (groot) — see smoke note below | macOS / CPU OK |
| `libero_isaac.py` | Isaac Sim | 1 | _TBD ([R8 / #15](https://github.com/strands-labs/robots-sim/issues/15))_ | RTX path-traced |
| `libero_isaac_fleet.py` | Isaac Sim | 4096 | _TBD ([R23 / #27](https://github.com/strands-labs/robots-sim/issues/27))_ | IsaacLab-style fleet RL |
| `libero_newton.py` | Newton / Warp | 1 | _TBD ([R12 / #19](https://github.com/strands-labs/robots-sim/issues/19))_ | CUDA only |
| `libero_newton_fleet.py` | Newton / Warp | 4096 | _TBD ([R12 / #19](https://github.com/strands-labs/robots-sim/issues/19))_ | fleet |

**Mock-policy smoke wall-time (reference only, not matrix-authoritative):**

- `libero_mujoco.py --policy mock --n-episodes 10 --seed 42` → ~0.8 s on
  a single-CPU dev box (success rate 0.0 — mock can't satisfy goals).

The `--policy=groot` MuJoCo number drops in once the upstream BDDL fix
([`strands-labs/robots#147`](https://github.com/strands-labs/robots/pull/147))
lands on PyPI and a contributor with a GPU + Docker measures it
against `libero_spatial/`.

The flagship driver
[`libero_backend_matrix.py`](https://github.com/strands-labs/robots-sim/issues/22)
(R15) walks all five rows and prints a unified table.

## Running the MuJoCo baseline

```bash
pip install 'strands-robots[sim-mujoco,benchmark-libero]'

# 1) Programmatic, deterministic, no LLM. R15 ingests this output.
python examples/libero_mujoco.py --policy mock --n-episodes 5

# 2) Strands-Agent + natural language. Requires a configured LLM
#    provider (Bedrock by default — see https://strandsagents.com/).
pip install strands-agents
python examples/libero_mujoco_agent.py --policy mock

# 3) Real eval against `libero_spatial/`. Programmatic file's docstring
#    has the three-step sequence (download subfolder → start service →
#    run); the agent file lets the agent run those steps itself based
#    on --task's suite.
python examples/libero_mujoco.py --policy groot --port 8000 --n-episodes 50
python examples/libero_mujoco_agent.py --policy groot --port 8000

# 4) Different LIBERO task; suite auto-derived from --task:
python examples/libero_mujoco.py \
    --task libero-spatial-pick_up_the_milk_and_place_it_in_the_basket
```

Each invocation produces an MP4 under `rollouts/YYYY_MM_DD/`. Filename
encodes `--task=<benchmark_name>`, `--policy=mock|groot`, `--seed=S`,
and either `--n_eps=N` (programmatic) or `--agent` marker (agent file)
so post-hoc analysis can tell what produced each file. The `rollouts/`
layout and timestamped name pattern are preserved from the deleted
`SimEnv` so existing scrapers keep working.

> **Note:** the `[sim-mujoco]` and `[benchmark-libero]` extras are
> currently on `strands-robots` `main` only and will land in the next
> PyPI release (`>= 0.4.0`). Until then, install from git:
> `pip install 'strands-robots[sim-mujoco,benchmark-libero] @ git+https://github.com/strands-labs/robots.git@main'`.

> **Note:** `load_libero_suite(...)` requires upstream
> [`strands-labs/robots#147`](https://github.com/strands-labs/robots/pull/147)
> (case-insensitive BDDL parsing) to register tasks from real LIBERO
> BDDL files. Without it every task is skipped and both example files
> raise on suite registration.

## Migration from the legacy 0.1.x API

`strands-robots-sim` 0.1.x shipped `SimEnv` / `SteppedSimEnv` plus a
natural-language `libero_example.py`. Those code paths moved upstream
in 0.2.0 — see [`MIGRATION.md`](MIGRATION.md) for the explicit
`SimEnv → libero_mujoco.py`, `agent("Run the task ...") →
libero_mujoco_agent.py`, and `SteppedSimEnv → R24 / #29 + upstream U6`
mapping.
