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
bare-Docker fallback) live in `libero/run_mujoco.py`'s docstring;
`libero/run_mujoco_agent.py` makes the agent itself run them based on
`--task`'s suite.

The mock invocation's wall-time is a smoke-test reference only; **the
canonical mujoco baseline number for the matrix table is the
`--policy=groot` measurement of `libero/run_mujoco.py`** against the
`libero_spatial/` sub-checkpoint.

## Backend matrix

Same task — `libero-spatial-pick_up_the_red_cube`, 10 episodes, seed
42 — on every available backend with success rate and wall-time
side-by-side. Numbers come from the *programmatic* file with
`--policy=groot` against the matching `libero_<suite>/` sub-checkpoint
unless a row says otherwise; mock-policy smoke runs are listed below
the table for reference.

| Example | Backend | `n_envs` | Wall-time @ success-rate | Notes |
|---|---|---|---|---|
| [`libero/run_mujoco.py`](libero/run_mujoco.py) | MuJoCo (in `strands-robots`) | 1 | ~9 s/ep @ 1.00 (groot, ZMQ client)[^1] | Reliably reaches 5/5 against post-[#188](https://github.com/strands-labs/robots/pull/188) `strands-robots`; macOS / CPU OK |
| `libero_isaac.py` | Isaac Sim | 1 | _TBD ([R8 / #15](https://github.com/strands-labs/robots-sim/issues/15))_ | RTX path-traced |
| `libero_isaac_fleet.py` | Isaac Sim | 4096 | _TBD ([R23 / #27](https://github.com/strands-labs/robots-sim/issues/27))_ | IsaacLab-style fleet RL |
| `libero_newton.py` | Newton / Warp | 1 | _TBD ([R12 / #19](https://github.com/strands-labs/robots-sim/issues/19))_ | CUDA only |
| `libero_newton_fleet.py` | Newton / Warp | 4096 | _TBD ([R12 / #19](https://github.com/strands-labs/robots-sim/issues/19))_ | fleet |

**Mock-policy smoke wall-time (reference only, not matrix-authoritative):**

- `libero/run_mujoco.py --policy mock --n-episodes 10 --seed 42` → ~3 s/ep on a
  single-CPU dev box with the LIBERO scene loaded (success rate 0.0 —
  mock can't satisfy goals). The pre-scene-loading version of this
  example was ~0.8 s/ep against a bare Panda; the ~2 s/ep delta is
  per-episode scene-gen + load (cached after first call) + scene-step
  cost.

The `--policy=groot` number above (~9 s/ep on an L4 with
`success_rate=1.00`) is measured against
`nvidia/GR00T-N1.7-LIBERO/libero_10` on `libero-10/SCENE5`, 5 episodes,
seeds 42 and 100 → 44.3-44.8 s wall-time end-to-end (engine + scene +
policy + ZMQ I/O). Acceptance criterion `success_rate > 0` is met.

For users who want server-side determinism (per-episode CUDA reseed
matching client-side `policy.reset(seed=...)`), an optional drop-in
docker wrapper is available at
[`libero/gr00t_server_deterministic_wrapper.py`](libero/gr00t_server_deterministic_wrapper.py).
The example file works WITHOUT this wrapper; it's only needed when
bit-exact run-to-run reproducibility matters.

The flagship driver `libero_backend_matrix.py` (R15) walks all five
rows and prints a unified table.

[^1]: Single-sample on the L4 reference dev box (`libero-10/SCENE5`,
    seed=42, n=5). Pre-#188 success rate was 0.20–0.60 across re-runs;
    post-#188 it stabilises at 1.00. See PR #26's history threads on
    [#187](https://github.com/strands-labs/robots/issues/187) /
    [#188](https://github.com/strands-labs/robots/pull/188) for the
    variance bisection.

## Running the MuJoCo baseline

```bash
pip install 'strands-robots[sim-mujoco,benchmark-libero]'

# 1) Programmatic, deterministic, no LLM. R15 ingests this output.
python examples/libero/run_mujoco.py --policy mock --n-episodes 5

# 2) Strands-Agent + natural language. Requires a configured LLM
#    provider (Bedrock by default — see https://strandsagents.com/).
pip install strands-agents
python examples/libero/run_mujoco_agent.py --policy mock

# 3) Real eval against `libero_spatial/`. Programmatic file's docstring
#    has the three-step sequence (download subfolder → start service →
#    run); the agent file lets the agent run those steps itself based
#    on --task's suite.
python examples/libero/run_mujoco.py --policy groot --port 8000 --n-episodes 50
python examples/libero/run_mujoco_agent.py --policy groot --port 8000

# 4) Different LIBERO task; suite auto-derived from --task:
python examples/libero/run_mujoco.py \
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
`SimEnv → libero/run_mujoco.py`, `agent("Run the task ...") →
libero/run_mujoco_agent.py`, and `SteppedSimEnv → R24 / #29 + upstream U6`
mapping.
