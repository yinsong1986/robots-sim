# Examples

Per-backend tutorials for running LIBERO benchmarks against each
`SimEngine` shipped by `strands-robots-sim` and its upstream sibling
`strands-robots`.

## Two execution patterns

Each backend ships **two sibling files** that mirror the two driver
patterns the deleted `SimEnv` API used to cover.

| File suffix | Driver | Replaces | Best for |
|---|---|---|---|
| `<benchmark>/run_<backend>.py` | **Programmatic** — Python script calls `sim.evaluate_benchmark(...)` directly | `SimEnv` | CI / benchmark numbers / R15 matrix table |
| `<benchmark>/run_<backend>_agent.py` | **Strands `Agent` + natural language** — script owns the deterministic plumbing (GR00T container lifecycle, scene pre-warm, MP4 recording); a single `agent("…")` call invokes `evaluate_benchmark` from natural-language kwargs and produces a prose summary | the natural-language entry point in the deleted `libero_example.py` | Demoing why a Strands integration buys you anything beyond direct calls |

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
which subfolder to use; both example files orchestrate the GR00T
container lifecycle (build → checkpoint download → start → wait-for-
ready → teardown) deterministically from the script via
`gr00t_inference(action='lifecycle', ...)`. The agent file's agent
isn't asked to manage Docker / HF cache — those are brittle for an
LLM and stay under Python control. Pass `--no-auto-server` to either
file to reuse an already-running container instead.

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
| [`libero/run_isaac.py`](libero/run_isaac.py) | Isaac Sim | 1 | _TBD ([R8 / #15](https://github.com/strands-labs/robots-sim/issues/15))_ | RTX path-traced; loads a real Franka USD via `add_robot(usd_path=...)`. Number pending the nightly GPU runner ([#17](https://github.com/strands-labs/robots-sim/issues/17) / [#59](https://github.com/strands-labs/robots-sim/pull/59)) |
| `libero/run_isaac_fleet.py` | Isaac Sim | 4096 | _TBD ([R23 / #27](https://github.com/strands-labs/robots-sim/issues/27))_ | IsaacLab-style fleet RL |
| `libero/run_newton.py` | Newton / Warp | 1 | _TBD ([R12 / #19](https://github.com/strands-labs/robots-sim/issues/19))_ | CUDA only |
| `libero/run_newton_fleet.py` | Newton / Warp | 4096 | _TBD ([R12 / #19](https://github.com/strands-labs/robots-sim/issues/19))_ | fleet |

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

# 3) Real eval against the matching `libero_<suite>/` sub-checkpoint.
#    Both files auto-orchestrate the GR00T container lifecycle from
#    the script (build → checkpoint download → start → teardown);
#    pass `--no-auto-server` to reuse an existing one.
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
> currently on `strands-robots` `main` only and will land in a future
> PyPI release. Until then, install from git:
> `pip install 'strands-robots[sim-mujoco,benchmark-libero] @ git+https://github.com/strands-labs/robots.git@main'`.

## Running the Isaac Sim backend

[`libero/run_isaac.py`](libero/run_isaac.py) and
[`libero/run_isaac_agent.py`](libero/run_isaac_agent.py) are the Isaac
Sim siblings of the MuJoCo files — same CLI shape, same two grep-stable
output lines, same `evaluate_benchmark(...)` / agent drivers. They
differ in three Isaac-specific ways:

- **Real-asset robot.** Instead of a LIBERO MJCF, the script loads a
  *real* robot via `add_robot(usd_path=...)` — by default Isaac Sim's
  bundled Franka Panda USD, resolved from the assets root over the
  public Omniverse CDN (no local Nucleus needed). Override with
  `--robot-usd PATH` or `--robot-urdf PATH`. (A real asset is required
  because the procedural builder is a kinematically-approximate
  stick-figure unusable by a LIBERO manipulation policy.)
- **Explicit camera.** Isaac doesn't auto-attach a viewport camera the
  way MuJoCo's `mjData` does, so the script makes an explicit
  `add_camera(...)` call at the LIBERO `agentview` vantage before the
  eval.
- **Isaac-specific container name** (`gr00t-libero-isaac`) so Isaac and
  MuJoCo `--policy=groot` runs don't clobber each other's containers on
  the same host.

```bash
pip install 'strands-robots-sim[isaac]' \
    'strands-robots[benchmark-libero] @ git+https://github.com/strands-labs/robots.git@main'

# 1) Programmatic smoke test (mock policy). Loads the default Franka USD.
python examples/libero/run_isaac.py --policy mock --n-episodes 5

# 1b) Bring your own robot asset:
python examples/libero/run_isaac.py --policy mock --robot-usd /path/to/robot.usd
python examples/libero/run_isaac.py --policy mock --robot-urdf /path/to/robot.urdf

# 2) Strands-Agent + natural language (needs `strands-agents` + an LLM
#    provider — Bedrock by default).
pip install strands-agents
python examples/libero/run_isaac_agent.py --policy mock

# 3) Real eval against the matching `libero_<suite>/` sub-checkpoint
#    (auto-orchestrates the GR00T container; `--no-auto-server` reuses one):
python examples/libero/run_isaac.py --policy groot --port 8000 --n-episodes 50
```

> **Requires Isaac Sim** installed separately on the host (it is **not**
> pip-installable — Omniverse Launcher / Isaac Lab / NGC Docker image,
> RTX GPU, CUDA 12+). On a non-Isaac host both scripts exit early with a
> diagnostic from `IsaacSimulation.is_available()` rather than crashing
> on the first `omni.*` import.

> **Status (landing):** CLI / control-flow / lint validation and the
> `is_available()` short-circuit are verified on a CPU-only dev box; the
> LIBERO suite + helper resolution paths are unit-checked against
> `strands-robots`. The full RTX eval (the matrix wall-time @
> success-rate number above) is **not yet run end-to-end** — it is
> gated on the nightly GPU runner
> ([#17](https://github.com/strands-labs/robots-sim/issues/17) /
> [#59](https://github.com/strands-labs/robots-sim/pull/59)) and the
> Phase-2 data-plane slices it rides on
> ([#61](https://github.com/strands-labs/robots-sim/pull/61) add_camera,
> [#62](https://github.com/strands-labs/robots-sim/pull/62) render
> frame-path, [#63](https://github.com/strands-labs/robots-sim/pull/63)
> USD-load, [#64](https://github.com/strands-labs/robots-sim/pull/64)
> URDF-load). `run_isaac_agent.py` is **draft scaffolding**: it runs
> end-to-end today but reports `success_rate=0.0` until the
> procedural-articulation + Isaac-recorder slices land.

## Migration from the legacy 0.1.x API

`strands-robots-sim` 0.1.x shipped `SimEnv` / `SteppedSimEnv` plus a
natural-language `libero_example.py`. Those code paths moved upstream
in 0.2.0 — see [`MIGRATION.md`](MIGRATION.md) for the explicit
`SimEnv → libero/run_mujoco.py`, `agent("Run the task ...") →
libero/run_mujoco_agent.py`, and `SteppedSimEnv → R24 / #29 + upstream U6`
mapping.
