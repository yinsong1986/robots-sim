# Examples

Per-backend tutorials for running LIBERO benchmarks against each
`SimEngine` shipped by `strands-robots-sim` and its upstream sibling
`strands-robots`.

## Two execution patterns

Each backend ships **two sibling files** that mirror the two driver
patterns the deleted `SimEnv` API used to cover.

| File suffix | Driver | Replaces | Best for |
|---|---|---|---|
| `<benchmark>/run_<backend>.py` | **Programmatic** — Python script calls `sim.evaluate_benchmark(...)` directly | `SimEnv` | CI / benchmark numbers / the flagship backend-matrix table |
| `<benchmark>/run_<backend>_agent.py` | **Strands `Agent` + natural language** — script owns the deterministic plumbing (GR00T container lifecycle, scene pre-warm, MP4 recording); a single `agent("…")` call invokes `evaluate_benchmark` from natural-language kwargs and produces a prose summary | the natural-language entry point in the deleted `libero_example.py` | Demoing why a Strands integration buys you anything beyond direct calls |

The programmatic files print two grep-stable lines (`benchmark_name=...`
and `policy=... task=... success_rate=... wall_time=...s`) that the
flagship driver
[`examples/libero/libero_backend_matrix.py`](libero/libero_backend_matrix.py)
subprocess-and-parses for the side-by-side comparison table. The agent
files are for human inspection — output is non-deterministic
LLM-generated prose, not matrix-ingested.

> **Iterative supervision** (the deleted `SteppedSimEnv` pattern) is
> deliberately *not* in this directory. With
> `nvidia/GR00T-N1.7-LIBERO/libero_<suite>/` finetuned end-to-end on
> its training distribution the policy executes the canonical task
> without stalling, so System-2 supervision over an in-distribution
> finetuned policy has nothing to actually decide. The canonical
> pattern doc for OOD-anchored iterative supervision lives upstream at
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

### `--policy groot` prerequisites & gotchas

Running `--policy groot` end-to-end (real GR00T-N1.7 VLA, served via
the `nvcr.io/nvidia/isaac-gr00t` container) reaches `success_rate=1.00`
on e.g. `libero-10/SCENE5` for both `run_mujoco.py` and
`run_mujoco_agent.py`. A few setup gotchas trip up first-time runs;
each fails *loudly* except the last, which silently zeroes the score:

- **Checkpoint dir must live outside `/home`.** `gr00t_inference`
  refuses to bind-mount any path under `/home` (a "protected host path"
  mount guard), so a checkpoint cache there aborts `start_container`
  with `refusing to mount … under protected host path '/home'`. Both
  example drivers now **default to a non-`/home` cache**
  (`$XDG_CACHE_HOME/strands_robots/checkpoints` when it's outside
  `/home`, else `/tmp/strands_robots/checkpoints`), so the OOTB run just
  works ([#125](https://github.com/strands-labs/robots-sim/issues/125),
  fixed in [#126](https://github.com/strands-labs/robots-sim/pull/126)).
  If you *override* `--checkpoint-dir` (or `$STRANDS_ROBOTS_CHECKPOINT_DIR`)
  with a `/home` path you'll hit the guard again — point it at a
  non-`/home` path such as `--checkpoint-dir /tmp/groot-ck`.
- **`pyzmq` is required** for the GR00T ZMQ client. Without it the eval
  fails *after* the model loads with
  `ImportError: 'zmq' is required for GR00T service inference`. If your
  GR00T policy extra doesn't pull it in, `pip install pyzmq`.
- **`numba` + `coverage>=7` silently zeroes `success_rate`.** When both
  are installed in the eval environment, `import numba` fails
  (`numba/misc/coverage_support.py` subclasses the removed
  `coverage.types.Tracer`). That makes the LIBERO adapter's OSC
  controller fail to install, so **GR00T actions silently no-op and
  `success_rate=0`** with only a buried `WARNING` — easy to misread as a
  bad policy. Uninstall the conflicting `coverage` (or pin
  `coverage<7`) and the same run returns `success_rate=1.00`. The
  silent-degrade behaviour is tracked upstream at
  [`strands-labs/robots#522`](https://github.com/strands-labs/robots/issues/522).

See [docs/troubleshooting.md](../docs/troubleshooting.md#-policy-groot-eval-failures)
for the exact error strings and one-line fixes.

## Backend matrix

Same task — `libero-spatial-pick_up_the_red_cube`, 10 episodes, seed
42 — on the two supported backends with success rate and wall-time
side-by-side. Numbers come from the *programmatic* file with
`--policy=groot` against the matching `libero_<suite>/` sub-checkpoint
unless a row says otherwise; mock-policy smoke runs are listed below
the table for reference.

| Example | Backend | `n_envs` | Wall-time @ success-rate | Notes |
|---|---|---|---|---|
| [`libero/run_mujoco.py`](libero/run_mujoco.py) | MuJoCo (in `strands-robots`) | 1 | ~9 s/ep @ 1.00 (groot, ZMQ client)[^1] | Reliably reaches 5/5 against post-[#188](https://github.com/strands-labs/robots/pull/188) `strands-robots`; macOS / CPU OK |
| [`libero/run_isaac.py`](libero/run_isaac.py) | Isaac Sim | 1 | _not yet measured — blocked on [#140](https://github.com/strands-labs/robots-sim/issues/140)_ | RTX real-time (`render_mode="rtx_realtime"`); loads a real Franka USD via `add_robot(usd_path=...)`. Data-plane slices landed in [PR #74](https://github.com/strands-labs/robots-sim/pull/74), but `run_isaac.py` currently crashes with an `IndexError` in `render()` during RTX warm-up before `evaluate_benchmark` runs, so **end-to-end validation does not hold today** ([#140](https://github.com/strands-labs/robots-sim/issues/140)). |

IsaacLab-style fleet RL (n_envs=4096) is surfaced by the flagship
matrix driver as a separate `run_isaac_fleet.py` row (`isaac-4096`),
which reads `unavailable` until that driver lands; see
[Running the matrix](#running-the-matrix) below.

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

The flagship driver
[`examples/libero/libero_backend_matrix.py`](libero/libero_backend_matrix.py)
([#22](https://github.com/strands-labs/robots-sim/issues/22))
walks every per-backend driver this checkout can execute (today MuJoCo +
Isaac Sim) and prints a unified table — see
[Running the matrix](#running-the-matrix) below.

[^1]: Single-sample on the L4 reference dev box (`libero-10/SCENE5`,
    seed=42, n=5). Pre-#188 success rate was 0.20–0.60 across re-runs;
    post-#188 it stabilises at 1.00. See PR #26's history threads on
    [#187](https://github.com/strands-labs/robots/issues/187) /
    [#188](https://github.com/strands-labs/robots/pull/188) for the
    variance bisection.

## Running the MuJoCo baseline

```bash
pip install 'strands-robots[sim-mujoco,benchmark-libero]'

# 1) Programmatic, deterministic, no LLM. The flagship matrix driver ingests this output.
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

> **Status (in progress — blocked on [#140](https://github.com/strands-labs/robots-sim/issues/140)):**
> Phase-2 data-plane slices (`add_camera`, render frame-path, USD-load,
> URDF-load) are merged on `main`, including the `is_available()`
> namespace-shim and the port from `omni.isaac.*` to `isaacsim.*`
> ([PR #74](https://github.com/strands-labs/robots-sim/pull/74)).
> **End-to-end validation does not hold today:** `run_isaac.py`
> currently crashes with an `IndexError` in `render()` during the RTX
> warm-up loop before `evaluate_benchmark` runs
> ([#140](https://github.com/strands-labs/robots-sim/issues/140)), so the
> matrix wall-time @ success-rate number for the Isaac row is not yet
> measured. On a non-Isaac host both scripts still exit early with a
> structured diagnostic via `IsaacSimulation.is_available()` rather than
> crashing on the first import. Once #140 is fixed and `run_isaac.py`
> runs end-to-end, this row reverts to validated wording.

## Running the matrix

[`libero/libero_backend_matrix.py`](libero/libero_backend_matrix.py)
is the flagship: one script, one LIBERO task, every per-backend driver
file that this checkout can actually execute, side-by-side. Missing
backends produce `unavailable` rows; backends whose `is_available()`
short-circuits fire produce `skip (...)` rows with the truncated
reason. Backends that succeed produce an `ok` row with measured
`success_rate` and `wall_time`. The script never imports backend
modules itself — it subprocesses each `run_<backend>.py` and parses
the two grep-stable lines documented above, so a missing backend
never crashes the matrix.

```bash
# 1) Whatever's installed -- mock policy by default, no GR00T
#    container needed:
python examples/libero/libero_backend_matrix.py

# 2) Limit which backend rows are attempted (faster smoke runs):
python examples/libero/libero_backend_matrix.py --backends mujoco,isaac-1

# 3) Real eval against the matching `libero_<suite>/` GR00T
#    sub-checkpoint (each driver auto-orchestrates its own GR00T
#    container; see the per-backend section above for setup):
python examples/libero/libero_backend_matrix.py --policy groot

# 4) Different LIBERO task (forwarded to every per-backend driver):
python examples/libero/libero_backend_matrix.py \
    --task libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_…
```

### Install combinations

The matrix script's row availability follows from which extras are
installed, since each `run_<backend>.py` only succeeds when its
backend can import:

```bash
# MuJoCo only (mujoco row → ok; isaac row → skip):
pip install 'strands-robots[sim-mujoco,benchmark-libero]'

# + Isaac Sim single-env (isaac-1 → ok on an Isaac Sim host):
pip install 'strands-robots-sim[isaac]' \
    'strands-robots[benchmark-libero]'
```

A row that reads `unavailable (no run_<backend>.py)` means the
per-backend driver file isn't in this checkout — the matrix script is
forward-compatible with rows that don't exist on disk, so it keeps
working as the staged plan tracked in
[#8](https://github.com/strands-labs/robots-sim/issues/8) evolves.

### Output format

The table is bracketed by stable markers so a downstream CI job can
locate it in a longer log:

```
=== libero_backend_matrix ===
Task: libero-spatial-pick_up_the_red_cube  (10 episodes, seed=42)
backend       success_rate   wall_time  status
----------------------------------------------------------------
mujoco                1.00       86.4s  ok
isaac-1                 --          --  skip (Isaac Sim is not available on this host: …)
=== /libero_backend_matrix ===
```

Pass `--backends mujoco,isaac-1` to attempt only a subset, and
`--per-backend-timeout 1200` to allow longer-running drivers (the
default 600 s is generous for 10-episode mock smoke; full
`--policy=groot` runs at 50 episodes can need more).

## Unique-capability demos

Not every example is a LIBERO benchmark. The files under
[`isaac/`](isaac/) demonstrate capabilities that are **specific to the
Isaac Sim backend** -- things the lightweight MuJoCo backend can't
replicate because it doesn't ship the Omniverse / USD / RTX stack
underneath. These examples sit outside the
`<benchmark>/run_<backend>.py` matrix convention because they're
showing what the Isaac backend is *for* at a higher level than
per-task evaluation.

| Example | Capability | Why Isaac-only | Tracking |
|---|---|---|---|
| [`isaac/isaac_replicator_synthdata.py`](isaac/isaac_replicator_synthdata.py) | NVIDIA Replicator domain-randomized synthetic-dataset generation (RGB + depth + semantic segmentation, randomized lighting / materials / camera poses) | The full Omniverse / USD stack unlocks `omni.replicator.core`. MuJoCo's renderer is rasterization-only and has no equivalent SDG / labelling pipeline. | [#16](https://github.com/strands-labs/robots-sim/issues/16) (this example) |

Each file under `isaac/` short-circuits with a structured diagnostic on
hosts without Isaac Sim (`IsaacSimulation.is_available()` check) so it's
safe to invoke in CI / dev-box smoke runs without crashing on the first
`omni.*` import.

## Migration from the legacy 0.1.x API

`strands-robots-sim` 0.1.x shipped `SimEnv` / `SteppedSimEnv` plus a
natural-language `libero_example.py`. Those code paths moved upstream
in 0.2.0 — see [`MIGRATION.md`](MIGRATION.md) for the explicit
`SimEnv → libero/run_mujoco.py`, `agent("Run the task ...") →
libero/run_mujoco_agent.py`, and `SteppedSimEnv → upstream U6
([`strands-labs/robots#136`](https://github.com/strands-labs/robots/issues/136))`
mapping.
