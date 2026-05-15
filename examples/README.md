# Examples

Per-backend tutorials for running LIBERO benchmarks against each
`SimEngine` shipped by `strands-robots-sim` and its upstream sibling
`strands-robots`.

## Two execution patterns

Each backend ships **two sibling files** that mirror the two patterns
the deleted `SimEnv` / `SteppedSimEnv` API used to cover.

| File suffix | Pattern | Replaces | Best for |
|---|---|---|---|
| `<backend>.py` | **One-shot** — `evaluate_benchmark(...)` runs to completion, prints success rate + wall-time | `SimEnv` | CI / benchmark numbers / R15 matrix table |
| `<backend>_stepped.py` | **Iterative supervision** — `start_policy(...)` + poll `get_state` / `render` in a System-2 loop | `SteppedSimEnv` | Agent-driven adaptation, demoing the iterative pattern |

The one-shot files print two grep-stable lines (`benchmark_name=...` and
`policy=... success_rate=... wall_time=...s`) that R15
([`libero_backend_matrix.py`](https://github.com/strands-labs/robots-sim/issues/22))
subprocess-and-parses for the side-by-side comparison table. The
stepped files are for human inspection — output is human-readable, not
matrix-ingested.

## Two policy choices

Both files in every pair accept `--policy {mock,groot}`:

| Flag | Provider | When | Reproducibility |
|---|---|---|---|
| `--policy mock` (default) | random-action stub in `strands_robots.policies.mock` | smoke tests / CI / no-GPU dev boxes / "did the plumbing work" sanity check | deterministic given `--seed` |
| `--policy groot` | NVIDIA GR00T VLA, served via `nvcr.io/nvidia/isaac-gr00t` Docker (or `gr00t_inference` Strands tool) on `--port 8000` against the public `nvidia/GR00T-N1.7-LIBERO` checkpoint | real LIBERO success-rate measurements | depends on the GR00T checkpoint + service config |

Service-start commands (Strands tool *and* bare-Docker fallback) live in
`libero_mujoco.py`'s docstring; `libero_mujoco_stepped.py` points back
at it rather than duplicating.

The mock invocation's wall-time is a smoke-test reference only; **the
canonical mujoco baseline number for the matrix table is the
`--policy=groot` measurement**.

## Backend matrix

Same task — first registered LIBERO spatial task, 10 episodes, seed 42 —
on every available backend with success rate and wall-time recorded
side-by-side. Numbers come from the *one-shot* file with `--policy=groot`
unless a row note says otherwise; mock-policy smoke runs are listed
below the table for reference.

| Example | Backend | `n_envs` | Wall-time @ success rate | Notes |
|---|---|---:|---|---|
| [`libero_mujoco.py`](libero_mujoco.py) | MuJoCo (in `strands-robots`) | 1 | TBD (groot) — see smoke note below | macOS / CPU OK |
| `libero_isaac.py` | Isaac Sim | 1 | _TBD ([R8 / #15](https://github.com/strands-labs/robots-sim/issues/15))_ | RTX path-traced |
| `libero_isaac_fleet.py` | Isaac Sim | 4096 | _TBD ([R23 / #27](https://github.com/strands-labs/robots-sim/issues/27))_ | IsaacLab-style fleet RL |
| `libero_newton.py` | Newton / Warp | 1 | _TBD ([R12 / #19](https://github.com/strands-labs/robots-sim/issues/19))_ | CUDA only |
| `libero_newton_fleet.py` | Newton / Warp | 4096 | _TBD ([R12 / #19](https://github.com/strands-labs/robots-sim/issues/19))_ | fleet |

**Mock-policy smoke wall-time (reference only, not matrix-authoritative):**

- `libero_mujoco.py --policy mock --n-episodes 10 --seed 42` → ~0.8 s on
  a single-CPU dev box (success rate 0.0 — mock can't satisfy goals).

The `--policy=groot` MuJoCo number drops in once the upstream BDDL fix
([`strands-labs/robots#147`](https://github.com/strands-labs/robots/pull/147))
lands on PyPI and a contributor with a GPU + Docker measures it.

The flagship driver
[`libero_backend_matrix.py`](https://github.com/strands-labs/robots-sim/issues/22)
(R15) walks all five rows and prints a unified table.

## Running the MuJoCo baseline

```bash
pip install 'strands-robots[sim-mujoco,benchmark-libero]'

# 1) Smoke test, no GPU needed:
python examples/libero_mujoco.py --policy mock --n-episodes 5

# 2) Iterative supervision pattern (also works without a GPU, mock policy):
python examples/libero_mujoco_stepped.py --policy mock --max-iters 20

# 3) Real eval against nvidia/GR00T-N1.7-LIBERO. Start the inference
#    service first per `libero_mujoco.py`'s docstring, then:
python examples/libero_mujoco.py --policy groot --port 8000 --n-episodes 50
```

Each invocation produces an MP4 under `rollouts/YYYY_MM_DD/`; filenames
encode `policy=mock|groot`, the seed, and the suite / episode count
(one-shot) or `--stepped` marker (iterative) so post-hoc analysis can
tell what produced each file. The filename convention is preserved from
the deleted `SimEnv` so existing `rollouts/` scrapers keep working.

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

`strands-robots-sim` 0.1.x shipped `SimEnv` / `SteppedSimEnv` with a
LIBERO env layer baked in. Those code paths moved upstream in 0.2.0 — see
[`MIGRATION.md`](MIGRATION.md) for the explicit
`SimEnv → libero_mujoco.py` and `SteppedSimEnv → libero_mujoco_stepped.py`
mapping.
