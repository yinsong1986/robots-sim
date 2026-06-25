# Examples

`strands-robots-sim` ships runnable example drivers under
[`examples/`](https://github.com/strands-labs/robots-sim/tree/main/examples)
that mirror the upstream `strands-robots/examples/` layout. Every Isaac
example has a programmatic sibling and a Strands-Agent natural-language
sibling so you can pick the driver that matches your audience.

## Two execution patterns per backend

| File | Driver | Best for |
|---|---|---|
| `<benchmark>/run_<backend>.py` | **Programmatic** — Python script calls `sim.evaluate_benchmark(...)` directly | CI / matrix tables / benchmark numbers |
| `<benchmark>/run_<backend>_agent.py` | **Strands `Agent` + natural language** — script owns the deterministic plumbing (GR00T container lifecycle, scene pre-warm, MP4 recording); a single `agent("…")` call invokes the eval and produces a prose summary | Demos / "what does a Strands integration buy?" |

The programmatic file prints two grep-stable lines that the flagship
`libero_backend_matrix.py` driver subprocess-and-parses for the
side-by-side table:

```
benchmark_name=libero-spatial-pick_up_the_red_cube
policy=groot task=libero-spatial-pick_up_the_red_cube success_rate=1.00 wall_time=44.3s
```

The agent file's output is non-deterministic LLM-generated prose — for
human inspection only, not for matrix ingestion.

## Two policy choices

Both files in every pair accept `--policy {mock,groot}`:

| Flag | Provider | When | Reproducibility |
|---|---|---|---|
| `--policy mock` (default) | random-action stub from `strands_robots.policies.mock` | Smoke tests / CI / no-GPU dev boxes / "did the plumbing work" sanity check | Deterministic given `--seed` |
| `--policy groot` | NVIDIA GR00T VLA, served via `nvcr.io/nvidia/isaac-gr00t` Docker on `--port 8000` against the suite-specific `nvidia/GR00T-N1.7-LIBERO/libero_<suite>/` checkpoint | Real LIBERO success-rate measurements | Depends on the GR00T checkpoint + service config |

Both example files orchestrate the GR00T container lifecycle (build →
checkpoint download → start → wait-for-ready → teardown) deterministically
from the script via `gr00t_inference(action='lifecycle', ...)`. The agent
file's agent isn't asked to manage Docker / HF cache — those are brittle
for an LLM and stay under Python control. Pass `--no-auto-server` to
either file to reuse an already-running container instead.

The checkpoint is cached under a **non-`/home`** path by default
(`$STRANDS_ROBOTS_CHECKPOINT_DIR`, an outside-`/home`
`$XDG_CACHE_HOME/strands_robots/checkpoints`, or
`/tmp/strands_robots/checkpoints`). `gr00t_inference`'s `start_container`
step refuses to bind-mount any path under `/home` (a "protected host
path" guard), so a `/home` cache would abort the lifecycle; the non-`/home`
default keeps `--policy groot` working out-of-the-box. Override with
`--checkpoint-dir <path>` if you want the cache elsewhere (must also be
outside `/home`).

## Backend matrix (LIBERO)

Same task — `libero-spatial-pick_up_the_red_cube`, 10 episodes, seed 42 —
on the two supported backends, with success rate and wall-time recorded
side-by-side. Numbers come from the *programmatic* file with
`--policy=groot` against the matching `libero_<suite>/` sub-checkpoint
unless a row says otherwise; mock-policy smoke runs are reference only.

| Example | Backend | `n_envs` | Renderer | Why use this row | Wall-time @ success-rate |
|---|---|---:|---|---|---|
| [`libero/run_mujoco.py`](https://github.com/strands-labs/robots-sim/tree/main/examples/libero/run_mujoco.py) | MuJoCo (in `strands-robots`) | 1 | Software / GL | Default; macOS + Apple Silicon OK; fast iteration | ~9 s/ep @ 1.00 (groot)[^1] |
| [`libero/run_isaac.py`](https://github.com/strands-labs/robots-sim/tree/main/examples/libero/run_isaac.py) | Isaac Sim | 1 | RTX path-traced | Photoreal eval, demo videos, paper-grade frames | _measured by [`libero_backend_matrix.py`](https://github.com/strands-labs/robots-sim/tree/main/examples/libero/libero_backend_matrix.py); lifecycle validated in [PR #74](https://github.com/strands-labs/robots-sim/pull/74)_ |

IsaacLab-style fleet RL (`n_envs=4096`) is surfaced by the flagship
matrix driver `libero_backend_matrix.py` as a separate
`run_isaac_fleet.py` row (`isaac-4096`), which reads `unavailable` until
that driver lands. The flagship driver walks the rows and prints a
unified table; each row also has a stand-alone example you can run in
isolation. Roadmap and follow-ups are tracked under the umbrella
[#8](https://github.com/strands-labs/robots-sim/issues/8).

[^1]: Single-sample on the L4 reference dev box (`libero-10/SCENE5`,
    seed=42, n=5). Pre-`strands-labs/robots#188` success rate was 0.20–0.60
    across re-runs; post-#188 it stabilises at 1.00.

## Running the Isaac Sim baseline

```bash
pip install 'strands-robots-sim[isaac]' \
    'strands-robots[benchmark-libero] @ git+https://github.com/strands-labs/robots.git@main'

# 1) Programmatic smoke test (mock policy). Loads the default Franka USD.
python examples/libero/run_isaac.py --policy mock --n-episodes 5

# 1b) Bring your own robot asset:
python examples/libero/run_isaac.py --policy mock --robot-usd /path/to/robot.usd
python examples/libero/run_isaac.py --policy mock --robot-urdf /path/to/robot.urdf

# 2) Strands-Agent + natural language (needs `strands-agents` + an LLM provider):
pip install strands-agents
python examples/libero/run_isaac_agent.py --policy mock

# 3) Real eval against the matching `libero_<suite>/` sub-checkpoint
#    (auto-orchestrates the GR00T container; pass --no-auto-server to reuse one):
python examples/libero/run_isaac.py --policy groot --port 8000 --n-episodes 50
```

Each invocation produces an MP4 under `rollouts/YYYY_MM_DD/`. Filename
encodes `--task=<benchmark_name>`, `--policy=mock|groot`, `--seed=S`, and
either `--n_eps=N` (programmatic) or `--agent` marker (agent file) so
post-hoc analysis can tell what produced each file.

## Isaac-specific gotchas

The Isaac drivers differ from the MuJoCo ones in three load-bearing ways:

1. **Real-asset robot.** Instead of a LIBERO MJCF, the script loads a
   *real* robot via `add_robot(usd_path=...)` — by default Isaac Sim's
   bundled Franka Panda USD, resolved from the assets root over the
   public Omniverse CDN (no local Nucleus needed). Override with
   `--robot-usd PATH` or `--robot-urdf PATH`. (A real asset is required
   because the procedural builder is a kinematically-approximate
   stick-figure unusable by a LIBERO manipulation policy.)
2. **Explicit camera.** Isaac doesn't auto-attach a viewport camera the
   way MuJoCo's `mjData` does, so the script makes an explicit
   `add_camera(...)` call at the LIBERO `agentview` vantage before the
   eval.
3. **Isaac-specific container name** (`gr00t-libero-isaac`) so Isaac and
   MuJoCo `--policy=groot` runs don't clobber each other's containers on
   the same host.

> **Requires Isaac Sim** installed separately (it is **not** pip-installable —
> Omniverse Launcher / Isaac Lab / NGC Docker, RTX GPU, CUDA 12+). On a
> non-Isaac host both scripts exit early with a diagnostic from
> `IsaacSimulation.is_available()` rather than crashing on the first
> `omni.*` import.

## Other Isaac examples

In addition to the LIBERO drivers, the repo ships two visually-driven
demos under `examples/`:

- [`examples/isaac_gs/`](https://github.com/strands-labs/robots-sim/tree/main/examples/isaac_gs) —
  Isaac Sim + 3DGS hybrid-render digital-twin example.
- [`examples/so101_curobo/`](https://github.com/strands-labs/robots-sim/tree/main/examples/so101_curobo) —
  SO-101 synthetic-data generation with cuRobo on the Isaac Sim backend.

Each subdir has its own README with run instructions.

### SO-101 cuRobo pick-and-place

<video controls muted loop playsinline width="560" poster="../../assets/so101_oblique.png">
  <source src="../../assets/so101_pickplace.mp4" type="video/mp4">
  Your browser does not support the video tag.
  <a href="../../assets/so101_pickplace.mp4">Download the clip</a>.
</video>

[`examples/so101_curobo/app.py`](https://github.com/strands-labs/robots-sim/tree/main/examples/so101_curobo)
plans a collision-aware pick-and-place with cuRobo and replays it on the Isaac
Sim PhysX backend: the gripper grasps and lifts the cube, carries it to the
open-top bin, and records a LeRobot dataset with `state`, `action`, and three
camera views (front / top-down / oblique, shown above).

![SO-101 pick-and-place, top-down camera](../assets/so101_topdown.png)

*The same episode from the top-down camera — the SO-101 arm, the red cube, and
the green open-top bin.*

## Cross-references

- [Getting Started → Quickstart](../getting-started/quickstart.md) — the
  shortest possible Isaac example.
- [Simulation → World Building](../simulation/world-building.md) — what
  the example drivers call under the hood.
- [Backends → Isaac Sim](../backends/isaac.md) — the full backend
  reference including procedural builders and loaders.
- [Troubleshooting](../troubleshooting.md) — common failures running the
  example drivers.
