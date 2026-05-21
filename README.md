# Strands Robots Simulation

> Heavy NVIDIA simulation backends for [`strands-robots`](https://github.com/strands-labs/robots): **Isaac Sim** (USD + IsaacLab 3.0) and **Newton/Warp** (GPU-native, 4096+ envs, differentiable).

For the default lightweight **MuJoCo** backend, use [`strands-robots`](https://github.com/strands-labs/robots) directly — both backends in this repo plug into the same `Simulation` AgentTool / `SimEngine` ABC and load via entry points, so the user-facing API is identical across all three.

> **Status:** v0.2.0 is a re-scoped foundation release. The legacy `SimEnv` / `SteppedSimEnv` / GR00T policy code path moved upstream — see [`examples/MIGRATION.md`](examples/MIGRATION.md). Backend code lands in v0.3.0+ (Isaac) and v0.4.0+ (Newton). Track the rollout in [#8](https://github.com/strands-labs/robots-sim/issues/8).

---

## Backend matrix

| Capability | MuJoCo<br/>(`strands-robots`) | **Newton**<br/>(this repo) | **Isaac Sim**<br/>(this repo) |
|---|:---:|:---:|:---:|
| Native GPU | — | ✅ Warp | ✅ PhysX |
| Apple Silicon | ✅ | ❌ (CUDA only) | ❌ |
| Photoreal rendering | — | basic OpenGL | ✅ RTX path-traced |
| `num_envs` per GPU | 1–8 | **4096+** | 4096+ |
| Differentiable sim | — | ✅ (Warp autodiff) | partial |
| Soft bodies / cloth / MPM | — | ✅ | ✅ |
| USD scene format | — | partial | ✅ native |
| Synthetic data (Replicator) | — | — | ✅ |
| Download size | ~50 MB | ~500 MB | ~30 GB |
| Setup friction | low | medium | high |

Authoritative sources: [`strands-labs/robots#96`](https://github.com/strands-labs/robots/issues/96) (Newton design), [`strands-labs/robots#97`](https://github.com/strands-labs/robots/issues/97) (Isaac Sim design).

- **Pick MuJoCo** for fast iteration, debugging, and macOS / Apple Silicon contributors.
- **Pick Newton** for differentiable physics, multi-solver workloads, or fleet RL where rendering can be off.
- **Pick Isaac Sim** for fleet RL with photoreal observations, USD-native scenes, or Replicator synth-data.

---

## Install

System requirements: **NVIDIA RTX GPU, Ubuntu 22.04+, CUDA 12+**. macOS / Apple Silicon contributors should install [`strands-robots`](https://github.com/strands-labs/robots) directly and skip this repo.

```bash
# Isaac Sim (~30 GB SDK download on first run)
pip install 'strands-robots-sim[isaac]'

# Newton + Warp
pip install 'strands-robots-sim[newton]'

# Both
pip install 'strands-robots-sim[all]'
```

`strands-robots` (and the default MuJoCo backend) are pulled in transitively.

---

## Quick start

> Snippets are written against the post-R6 entry-point API. They become copy-paste-runnable when [R6 / #13](https://github.com/strands-labs/robots-sim/issues/13) and [R7 / #14](https://github.com/strands-labs/robots-sim/issues/14) (Isaac) or [R11 / #18](https://github.com/strands-labs/robots-sim/issues/18) (Newton) ship.

### Isaac Sim — photorealistic single-env

```python
import strands_robots_sim                       # registers "isaac" / "newton" via entry points
from strands_robots.simulation import create_simulation

sim = create_simulation("isaac", rtx_mode="path_traced", headless=True)
sim.create_world()
sim.add_robot("so100")
sim.step(100)
frame = sim.render(camera_name="default")
```

### Newton — 4096-env fleet

```python
import strands_robots_sim
from strands_robots.simulation import create_simulation

sim = create_simulation("newton", num_envs=4096, solver="mujoco")
sim.create_world()
sim.add_robot("so100")
sim.step(1000)
state = sim.get_state()                         # batched [4096, ...] tensors
```

---

## LIBERO backend matrix

Same task — `libero-spatial-pick_up_the_red_cube` (Panda picks a red cube from a tabletop and places it in a target zone), 50 episodes, seed 42 — run on every available backend with success rate and wall-time recorded side-by-side. Numbers are committed as each example lands, measured on a reference machine (recorded in each example file's header).

The flagship driver `examples/libero_backend_matrix.py` ([R15 / #22](https://github.com/strands-labs/robots-sim/issues/22)) walks all five rows and prints a unified table; each row also has a stand-alone example you can run in isolation.

| Backend | `n_envs` | Renderer | Why use this row | Wall-time | Example | Issue |
|---|---:|---|---|---|---|---|
| `mujoco` | 1 | software / GL | Default; macOS + Apple Silicon OK; fast iteration | ~54 s/ep @ 0.60-0.92 (groot)* | `libero_mujoco.py` | [R5 / #12](https://github.com/strands-labs/robots-sim/issues/12) |
| `isaac` | 1 | RTX path-traced | Photoreal eval, demo videos, paper-grade frames | TBD | `libero_isaac.py` | [R8 / #15](https://github.com/strands-labs/robots-sim/issues/15) |
| `isaac` | 4096 | RTX off / minimal | IsaacLab-style fleet RL with USD scenes | TBD | `libero_isaac_fleet.py` | [R23 / #27](https://github.com/strands-labs/robots-sim/issues/27) |
| `newton` | 1 | OpenGL | GPU-physics baseline; entry point for diffsim work | TBD | `libero_newton.py` | [R12 / #19](https://github.com/strands-labs/robots-sim/issues/19) |
| `newton` | 4096 | OpenGL / null | Multi-solver fleet RL, lowest per-env compute | TBD | `libero_newton_fleet.py` | [R12 / #19](https://github.com/strands-labs/robots-sim/issues/19) |

\* L4 / Docker dev box, `nvidia/GR00T-N1.7-LIBERO/libero_10` against `libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_…` (5 episodes, seed 42), measured 2026-05-21 against `strands-robots` `main` post-[#186](https://github.com/strands-labs/robots/pull/186). The full upstream catch-up wave is in: [#168](https://github.com/strands-labs/robots/pull/168) (rounds 36-44) + [#172](https://github.com/strands-labs/robots/pull/172) (closes #169: ZMQ wire-format) + [#173](https://github.com/strands-labs/robots/pull/173) (closes #170: BDDL evaluator) + [#175](https://github.com/strands-labs/robots/pull/175) (closes #171 + #176: MuJoCoSimEngine state + OSC + gripper + suffix fixes) + [#180](https://github.com/strands-labs/robots/pull/180) (per-episode reseed) + [#184](https://github.com/strands-labs/robots/pull/184) (MJCF inertia preservation) + [#186](https://github.com/strands-labs/robots/pull/186) (retire `LiberoOffScreenRenderEngine`; `MuJoCoSimEngine` is now byte-equivalent to upstream LIBERO). PR #186 reports mean `success_rate=0.92`. Single-sample re-runs at the same seed have produced 0.60 / 0.80 / 0.92 in different processes — bounded variance from CUDA non-determinism in the GR00T docker server. Acceptance: `success_rate > 0`, not a specific number.

---

## Status & roadmap

Tracking umbrella: [`#8`](https://github.com/strands-labs/robots-sim/issues/8).

- **Stage 1 — Foundation cleanup**
  - [x] R2 / [#9](https://github.com/strands-labs/robots-sim/issues/9) — remove legacy `SimEnv` / `SteppedSimEnv` / Libero env layer
  - [x] R3 / [#10](https://github.com/strands-labs/robots-sim/issues/10) — remove duplicated GR00T policy / inference tool / tests / scripts / docs
  - [x] R4 / [#11](https://github.com/strands-labs/robots-sim/issues/11) — README rewrite (this file)
- **Stage 2 — MuJoCo LIBERO baseline**
  - [ ] R5 / [#12](https://github.com/strands-labs/robots-sim/issues/12) — `examples/libero_mujoco.py` + `examples/README.md`
- **Stage 3 — Isaac Sim** (ships with v0.3.0)
  - [ ] R6 / [#13](https://github.com/strands-labs/robots-sim/issues/13) — entry-point backend registration
  - [ ] R7 / [#14](https://github.com/strands-labs/robots-sim/issues/14) — `IsaacSimulation(SimEngine)` backend
  - [ ] R8 / [#15](https://github.com/strands-labs/robots-sim/issues/15) — `examples/libero_isaac.py`
  - [ ] R9 / [#16](https://github.com/strands-labs/robots-sim/issues/16) — `examples/isaac_replicator_synthdata.py`
  - [ ] R10 / [#17](https://github.com/strands-labs/robots-sim/issues/17) — nightly GPU CI
- **Stage 4 — Newton** (ships with v0.4.0)
  - [ ] R11 / [#18](https://github.com/strands-labs/robots-sim/issues/18) — `NewtonSimulation(SimEngine)` backend
  - [ ] R12 / [#19](https://github.com/strands-labs/robots-sim/issues/19) — Newton LIBERO examples
  - [ ] R13 / [#20](https://github.com/strands-labs/robots-sim/issues/20) — `examples/newton_diffsim_toy.py`
  - [ ] R14 / [#21](https://github.com/strands-labs/robots-sim/issues/21) — extend nightly GPU CI for Newton
- **Stage 5 — Flagship** (ships with v0.5.0)
  - [ ] R15 / [#22](https://github.com/strands-labs/robots-sim/issues/22) — `examples/libero_backend_matrix.py`

Migrating from the 0.1.x API: [`examples/MIGRATION.md`](examples/MIGRATION.md).

---

## Contributing

PRs welcome. `hatch run lint` (black / isort / flake8) and `hatch run test` (an import smoke until backend code lands) before submitting. Backend-specific tests will live under `tests/isaac/` (R7) and `tests/newton/` (R11). GPU CI is tracked separately in [#17](https://github.com/strands-labs/robots-sim/issues/17) (Isaac) and [#21](https://github.com/strands-labs/robots-sim/issues/21) (Newton).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

## Links

- [`strands-labs/robots`](https://github.com/strands-labs/robots) — default MuJoCo backend, `Simulation` AgentTool, LIBERO adapter
- [Strands Agents SDK](https://github.com/strands-agents/sdk-python)
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) ([IsaacLab](https://isaac-sim.github.io/IsaacLab/))
- [Newton](https://github.com/newton-physics/newton) (built on [NVIDIA Warp](https://github.com/NVIDIA/warp))
