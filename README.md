# Strands Robots Simulation

> Heavy NVIDIA simulation backends for [`strands-robots`](https://github.com/strands-labs/robots): **Isaac Sim** (USD + IsaacLab 3.0) and **Newton/Warp** (GPU-native, 4096+ envs, differentiable).

For the default lightweight **MuJoCo** backend, use [`strands-robots`](https://github.com/strands-labs/robots) directly — both backends in this repo plug into the same `Simulation` AgentTool / `SimEngine` ABC and load via entry points, so the user-facing API is identical across all three.

> **Status:** v0.2.0 was the re-scoped foundation release. The legacy `SimEnv` / `SteppedSimEnv` / GR00T policy code path moved upstream — see [`examples/MIGRATION.md`](examples/MIGRATION.md). **Isaac Sim Phase 1** (R7) has now landed in `main` — entry-point registration, `IsaacConfig`, `IsaacSimulation` lifecycle scaffolding, the procedural-robot builders (SO-100 / Panda / G1), and URDF / MJCF / USD loaders are all working today; the still-no-op data-plane methods (`add_object`, `add_camera`, `replicate`, the per-`IsaacSimulation` `_load_*_robot` stubs) are Phase 2 work. **Newton** lands in v0.4.0+. Track the rollout in [#8](https://github.com/strands-labs/robots-sim/issues/8).

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

> Isaac Sim Phase 1 has shipped — the snippet below is copy-paste-runnable today on a host with Isaac Sim 2024.x+ installed. Newton snippets become copy-paste-runnable when [R11 / #18](https://github.com/strands-labs/robots-sim/issues/18) ships.

### Isaac Sim — photorealistic single-env

```python
import strands_robots_sim                       # registers "isaac" / "newton" via entry points
from strands_robots.simulation import create_simulation

sim = create_simulation("isaac", render_mode="rtx_pathtracing", headless=True)
sim.create_world()
sim.add_robot("so100")          # procedural; no asset files needed
sim.step(100)
frame = sim.render(camera_name="default")
```

For URDF / MJCF / USD ingestion, use the loader module directly:

```python
from strands_robots_sim.isaac.loaders import load_urdf, load_mjcf, load_usd

panda = load_urdf("/path/to/panda.urdf")        # stdlib XML, no extra deps
print(panda.num_joints, panda.joint_names)
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
| `mujoco` | 1 | software / GL | Default; macOS + Apple Silicon OK; fast iteration | ~9 s/ep @ 1.00 (groot)* | `libero/run_mujoco.py` | [R5 / #12](https://github.com/strands-labs/robots-sim/issues/12) |
| `isaac` | 1 | RTX path-traced | Photoreal eval, demo videos, paper-grade frames | TBD | `libero_isaac.py` | [R8 / #15](https://github.com/strands-labs/robots-sim/issues/15) |
| `isaac` | 4096 | RTX off / minimal | IsaacLab-style fleet RL with USD scenes | TBD | `libero_isaac_fleet.py` | [R23 / #27](https://github.com/strands-labs/robots-sim/issues/27) |
| `newton` | 1 | OpenGL | GPU-physics baseline; entry point for diffsim work | TBD | `libero_newton.py` | [R12 / #19](https://github.com/strands-labs/robots-sim/issues/19) |
| `newton` | 4096 | OpenGL / null | Multi-solver fleet RL, lowest per-env compute | TBD | `libero_newton_fleet.py` | [R12 / #19](https://github.com/strands-labs/robots-sim/issues/19) |

\* L4 / Docker dev box, `nvidia/GR00T-N1.7-LIBERO/libero_10` against `libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_…` (5 episodes, seeds 42 + 100), measured 2026-05-21 against `strands-robots` post-[#188](https://github.com/strands-labs/robots/pull/188). The full upstream catch-up wave is in: [#168](https://github.com/strands-labs/robots/pull/168) (rounds 36-44) + [#172](https://github.com/strands-labs/robots/pull/172) (closes #169) + [#173](https://github.com/strands-labs/robots/pull/173) (closes #170) + [#175](https://github.com/strands-labs/robots/pull/175) (closes #171 + #176) + [#180](https://github.com/strands-labs/robots/pull/180) (closes #179) + [#184](https://github.com/strands-labs/robots/pull/184) (closes #181) + [#186](https://github.com/strands-labs/robots/pull/186) (closes #178) + [#188](https://github.com/strands-labs/robots/pull/188) (closes #187: spec-driven instruction fallback + per-episode `policy.reset(seed=)` plumbing). Pre-#188 the ZMQ path returned `success_rate=0.20-0.60` because language-conditioned GR00T was getting an empty instruction. Post-#188 hits 5/5 reliably across seeds.

---

## Status & roadmap

Tracking umbrella: [`#8`](https://github.com/strands-labs/robots-sim/issues/8).

- **Stage 1 — Foundation cleanup**
  - [x] R2 / [#9](https://github.com/strands-labs/robots-sim/issues/9) — remove legacy `SimEnv` / `SteppedSimEnv` / Libero env layer
  - [x] R3 / [#10](https://github.com/strands-labs/robots-sim/issues/10) — remove duplicated GR00T policy / inference tool / tests / scripts / docs
  - [x] R4 / [#11](https://github.com/strands-labs/robots-sim/issues/11) — README rewrite (this file)
- **Stage 2 — MuJoCo LIBERO baseline**
  - [x] R5 / [#12](https://github.com/strands-labs/robots-sim/issues/12) — `examples/libero/run_mujoco.py` + `examples/README.md`
- **Stage 3 — Isaac Sim** (Phase 1 in `main`; full backend ships with v0.3.0)
  - [x] R6 / [#13](https://github.com/strands-labs/robots-sim/issues/13) — entry-point backend registration ([#44](https://github.com/strands-labs/robots-sim/pull/44))
  - [x] R7 (Phase 1) / [#14](https://github.com/strands-labs/robots-sim/issues/14) — `IsaacSimulation(SimEngine)` skeleton + lifecycle + procedural builders + loaders ([#45](https://github.com/strands-labs/robots-sim/pull/45) / [#46](https://github.com/strands-labs/robots-sim/pull/46) / [#47](https://github.com/strands-labs/robots-sim/pull/47) / [#51](https://github.com/strands-labs/robots-sim/pull/51))
  - [ ] R7 (Phase 2) / [#14](https://github.com/strands-labs/robots-sim/issues/14) — data-plane wiring (`add_object` / `add_camera` / `replicate` / articulation construction)
  - [ ] R8 / [#15](https://github.com/strands-labs/robots-sim/issues/15) — `examples/libero/run_isaac.py`
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
