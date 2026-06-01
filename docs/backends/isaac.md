# Isaac Sim Backend

GPU-native photorealistic simulation backend for `strands-robots-sim` using NVIDIA Isaac Sim.

## Overview

`IsaacSimulation` provides:

- **Photorealistic rendering** via Omniverse RTX (path-traced, ground-truth depth, semantic segmentation)
- **Asset pipeline**: USD-native scenes, NVIDIA Nucleus assets
- **Sensors**: RTX cameras, LiDAR, depth, contact, IMU (GPU-batched)
- **Synthetic data generation**: Replicator pipeline for domain randomization
- **Fleet replication**: parallel environments via `omni.isaac.cloner.Cloner`
- **Isaac Lab integration**: GPU-accelerated RL environments


> **Phase 1 status (skeleton).** This release ships the SimEngine-shaped surface, lazy-import scaffolding, the procedural-robot dataclass + builders (SO-100 / Panda / G1), the URDF / MJCF / USD loader module, and the Phase 2 `add_object` / `remove_object` data-plane wiring (shape primitives via `omni.isaac.core.objects.{Dynamic,Fixed}{Cuboid,Sphere,Cylinder,Capsule}`). **Working today**: `IsaacConfig`, `IsaacSimulation.is_available()`, world / lifecycle (`create_world` / `destroy` / `cleanup`), procedural builders via `add_robot("so100" | "panda" | "unitree_g1")`, scene primitives via `add_object` / `remove_object`, the `isaac.loaders.load_urdf` / `load_mjcf` / `load_usd` functions for ingesting external robot description files into a `ProceduralRobot` dataclass, and `render`'s RTX frame-extraction path against a Phase-2 ``add_camera`` handle (real `get_rgba` / `get_depth` calls; falls back to blank frames in `headless` mode, when no camera is configured, or against a Phase-1 camera with no RTX handle attached). **Still no-op in this phase**: the remaining data-plane wiring on `IsaacSimulation` itself — `add_camera`, `replicate`, the per-`IsaacSimulation` `_load_usd_robot` / `_load_urdf_robot` private methods, and articulation-touching paths under `get_observation` / `send_action` — currently return `status: "success"` without instantiating the underlying USD prim or articulation handle. Following the documented Quick Start on a real Isaac Sim install will therefore observe `get_observation()` returning `{}` for those paths — no exception is raised. The remaining data-plane wiring (articulation construction, sensor / replicator integration) lands in subsequent Phase 2 slices and Phase 3+. Treat the Phase-1 surface as an integration contract for the still-no-op methods; the loaders module + `add_object` / `remove_object` are the working paths for ingestion + scene primitives today, and `render` is the working frame-extraction path once a Phase-2 camera handle exists.

## Installation

Isaac Sim is **not installable from PyPI**. It is an NVIDIA Omniverse Kit application that must be installed separately.

### Option 1: NVIDIA Omniverse Launcher (recommended)

1. Download [NVIDIA Omniverse Launcher](https://developer.nvidia.com/omniverse)
2. Install **Isaac Sim 2024.x** (or newer) from the Exchange tab
3. Install Python dependencies:

```bash
pip install 'strands-robots-sim[isaac]'
```

### Option 2: Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh -i
pip install 'strands-robots-sim[isaac]'
```

### Option 3: Docker

```bash
docker pull nvcr.io/nvidia/isaac-sim:4.5.0
docker run --gpus all -it nvcr.io/nvidia/isaac-sim:4.5.0
# Inside container:
pip install 'strands-robots-sim[isaac]'
```

## Requirements

- NVIDIA GPU (RTX 2070+ or A100/H100 for fleet training)
- CUDA 12.0+
- Isaac Sim 2024.x or newer
- Linux (Ubuntu 22.04+ recommended)
- Python 3.10+

## Quick Start

```python
from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig

# Check availability
available, reason = IsaacSimulation.is_available()
if not available:
    print(f"Isaac Sim not available: {reason}")
    exit(1)

# Create simulation
config = IsaacConfig(
    num_envs=1,
    headless=True,
    render_mode="rtx_realtime",
)
sim = IsaacSimulation(config)

# Create world and add robot
sim.create_world()
sim.add_robot("so100")
sim.add_camera("front_cam", position=[1.0, 0.0, 0.5])

# Step and render
sim.step(100)
result = sim.render("front_cam")
rgb = result["rgb"]  # (H, W, 3) uint8

# Clean up
sim.destroy()
```

## Fleet Training (Multi-Env)

```python
config = IsaacConfig(num_envs=1024, headless=True)
sim = IsaacSimulation(config)
sim.create_world()
sim.add_robot("so100")
sim.replicate(1024)  # 1024 parallel environments

for step in range(10000):
    sim.step(1)
    obs = sim.get_observation("so100")  # batched across all envs
    # ... RL training loop ...

sim.destroy()
```

## Entry-Point Discovery

Isaac Sim registers as a `strands_robots.backends` entry point:

```python
from importlib.metadata import entry_points

for ep in entry_points(group='strands_robots.backends'):
    print(ep.name, '->', ep.value)
# isaac -> strands_robots_sim.isaac.simulation:IsaacSimulation
# isaac_sim -> strands_robots_sim.isaac.simulation:IsaacSimulation
```

## Configuration

### `IsaacConfig` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_envs` | 1 | Number of parallel environments |
| `device` | "cuda:0" | CUDA device |
| `headless` | True | Run without GUI |
| `physics_dt` | 1/120 s | Physics timestep |
| `rendering_dt` | 1/30 s | Rendering timestep |
| `render_mode` | "headless" | "headless", "rtx_realtime", or "rtx_pathtracing" |
| `gravity` | (0, 0, -9.81) | Gravity vector (Z-up) |
| `camera_width` | 640 | Default camera width |
| `camera_height` | 480 | Default camera height |
| `enable_rtx_sensors` | True | Enable RTX-accelerated sensors |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STRANDS_ISAAC_HEADLESS` | - | Override headless mode ("true"/"false") |
| `STRANDS_ISAAC_RTX_PATHTRACING` | - | Enable RTX pathtracing ("true"/"false") |
| `STRANDS_ISAAC_NUCLEUS_URL` | - | Override Omniverse Nucleus server URL |

## Procedural Robots

The following robots can be added without any asset files:

- `so100` (aliases: `so-100`, `so_100`, `so101`) -- 6-DOF tabletop arm
- `panda` (aliases: `franka`, `franka_panda`) -- 7-DOF manipulator
- `unitree_g1` (aliases: `g1`) -- 21-DOF humanoid (simplified). The six 2-DOF compound joints (hips / ankles / shoulder-yaw + elbow on each arm) are split through massless intermediate `*_link` bodies so the kinematic graph is a valid tree by construction.

```python
sim.add_robot("so100")  # procedural, no asset files needed
sim.add_robot("panda")
sim.add_robot("g1", data_config="unitree_g1")
```

Every procedural builder validates the kinematic graph at construction time via `_validate_kinematic_tree`: a robot whose joint set has a duplicate `(parent_body, child_body)` edge fails fast with `ValueError` listing the offending bodies + joint names. Validation is **fail-first by default** with no env-var escape hatch — shipping a knowingly-broken robot has no good use case in this package.

## Loading External Description Files (URDF / MJCF / USD)

The `strands_robots_sim.isaac.loaders` module produces `ProceduralRobot` dataclass instances from existing robot description files, so callers don't have to add a new `_build_*` function for every robot they need. Three formats are supported:

```python
from strands_robots_sim.isaac.loaders import load_urdf, load_mjcf, load_usd

# URDF -- stdlib XML; no third-party deps required.
panda_urdf = load_urdf("/path/to/panda.urdf")

# MJCF (MuJoCo XML) -- stdlib XML; LIBERO scenes, robosuite assets.
panda_mjcf = load_mjcf("/opt/conda/.../robosuite/models/assets/robots/panda/robot.xml")

# USD -- requires `pxr` (ships in the [isaac] extra).
panda_usd = load_usd("/path/to/panda.usda")

# All three return the same dataclass shape.
print(panda_urdf.num_joints, panda_urdf.joint_names)
```

The loaders share failure semantics: missing path raises `FileNotFoundError`, malformed document raises `ValueError` with the offending element + path, and an empty document (zero links / joints / bodies) also raises `ValueError`. Loaders never silently return a phantom robot.

The hardcoded `_build_*` functions in `procedural.py` remain as a zero-dep, testable fallback used when no description file is configured. Loaders layer on top.

The loader module is verified against the seven robosuite-bundled MJCFs that the `strands-robots` LIBERO adapter consumes (`panda` / `iiwa` / `kinova3` / `jaco` / `sawyer` / `ur5e` / `baxter`); the parity tests live in `strands_robots_sim/isaac/tests/test_loaders.py::TestRobosuiteMjcfParity`.

## Comparison with Newton Backend

| Feature | Newton (Warp) | Isaac Sim |
|---------|:---:|:---:|
| Physics parallelism | 4096+ envs | 1024 envs |
| Rendering | OpenGL/null | RTX photorealistic |
| USD native | Partial | Full |
| Sensors (camera, LiDAR) | Basic | RTX GPU-batched |
| Synthetic data gen | No | Replicator |
| Soft body/cloth | Yes (VBD) | Yes (PhysX) |
| Differentiable sim | Yes (Warp tape) | No |
| Install size | ~500MB | ~30GB |
| Use case | Fast RL training | Photorealistic sim2real |

## Architecture

```
strands_robots_sim/isaac/
    __init__.py         PEP 562 lazy exports (zero omni overhead on import)
    _install.py         Single source of truth for Isaac Sim install metadata
                        (docker image tag, Omniverse Launcher hint, Isaac Lab
                        bootstrap) — composes ImportError messages and the
                        is_available() reason string from these constants
    config.py           IsaacConfig dataclass + validation
    simulation.py       IsaacSimulation(SimEngine) -- main backend class
    procedural.py       SO-100 / Panda / G1 builders + kinematic-tree guard
    loaders.py          URDF / MJCF / USD -> ProceduralRobot loaders
    stages.py           USD stage management (Phase 2)
    sensors.py          RTX camera, LiDAR wrappers (Phase 3)
    replicator.py       Domain randomization (Phase 3)
    tests/
        test_unit.py                          Mocked tests (no GPU)
        test_entrypoint.py                    Entry-point + lazy-import surface
        test_get_observation_diagnostic_logs.py   WARNING/DEBUG level pins
        test_procedural_g1_dof.py             G1 DOF-count drift pin
        test_procedural_kinematic_guard.py    Fail-first kinematic-tree pin
        test_loaders.py                       URDF / MJCF / USD round-trip +
                                              robosuite real-asset parity tests
        test_gpu_integ.py                     GPU tests (STRANDS_GPU_TEST=1)
```

## Thread Safety

- All mutable state protected by `threading.RLock`
- `step()` must not run concurrently with `add_robot()`
- `SimulationApp` is a process-wide singleton (never create more than one)
- `destroy()` clears the World but does NOT shut down `SimulationApp`

## Testing

```bash
# Unit tests (no GPU required)
pytest strands_robots_sim/isaac/tests/test_unit.py -v
pytest strands_robots_sim/isaac/tests/test_entrypoint.py -v
pytest strands_robots_sim/isaac/tests/test_loaders.py -v
pytest strands_robots_sim/isaac/tests/test_procedural_g1_dof.py -v
pytest strands_robots_sim/isaac/tests/test_procedural_kinematic_guard.py -v

# Or all of the above in one go (skips the GPU file by default):
pytest strands_robots_sim/isaac/tests/ --ignore=strands_robots_sim/isaac/tests/test_gpu_integ.py

# GPU integration tests (requires Isaac Sim)
STRANDS_GPU_TEST=1 pytest strands_robots_sim/isaac/tests/test_gpu_integ.py -v
```
