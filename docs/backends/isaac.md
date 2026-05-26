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


> **Phase 1 status (skeleton).** This release ships the SimEngine-shaped surface and lazy-import scaffolding only. Several methods on `IsaacSimulation` (`add_robot` on the procedural branch, `_load_usd_robot`, `_load_urdf_robot`, `add_object`, `add_camera`, `replicate`) currently return `status: "success"` without instantiating the underlying USD prim or articulation handle. Following the documented Quick Start on a real Isaac Sim install will therefore observe `get_observation()` returning `{}` and `render()` returning blank frames -- no exception is raised. The data-plane wiring (USD stage management, articulation construction, sensor / replicator integration) lands in Phase 2 and later. Treat the Phase-1 surface as an integration contract, not as a working physics path.

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
- `unitree_g1` (aliases: `g1`) -- 21-DOF humanoid (simplified)

```python
sim.add_robot("so100")  # procedural, no asset files needed
sim.add_robot("panda")
sim.add_robot("g1", data_config="unitree_g1")
```

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
    config.py           IsaacConfig dataclass
    simulation.py       IsaacSimulation(SimEngine) -- main backend class
    procedural.py       SO-100 / Panda / G1 USD prim builders
    stages.py           USD stage management (Phase 2)
    sensors.py          RTX camera, LiDAR wrappers (Phase 3)
    replicator.py       Domain randomization (Phase 3)
    tests/
        test_unit.py         Mocked tests (no GPU)
        test_entrypoint.py   Entry-point verification
        test_gpu_integ.py    GPU tests (STRANDS_GPU_TEST=1)
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

# GPU integration tests (requires Isaac Sim)
STRANDS_GPU_TEST=1 pytest strands_robots_sim/isaac/tests/test_gpu_integ.py -v
```
