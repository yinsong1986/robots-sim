# Quickstart

Bring up an Isaac Sim world, drop a robot in, render an RTX frame.

## Prerequisites

- Isaac Sim installed and verified (see [Installation](installation.md)).
- `strands-robots-sim[isaac]` installed in the same Python environment.

## Hello, RTX

```python
from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig

sim = IsaacSimulation(IsaacConfig(
    render_mode="rtx_realtime",                # or "rtx_pathtracing" for path-traced
    headless=True,
))
sim.create_world()
sim.add_robot("so100")                         # procedural builder, no asset files
sim.add_object(name="cube", shape="cuboid",
               position=[0.4, 0.0, 0.05], scale=[0.05, 0.05, 0.05])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.0, 0.0, 0.1])

sim.step(120)                                  # ~1 s of sim time
frame = sim.render(camera_name="front")        # {"rgb": (H, W, 3) uint8, "depth": ...}

sim.destroy()
```

!!! note "`create_simulation('isaac')` is not wired up yet"

    `IsaacSimulation` is registered as a `strands_robots.backends` entry
    point, but the released `strands-robots` floor (`>=0.3.8,<0.4`) does not
    walk that group from `create_simulation`, so
    `create_simulation("isaac")` raises
    `ValueError: Unknown simulation backend: 'isaac'`. Construct
    `IsaacSimulation(IsaacConfig(...))` directly until the upstream walker
    ships ([`strands-labs/robots#131`](https://github.com/strands-labs/robots/issues/131));
    the kwargs are identical, forwarded into `IsaacConfig`.

What happened:

1. `IsaacSimulation(IsaacConfig(...))` constructs the backend directly — the
   supported path until `create_simulation("isaac")` resolves through the
   `strands_robots.backends` entry point.
2. `create_world()` spins up a `SimulationApp`, opens a USD stage, and
   adds a ground plane.
3. `add_robot("so100")` runs the procedural SO-100 builder — no asset
   files needed, no Nucleus required.
4. `add_object(...)` and `add_camera(...)` author scene primitives via
   the `omni.isaac.core` API.
5. `step(120)` advances PhysX 120 substeps.
6. `render(...)` returns an RGBA frame plus depth from the configured RTX
   sensor.

## Bring your own robot asset

The procedural builders ship a kinematically-approximate SO-100 / Panda /
G1 — useful for smoke tests, not enough for LIBERO-style manipulation. Use
the loaders / `add_robot(usd_path=...)` for real robots:

```python
sim.add_robot(name="panda", usd_path="/path/to/panda.usda")
# or:
sim.add_robot(name="panda", urdf_path="/path/to/panda.urdf")
```

For pre-loading description files into a `ProceduralRobot` dataclass
(useful when introspecting joint counts before adding the robot):

```python
from strands_robots_sim.isaac.loaders import load_urdf, load_mjcf, load_usd

panda = load_urdf("/path/to/panda.urdf")
print(panda.num_joints, panda.joint_names)
# 7 ['panda_joint1', 'panda_joint2', ...]
```

See [Simulation → World Building](../simulation/world-building.md) for the
full `add_robot` / `add_object` / `add_camera` reference.

## Run a benchmark

`strands-robots-sim` ships two end-to-end LIBERO drivers under
`examples/libero/`:

| Driver | Purpose |
|---|---|
| `examples/libero/run_isaac.py` | Programmatic — calls `evaluate_benchmark(...)` directly. CI / matrix-table input. |
| `examples/libero/run_isaac_agent.py` | Strands `Agent` + natural language — invokes the same eval through one `agent("...")` call. |

```bash
# Smoke test (mock policy, no GPU policy server):
python examples/libero/run_isaac.py --policy mock --n-episodes 5

# Bring your own robot asset:
python examples/libero/run_isaac.py --policy mock --robot-usd /path/to/robot.usd

# Real eval against an NVIDIA GR00T checkpoint (auto-orchestrates the GR00T
# Docker container; pass --no-auto-server to reuse one):
python examples/libero/run_isaac.py --policy groot --port 8000 --n-episodes 50
```

Both files print two grep-stable lines that the flagship
`libero_backend_matrix.py` driver subprocess-and-parses for the
side-by-side table:

```
benchmark_name=libero-spatial-pick_up_the_red_cube
policy=groot task=libero-spatial-pick_up_the_red_cube success_rate=1.00 wall_time=44.3s
```

See [Examples → Overview](../examples/overview.md) for the full driver
matrix and the LIBERO-specific gotchas.

## Driving from a Strands Agent

The same `IsaacSimulation` instance plugs into the upstream `Simulation`
AgentTool — call it with a method name from natural language, get back a
`{"status", "content"}` payload:

```python
from strands import Agent
from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig

sim = IsaacSimulation(IsaacConfig(render_mode="rtx_realtime", headless=True))
sim.create_world()
sim.add_robot("so100")

agent = Agent(tools=[sim])
agent("Add a top-down camera at z=1.5 looking at the origin, "
      "step 100 frames, then render it")
```

The agent picks the right `add_camera`, `step`, and `render` actions out
of the AgentTool's full action set; you never have to teach it the Isaac
API explicitly.

## Next

- [Architecture](../architecture.md) — how the plugin contract works.
- [Simulation → World Building](../simulation/world-building.md) — the
  full `add_robot` / `add_object` / `add_camera` reference.
- [Simulation → Domain Randomization](../simulation/domain-randomization.md) —
  Replicator synth-data pipeline.
- [Examples → Overview](../examples/overview.md) — runnable LIBERO drivers.
