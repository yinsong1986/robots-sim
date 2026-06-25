# World building

`IsaacSimulation` exposes the same five-call shape MuJoCo does:
`create_world` → `add_robot` → `add_object` → `add_camera` → `step` →
`render`. The signatures are deliberately MuJoCo-compatible so a scene
authored against the upstream backend ports here without code changes.

## `create_world(...)`

Spins up a USD stage and (by default) drops a ground plane:

```python
sim.create_world()
sim.create_world(physics_dt=1.0 / 240.0, ground_plane=False)
```

Kwargs override the matching `IsaacConfig` fields for this world only;
they don't mutate the config. Calling `create_world()` twice is an error
— `destroy()` first.

## `add_robot(...)`

Three branches, picked by which kwargs you pass.

### Branch 1 — procedural builder (no asset files)

```python
sim.add_robot("so100")                       # 6-DOF tabletop arm
sim.add_robot("panda", position=[0.5, 0, 0]) # 7-DOF Franka
sim.add_robot("g1", data_config="unitree_g1") # 21-DOF humanoid
```

| Name | Aliases | DOF | Use for |
|---|---|---|---|
| `so100` | `so-100`, `so_100`, `so101` | 6 | LIBERO baseline tabletop |
| `panda` | `franka`, `franka_panda` | 7 | Manipulation primitives |
| `unitree_g1` | `g1` | 21 | Humanoid locomotion / locomanip |

Each builder validates the kinematic graph at construction with
`_validate_kinematic_tree`: a robot whose joint set has a duplicate
`(parent_body, child_body)` edge fails fast with `ValueError` listing the
offending bodies + joint names. There is no env-var escape hatch —
shipping a knowingly-broken robot has no good use case.

The procedural builders are **kinematically approximate** stick figures.
They are unsuitable as drop-in LIBERO substitutes and are deliberately
documented as such — for LIBERO eval, use the URDF / USD branches below.

### Branch 2 — USD asset (Nucleus-native)

```python
sim.add_robot(name="panda", usd_path="/path/to/franka.usda")
sim.add_robot(name="panda", usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/.../franka.usda")
```

The simulation parses the USD prim, instantiates an
`isaacsim.core.api.articulations.Articulation`, and wires `send_action` /
`get_observation` against its joint set. This is the **recommended path**
for LIBERO and most production scenes.

### Branch 3 — URDF asset (auto-converted to USD)

```python
sim.add_robot(name="panda", urdf_path="/path/to/panda.urdf")
```

Conversion runs through the direct `isaacsim.asset.importer.urdf`
interface (no detour through the legacy `omni.importer.urdf` extension).
The converted USD is cached per session.

### Loaders — introspect first, add second

Pre-load a description file into a `ProceduralRobot` dataclass before
adding it. Useful for branching on joint count / joint names without
booting the full `IsaacSimulation`:

```python
from strands_robots_sim.isaac.loaders import load_urdf, load_mjcf, load_usd

panda = load_urdf("/path/to/panda.urdf")        # stdlib XML, no deps
panda = load_mjcf("/path/to/robot.xml")          # robosuite / LIBERO MJCF
panda = load_usd("/path/to/panda.usda")          # needs `pxr` from [isaac]

print(panda.num_joints, panda.joint_names)
```

Failure semantics are uniform across all three:

- Missing path → `FileNotFoundError`.
- Malformed document → `ValueError` with the offending element + path.
- Empty document (zero links / joints / bodies) → `ValueError`.

Loaders never silently return a phantom robot.

The MJCF loader is verified against the seven robosuite-bundled MJCFs the
upstream LIBERO adapter consumes (`panda` / `iiwa` / `kinova3` / `jaco` /
`sawyer` / `ur5e` / `baxter`); parity tests live in
`strands_robots_sim/isaac/tests/test_loaders.py::TestRobosuiteMjcfParity`.

## `add_object(...)`

Authors a USD primitive and wraps it in
`isaacsim.core.api.objects.{Dynamic,Fixed}{Cuboid,Sphere,Cylinder,Capsule}`:

```python
sim.add_object(name="cube", shape="cuboid",
               position=[0.4, 0.0, 0.05],
               scale=[0.05, 0.05, 0.05],
               mass=0.1)

sim.add_object(name="ball", shape="sphere",
               position=[0.0, 0.4, 0.05], size=[0.04])

sim.add_object(name="wall", shape="cuboid",
               position=[0.0, 0.5, 0.5],
               scale=[1.0, 0.01, 1.0], is_static=True)
```

`shape` accepts `"box"` (canonical), `"sphere"`, `"cylinder"`,
`"capsule"`, and `"cuboid"` (an alias for `"box"` that mirrors Isaac's
`DynamicCuboid` class name). Dimensions are passed via `size=`; `scale=`
is accepted as an alias for `size=`. The result `json` always reports the
canonical `"box"`. Per-shape `size` conventions: box `[w, h, d]`, sphere
`[radius]`, cylinder / capsule `[radius, height]`.

| Shape | `isaacsim.core.api.objects` |
|---|---|
| `"box"` / `"cuboid"` | `DynamicCuboid` / `FixedCuboid` |
| `"sphere"` | `DynamicSphere` / `FixedSphere` |
| `"cylinder"` | `DynamicCylinder` / `FixedCylinder` |
| `"capsule"` | `DynamicCapsule` / `FixedCapsule` |

`is_static=True` produces a `Fixed*`; otherwise `Dynamic*`. Static
objects collide but don't move — use them for tables, walls, fixtures.

## `add_camera(...)`

```python
sim.add_camera(
    name="front",
    position=[1.2, 0.0, 0.6],
    target=[0.0, 0.0, 0.1],
    width=1280, height=720,
    fov=60.0,    # horizontal field of view, degrees
)
```

A camera is an `isaacsim.sensors.camera.Camera` prim with look-at + FOV
wired. The `fov` (horizontal field of view, in degrees) is mapped onto
the camera's focal length using the standard pinhole relation against the
USD-default 24 mm horizontal aperture — there is no separate
`horizontal_aperture_mm` / `focal_length_mm` knob.
`render(camera_name="front")` returns a dict with RGB, depth, and (if
enabled) ground-truth segmentation:

```python
frame = sim.render(camera_name="front")
rgb   = frame["rgb"]      # (H, W, 3) uint8
depth = frame["depth"]    # (H, W) float32, meters
```

In `headless` mode `render(...)` returns a blank frame so calling code
doesn't have to special-case the no-render path. The frame-extraction
path is real (`get_rgba` / `get_depth` against the camera's RTX handle)
in `rtx_realtime` and `rtx_pathtracing` modes.

In an RTX render mode, `add_camera` **warms up** the new camera before it
returns: Isaac's RTX pipeline does not accumulate a frame until the world is
stepped with rendering enabled, so a brand-new camera's first `get_rgba()`
returns a malformed / empty buffer (shape `(0,)`). `add_camera` steps the
world up to `STRANDS_ISAAC_CAMERA_WARMUP_STEPS` times (default 10) and
returns as soon as the camera yields a valid frame, so the first
`render(...)` / recording capture after `add_camera` sees a real frame
instead of dropping the opening frames of a rollout. The warm-up is skipped
in `headless` mode (no RTX frames are produced) and can be disabled with
`STRANDS_ISAAC_CAMERA_WARMUP_STEPS=0`. It is best-effort: a step failure is
logged and never fails `add_camera` (the camera is still registered).

## Stepping the world

```python
sim.step(120)                                # 120 substeps at physics_dt
sim.step(1)                                  # one tick (RL inner loop)
```

`step()` advances PhysX `n_steps` times. Render cadence is independent —
the renderer ticks at `IsaacConfig.rendering_dt`, decoupled from physics.

## Sending actions / reading state

```python
joint_names = sim.robot_joint_names("panda")
print(joint_names)
# ['panda_joint1', 'panda_joint2', ..., 'panda_finger_joint1', ...]

# Send a joint-position action:
action = {name: 0.0 for name in joint_names}
sim.send_action(action, robot_name="panda")
sim.step(1)

# Read observation back:
obs = sim.get_observation(robot_name="panda")
# obs == {"panda_joint1": 0.0, "panda_joint2": 0.0, ..., "panda_finger_joint1": 0.0}
# i.e. a flat dict of joint-position floats keyed by joint name (in
# `robot_joint_names(robot_name)` order). Index it as obs[joint_name].
```

`get_observation` returns a **flat dict** of joint positions keyed by
joint name — `obs[joint_name]` is the current position float. It is
**not** nested under `"joint_positions"` / `"joint_velocities"`. The
dict is empty (`{}`) before `create_world()`, for an unknown /
ambiguous `robot_name`, or when the robot's articulation handle is not
yet initialised (each case is logged for diagnosis).

`send_action` accepts either a dict keyed by joint name or a flat list /
array in `joint_names` order. On Isaac Sim 6.0 it drives PD position
targets via the articulation's
`apply_action(ArticulationAction(joint_positions=...))`.
`get_observation(skip_images=True)` skips camera rendering when only the
joint state matters (10x speedup in mock-policy smoke loops).

## Footguns

- **Planes are static.** `add_object(shape="cuboid", is_static=False)`
  with a near-zero z-thickness will sink into the ground plane;
  pass `is_static=True` instead.
- **Aim cameras.** `target == position` errors. Pass a distinct
  look-at point.
- **Name collisions.** Robots, objects, cameras share one name table.
  `ValueError` on duplicates.
- **`step()` while `send_action` is in flight blocks.** All
  state-mutating calls hold the same `RLock`. This is by design.
- **Procedural robots aren't drop-in for LIBERO.** Load a real Panda
  USD / URDF for any manipulation eval — the procedural builder is for
  smoke tests / kinematics validation only.

## Next

- [Domain Randomization](domain-randomization.md) — Replicator pipeline.
- [Backends → Isaac Sim](../backends/isaac.md) — full backend reference.
- [API Reference](../api-reference.md) — exact class signatures.
