# API Reference

Hand-curated reference for the public surface of `strands-robots-sim`.
For the upstream `Simulation` AgentTool, `SimEngine` ABC, and policy
provider classes, see the
[`strands-robots` API docs](https://strands-labs.github.io/robots/api-reference/).

## Module layout

```
strands_robots_sim/
├── __init__.py            # PEP 562 lazy exports
└── isaac/
    ├── __init__.py        # IsaacConfig, IsaacSimulation lazy exports
    ├── _install.py        # Single source of truth for install metadata
    ├── config.py          # IsaacConfig dataclass + validation
    ├── simulation.py      # IsaacSimulation(SimEngine) -- main backend
    ├── procedural.py      # SO-100 / Panda / G1 builders + tree validator
    ├── loaders.py         # URDF / MJCF / USD -> ProceduralRobot
    └── tests/             # unit + GPU-integration tests
```

## `IsaacConfig`

```python
from strands_robots_sim.isaac import IsaacConfig
```

Dataclass owning all simulation-wide configuration. Constructor signature
(see [Simulation → Overview](simulation/overview.md) for the parameter
matrix):

```python
IsaacConfig(
    num_envs: int = 1,
    device: str = "cuda:0",
    headless: bool = True,
    physics_dt: float = 1.0 / 120.0,
    rendering_dt: float = 1.0 / 30.0,
    render_mode: str = "headless",          # "headless" / "rtx_realtime" / "rtx_pathtracing"
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ground_plane: bool = True,
    stage_path: str = "/World",
    nucleus_url: str | None = None,
    camera_width: int = 640,
    camera_height: int = 480,
    enable_rtx_sensors: bool = True,
    verbose: bool = False,
    extra: dict[str, Any] = field(default_factory=dict),
)
```

Validation runs in `__post_init__`:

- `render_mode` must be one of `RENDER_MODES = ("headless", "rtx_realtime", "rtx_pathtracing")`.
- `num_envs >= 1`.
- `device` must look like `"cuda:N"` or be a string that
  `torch.device(...)` would accept.
- `physics_dt > 0` and `rendering_dt > 0`.

## `IsaacSimulation`

```python
from strands_robots_sim.isaac import IsaacSimulation
```

`SimEngine` subclass. Methods are split into lifecycle, scene authoring,
physics / observation, and rendering.

### Static / class methods

```python
IsaacSimulation.is_available() -> tuple[bool, str | None]
```

Pre-flight check. Returns `(True, None)` on a healthy machine, or
`(False, reason)` on a CPU-only / non-Isaac box. Critically, this does
**not** import `omni.*` — call it before constructing `IsaacSimulation` if
you want to fall back gracefully.

### Lifecycle

```python
IsaacSimulation(config: IsaacConfig | None = None, **kwargs: Any)
```

Boots the process-wide `SimulationApp` singleton. `**kwargs` are forwarded
into `IsaacConfig(...)` if `config` is `None`.

```python
sim.create_world(**kwargs) -> dict
sim.destroy() -> dict
sim.cleanup() -> None                       # idempotent; tears SimulationApp down
sim.reset(env_ids: list[int] | None = None) -> dict
sim.step(n_steps: int = 1) -> dict
sim.get_state() -> dict
```

`__enter__` / `__exit__` are wired to `cleanup()` so `with
IsaacSimulation(...) as sim:` works.

### Scene authoring

```python
sim.add_robot(
    name: str,
    *,
    usd_path: str | None = None,
    urdf_path: str | None = None,
    position: list[float] | None = None,
    data_config: str | None = None,
    **kwargs,
) -> dict

sim.remove_robot(name: str) -> dict
sim.list_robots() -> list[str]
sim.robot_joint_names(robot_name: str) -> list[str]

sim.add_object(
    name: str,
    *,
    shape: str,                             # "cuboid" / "sphere" / "cylinder" / "capsule"
    position: list[float],
    scale: list[float] | None = None,
    radius: float | None = None,
    mass: float | None = None,
    is_static: bool = False,
    color: list[float] | None = None,
) -> dict

sim.remove_object(name: str) -> dict

sim.add_camera(
    name: str,
    *,
    position: list[float],
    target: list[float] | None = None,
    width: int | None = None,
    height: int | None = None,
    horizontal_aperture_mm: float | None = None,
    focal_length_mm: float | None = None,
) -> dict
```

See [Simulation → World Building](simulation/world-building.md) for
worked examples.

### Physics + observation

```python
sim.send_action(
    action: dict[str, float] | list[float],
    robot_name: str | None = None,
    n_substeps: int = 1,
) -> dict

sim.get_observation(
    robot_name: str | None = None,
    *,
    skip_images: bool = False,
) -> dict
```

`action` accepts a dict keyed by joint name or a flat list / array in
`robot_joint_names(robot_name)` order. `skip_images=True` skips camera
rendering when only joint state matters.

### Rendering

```python
sim.render(
    camera_name: str = "default",
    width: int | None = None,
    height: int | None = None,
) -> dict
```

Returns `{"rgb": ndarray (H, W, 3) uint8, "depth": ndarray (H, W) float32, ...}`
when a camera is attached and `render_mode != "headless"`. Returns blank
frames otherwise (headless / no camera attached) so calling code does not
have to special-case the no-render path.

## `ProceduralRobot` and the loaders

```python
from strands_robots_sim.isaac.loaders import load_urdf, load_mjcf, load_usd
```

All three return the same dataclass shape:

```python
@dataclass
class ProceduralRobot:
    name: str
    bodies: list[Body]
    joints: list[Joint]
    base_link: str

    @property
    def num_joints(self) -> int: ...

    @property
    def joint_names(self) -> list[str]: ...
```

Failure semantics are uniform across loaders:

| Condition | Exception |
|---|---|
| Path does not exist | `FileNotFoundError` |
| Document fails to parse | `ValueError` (with element + path) |
| Empty document (no links / joints / bodies) | `ValueError` |

The MJCF loader is verified against the seven robosuite-bundled MJCFs the
upstream LIBERO adapter consumes (`panda` / `iiwa` / `kinova3` / `jaco` /
`sawyer` / `ur5e` / `baxter`).

## Procedural builders

```python
from strands_robots_sim.isaac.procedural import (
    ProceduralRobot,
    build_so100,
    build_panda,
    build_unitree_g1,
    _validate_kinematic_tree,                # public-ish: used by tests
)
```

The builders construct `ProceduralRobot` instances without any asset
files. `_validate_kinematic_tree(robot)` raises `ValueError` if the joint
graph has a duplicate `(parent_body, child_body)` edge — the validator
runs at every builder's construction, fail-first.

## `_install.py` constants

```python
from strands_robots_sim.isaac._install import (
    DOCKER_IMAGE_TAG,                       # "nvcr.io/nvidia/isaac-sim:4.5.0"
    OMNIVERSE_LAUNCHER_HINT,
    ISAAC_LAB_BOOTSTRAP_HINT,
    INSTALL_HELP,                           # full multiline help string
)
```

These are the single source of truth for the `is_available()` reason
string and any `ImportError` rendered to the user. Updating Isaac Sim
versions = update one constant.

## Entry-point registration

```toml
# pyproject.toml
[project.entry-points."strands_robots.backends"]
isaac     = "strands_robots_sim.isaac.simulation:IsaacSimulation"
isaac_sim = "strands_robots_sim.isaac.simulation:IsaacSimulation"
```

Both names resolve to the same class. `create_simulation("isaac", ...)`
upstream does:

```python
import importlib.metadata
ep = next(e for e in importlib.metadata.entry_points(group="strands_robots.backends")
          if e.name == "isaac")
cls = ep.load()
return cls(**kwargs)
```

## See also

- [Architecture](architecture.md) — the plugin contract this surface implements.
- [Simulation → Overview](simulation/overview.md) — config + lifecycle in plain English.
- [Backends → Isaac Sim](backends/isaac.md) — the full backend reference,
  including procedural builders and tests.
