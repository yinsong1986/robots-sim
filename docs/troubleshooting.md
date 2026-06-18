# Troubleshooting

Common failures running `strands-robots-sim` on Isaac Sim hosts and how
to diagnose them. The first thing to try in almost every case:

```python
from strands_robots_sim.isaac import IsaacSimulation
print(IsaacSimulation.is_available())
```

`is_available()` returns `(True, None)` on a healthy machine and a
structured `(False, reason)` tuple on every other configuration. The
reason string includes the failing `omni.*` import and the relevant
install hint from `_install.py`.

## `is_available()` returned False

### `Isaac Sim is not installed`

**Symptom:** `is_available()` returns
`(False, "Isaac Sim is not installed. Install via Omniverse Launcher / Isaac Lab / NGC Docker. ...")`

**Cause:** No `omni.*` modules are importable in the active Python.

**Fix:** Follow [Getting Started → Installation](getting-started/installation.md).
Pick one of:

- Omniverse Launcher → install Isaac Sim 4.x → activate `setup_python_env.sh`.
- Isaac Lab → `./isaaclab.sh -i` → `source _isaac_sim/setup_python_env.sh`.
- NGC Docker → `docker run --gpus all nvcr.io/nvidia/isaac-sim:4.5.0`.

`pip install 'strands-robots-sim[isaac]'` alone is **not** sufficient —
Isaac Sim itself is not on PyPI.

### `omni.isaac.core failed to import`

**Symptom:** `is_available()` returns a `(False, ...)` tuple naming
`omni.isaac.core` or one of its submodules as the failing import.

**Cause:** Isaac Sim is partially installed, or you have a Python
environment that doesn't have Isaac Sim's site-packages on
`sys.path`.

**Fix:** Run the script with Isaac Sim's bundled Python:

```bash
~/.local/share/ov/pkg/isaac-sim-4.x/python.sh script.py
# or, inside the docker container:
/isaac-sim/python.sh script.py
```

`source setup_python_env.sh` (Omniverse) or `source
_isaac_sim/setup_python_env.sh` (Isaac Lab) achieves the same effect for
an interactive shell.

### `CUDA driver version is insufficient`

**Symptom:** `(False, "CUDA driver version is insufficient for the CUDA runtime version")`

**Cause:** Isaac Sim 4.5 needs CUDA 12.0+ and a recent NVIDIA driver
(>=535 for Ubuntu 22.04 stock).

**Fix:** Upgrade the NVIDIA driver, **then** the CUDA runtime if needed.
On the dev machine:

```bash
nvidia-smi                                   # check driver version
sudo apt-get install nvidia-driver-535       # or newer
sudo reboot
```

## `omni.*` import errors at runtime

### `ImportError: cannot import name 'SimulationApp' from 'omni.isaac.kit'`

**Cause:** Isaac Sim 4.5 moved `SimulationApp` to the `isaacsim`
namespace (`isaacsim.SimulationApp`). The `strands-robots-sim` package
tries the modern path first and falls back to the legacy
`omni.isaac.kit.SimulationApp`; seeing this error means neither resolved,
which usually points at a partial / pre-4.x Isaac Sim install.

**Fix:** Upgrade to the pinned Isaac Sim 4.5 (`nvcr.io/nvidia/isaac-sim:4.5.0`).

### `ModuleNotFoundError: No module named 'pxr'`

**Cause:** The `usd-core` PyPI wheel is missing. The `[isaac]` extra
pins it, so this usually means you skipped the extra:

```bash
pip install 'strands-robots-sim[isaac]'      # not just 'strands-robots-sim'
```

### `RuntimeError: SimulationApp already initialized`

**Cause:** You created two `IsaacSimulation` instances trying to use
different `SimulationApp` configurations (e.g. one `headless=True`, one
`headless=False`).

**Fix:** `SimulationApp` is process-wide. Use one configuration per
process; spawn subprocesses for distinct configs:

```python
import multiprocessing as mp

from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig

def child(headless: bool):
    sim = IsaacSimulation(IsaacConfig(headless=headless))
    ...

mp.set_start_method("spawn")
mp.Process(target=child, args=(True,)).start()
mp.Process(target=child, args=(False,)).start()
```

`destroy()` clears the world but intentionally does **not** tear down
`SimulationApp` — Kit cannot re-bootstrap inside a single process.

## GPU / driver problems

### `RuntimeError: PhysX failed to initialize`

**Cause:** Usually a driver mismatch or no CUDA-capable device on the
host.

**Fix:**

```bash
nvidia-smi                                   # confirm GPU + driver
echo $CUDA_VISIBLE_DEVICES                   # confirm not masked out
```

If `nvidia-smi` itself fails, the driver isn't loaded — fix that first.

### `Vulkan loader: failed to find vulkan-1`

**Cause:** Isaac Sim's RTX path needs Vulkan 1.3+ on the host.

**Fix:**

```bash
sudo apt-get install -y libvulkan1 vulkan-tools
vulkaninfo --summary                          # should print the GPU
```

## Rendering returns blank frames

`render(camera_name="...")` returning all-black is **expected** in three
cases (the engine returns blank frames intentionally rather than crashing):

1. **`headless` render mode.** The RTX pipeline is disabled — switch to
   `rtx_realtime` or `rtx_pathtracing`.
2. **No `add_camera(...)` call.** The `"default"` camera does not exist
   until you create it. Add an explicit camera:

   ```python
   sim.add_camera(name="front", position=[1.2, 0, 0.6], target=[0, 0, 0.1])
   frame = sim.render(camera_name="front")
   ```

3. **Camera with no RTX handle attached.** Common when a camera was
   added before `create_world()` finished asynchronously. Step the
   world a few ticks before reading frames:

   ```python
   sim.step(5)
   frame = sim.render(camera_name="front")
   ```

If none of these apply and you still get black frames, dump the camera
state:

```python
print(sim.get_state()["cameras"])
# [{"name": "front", "rtx_handle": "<...>", "ready": True/False, ...}]
```

`ready: False` means RTX is still initializing — give it more frames.

## Adapter / LIBERO failures

### `ValueError: procedural builder is not LIBERO-compatible`

**Cause:** You called `sim.add_robot("panda")` (the procedural Panda
builder) and then tried to run a LIBERO benchmark against it. Procedural
builders are kinematically approximate stick figures, and LIBERO policies
expect the real Franka kinematics + masses.

**Fix:** Load the real Franka USD or URDF instead:

```python
sim.add_robot(name="panda", usd_path="/path/to/franka.usda")
# or:
sim.add_robot(name="panda", urdf_path="/path/to/panda.urdf")
```

The example drivers in `examples/libero/run_isaac.py` already do this —
copy their default asset path or pass `--robot-usd PATH`.

### `success_rate = 0.0` in `examples/libero/run_isaac_agent.py`

**Cause:** Procedural robots don't construct an `Articulation` handle, so
a LIBERO eval driven against a procedural robot reads zero joint state.
The lifecycle (`create_world` → `add_robot(usd_path=...)` → `add_camera`
→ `step` → `render`) is validated end-to-end on Isaac Sim 4.5 (see
[PR #74](https://github.com/strands-labs/robots-sim/pull/74)); the
remaining gap is that `evaluate_benchmark` reaches the LIBERO suite
loader under Isaac Sim's bundled Python before it can score episodes.
Track the rollout in the umbrella
[`#8`](https://github.com/strands-labs/robots-sim/issues/8).

**Workaround:** Load the real Franka USD/URDF (not a procedural robot),
and use `examples/libero/run_isaac.py` (programmatic) for matrix-quality
numbers; the agent file is for demos.

## Where to file bugs

- New issue: <https://github.com/strands-labs/robots-sim/issues/new>.
- Include the output of `IsaacSimulation.is_available()`, the
  `nvidia-smi` summary, the Isaac Sim version, and the failing snippet.
- For policy / inference issues (GR00T container lifecycle, ZMQ wire
  format), file against `strands-labs/robots` instead — that's where the
  policy providers live.
