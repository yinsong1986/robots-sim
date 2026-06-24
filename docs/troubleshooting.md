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

- Omniverse Launcher → install Isaac Sim 6.0 → activate `setup_python_env.sh`.
- Isaac Lab → `./isaaclab.sh -i` → `source _isaac_sim/setup_python_env.sh`.
- NGC Docker → `docker run --gpus all nvcr.io/nvidia/isaac-sim:6.0`.

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
~/.local/share/ov/pkg/isaac-sim-6.0/python.sh script.py
# or, inside the docker container:
/isaac-sim/python.sh script.py
```

`source setup_python_env.sh` (Omniverse) or `source
_isaac_sim/setup_python_env.sh` (Isaac Lab) achieves the same effect for
an interactive shell.

### `CUDA driver version is insufficient`

**Symptom:** `(False, "CUDA driver version is insufficient for the CUDA runtime version")`

**Cause:** Isaac Sim 6.0 needs CUDA 12.0+ and a recent NVIDIA driver
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

**Cause:** Isaac Sim 6.0 serves `SimulationApp` from the `isaacsim`
namespace (`isaacsim.SimulationApp`). The `strands-robots-sim` package
tries the modern path first and falls back to the legacy
`omni.isaac.kit.SimulationApp`; seeing this error means neither resolved,
which usually points at a partial / pre-6.0 Isaac Sim install.

**Fix:** Upgrade to the pinned Isaac Sim 6.0 (`nvcr.io/nvidia/isaac-sim:6.0`).

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

If none of these apply and you still get black frames, check how many
cameras the simulation thinks it has via the state envelope:

```python
state = sim.get_state()["content"][0]["json"]
print(state["num_cameras"])
# e.g. 1  (the "front" camera you added above)
```

`get_state()` returns the standard
`{"status", "content": [{"text", "json": {...}}]}` envelope; the `json`
payload carries scalar counts (`num_cameras`, `num_robots`, `num_objects`,
…), not a per-camera list. A `num_cameras` of `0` means no camera was ever
added — call `add_camera(...)` first.

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
→ `step` → `render`) is validated end-to-end on Isaac Sim 6.0 (see
[PR #74](https://github.com/strands-labs/robots-sim/pull/74)); the
remaining gap is that `evaluate_benchmark` reaches the LIBERO suite
loader under Isaac Sim's bundled Python before it can score episodes.
Track the rollout in the umbrella
[`#8`](https://github.com/strands-labs/robots-sim/issues/8).

**Workaround:** Load the real Franka USD/URDF (not a procedural robot),
and use `examples/libero/run_isaac.py` (programmatic) for matrix-quality
numbers; the agent file is for demos.

## `--policy groot` eval failures

Running the LIBERO examples with `--policy groot`
(`examples/libero/run_mujoco.py`, `run_mujoco_agent.py`) brings up the
GR00T-N1.7 VLA in the `nvcr.io/nvidia/isaac-gr00t` container. Three
setup gotchas trip up first-time runs — the first two fail loudly, the
third silently zeroes the score. See the
[`--policy groot` prerequisites](https://github.com/strands-labs/robots-sim/blob/main/examples/README.md#-policy-groot-prerequisites--gotchas)
in the examples README for the overview.

### `refusing to mount … under protected host path '/home'`

**Symptom:** `start_container` aborts during the GR00T lifecycle with a
message about refusing to bind-mount a path under `/home`.

**Cause:** `gr00t_inference` refuses to bind-mount any checkpoint cache
under `/home` (a "protected host path" guard). The historical default
checkpoint location lived under `/home`, so the two defaults were
mutually inconsistent
([#125](https://github.com/strands-labs/robots-sim/issues/125)).

**Fix:** The example drivers now default to a non-`/home` cache
(fixed in [#126](https://github.com/strands-labs/robots-sim/pull/126)),
so the OOTB run works. If you override the location, keep it off
`/home`:

```bash
python examples/libero/run_mujoco.py --policy groot --checkpoint-dir /tmp/groot-ck
# or, equivalently:
export STRANDS_ROBOTS_CHECKPOINT_DIR=/tmp/groot-ck
```

### `ImportError: 'zmq' is required for GR00T service inference`

**Symptom:** The eval fails *after* the GR00T model loads (so the
container is up and the checkpoint downloaded) with this `ImportError`.

**Cause:** The GR00T ZMQ client needs `pyzmq`, which the policy extra
doesn't always pull in.

**Fix:**

```bash
pip install pyzmq
```

### `success_rate = 0.0` with a buried `numba` / `coverage` warning

**Symptom:** A `--policy groot` run completes with `success_rate=0` even
though the container, checkpoint download, and ZMQ service all came up
cleanly. The only clue is a buried `WARNING` about the OSC controller
failing to install, preceded by a `numba` import error.

**Cause:** When both `numba` and `coverage>=7` are installed in the
eval environment, `import numba` fails because
`numba/misc/coverage_support.py` subclasses the removed
`coverage.types.Tracer`. That makes the LIBERO adapter's OSC controller
fail to install, so **GR00T's actions silently no-op** — the policy is
fine, but no torques reach the robot, so every episode fails. It's easy
to misread this as a bad policy.

**Fix:** Remove the conflicting `coverage` (or pin `coverage<7`) in the
eval environment:

```bash
pip uninstall coverage      # or: pip install 'coverage<7'
```

The same run then returns `success_rate=1.00`. The silent-degrade
behaviour (no hard error, just a buried warning) is tracked upstream at
[`strands-labs/robots#522`](https://github.com/strands-labs/robots/issues/522).

## Where to file bugs

- New issue: <https://github.com/strands-labs/robots-sim/issues/new>.
- Include the output of `IsaacSimulation.is_available()`, the
  `nvidia-smi` summary, the Isaac Sim version, and the failing snippet.
- For policy / inference issues (GR00T container lifecycle, ZMQ wire
  format), file against `strands-labs/robots` instead — that's where the
  policy providers live.
