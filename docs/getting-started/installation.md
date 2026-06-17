# Installation

`strands-robots-sim` ships **two** install layers:

1. **Isaac Sim itself** — the Omniverse Kit application + Python runtime.
   Not on PyPI; install once per machine via Omniverse Launcher, Isaac Lab,
   or the NGC Docker image.
2. **The `strands-robots-sim` Python package** — a thin pip-installable
   plugin that registers `IsaacSimulation` as a `strands_robots.backends`
   entry point and adds the pip-installable Isaac Sim companion deps
   (`isaacsim`, `isaaclab`, and `usd-core` for USD authoring).

Order matters: install Isaac Sim **first**, then `pip install
'strands-robots-sim[isaac]'` into the same Python environment.

## System requirements

- **GPU** — NVIDIA RTX 2070+ (RTX 3090 / A100 / L4 / H100 for fleet training).
- **OS** — Ubuntu 22.04+ (Linux only; macOS / Apple Silicon are not
  supported by Isaac Sim).
- **CUDA** — 12.0+, with a recent NVIDIA driver matching the Isaac Sim release.
- **Python** — 3.10, 3.11, or 3.12 (Isaac Sim 4.5 ships its own 3.10
  embedded interpreter; mirror that version in your venv).
- **Disk** — ~30 GB for the Isaac Sim SDK on first run, plus ~5 GB for
  the cached USD assets.

If you only need fast iteration / Apple Silicon / CI smoke tests, install
[`strands-robots`](https://github.com/strands-labs/robots) directly and use
the MuJoCo backend; the agent contract is the same.

## Step 1 — install Isaac Sim

Pick one of three paths.

=== "Omniverse Launcher (recommended)"

    1. Download the [NVIDIA Omniverse Launcher](https://developer.nvidia.com/omniverse).
    2. Sign in with an NVIDIA developer account.
    3. From the **Exchange** tab, install **Isaac Sim 4.x**.
    4. Optionally launch the app once to confirm it boots.

    The launcher places Isaac Sim under `~/.local/share/ov/pkg/isaac-sim-4.x/`.

=== "Isaac Lab"

    [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) bundles Isaac Sim
    plus the IsaacLab fleet-RL framework:

    ```bash
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    ./isaaclab.sh -i              # downloads Isaac Sim + sets up the venv
    source _isaac_sim/setup_python_env.sh
    ```

    Pick this if you also want IsaacLab's vectorized envs / RL recipes.

=== "Docker (NGC)"

    The NVIDIA-published Isaac Sim image is the lowest-friction option for
    cloud / CI runners:

    ```bash
    docker pull nvcr.io/nvidia/isaac-sim:4.5.0
    docker run --gpus all -it --rm \
        -e ACCEPT_EULA=Y \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        nvcr.io/nvidia/isaac-sim:4.5.0
    ```

    Inside the container, Isaac Sim is at `/isaac-sim/` with a Python
    runtime at `/isaac-sim/python.sh`.

## Step 2 — install `strands-robots-sim`

Use the same Python environment that Isaac Sim's `python.sh` /
`setup_python_env.sh` activates. Then:

```bash
pip install 'strands-robots-sim[isaac]'
```

The `[isaac]` extra pulls in the pip-installable companion deps that match
the documented Isaac Sim **4.5.x** image: `isaacsim==4.5.*` (the PyPI shim
exposing Kit's Python API), `isaaclab>=2.0,<3.0` (Isaac Lab's task / RL
utilities, the 2.x line paired with Isaac Sim 4.5), and `usd-core` (USD
scene authoring). `strands-robots` is pulled in transitively, which gives
you the upstream `Simulation` AgentTool, `create_simulation()` factory,
and policy providers.

## Step 3 — verify the install

```python
import strands_robots_sim                      # registers entry points
from strands_robots_sim.isaac.simulation import IsaacSimulation

available, reason = IsaacSimulation.is_available()
print("isaac available:", available, "(reason:", reason, ")")
```

On a healthy machine this prints `isaac available: True (reason: None)`.
On a CPU-only / non-Isaac box it returns a structured diagnostic
explaining which `omni.*` import failed — see
[Troubleshooting](../troubleshooting.md).

You can also confirm the entry-point registration:

```python
from importlib.metadata import entry_points

for ep in entry_points(group="strands_robots.backends"):
    print(ep.name, "->", ep.value)
# isaac     -> strands_robots_sim.isaac.simulation:IsaacSimulation
# isaac_sim -> strands_robots_sim.isaac.simulation:IsaacSimulation
```

## From source

```bash
git clone https://github.com/strands-labs/robots-sim
cd robots-sim
pip install -e '.[isaac,dev]'
```

`hatch run lint` and `hatch run test` run the full unit-test slice (no GPU
required); `STRANDS_GPU_TEST=1 pytest strands_robots_sim/isaac/tests/test_gpu_integ.py`
runs the GPU integration tests.

## Next

- [Quickstart](quickstart.md) — bring up an Isaac world and step it.
- [Architecture](../architecture.md) — how the entry-point plugin contract works.
- [Backends → Isaac Sim](../backends/isaac.md) — the full backend reference.
