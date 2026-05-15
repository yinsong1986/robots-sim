# Migration: `strands-robots-sim` 0.1.x → 0.2.0

`strands-robots-sim` 0.2.0 is a re-scoping release. The legacy `SimEnv`,
`SteppedSimEnv`, and Libero-direct environment layer have been **removed**.
That lightweight MuJoCo + LIBERO code path now lives in
[`strands-labs/robots`](https://github.com/strands-labs/robots), accessible
via the `Simulation` AgentTool and the `LiberoAdapter` benchmark plugin
([strands-labs/robots#110](https://github.com/strands-labs/robots/issues/110) /
[PR #130](https://github.com/strands-labs/robots/pull/130)).

This release is **breaking**: importing `SimEnv` or `SteppedSimEnv` from
`strands_robots_sim` raises `ImportError` with a pointer back to this file.

Going forward, `strands-robots-sim` is the home for **heavy, NVIDIA-GPU-only**
simulation backends (Isaac Sim, Newton/Warp) that plug into `strands-robots`
through its `SimEngine` ABC. See the umbrella issue
[strands-labs/robots-sim#8](https://github.com/strands-labs/robots-sim/issues/8)
for the full re-scope and roadmap.

---

## Install

```bash
# Lightweight LIBERO + MuJoCo (default; replaces this package's old `[sim]` extra)
pip install 'strands-robots[sim-mujoco]'

# Heavy GPU-only backends ship in later 0.x releases:
# pip install 'strands-robots-sim[isaac]'    # Isaac Sim     — Stage 3
# pip install 'strands-robots-sim[newton]'   # Newton / Warp — Stage 4
```

---

## API mapping

| Before (0.1.x, this package) | After (0.2.0+, `strands-robots`) | Why the shape changed |
|---|---|---|
| `from strands_robots_sim import SimEnv` | `from strands_robots.simulation import Simulation` — runnable example: [`examples/libero_mujoco.py`](libero_mujoco.py) | The agent-facing async lifecycle is now the 58-action `Simulation` AgentTool; episode rollout is one of those actions. See its `action=` enum. |
| `SimEnv(env_type="libero", task_suite="libero_spatial")` | `from strands_robots.benchmarks.libero import load_libero_suite` then `load_libero_suite("libero_spatial")` | Benchmarks register globally through `BenchmarkProtocol`; the simulation engine is selected separately (default MuJoCo). |
| `agent.tool.my_sim(action="execute", instruction="pick up the red block", policy_port=8000, max_episodes=50, ...)` | `sim.evaluate_benchmark(benchmark_name="libero-spatial-pick_up_the_red_block", policy_provider="groot", policy_config={"host": "localhost", "port": 8000, "data_config": "libero_panda"}, n_episodes=50, seed=42)` — runnable: [`examples/libero_mujoco.py --policy groot`](libero_mujoco.py) | Tasks are addressed by canonical `libero-<suite>-<task>` IDs rather than suite + free-form instruction. The GR00T provider's `data_config` key is `"libero_panda"` (not the bare `"libero"` the legacy SimEnv used). Success rate / wall-time are returned directly. |
| `from strands_robots_sim import SteppedSimEnv` | `from strands_robots.simulation import Simulation` — runnable example: [`examples/libero_mujoco_stepped.py`](libero_mujoco_stepped.py) | There is no separate stepped class anymore — iterative control is a *usage pattern* on the same `Simulation` tool. |
| `SteppedSimEnv(...).execute_steps(...)` (System-2 reads camera every N steps) | `sim.start_policy(policy_provider="groot", policy_config={...}, instruction="...")` + poll `sim.get_state(...)` / `sim.render(...)` between System-2 turns — runnable: [`examples/libero_mujoco_stepped.py`](libero_mujoco_stepped.py) | Step batching is no longer baked into the tool API. Bring your own polling cadence. See the upstream iterative-control doc, [strands-labs/robots#136](https://github.com/strands-labs/robots/issues/136) (U6). |
| `agent.tool.my_sim(record_video=True)` → `rollouts/YYYY_MM_DD/...mp4` | `sim.start_cameras_recording(cameras=[...], output_dir="rollouts/YYYY_MM_DD", name=...)` + `sim.stop_cameras_recording()` (the example files do this around `evaluate_benchmark` / the `start_policy` loop) | The `rollouts/YYYY_MM_DD/<timestamp>--<metadata>__<camera>.mp4` filename convention is preserved by the example files; per-episode segmentation needs upstream `record_video=` plumbing on `evaluate_benchmark` and is filed as a follow-up. |
| `pip install 'strands-robots-sim[sim]'` (libero / robosuite / scipy / mujoco / gymnasium) | `pip install 'strands-robots[sim-mujoco,benchmark-libero]'` | The lightweight backend stack moved upstream. Heavy GPU backends (Isaac, Newton) will live behind `[isaac]` / `[newton]` extras in this repo. |

---

## Side-by-side example

### Before — 0.1.x with this package

```python
from strands import Agent
from strands_robots_sim import SimEnv, gr00t_inference

sim = SimEnv(
    tool_name="my_sim",
    env_type="libero",
    task_suite="libero_spatial",
)

agent = Agent(tools=[sim, gr00t_inference])

agent.tool.my_sim(
    action="execute",
    instruction="pick up the red block",
    policy_port=8000,
    max_episodes=50,
    max_steps_per_episode=500,
)
```

### After — 0.2.0 with `strands-robots` (default MuJoCo backend)

For the runnable version with `--policy {mock,groot}` flag, MP4
recording, and the GR00T service-start commands, see
[`examples/libero_mujoco.py`](libero_mujoco.py). Minimal shape:

```python
from strands_robots.simulation import Simulation
from strands_robots.benchmarks.libero import load_libero_suite

sim = Simulation(tool_name="sim", mesh=False)
sim.create_world()
sim.add_robot("panda", data_config="panda")
load_libero_suite("libero_spatial")

sim.evaluate_benchmark(
    benchmark_name="libero-spatial-pick_up_the_red_block",
    robot_name="panda",
    policy_provider="groot",
    policy_config={
        "host": "localhost",
        "port": 8000,
        "data_config": "libero_panda",   # NB: not the bare "libero" the legacy SimEnv used
    },
    n_episodes=50,
    seed=42,
)
```

### After — same task on Isaac Sim (Stage 3, future)

```python
import strands_robots_sim  # registers "isaac" via entry points
from strands_robots.simulation import create_simulation
from strands_robots.benchmarks.libero import load_libero_suite

sim = create_simulation("isaac", rtx_mode="path_traced", headless=True)
sim.create_world()
load_libero_suite("libero_spatial")

sim.evaluate_benchmark(
    benchmark_name="libero-spatial-pick_up_the_red_block",
    n_episodes=50,
    seed=42,
)
```

### After — Newton fleet (Stage 4, future)

```python
sim = create_simulation("newton", num_envs=4096, solver="mujoco")
sim.create_world()
load_libero_suite("libero_spatial")
sim.evaluate_benchmark(benchmark_name="libero-spatial-pick_up_the_red_block",
                      n_episodes=50, seed=42)
```

---

## Iterative control (replacement for `SteppedSimEnv`)

`SteppedSimEnv` baked a "run N steps, then return camera + state to System-2"
loop into the tool. The replacement is the upstream `start_policy` + polling
pattern on the same `Simulation` AgentTool. For the runnable version with
MP4 recording, see [`examples/libero_mujoco_stepped.py`](libero_mujoco_stepped.py).
Minimal shape:

```python
import time
from strands_robots.simulation import Simulation

sim = Simulation(tool_name="sim", mesh=False)
sim.create_world()
sim.add_robot("panda", data_config="panda")

sim.start_policy(
    robot_name="panda",
    policy_provider="groot",
    policy_config={"host": "localhost", "port": 8000, "data_config": "libero_panda"},
    instruction="pick up the red block",
    duration=30.0,
)

for _ in range(50):
    time.sleep(0.5)
    state_resp = sim.get_state()           # status envelope, NOT a flat dict
    frame_resp = sim.render(camera_name="default")
    # ↓ Real System-2: inspect state_resp / frame_resp; may call
    #   sim.stop_policy() then sim.start_policy(...) again with a new
    #   instruction; or `break` to end the session.

sim.stop_policy(robot_name="panda")
```

`get_state()` and `render()` both return the standard
`{"status": ..., "content": [...]}` envelope rather than flat data —
the System-2 hook reads from there. The canonical write-up is upstream
in [strands-labs/robots#136](https://github.com/strands-labs/robots/issues/136)
(U6).

---

## MP4 output

Both example files preserve the deleted `SimEnv`'s `rollouts/YYYY_MM_DD/`
directory layout and timestamped filename convention. Each invocation
writes one MP4 whose filename encodes `policy=mock` / `policy=groot`,
the seed, and either the suite + episode count (one-shot) or a
`--stepped` marker (iterative). One MP4 *per run* today; per-episode
segmentation needs upstream `record_video=` plumbing on
`evaluate_benchmark` and is filed as a follow-up — see PR description
on [`strands-labs/robots-sim#26`](https://github.com/strands-labs/robots-sim/pull/26).

---

## See also

- Upstream `Simulation` AgentTool: <https://github.com/strands-labs/robots#simulation-mujoco>
- LIBERO adapter: [strands-labs/robots#110](https://github.com/strands-labs/robots/issues/110), [PR #130](https://github.com/strands-labs/robots/pull/130)
- Iterative-control pattern: [strands-labs/robots#136](https://github.com/strands-labs/robots/issues/136) (U6)
- Re-scope umbrella: [strands-labs/robots-sim#8](https://github.com/strands-labs/robots-sim/issues/8)
