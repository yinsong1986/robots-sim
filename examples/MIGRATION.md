# Migration: `strands-robots-sim` 0.2.x → 0.3.0

`strands-robots-sim` 0.3.0 is a re-scoping release. The legacy `SimEnv`,
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

| Before (0.2.x, this package) | After (0.3.0, `strands-robots`) | Why the shape changed |
|---|---|---|
| `from strands_robots_sim import SimEnv` | `from strands_robots.simulation import Simulation` | The agent-facing async lifecycle is now the 58-action `Simulation` AgentTool; episode rollout is one of those actions. See its `action=` enum. |
| `SimEnv(env_type="libero", task_suite="libero_spatial")` | `from strands_robots.benchmarks.libero import load_libero_suite` then `load_libero_suite("libero_spatial")` | Benchmarks register globally through `BenchmarkProtocol`; the simulation engine is selected separately (default MuJoCo). |
| `agent.tool.my_sim(action="execute", instruction="pick up the red block", policy_port=8000, max_episodes=50, ...)` | `sim.evaluate_benchmark(benchmark_name="libero-spatial-pick_up_the_red_block", policy_provider="groot", policy_port=8000, n_episodes=50, seed=42)` | Tasks are addressed by canonical `libero-<suite>-<task>` IDs rather than suite + free-form instruction. Success rate / wall-time are returned directly instead of streamed via tool status. |
| `from strands_robots_sim import SteppedSimEnv` | `from strands_robots.simulation import Simulation` | There is no separate stepped class anymore — iterative control is a *usage pattern* on the same `Simulation` tool. |
| `SteppedSimEnv(...).execute_steps(...)` (System-2 reads camera every N steps) | `sim.start_policy(policy_provider="groot", ...)` + poll `sim.get_state(...)` / `sim.render(...)` between System-2 turns | Step batching is no longer baked into the tool API. Bring your own polling cadence. See the upstream iterative-control doc, [strands-labs/robots#136](https://github.com/strands-labs/robots/issues/136) (U6). |
| `pip install 'strands-robots-sim[sim]'` (libero / robosuite / scipy / mujoco / gymnasium) | `pip install 'strands-robots[sim-mujoco]'` | The lightweight backend stack moved upstream. Heavy GPU backends (Isaac, Newton) will live behind `[isaac]` / `[newton]` extras in this repo. |

---

## Side-by-side example

### Before — 0.2.x with this package

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

### After — 0.3.0 with `strands-robots` (default MuJoCo backend)

```python
from strands_robots.simulation import Simulation
from strands_robots.benchmarks.libero import load_libero_suite

sim = Simulation(tool_name="sim", mesh=False)
sim.create_world()
load_libero_suite("libero_spatial")

sim.evaluate_benchmark(
    benchmark_name="libero-spatial-pick_up_the_red_block",
    policy_provider="groot",
    policy_port=8000,
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
pattern on the same `Simulation` AgentTool:

```python
sim.start_policy(policy_provider="groot", policy_port=8000,
                 task="pick up the red block")

while not sim.is_done():
    state = sim.get_state()
    frame = sim.render(camera="agentview")
    # ... agent / System-2 inspects state & frame, may call sim.stop_policy()
    # and re-issue start_policy(...) with a revised instruction
    time.sleep(0.5)
```

The exact polling cadence and System-2 hand-off is documented upstream in
[strands-labs/robots#136](https://github.com/strands-labs/robots/issues/136)
(U6). Until that doc lands, treat the snippet above as the canonical pattern.

---

## See also

- Upstream `Simulation` AgentTool: <https://github.com/strands-labs/robots#simulation-mujoco>
- LIBERO adapter: [strands-labs/robots#110](https://github.com/strands-labs/robots/issues/110), [PR #130](https://github.com/strands-labs/robots/pull/130)
- Iterative-control pattern: [strands-labs/robots#136](https://github.com/strands-labs/robots/issues/136) (U6)
- Re-scope umbrella: [strands-labs/robots-sim#8](https://github.com/strands-labs/robots-sim/issues/8)
