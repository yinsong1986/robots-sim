# SO-101 synthetic data generation with cuRobo (Isaac / MuJoCo)

Interactive demo for **strands-labs/robots-sim#67**: set up an SO-101 tabletop
pick-and-place world, plan a motion with **[cuRobo](https://nvlabs.github.io/curobo/)**,
execute it in simulation, and record the rollout as a **LeRobot dataset** for
policy training — repeatable across randomized scenes for scale.

It is the cuRobo/motion-planning counterpart to the Replicator synthetic-data
example (R9 / #16) and mirrors the shape of `examples/mujoco_gs/`.

## What runs today vs. what needs more runtime

The control + data-collection loop is **backend-agnostic**: the planner emits
joint targets and the executor/collector speak the `SimEngine` surface, so the
*same* code runs on MuJoCo today and Isaac once installed.

| Capability | Status |
|---|---|
| SO-101 scene + execute + **LeRobot dataset** recording | ✅ works now (MuJoCo, real SO-101) |
| Scripted joint-space pick-and-place planner | ✅ works now (demonstrative motion; grasps not guaranteed) |
| CPU/CI smoke (state+action, no GL) | ✅ `smoke_test.py` |
| Strands agent + Gradio UI | ✅ works now (buttons always; chat needs an LLM backend) |
| **cuRobo** collision-aware planning | ✅ **installs + drives the full pick-place on driver 550 (validated, #67 T3/T4/T5)**; `--planner curobo`. Uses **position-only** IK (the 5-DOF SO-101 can't hit arbitrary 6-DOF poses), producing a collision-free reach→grasp→lift→place→release the arm executes. Physical-grasp success (cube actually transported) needs approach-orientation tuning — see T5 note. |
| **Isaac Sim** backend (`--backend isaac`) | ⛔ falls back to MuJoCo until the runtime + `create_simulation("isaac")` (T1) are present |

Missing cuRobo / Isaac / lerobot / LLM each disable only their own feature with
an actionable message — the app still loads and the loop is still demonstrable.

## Quick start

```bash
# from the repo root
pip install -r examples/so101_curobo/requirements.txt

# headless CI smoke (no GPU/GL): build -> scripted plan -> record -> reload
python -m examples.so101_curobo.smoke_test

# headless data generation (writes a LeRobot dataset to --root)
MUJOCO_GL=egl python -m examples.so101_curobo.app --smoke --episodes 5 \
    --root /tmp/so101_curobo_ds

# interactive Gradio app (camera preview + buttons + chat)
MUJOCO_GL=egl python -m examples.so101_curobo.app --server-port 7863
```

Load a recorded dataset back (no Hub round-trip):

```python
from strands_robots.dataset_recorder import load_lerobot_episode
ds, start, length = load_lerobot_episode("local/so101_curobo_pickplace", 0, root="/tmp/so101_curobo_ds")
```

## Architecture

```
Strands Agent (Gradio UI)            agent.py / app.py
   │  natural language → tools
   ▼
SO101CuroboDemo                       controller.py
   ├─ sim     = make_sim("mujoco"|"isaac")            scene.py   (SimEngine)
   │            build_pick_place_scene(): SO-101 + cube + bin + cameras
   ├─ planner = make_planner()                        planner.py
   │            ScriptedPlanner | CuroboMotionPlanner → JointTrajectory
   └─ collector = LeRobotDataCollector                collector.py
                send_action(waypoint) + add_frame(obs, action) per step,
                save_episode() + finalize() → LeRobot v3.0 dataset; success check
```

The collector uses the tested `strands_robots.dataset_recorder.DatasetRecorder`
recipe: `create() → add_frame()* → save_episode() → finalize()`.

## Flipping to Isaac + cuRobo

- **Isaac:** `--backend isaac` calls `create_simulation("isaac", render_mode="rtx_realtime")`.
  Needs the Isaac Sim runtime (~30 GB) and backend registration (#67 **T1**), plus a
  faithful SO-101 USD via `add_robot(usd_path=...)` (**T2**). Falls back to MuJoCo otherwise.
- **cuRobo:** `--planner curobo` (+ `--curobo-urdf` / `SO101_URDF`). **Validated on
  driver 550 / CUDA 12.4 / L4** (the docs' driver ≥ 580 is conservative — CUDA 12.x
  kernels run on a 12.4 driver). Install recipe:
  ```bash
  export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
  python -m venv --system-site-packages .venv && source .venv/bin/activate
  pip install -U pip setuptools wheel ninja
  git clone --depth 1 https://github.com/NVlabs/curobo && cd curobo
  sed -i '/Topic :: Scientific\/Engineering :: Robotics/d' pyproject.toml  # newer setuptools rejects it
  pip install -e . --no-build-isolation
  pip install 'cuda-core[cu12]'   # the refactored cuRobo's runtime kernel backend (required)
  ```
  `CuroboMotionPlanner` builds the SO-101 model from a URDF via the new
  `RobotBuilder` (T4, auto-derives the 5-DOF arm chain to `gripper_frame_link`)
  and chains `MotionPlanner.plan_pose` segments into the full pick-place (T5,
  validated end-to-end: a 414-waypoint collision-free reach→grasp→lift→place→
  release the MuJoCo arm executes, recorded as a LeRobot episode).
  **5-DOF handling:** the SO-101 has only 5 arm DOF, so a fully-constrained
  6-DOF pose goal is usually infeasible. The planner uses **position-only**
  tracking (`ToolPoseCriteria.track_position`, `position_only=True`), leaving
  orientation free so tabletop targets are reachable; the bin
  (`scene.DEFAULT_PLACE_POSITION`) is set within the arm's reach.
  **Remaining tuning:** because orientation is free, the gripper reaches the
  cube but isn't guaranteed aligned to physically grip it (so the per-episode
  *grasp-success* label is often False). Constraining the approach axis
  (top-down, roll free) via `ToolPoseCriteria.track_orientation` + a tuned grasp
  is the next refinement; the collision-free motion + labeled data generation
  already work. If a segment is unreachable the planner logs and falls back to
  the scripted planner so an episode is still recorded.

## Issue #67 task mapping

| Task | Where | State |
|---|---|---|
| T1 backend registration | `scene.make_sim("isaac")` | stub + clear error |
| T2 faithful SO-101 asset | `add_robot(usd_path=...)` hook | MuJoCo SO-101 used now |
| T3 cuRobo install validation | `planner.CUROBO_INSTALL_HINT` | ✅ validated on driver 550 (recipe above) |
| T4 cuRobo SO-101 config | `CuroboMotionPlanner._ensure` (`RobotBuilder`) | ✅ builds the 5-DOF model from URDF |
| T5 `CuroboMotionPlanner` | `planner.py` | ✅ drives the full pick-place (position-only IK, validated MuJoCo execute + record); grasp-orientation tuning for success>0 pending |
| T6 executor + gripper | `collector._execute_and_record` | ✅ |
| T7 `LeRobotDataCollector` | `collector.py` | ✅ (multi-episode, success check) |
| T8 domain randomization | `record_dataset(randomize=True)` → `sim.randomize` | ✅ basic |
| T9 agent + Gradio app | `agent.py`, `app.py` | ✅ |
| T10 docs + smoke test | this file, `smoke_test.py` | ✅ |

## Scope

The cuRobo path targets **rigid, quasi-static, prehensile** tasks (pick/place,
reach, push). Contact-rich / deformable / dynamic tasks should route to
teleop/RL, not this generator.
