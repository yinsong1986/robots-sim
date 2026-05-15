#!/usr/bin/env python3
"""LIBERO on MuJoCo with iterative System-2 supervision.

Replacement for the deleted ``SteppedSimEnv`` pattern. Instead of one
``evaluate_benchmark`` call that runs the policy to completion (see
``libero_mujoco.py``), this file demonstrates the iterative supervision
loop: kick off a ``start_policy`` worker in the background, then poll
``get_state`` / ``render`` every N seconds. A real System-2 agent slots
into the polling loop and can interrupt the worker, change the
instruction, or stop early based on what it sees.

The canonical write-up of the technique lives upstream at
`strands-labs/robots#136 <https://github.com/strands-labs/robots/issues/136>`_
(U6); this file is the LIBERO-specific runnable instance referenced
from ``examples/MIGRATION.md``.

Saves one MP4 of the full supervision session under
``rollouts/YYYY_MM_DD/``.

Usage
-----
::

    # 1) Smoke test (mock policy, no GPU):
    python examples/libero_mujoco_stepped.py --policy mock

    # 2) Real run against `nvidia/GR00T-N1.7-LIBERO`. Service must be
    #    running on `--port` first — see `libero_mujoco.py`'s docstring
    #    for the start commands; intentionally not duplicated here:
    python examples/libero_mujoco_stepped.py --policy groot --port 8000

Requires
--------
``pip install 'strands-robots[sim-mujoco,benchmark-libero]'``

Notes on the demo
-----------------
* The supervision loop runs against the default Panda + ``default``
  camera (the one ``create_world`` provides) rather than the LIBERO
  task's BDDL scene + ``agentview`` camera. A fully-faithful LIBERO
  step-eval would call ``spec.on_episode_start(sim, rng)`` to load the
  scene per task; that public-method ergonomics gap is filed as a
  follow-up and not required for the *pattern* this file demonstrates.
* ``get_state()`` returns the standard
  ``{"status": ..., "content": [{"text": ...}, {"json": ...}]}``
  envelope, **not** a flat dict. There is no ``state["reward"]`` key —
  reward is only surfaced inside ``evaluate_benchmark``'s JSON
  payload. The supervision-loop exit here is therefore based on
  ``--max-iters`` (and the stub system-2 hook), not on a reward signal.
* ``render(camera_name=...)`` likewise returns a status dict, not a
  ``np.ndarray``. The PNG bytes live at
  ``frame_resp["content"][1]["image"]["source"]["bytes"]``; helper
  ``strands_robots.simulation.policy_runner._extract_frame_ndarray``
  unpacks it if you need a numpy array for downstream model code.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time

from strands_robots.simulation import Simulation


# How often the System-2 hook observes camera + state (Hz). 2 Hz is
# slow enough that the LLM round-trip in a real agent doesn't dominate
# wall-time, fast enough to react before the policy completes a typical
# ~10 s episode.
OBSERVE_HZ = 2.0


def _date_dir(date_root: str = "rollouts") -> str:
    out = os.path.join(date_root, _dt.date.today().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["mock", "groot"], default="mock")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-iters", type=int, default=50, help="Max supervision-loop observations.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.policy == "groot":
        start_kwargs = {
            "policy_provider": "groot",
            "policy_config": {
                "host": "localhost",
                "port": args.port,
                "data_config": "libero_panda",
            },
        }
    else:
        start_kwargs = {"policy_provider": "mock"}

    sim = Simulation(tool_name="libero_stepped", mesh=False)
    try:
        sim.create_world()
        sim.add_robot("panda", data_config="panda")

        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        rec_name = (
            f"{ts}--stepped--task=libero-spatial-pick_up_the_red_cube"
            f"--seed={args.seed}--policy={args.policy}"
        )
        video_dir = _date_dir()
        sim.start_cameras_recording(cameras=["default"], output_dir=video_dir, name=rec_name)

        observed = 0
        try:
            sim.start_policy(
                robot_name="panda",
                instruction="pick up the red cube",
                duration=30.0,
                **start_kwargs,
            )

            # ─── Iterative supervision loop (the System-2 hook) ──────────────
            # Each iteration represents one System-2 "look at the world,
            # decide whether to keep going" turn. In production this is
            # where a Strands agent inspects `state_resp` + `frame_resp`
            # and chooses to continue / re-issue / stop.
            for step in range(args.max_iters):
                time.sleep(1.0 / OBSERVE_HZ)

                state_resp = sim.get_state()  # noqa: F841 — exercised by the agent stub below.
                frame_resp = sim.render(camera_name="default")  # noqa: F841

                # ↓ Replace this block with a real Strands agent call.
                #
                # The agent receives `state_resp` (text summary of
                # sim_time / step_count / robots) and `frame_resp` (PNG
                # bytes of the camera view) and may:
                #
                #   - continue (just `continue` to the next iteration)
                #   - call `sim.stop_policy(robot_name="panda")` then
                #     `sim.start_policy(...)` with a new `instruction=`
                #     to retarget mid-rollout
                #   - `break` to end the session early
                #
                # In this demo we only count iterations and let the
                # background policy worker finish naturally.

                observed = step + 1
        finally:
            sim.stop_policy(robot_name="panda")
            sim.stop_cameras_recording()

        video_path = os.path.join(video_dir, f"{rec_name}__default.mp4")
        # Single grep-stable line — different format from the one-shot
        # file (no `success_rate=` / `wall_time=`); R15's matrix flagship
        # does NOT ingest stepped runs, so the format is for human
        # inspection only.
        print(f"policy={args.policy}  observe_iters={observed}  videos={video_path}")
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
