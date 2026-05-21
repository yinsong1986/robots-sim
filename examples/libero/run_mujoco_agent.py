#!/usr/bin/env python3
"""LIBERO on MuJoCo, driven by a Strands ``Agent`` in natural language.

The agent receives a single prompt describing the eval, picks
``evaluate_benchmark`` on the registered ``Simulation`` tool, sets the
kwargs from prompt context, runs, and returns a natural-language
summary. For ``--policy=groot`` the agent itself starts and stops the
GR00T inference service through the ``gr00t_inference`` tool — no
scripted ``gr00t_inference(action="start", ...)`` call from Python.

This is the canonical replacement for the natural-language entry point
the deleted ``examples/libero_example.py`` shipped pre-rescope. It's
the "this is why ``strands-robots`` exists" demo: a user describes what
they want, the agent orchestrates the pieces.

Usage
-----
::

    # 1) Smoke test:
    python examples/libero/run_mujoco_agent.py --policy mock

    # 2) Real run; agent starts the GR00T service against the right
    #    `libero_<suite>/` sub-checkpoint of `nvidia/GR00T-N1.7-LIBERO`
    #    when asked. Pre-condition: HF auth + Docker access; the agent
    #    runs the download / start commands itself.
    python examples/libero/run_mujoco_agent.py --policy groot --port 8000

    # 3) Different LIBERO task; agent picks the matching subfolder.
    python examples/libero/run_mujoco_agent.py \\
        --task libero-spatial-pick_up_the_milk_and_place_it_in_the_basket

Requires
--------
- ``pip install 'strands-robots[sim-mujoco,benchmark-libero]' strands-agents``
- A configured LLM provider for Strands. Default is Anthropic Claude via
  AWS Bedrock — see https://strandsagents.com/ for setup. Without one the
  ``Agent(...)`` call below raises an authentication / configuration
  error pointing at the SDK setup docs.
- For ``--policy=groot``: Docker + an NVIDIA GPU + ~30 GB free disk for
  the GR00T checkpoint. The agent itself runs the download and
  service-start commands via the ``gr00t_inference`` tool, so the host
  needs ``hf`` (HuggingFace CLI) and ``docker`` on ``PATH``.

Notes
-----
- Output is non-deterministic by design (LLM-generated summary). R15
  does not ingest this file; the deterministic numbers live in
  ``run_mujoco.py`` (sibling file).
- Records video to ``rollouts/YYYY_MM_DD/`` via the agent's
  ``record_video=True`` choice; filename includes the ``--agent`` marker
  and ``policy=mock|groot`` so per-file post-hoc analysis can tell which
  driver produced it.
- An iterative-supervision (``SteppedSimEnv`` replacement) variant
  deliberately doesn't live here — see
  `R24 / #29 <https://github.com/strands-labs/robots-sim/issues/29>`_
  for the OOD-anchored runnable demo and upstream
  `U6 / #136 <https://github.com/strands-labs/robots/issues/136>`_ for
  the canonical pattern doc.
"""

from __future__ import annotations

import argparse

from strands import Agent

from strands_robots.benchmarks.libero import load_libero_suite
from strands_robots.simulation import Simulation
from strands_robots.tools import gr00t_inference


def _suite_for_task(task: str) -> str:
    parts = task.split("-", 2)
    if len(parts) < 3 or parts[0] != "libero":
        raise ValueError(
            f"--task must look like 'libero-<suite>-<task_stem>', got {task!r}."
        )
    return f"libero_{parts[1]}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["mock", "groot"], default="mock")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--task",
        default="libero-spatial-pick_up_the_red_cube",
        help="Any registered LIBERO benchmark name; suite is auto-derived.",
    )
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    suite = _suite_for_task(args.task)

    sim = Simulation(tool_name="libero_sim", mesh=False)
    # Pre-register the LIBERO suite so the agent only has to choose by
    # benchmark name; we don't make it run BDDL discovery.
    load_libero_suite(suite)

    agent = Agent(tools=[sim, gr00t_inference])

    try:
        # For --policy=groot, the agent starts the inference service
        # itself. The HF repo `nvidia/GR00T-N1.7-LIBERO` is a tree of
        # four sub-checkpoints (`libero_spatial/`, `libero_10/`,
        # `libero_object/`, `libero_goal/`); we tell the agent which
        # subfolder to download and serve based on --task's suite.
        if args.policy == "groot":
            agent(
                f"Start the GR00T inference service on port {args.port} using "
                f"the `{suite}` subfolder of `nvidia/GR00T-N1.7-LIBERO` and "
                f"data_config 'libero_panda'. If the subfolder isn't already at "
                f"`checkpoints/GR00T-N1.7-LIBERO/{suite}/`, download it first "
                f"with `hf download nvidia/GR00T-N1.7-LIBERO --include "
                f"'{suite}/*' --local-dir checkpoints/GR00T-N1.7-LIBERO`."
            )
            policy_phrase = (
                f"using the GR00T policy on localhost:{args.port} with "
                f"`policy_provider='groot'` and "
                f"`policy_config={{'host': 'localhost', 'port': {args.port}, "
                f"'data_config': 'libero_panda'}}`"
            )
        else:
            policy_phrase = "using the mock policy (`policy_provider='mock'`)"

        # The actual one-shot eval — agent picks `evaluate_benchmark`
        # from the registered Simulation tool, sets kwargs from prompt
        # context, then summarises in natural language.
        result = agent(
            f"Set up a MuJoCo simulation: create a world, add a Franka Panda "
            f"robot named 'panda' with data_config='panda'. Then run the "
            f"LIBERO benchmark '{args.task}' for {args.n_episodes} episodes "
            f"with seed {args.seed}, {policy_phrase}, on the 'panda' robot. "
            f"Record video into the `rollouts/` directory with a filename "
            f"that ends `--agent--policy={args.policy}`. "
            f"When the eval finishes, tell me the success rate and the total "
            f"wall-time. Then destroy the simulation world."
        )
        print(result)

        if args.policy == "groot":
            agent(f"Stop the GR00T inference service on port {args.port}.")
    finally:
        # Belt-and-braces cleanup in case the agent didn't call destroy.
        try:
            sim.destroy()
        except Exception:
            pass


# Optional follow-up showing System-2 multi-turn reasoning across runs.
# Drop this in `main()` after the first `print(result)` to see how the
# same agent compounds context across calls:
#
#     agent(
#         "If the success rate from the last run was below 0.5, run the "
#         "same task again with seed 43 and tell me whether the gap is "
#         "policy variance or a systematic failure mode. If it's variance, "
#         "give me the mean and stddev across the two runs. If it's "
#         "systematic, suggest a single follow-up benchmark to confirm."
#     )
#
# For an iterative-supervision pattern (System-2 observes camera + state
# *during* a rollout), see R24 / #29 — that example is anchored on OOD
# scenarios where supervision actually earns its complexity.

if __name__ == "__main__":
    main()
