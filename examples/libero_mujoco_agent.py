#!/usr/bin/env python3
"""LIBERO on MuJoCo, driven through a Strands Agent in natural language.

The whole point of the upstream ``Simulation`` being an ``AgentTool`` is that
an LLM can pick the right action from a free-form prompt. This file shows
that loop end to end: ask the agent to run a benchmark in plain English,
let it dispatch ``evaluate_benchmark`` itself, and get a natural-language
summary back. This is the headline demo of why a Strands integration buys
you anything beyond direct API calls.

For a deterministic, no-LLM-needed version of the same task (used by R15's
backend matrix), see ``libero_mujoco.py``.

Requires::

    pip install 'strands-robots[sim-mujoco,benchmark-libero]' strands-agents

Plus a configured LLM provider. Strands defaults to Anthropic Claude via
AWS Bedrock — see https://strandsagents.com/ for setup. Without one, the
``Agent(...)`` call below will raise an authentication / configuration
error pointing at the SDK setup docs.

Output is non-deterministic by design (LLM-generated summary), so this file
is **not** ingested by R15.
"""

from __future__ import annotations

from strands import Agent

from strands_robots.benchmarks.libero import load_libero_suite
from strands_robots.simulation import Simulation


def main() -> None:
    sim = Simulation(tool_name="libero_sim", mesh=False)

    # Pre-register the LIBERO tasks so the agent only has to choose by name
    # — we don't want to teach the LLM the BDDL discovery dance.
    registered = load_libero_suite("libero_spatial")
    if not registered:
        raise RuntimeError(
            "load_libero_suite('libero_spatial') registered 0 tasks. "
            "Try `pip install strands-robots[benchmark-libero]`."
        )
    benchmark_name = next(iter(registered))

    agent = Agent(tools=[sim])
    try:
        # Natural-language driver: the LLM is responsible for
        #   create_world → add_robot → evaluate_benchmark → destroy
        # in the right order, and for summarising the result.
        response = agent(
            "You have access to a simulation tool. Please:\n"
            "1. Create a MuJoCo world.\n"
            "2. Add a Franka Panda robot named 'panda' with data_config='panda'.\n"
            f"3. Run the LIBERO benchmark '{benchmark_name}' with the mock\n"
            "   policy for 10 episodes, seed 42, on the 'panda' robot.\n"
            "4. Report the success rate and wall-time clearly.\n"
            "5. Destroy the world to clean up."
        )
        print(response)
    finally:
        # Belt-and-braces cleanup in case the LLM didn't call destroy.
        try:
            sim.destroy()
        except Exception:
            pass


# --- Deterministic alternative (no LLM in the loop) -------------------------
# If you want to use the same `Simulation` AgentTool from an `Agent` *without*
# letting the LLM choose the action sequence, call the dispatcher directly:
#
#     agent = Agent(tools=[sim])
#     agent.tool.libero_sim(action="create_world")
#     agent.tool.libero_sim(action="add_robot",
#                           name="panda", data_config="panda")
#     agent.tool.libero_sim(action="evaluate_benchmark",
#                           benchmark_name=benchmark_name,
#                           robot_name="panda",
#                           policy_provider="mock",
#                           n_episodes=10, seed=42)
#     agent.tool.libero_sim(action="destroy")
#
# Useful when you want strands' tool registration ergonomics but a CI-stable
# call sequence. Functionally equivalent to libero_mujoco.py's direct API.

if __name__ == "__main__":
    main()
