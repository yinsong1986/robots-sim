#!/usr/bin/env python3
"""LIBERO on the default MuJoCo backend shipped by ``strands-robots``.

Direct-API flow — deterministic, no LLM in the loop. This is the **baseline
row** in the backend comparison matrix (see ``examples/README.md``); R15's
``libero_backend_matrix.py`` ingests this script's stdout for the table, so
the ``benchmark_name=...`` / ``success_rate=... wall_time=...s`` lines below
are intentionally grep-stable.

For a Strands-Agent + natural-language version of the same task, see
``libero_mujoco_agent.py``.

Requires::

    pip install 'strands-robots[sim-mujoco,benchmark-libero]'

No imports from ``strands_robots_sim`` — this example exercises the upstream
``Simulation`` AgentTool and the LIBERO benchmark adapter directly.
"""

from __future__ import annotations

import time

from strands_robots.benchmarks.libero import load_libero_suite
from strands_robots.simulation import Simulation


def main() -> None:
    # `mesh=False` keeps Simulation standalone (no peer-mesh side effects).
    sim = Simulation(tool_name="libero_sim", mesh=False)
    try:
        sim.create_world()

        # LiberoAdapter.default_robot is "panda" — `data_config="panda"`
        # makes the benchmark's compatibility check pass without us having
        # to pin a URDF path.
        sim.add_robot("panda", data_config="panda")

        # Bulk-register every BDDL task in the suite under
        # `libero-spatial-<task_stem>` keys.
        registered = load_libero_suite("libero_spatial")
        if not registered:
            raise RuntimeError(
                "load_libero_suite('libero_spatial') registered 0 tasks. "
                "Is the `libero` package installed? "
                "Try `pip install strands-robots[benchmark-libero]`."
            )

        # First registered task — robust against LIBERO version drift in
        # exact task names. Pin a specific one (e.g.
        # "libero-spatial-pick_up_the_black_bowl_between_..." ) if your
        # comparison needs to hold a fixed task across runs / backends.
        benchmark_name = next(iter(registered))

        t0 = time.time()
        result = sim.evaluate_benchmark(
            benchmark_name=benchmark_name,
            robot_name="panda",
            policy_provider="mock",  # swap "groot" / "lerobot" + policy_config={...} for real eval
            n_episodes=10,
            seed=42,
        )
        wall_time = time.time() - t0

        # Tool-style result: { "status": ..., "content": [{"text": ...}, {"json": {...}}] }
        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        success_rate = json_payload["success_rate"]

        # R15 parses these two lines — keep the format stable.
        print(f"benchmark_name={benchmark_name}")
        print(f"success_rate={success_rate:.2f}  wall_time={wall_time:.1f}s")
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
