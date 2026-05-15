#!/usr/bin/env python3
"""LIBERO on the default MuJoCo backend shipped by ``strands-robots``.

One-shot evaluation flow — replaces the deleted ``SimEnv`` pattern. The
agent specifies the task once, the policy runs to completion, the script
prints a success rate and wall-time, and **an MP4 of the run is saved**
under ``rollouts/YYYY_MM_DD/``.

Used by R15's backend-matrix flagship; keeps two grep-stable lines
(``benchmark_name=...`` and ``policy=...  success_rate=...  wall_time=...s``)
so subprocess-and-parse stays trivial.

Usage
-----
::

    # 1) Smoke test, no GPU required:
    python examples/libero_mujoco.py --policy mock --n-episodes 5

    # 2) Real LIBERO eval against the public NVIDIA checkpoint
    #    (`nvidia/GR00T-N1.7-LIBERO`).
    #
    #    Step 2a — start a GR00T inference service. Either via the
    #    Strands tool wrapper:
    #
    #        from strands_robots.tools import gr00t_inference
    #        gr00t_inference(
    #            action="start",
    #            checkpoint_path="nvidia/GR00T-N1.7-LIBERO",
    #            port=8000,
    #            data_config="libero_panda",
    #        )
    #
    #    …or with the equivalent bare Docker invocation if you don't have
    #    Strands tools wired up:
    #
    #        docker run --gpus all -p 8000:8000 \\
    #            nvcr.io/nvidia/isaac-gr00t:latest serve \\
    #            --checkpoint nvidia/GR00T-N1.7-LIBERO \\
    #            --data-config libero_panda \\
    #            --port 8000
    #
    #    Step 2b — run the eval against it:
    python examples/libero_mujoco.py --policy groot --port 8000 --n-episodes 50

Requires
--------
``pip install 'strands-robots[sim-mujoco,benchmark-libero]'``

Imports only from ``strands_robots`` — proves the plugin-repo shape
works without any heavy backend installed.

Notes on the MP4 output
-----------------------
``Simulation.evaluate_benchmark`` does not currently expose a
per-episode ``record_video`` plumb (see PR description for the upstream
gap). This script wraps the whole run in a single
``start_cameras_recording`` / ``stop_cameras_recording`` pair, so each
invocation produces **one** MP4 capturing every episode in sequence.
Filename encodes ``policy=mock`` / ``policy=groot``, the suite, the
episode count and the seed so post-hoc analysis can tell what produced
it; once upstream grows per-episode video plumbing, splitting this into
N MP4s is a small change.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time

from strands_robots.benchmarks.libero import load_libero_suite
from strands_robots.simulation import Simulation


def _date_dir(date_root: str = "rollouts") -> str:
    out = os.path.join(date_root, _dt.date.today().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["mock", "groot"], default="mock")
    p.add_argument(
        "--port", type=int, default=8000, help="GR00T inference port (only used with --policy=groot)"
    )
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.policy == "groot":
        # `data_config="libero_panda"` is the registered key in
        # `strands_robots.policies.groot.DATA_CONFIG_MAP` for the LIBERO
        # Panda embodiment — note: NOT the bare "libero" string the
        # legacy SimEnv used. The factory forwards this kwarg verbatim
        # to `Gr00tPolicy.__init__`.
        policy_kwargs = {
            "policy_provider": "groot",
            "policy_config": {
                "host": "localhost",
                "port": args.port,
                "data_config": "libero_panda",
            },
        }
    else:
        policy_kwargs = {"policy_provider": "mock"}

    sim = Simulation(tool_name="libero_sim", mesh=False)
    try:
        sim.create_world()
        sim.add_robot("panda", data_config="panda")

        registered = load_libero_suite("libero_spatial")
        if not registered:
            raise RuntimeError(
                "load_libero_suite('libero_spatial') registered 0 tasks. "
                "Apply upstream fix from strands-labs/robots#147 if it isn't merged."
            )
        # First registered task — robust against LIBERO version drift in
        # exact task names. Pin a specific task in your own code if you
        # need a fixed comparison point across runs / backends.
        benchmark_name = next(iter(registered))

        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        rec_name = (
            f"{ts}--suite=libero-spatial--n_eps={args.n_episodes}"
            f"--seed={args.seed}--policy={args.policy}"
        )
        video_dir = _date_dir()
        sim.start_cameras_recording(cameras=["default"], output_dir=video_dir, name=rec_name)
        try:
            t0 = time.time()
            result = sim.evaluate_benchmark(
                benchmark_name=benchmark_name,
                robot_name="panda",
                n_episodes=args.n_episodes,
                seed=args.seed,
                **policy_kwargs,
            )
            wall_time = time.time() - t0
        finally:
            sim.stop_cameras_recording()

        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        success_rate = json_payload["success_rate"]
        video_path = os.path.join(video_dir, f"{rec_name}__default.mp4")

        # Two grep-stable lines for R15 to subprocess-and-parse. Keep the
        # exact format (`policy=`, `success_rate=`, `wall_time=`, `videos=`)
        # stable across rebases / refactors.
        print(f"benchmark_name={benchmark_name}")
        print(
            f"policy={args.policy}  success_rate={success_rate:.2f}  "
            f"wall_time={wall_time:.1f}s  videos={video_path}"
        )
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
