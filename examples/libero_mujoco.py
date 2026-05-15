#!/usr/bin/env python3
"""LIBERO on the default MuJoCo backend shipped by ``strands-robots``.

One-shot programmatic flow — replaces the deleted ``SimEnv`` pattern.
Scripted ``sim.evaluate_benchmark(...)`` call: the "if you just want to
run LIBERO from a Python script" file. For the natural-language /
``Agent``-driven version, see ``libero_mujoco_agent.py``.

Used by R15's backend-matrix flagship; keeps two grep-stable lines
(``benchmark_name=...`` and ``policy=... task=... success_rate=... wall_time=...s``)
so subprocess-and-parse stays trivial.

Usage
-----
::

    # 1) Smoke test, no GPU required:
    python examples/libero_mujoco.py --policy mock --n-episodes 5

    # 2) Real LIBERO eval against `nvidia/GR00T-N1.7-LIBERO`. The HF repo
    #    is a tree of four sub-checkpoints (`libero_spatial/`,
    #    `libero_10/`, `libero_object/`, `libero_goal/`) — pick the
    #    subfolder matching your `--task`. For the default
    #    `--task libero-spatial-pick_up_the_red_cube`, the right
    #    subfolder is `libero_spatial/`.
    #
    #    Step 2a — build the GR00T container from upstream
    #    (no pre-built image is published; build locally from the
    #    n1.7-release tag):
    #
    #        git clone --depth 1 --branch n1.7-release --recurse-submodules \\
    #            https://github.com/NVIDIA/Isaac-GR00T.git
    #        cd Isaac-GR00T && DOCKER_BUILDKIT=1 bash docker/build.sh
    #        # → produces image `gr00t:latest` (~28 GB)
    #
    #    Step 2b — download just the sub-checkpoint you need
    #    (`nvidia/Cosmos-Reason2-2B`, the VLM backbone, is a *gated*
    #    repo — accept the terms once at huggingface.co/nvidia/Cosmos-
    #    Reason2-2B and make sure your `~/.cache/huggingface/token`
    #    has access):
    #
    #        hf download nvidia/GR00T-N1.7-LIBERO \\
    #            --include 'libero_spatial/*' \\
    #            --local-dir checkpoints/GR00T-N1.7-LIBERO
    #
    #    Step 2c — start the container with HF token + cache mounted
    #    so the gated VLM backbone download works:
    #
    #        docker run -d --gpus all --ipc=host --name gr00t \\
    #            -v "$(pwd)/checkpoints":/data/checkpoints \\
    #            -v "$HOME/.cache/huggingface":/root/.cache/huggingface \\
    #            -e HF_TOKEN="$(cat ~/.cache/huggingface/token)" \\
    #            -p 8000:8000 \\
    #            gr00t tail -f /dev/null
    #
    #    Step 2d — start the inference server *inside* the container.
    #    N1.7 uses the new `gr00t.eval.run_gr00t_server` module, NOT
    #    the older `scripts/inference_service.py --server` entrypoint
    #    that the Strands `gr00t_inference` tool currently wraps:
    #
    #        docker exec -d gr00t bash -c '
    #            python -m gr00t.eval.run_gr00t_server \\
    #                --model-path /data/checkpoints/GR00T-N1.7-LIBERO/libero_spatial \\
    #                --embodiment-tag libero_sim \\
    #                --port 8000 \\
    #                --host 0.0.0.0 \\
    #                --use-sim-policy-wrapper'
    #
    #    The model loads in ~80 s on an L4 (~6 GB GPU memory). Server
    #    listens on port 8000 via ZMQ.
    #
    #    Step 2e — run the eval against it (THIS file):
    python examples/libero_mujoco.py --policy groot --port 8000 --n-episodes 50

    # 3) Different LIBERO suite + task. Suite is auto-derived from --task,
    #    so use the matching `libero_<suite>/` sub-checkpoint:
    python examples/libero_mujoco.py \\
        --policy groot --port 8000 \\
        --task libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_…

Requires
--------
``pip install 'strands-robots[sim-mujoco,benchmark-libero]'``

Imports only from ``strands_robots`` — proves the plugin-repo shape
works without any heavy backend installed.

Notes on the MP4 output
-----------------------
``Simulation.evaluate_benchmark`` does not expose a per-episode
``record_video`` plumb (see PR #26 for the upstream gap). This script
wraps the whole run in a single
``start_cameras_recording`` / ``stop_cameras_recording`` pair, so each
invocation produces **one** MP4 capturing every episode in sequence.
Filename encodes ``--task=<benchmark_name>``, ``--policy=mock|groot``,
``--n_eps=N``, and ``--seed=S`` so post-hoc analysis can tell what
produced it.

Verification status (`--policy=groot` end-to-end)
-------------------------------------------------
**Verified locally**:

- The ``nvidia/GR00T-N1.7-LIBERO/libero_<suite>/`` checkpoint loads on a
  single NVIDIA L4 (uses ~6 GB of 23 GB VRAM after warm-up).
- The new ``python -m gr00t.eval.run_gr00t_server --embodiment-tag
  libero_sim --use-sim-policy-wrapper`` entrypoint serves on port 8000
  and resolves the embodiment tag to ``EmbodimentTag.LIBERO_PANDA``.
- The strands-robots ``Gr00tPolicy(data_config="libero_panda")`` can
  connect to the server and serialise/deserialise messages.

**Blocked on upstream gaps** that the Step-2e command above doesn't
side-step today. All three are tracked in a single combined issue:
`strands-labs/robots#148 <https://github.com/strands-labs/robots/issues/148>`_.

1. ``Simulation`` doesn't auto-load LIBERO BDDL scenes — there's no
   ``agentview`` / wrist camera in the world, so ``video.image`` /
   ``video.wrist_image`` keys never reach the server. (#148, Failure 1)
2. ``Gr00tPolicy._build_service_observation`` adds only a batch dim,
   but the N1.7 server expects ``(B, T, H, W, C)`` for video and
   ``(B, T, D)`` for state with ``T=1``; state must be ``float32`` (not
   ``float64``). (#148, Failure 2)
3. The Strands ``gr00t_inference`` tool wraps the older
   ``scripts/inference_service.py --server`` entrypoint that no longer
   exists in N1.7. The bare-Docker workflow in Step 2d is the way until
   the tool is updated. (#148, Failure 3)

So ``--policy=groot`` exits cleanly today only after points 1-3 are
resolved upstream. Until then, ``--policy=mock`` is the contract you
can rely on; PR #26's matrix-table number stays TBD pending those
upstream merges (or a contributor with a way to side-step them
locally — patches welcome).
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


def _suite_for_task(task: str) -> str:
    """Auto-derive a LIBERO suite name from a benchmark task ID.

    Task IDs follow the ``libero-<suite>-<task_stem>`` pattern (see
    ``strands_robots.benchmarks.libero.suite._format_registry_name``),
    so the suite is the second hyphen-separated segment.

    >>> _suite_for_task("libero-spatial-pick_up_the_red_cube")
    'libero_spatial'
    >>> _suite_for_task("libero-10-LIVING_ROOM_SCENE5_…")
    'libero_10'
    """
    parts = task.split("-", 2)
    if len(parts) < 3 or parts[0] != "libero":
        raise ValueError(
            f"--task must look like 'libero-<suite>-<task_stem>', got {task!r}. "
            "See `load_libero_suite` for registered names."
        )
    return f"libero_{parts[1]}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["mock", "groot"], default="mock")
    p.add_argument(
        "--port", type=int, default=8000, help="GR00T inference port (only used with --policy=groot)"
    )
    p.add_argument(
        "--task",
        default="libero-spatial-pick_up_the_red_cube",
        help="Any registered LIBERO benchmark name; suite is auto-derived.",
    )
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    suite = _suite_for_task(args.task)

    if args.policy == "groot":
        # Client-side `data_config="libero_panda"` — this is the registered
        # key in `strands_robots.policies.groot.DATA_CONFIG_MAP` that tells
        # the local `Gr00tPolicy` how to format LIBERO observations into the
        # GR00T-N1.7 input layout. Note this is *separate from* the server's
        # `--embodiment-tag libero_sim` (an alias of `LIBERO_PANDA` per the
        # checkpoint's `embodiment_id.json`); the two sides happen to mean
        # the same thing but the strings are not interchangeable.
        # Verified locally against `nvidia/GR00T-N1.7-LIBERO/libero_10` —
        # client `libero_panda` + server `libero_sim` is the working pair.
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

        registered = load_libero_suite(suite)
        if not registered:
            raise RuntimeError(
                f"load_libero_suite({suite!r}) registered 0 tasks. "
                "Apply upstream fix from strands-labs/robots#147 if it isn't merged."
            )
        if args.task not in registered:
            # The spec's default name `libero-spatial-pick_up_the_red_cube`
            # is aspirational — real LIBERO ships ~10 spatial tasks but
            # none of them is literally that string. If we hit the default
            # and it doesn't resolve, fall back to the first registered
            # task with a clear note. User-supplied unknown tasks still
            # error loudly.
            if args.task == "libero-spatial-pick_up_the_red_cube":
                fallback = next(iter(registered))
                print(
                    f"NOTE: default --task {args.task!r} isn't in real LIBERO "
                    f"(it's the spec's aspirational placeholder); falling back "
                    f"to first registered task {fallback!r}."
                )
                args.task = fallback
            else:
                raise RuntimeError(
                    f"--task {args.task!r} is not in the {suite} suite. "
                    f"Available: {sorted(registered)[:3]}…"
                )

        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        rec_name = (
            f"{ts}--task={args.task}--n_eps={args.n_episodes}"
            f"--seed={args.seed}--policy={args.policy}"
        )
        video_dir = _date_dir()
        sim.start_cameras_recording(cameras=["default"], output_dir=video_dir, name=rec_name)
        try:
            t0 = time.time()
            result = sim.evaluate_benchmark(
                benchmark_name=args.task,
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
        # exact format (`policy=`, `task=`, `success_rate=`, `wall_time=`,
        # `videos=`) stable across rebases / refactors.
        print(f"benchmark_name={args.task}")
        print(
            f"policy={args.policy}  task={args.task}  "
            f"success_rate={success_rate:.2f}  "
            f"wall_time={wall_time:.1f}s  videos={video_path}"
        )
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
