#!/usr/bin/env python3
"""LIBERO on MuJoCo, driven by a Strands ``Agent`` in natural language.

The agent receives a single prompt describing the eval, picks
``evaluate_benchmark`` on the registered ``Simulation`` tool, sets the
kwargs from prompt context, runs, and returns a natural-language
summary. This is the canonical replacement for the natural-language
entry point the deleted ``examples/libero_example.py`` shipped pre-
rescope: a user describes what they want, the agent picks the right
tool and runs it.

What the script handles deterministically (NOT the agent)
---------------------------------------------------------
LLM agents are reliable at "pick the right tool from a small set, fill
its kwargs from prompt context, summarise the result". They are *not*
reliable at multi-step infrastructure orchestration where each step
has a brittle invariant (docker container names, HF-cache locations,
recorder API selection, scene pre-warm timing). So this script keeps
the latter under deterministic Python control and gives the agent the
one decision it's actually good at: invoking ``evaluate_benchmark``.

Owned by the script:

* GR00T inference container lifecycle (start, wait-for-load, teardown
  on exit) via ``gr00t_inference(action='lifecycle', ...)`` — same
  block as ``run_mujoco.py``. Idempotent: reuses an already-running
  container with the matching name on ``--port``; no redundant
  checkpoint re-download if the cache is already populated.
* LIBERO scene pre-warm: ``spec._generate_scene_from_bddl()`` →
  ``sim.load_scene(...)`` → ``spec.prewarm(sim)``. Without this the
  GR00T server rejects the first observation with ``Video key
  'video.image' must be in observation`` because the scene's cameras
  haven't been registered yet.
* MP4 recording via ``start_cameras_recording_synchronous`` —
  introduced upstream in ``strands-labs/robots#192`` (closes #191).
  Returns an ``on_frame`` closure that the script wires into
  ``evaluate_benchmark(on_frame=...)`` so frames are rendered on the
  eval thread at known sync points (post-step). The legacy daemon-
  thread recorder (``start_cameras_recording``) races with the eval
  thread on shared ``mjData`` when the eval runs under Strands
  ``Agent`` tool dispatch, producing 2-3% frame capture rate plus
  greenish GL clear-colour artifacts; the synchronous mode eliminates
  the race entirely. The agent does *not* pick a recorder API — earlier
  shapes of this script let the agent decide and it consistently picked
  LeRobot's ``Dataset`` recorder which then crashed on
  ``[Errno 17] File exists: 'rollouts'``.

The ``on_frame`` closure can't cross Strands' tool-dispatch JSON
boundary (closures aren't JSON-serializable), so the eval call is
exposed to the agent through a thin ``@tool`` wrapper defined in
``main()`` that captures ``sim`` / ``video_dir`` / ``rec_name`` from
the outer scope. The agent picks the wrapper and fills its kwargs
from natural language; the closure stays in Python.

Owned by the agent: the single ``evaluate_benchmark(...)`` call with
benchmark_name + n_episodes + seed + policy_provider + policy_config
filled from natural language, plus the natural-language summary at
the end.

Usage
-----
::

    # 1) Smoke test (mock policy; no GPU / Docker needed):
    python examples/libero/run_mujoco_agent.py --policy mock --n-episodes 5

    # 2) Real run against `nvidia/GR00T-N1.7-LIBERO`. Script auto-
    #    orchestrates the GR00T inference container (idempotent). Pre-
    #    condition: HF token at `~/.cache/huggingface/token` (gated
    #    Cosmos-Reason2-2B backbone) + Docker + an NVIDIA GPU.
    python examples/libero/run_mujoco_agent.py --policy groot --port 8000 --n-episodes 5

    # 2b) Reuse an already-running container instead of letting the
    #     script manage one (e.g. for multi-eval sessions):
    python examples/libero/run_mujoco_agent.py --policy groot --no-auto-server --port 8000

    # 3) Different LIBERO suite + task; suite auto-derived from --task,
    #    so the lifecycle tool downloads the matching `libero_<suite>/`
    #    sub-checkpoint:
    python examples/libero/run_mujoco_agent.py \\
        --policy groot --port 8000 \\
        --task libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_…

Requires
--------
- ``pip install 'strands-robots[sim-mujoco,benchmark-libero]' strands-agents``
- A configured LLM provider for Strands. Default is Anthropic Claude
  via AWS Bedrock — see https://strandsagents.com/ for setup. Without
  one the ``Agent(...)`` call below raises an authentication /
  configuration error pointing at the SDK setup docs.
- For ``--policy=groot``: Docker + an NVIDIA GPU + ~30 GB free disk for
  the GR00T checkpoint (cached across re-runs).

Notes
-----
- Output is non-deterministic by design (LLM-generated summary). R15
  does not ingest this file; the deterministic numbers live in
  ``run_mujoco.py`` (sibling file).
- Records video to ``rollouts/YYYY_MM_DD/`` with a filename that ends
  ``--policy=mock|groot--agent`` so post-hoc analysis can tell which
  driver produced it.
- An iterative-supervision (``SteppedSimEnv`` replacement) variant
  deliberately doesn't live here — see R24 / #29 for the OOD-anchored
  runnable demo.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time

from strands import Agent, tool

from strands_robots.benchmarks.libero import load_libero_suite
from strands_robots.simulation import Simulation
from strands_robots.tools import gr00t_inference


def _date_dir(date_root: str = "rollouts") -> str:
    out = os.path.join(date_root, _dt.date.today().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    return out


def _suite_for_task(task: str) -> str:
    """Auto-derive a LIBERO suite name from a benchmark task ID.

    Same shape as ``run_mujoco.py``'s helper; see that file for the
    canonical doctest examples.
    """
    parts = task.split("-", 2)
    if len(parts) < 3 or parts[0] != "libero":
        raise ValueError(
            f"--task must look like 'libero-<suite>-<task_stem>', got {task!r}. "
            "See `load_libero_suite` for registered names."
        )
    return f"libero_{parts[1]}"


def _bring_up_gr00t_server(args: argparse.Namespace, suite: str) -> dict | None:
    """Start the GR00T inference container and block until model is loaded.

    Mirrors the lifecycle block in ``run_mujoco.py`` so the agent file
    has identical "real-eval" plumbing. Returns the lifecycle handle
    (or ``None`` if ``--policy=mock`` / ``--no-auto-server``).
    """
    if args.policy != "groot" or not args.auto_server:
        return None

    from pathlib import Path
    import subprocess
    from time import monotonic, sleep

    hf_token_path = Path("~/.cache/huggingface/token").expanduser()
    if not hf_token_path.is_file():
        raise RuntimeError(
            "--policy groot needs an HF token (Cosmos-Reason2-2B is gated). "
            "Run `huggingface-cli login` first, then retry."
        )
    result = gr00t_inference(
        action="lifecycle",
        lifecycle="full",
        image_name=args.image,
        hf_repo="nvidia/GR00T-N1.7-LIBERO",
        hf_subfolder=suite,
        hf_local_dir=args.checkpoint_dir,
        container_name=args.container,
        hf_token=hf_token_path.read_text().strip(),
        checkpoint_path=f"/data/checkpoints/{suite}",
        embodiment_tag="libero_sim",
        protocol="n1.7",
        use_sim_policy_wrapper=True,
        port=args.port,
    )
    if result.get("status") != "success":
        raise RuntimeError(f"gr00t_inference lifecycle=full failed: {result}")
    print(f"[setup] {result.get('message')}")

    # Same readiness wait as run_mujoco.py — the lifecycle tool returns
    # success when the port is bound, but the model loads asynchronously
    # after that. Block until GPU memory crosses a heuristic threshold.
    deadline = monotonic() + 180
    loaded_threshold_mib = 10_000  # N1.7 is ~6 GB on the L4
    while monotonic() < deadline:
        try:
            used = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ]
                )
                .decode()
                .strip()
                .splitlines()[0]
            )
        except Exception:
            used = 0
        if used > loaded_threshold_mib:
            print(f"[setup] GR00T model loaded (gpu_mem={used} MiB)")
            return result
        sleep(5)
    raise RuntimeError(
        "GR00T model didn't reach load threshold within 180 s. "
        "Check `docker logs <container>` for stderr."
    )


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
    p.add_argument(
        "--auto-server",
        dest="auto_server",
        action="store_true",
        default=True,
        help="(--policy=groot only) Bring up the GR00T inference service via "
        "`gr00t_inference(action='lifecycle', lifecycle='full', ...)` before "
        "the eval and tear it down on exit. Default: enabled.",
    )
    p.add_argument(
        "--no-auto-server",
        dest="auto_server",
        action="store_false",
        help="(--policy=groot only) Don't manage the inference service; "
        "expect one to already be listening on `--port`.",
    )
    p.add_argument(
        "--image",
        default="gr00t:latest",
        help="(--auto-server only) Docker image tag of the GR00T container.",
    )
    p.add_argument(
        "--container",
        default="gr00t-libero-mujoco",
        help="(--auto-server only) Docker container name to (re)use.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="(--auto-server only) Where to cache the HF checkpoint. "
        "Default: `~/.cache/strands_robots/checkpoints/`.",
    )
    args = p.parse_args()

    suite = _suite_for_task(args.task)

    # Build the policy_config dict that the agent will pass through to
    # `evaluate_benchmark`. We construct it deterministically here so
    # the agent doesn't have to invent dict literals from the prompt.
    if args.policy == "groot":
        policy_phrase = (
            f"using the GR00T policy with `policy_provider='groot'` and "
            f"`policy_config={{'host': 'localhost', 'port': {args.port}, "
            f"'data_config': 'libero_panda', 'groot_version': 'n1.7'}}`"
        )
    else:
        policy_phrase = "using the mock policy (`policy_provider='mock'`)"

    server_handle = _bring_up_gr00t_server(args, suite)

    sim = Simulation(tool_name="libero_sim", mesh=False)
    try:
        sim.create_world()
        # Pre-add a Panda named ``robot`` so:
        #   1. evaluate_benchmark's pre-flight check (`No robots in
        #      sim`) passes BEFORE on_episode_start runs scene loading.
        #   2. The resolved-name `evaluate_benchmark` picks up here
        #      survives the rename that LIBERO scene MJCFs do — the
        #      scenes ship a Franka Panda named `robot` (LIBERO/
        #      RoboSuite convention), so picking the same name here
        #      keeps the resolved robot stable across `on_episode_start`.
        sim.add_robot("robot", data_config="panda")

        registered = load_libero_suite(suite)
        if not registered:
            raise RuntimeError(
                f"load_libero_suite({suite!r}) registered 0 tasks. "
                "Apply upstream fix from strands-labs/robots#147 if it isn't merged."
            )
        if args.task not in registered:
            # Spec's default `libero-spatial-pick_up_the_red_cube` is
            # aspirational; fall back to the first registered task.
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
            f"--seed={args.seed}--policy={args.policy}--agent"
        )
        video_dir = _date_dir()
        recording_cameras = (
            ["image", "wrist_image"] if args.policy == "groot" else ["default"]
        )

        # Pre-warm the LIBERO scene so the cameras the GR00T server
        # expects (`image`, `wrist_image`) are registered before
        # recording / inference starts. Same block as run_mujoco.py;
        # see that file for the longer rationale on ordering.
        if args.policy == "groot":
            from strands_robots.simulation.benchmark import get_benchmark

            spec = get_benchmark(args.task)
            if spec.scene_path is None and getattr(spec, "_auto_generate_scene", False):
                generated = spec._generate_scene_from_bddl()
                if generated:
                    spec.scene_path = generated
            if spec.scene_path:
                sim.load_scene(spec.scene_path)
                if hasattr(spec, "prewarm"):
                    spec.prewarm(sim)
                if "robot" not in sim.list_robots():
                    sim.add_robot("robot", data_config="panda")

        # Start the synchronous recorder on the script's main thread
        # (subprocess.Popen / signal-handling-sensitive setup is safer
        # here than on the Strands worker thread that the agent's tool
        # dispatch will run on). We capture `on_frame_cb` / `finalize_cb`
        # in the outer closure: `on_frame_cb` is wired into the eval
        # via the @tool wrapper below; `finalize_cb` is invoked from
        # main after the agent returns. Calling finalize from the
        # worker thread (which is what an in-wrapper finalize did)
        # crashes imageio's FFmpeg pipe with `BrokenPipeError` plus
        # glibc `malloc_consolidate()` errors — the FFmpeg subprocess
        # is sensitive to which thread spawned it.
        start_result = sim.start_cameras_recording_synchronous(
            cameras=recording_cameras,
            output_dir=video_dir,
            name=rec_name,
            # Explicit dims matching `run_mujoco.py`'s daemon-thread
            # output. Without these the eval-thread renderer cache may
            # produce variable-shape arrays in the buffer that crash
            # imageio's FFmpeg encoder mid-stream.
            width=640,
            height=480,
        )
        if start_result.get("status") != "success":
            raise RuntimeError(
                f"start_cameras_recording_synchronous failed: {start_result}"
            )
        sync = next(c["json"] for c in start_result["content"] if "json" in c)
        on_frame_cb = sync["on_frame"]
        finalize_cb = sync["finalize"]

        # Define a thin `@tool` wrapper that calls
        # `evaluate_benchmark(on_frame=...)` with the synchronous
        # recorder's on_frame closure. Necessary because the closure
        # can't cross Strands' tool-dispatch JSON boundary; the
        # wrapper keeps it in Python scope. The agent picks the
        # wrapper and fills its kwargs from natural language.
        # See `strands-labs/robots#191` (issue) / PR #192 (fix) for
        # the synchronous-mode API design and the daemon-thread
        # artifact it replaces.
        @tool
        def run_libero_eval(
            benchmark_name: str,
            n_episodes: int,
            seed: int,
            policy_provider: str,
            policy_config: dict,
        ) -> dict:
            """Run a LIBERO benchmark with per-step synchronous camera recording.

            Wires the outer-scope ``on_frame_cb`` into
            ``evaluate_benchmark(on_frame=...)`` so the eval thread
            renders cameras at known sync points (post-step),
            eliminating the daemon-recorder / ``mjData`` race that
            otherwise produces greenish gradient frames under
            multi-threaded eval (Strands ``Agent`` tool dispatch
            from inside an asyncio event loop).

            The ``finalize_cb`` for flushing buffers to MP4 is
            invoked from the script main thread after this tool
            returns — keeping FFmpeg's subprocess.Popen off the
            Strands worker thread.

            Args:
                benchmark_name: Registered LIBERO benchmark task ID
                    (e.g. ``libero-10-LIVING_ROOM_SCENE5_…``).
                n_episodes: Number of episodes to run.
                seed: Master RNG seed.
                policy_provider: ``"mock"`` or ``"groot"``.
                policy_config: Provider-specific kwargs.

            Returns:
                The standard ``evaluate_benchmark`` result dict
                (``success_rate``, per-episode steps, etc.).
            """
            return sim.evaluate_benchmark(
                benchmark_name=benchmark_name,
                n_episodes=n_episodes,
                seed=seed,
                policy_provider=policy_provider,
                policy_config=policy_config,
                robot_name="robot",
                on_frame=on_frame_cb,
            )

        agent = Agent(tools=[run_libero_eval])
        t0 = time.time()
        try:
            result = agent(
                f"Use the `run_libero_eval` tool to run the LIBERO benchmark "
                f"'{args.task}' for {args.n_episodes} episodes with seed "
                f"{args.seed}, {policy_phrase}. The world, robot, scene, and "
                f"output paths have already been set up — just call the tool "
                f"with these kwargs. Once it returns, parse the `success_rate` "
                f"field from the JSON payload and report it as a percentage of "
                f"the {args.n_episodes} episodes."
            )
        finally:
            # Always finalize, even if the agent / eval raised. Runs
            # on the main thread (not the Strands worker), so FFmpeg
            # subprocess.Popen / Python signal handling are happy.
            finalize_result = finalize_cb()
            print(f"[finalize] {finalize_result.get('content', [{}])[0].get('text', '?')}")
        wall_time = time.time() - t0
        print(result)
        video_path = os.path.join(video_dir, f"{rec_name}__{recording_cameras[0]}.mp4")
        print(
            f"[agent-eval] policy={args.policy} task={args.task} "
            f"wall_time={wall_time:.1f}s videos={video_path}"
        )
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
        # Tear down the GR00T inference container if we brought it up.
        if server_handle is not None:
            gr00t_inference(
                action="lifecycle", lifecycle="teardown", container_name=args.container
            )


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
