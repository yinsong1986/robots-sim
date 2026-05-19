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

    # 2) Real LIBERO eval against `nvidia/GR00T-N1.7-LIBERO`. By default
    #    the script auto-orchestrates the GR00T inference service via the
    #    upstream `gr00t_inference(action="lifecycle", lifecycle="full",
    #    ...)` tool — it builds the n1.7 container if missing, downloads
    #    the right `libero_<suite>/` sub-checkpoint, runs the container,
    #    and starts the inference server before the eval, then tears down
    #    on exit. Each step is idempotent so re-runs are cheap.
    #
    #    Pre-condition: HF token at `~/.cache/huggingface/token` with
    #    access to `nvidia/Cosmos-Reason2-2B` (the gated VLM backbone) +
    #    Docker + an NVIDIA GPU.
    python examples/libero_mujoco.py --policy groot --port 8000 --n-episodes 50

    # 2b) If you'd rather manage the inference service yourself
    #     (multi-eval session, custom container config, etc.), pass
    #     --no-auto-server and run the lifecycle tool ahead of time.
    #     The setup commands live in `strands-labs/robots#148`'s
    #     "Reproduction" section.
    python examples/libero_mujoco.py --policy groot --no-auto-server --port 8000

    # 3) Different LIBERO suite + task. Suite is auto-derived from --task,
    #    so the lifecycle tool downloads the matching `libero_<suite>/`
    #    sub-checkpoint:
    python examples/libero_mujoco.py \\
        --policy groot --port 8000 \\
        --task libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_…

    # 4) Round 43 (#168) — upstream-aligned engine.
    #    `--engine=libero_offscreen_render` swaps the legacy MuJoCo backend
    #    (auto-generated scene + custom OSC controller) for one that wraps
    #    upstream's `OffScreenRenderEnv` directly. Physics + rendering +
    #    action dispatch all delegate to robosuite, matching NVIDIA's
    #    reference eval setup byte-for-byte. Required for `success_rate>0`
    #    on the in-process variant (round 44 verified 5/5 in 73 s on
    #    libero-10/SCENE5; with the `--policy=groot --auto-server` (ZMQ
    #    client) variant the same checkpoint still returns 0/5 — separate
    #    upstream investigation track per PR #168).
    python examples/libero_mujoco.py \\
        --policy groot --engine libero_offscreen_render \\
        --task libero-10-LIVING_ROOM_SCENE5_…

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

Recording is gated to the legacy ``--engine=mujoco`` backend.
``--engine=libero_offscreen_render`` skips it because the upstream
``OffScreenRenderEnv`` doesn't expose a per-call recorder; rollout
inspection on that backend uses the run-time ``--policy=groot`` log
output + offline frame dumps.

Verification status (`--policy=groot` end-to-end, after PR #175 lands)
----------------------------------------------------------------------
After [`strands-labs/robots#168`](https://github.com/strands-labs/robots/pull/168)
(rounds 36-44 — squashed at upstream `34f8c37`) +
[`#172`](https://github.com/strands-labs/robots/pull/172) (closes #169:
ZMQ wire-format `image_rotation_180` in LOCAL mode + engine V-flip
correction) + [`#173`](https://github.com/strands-labs/robots/pull/173)
(closes #170: BDDL evaluator agreement with `env.check_success`) +
[`#175`](https://github.com/strands-labs/robots/pull/175) (closes #171
+ #176: MuJoCoSimEngine state observation parity, OSC torque parity,
gripper home pose, BDDL `_main` suffix fallback), this example file's
``--policy=groot`` paths reach `success_rate > 0` end-to-end on real
hardware. Validated 2026-05-19 on the L4 / Docker dev box against
`nvidia/GR00T-N1.7-LIBERO/libero_10`,
`libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_…`, 5 episodes, seed 42:

==================================  ============  ==========  ===========================
``--engine``                        success_rate  wall_time   notes
==================================  ============  ==========  ===========================
``mujoco`` (legacy default)             **0.80**  168 s       4/5 — best after PR #175
``libero_offscreen_render``             **0.40**  389 s       2/5 — wraps upstream env
``mujoco`` (pre-PR #175, 2026-05-18)        0.00  596 s       0/5 — gap predates fix
==================================  ============  ==========  ===========================

The `mujoco` engine OUTPERFORMS `libero_offscreen_render` after
PR #175 because PR #175 specifically tuned the MuJoCoSimEngine path
(state byte-equivalent at canonical init, OSC torque parity at
identical state, settle step in `on_episode_start`, gripper home
pose in snapshot branch). The offscreen engine uses upstream's
``OffScreenRenderEnv`` directly so doesn't need analogous tuning,
but doesn't share the speedup either.

NON-DETERMINISM: success rate varies run-to-run because the eval's
torch / cuDNN globals aren't seeded by ``_set_eval_seed`` when called
through the in-process diagnostic at
``/tmp/opencode/eval-runs/r44_inprocess_eval.py``; the same seed has
yielded both 5/5 and 4/5 in different runs. The example file goes
through ``PolicyRunner._evaluate_with_spec`` which DOES call
``_set_eval_seed`` (round 38), so its number should be more stable —
but the ZMQ docker server has its own non-determinism sources we
haven't bisected. Acceptance criterion is "success_rate > 0", not
a specific number.

Round-by-round chronicle:
- PR #168 rounds 36-44: structural alignment to NVIDIA's reference
  (`action_horizon=8`, `max_steps=720`, image V-flip, image dims,
  gripper polarity, V-flip, dim publication, RLDS->robosuite gripper
  transform, plus `LiberoOffScreenRenderEngine` SimEngine backend)
- PR #172 (closes #169): ZMQ-wire image rotation correction so policy
  via ZMQ sees the same orientation as in-process
- PR #173 (closes #170): BDDL evaluator agrees with
  ``env.check_success`` so successful trajectories actually count
- PR #175 (closes #171 + #176): OSC torque parity + state parity
  + gripper home pose + BDDL ``_main`` suffix fallback

See PR #168 + PR #175 comments for the full round-by-round bisect.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time

from strands_robots.benchmarks.libero import load_libero_suite
from strands_robots.simulation import Simulation, create_simulation


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
        "--engine",
        choices=["mujoco", "libero_offscreen_render"],
        default="mujoco",
        help=(
            "Simulation backend. Default ``mujoco`` is the general-purpose "
            "MuJoCoSimEngine that auto-generates LIBERO scenes via the "
            "``libero`` package's procedural generator and drives them "
            "through our custom OSC controller. ``libero_offscreen_render`` "
            "(landed in strands-labs/robots#168 round 43) wraps upstream's "
            "``OffScreenRenderEnv`` directly — physics + rendering + action "
            "dispatch all delegate to robosuite, matching NVIDIA's reference "
            "eval setup byte-for-byte. Use the latter when running against "
            "``nvidia/GR00T-N1.7-LIBERO`` (round 44 verified ``success_rate=1.0`` "
            "via this backend + in-process policy on libero-10/SCENE5)."
        ),
    )
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

    # When `--policy=groot --auto-server` (default), bring up the GR00T
    # inference service via the upstream lifecycle tool: build the n1.7
    # container if missing → download the right `libero_<suite>/`
    # sub-checkpoint → start the container → start the server. Each
    # sub-step is idempotent so re-runs are cheap. Pass `--no-auto-server`
    # if you're managing the service yourself.
    server_handle = None
    if args.policy == "groot" and args.auto_server:
        from pathlib import Path

        from strands_robots.tools import gr00t_inference

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
            # The lifecycle tool mounts `hf_local_dir` (or its default cache
            # dir when `None`) → `/data/checkpoints`, and the HF download
            # places `<suite>/...` directly under that. So the in-container
            # path is `/data/checkpoints/<suite>`, NOT
            # `/data/checkpoints/GR00T-N1.7-LIBERO/<suite>`.
            checkpoint_path=f"/data/checkpoints/{suite}",
            embodiment_tag="libero_sim",
            protocol="n1.7",
            use_sim_policy_wrapper=True,
            port=args.port,
        )
        if result.get("status") != "success":
            raise RuntimeError(f"gr00t_inference lifecycle=full failed: {result}")
        server_handle = result
        print(f"[setup] {result.get('message')}")

        # The lifecycle tool returns success when the server's port is
        # bound, but the model itself loads asynchronously after that —
        # a too-eager `evaluate_benchmark` call can race the load and
        # hang on the first inference request. Wait until GPU memory
        # crosses a heuristic load-complete threshold before continuing.
        # Filed upstream as part of #148's lifecycle-readiness follow-up;
        # remove this loop once `gr00t_inference` blocks until ready.
        import subprocess
        from time import sleep, monotonic

        deadline = monotonic() + 180
        loaded_threshold_mib = 10_000  # N1.7 model is ~6 GB on the L4
        while monotonic() < deadline:
            try:
                used = int(
                    subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
                    )
                    .decode()
                    .strip()
                    .splitlines()[0]
                )
            except Exception:
                used = 0
            if used > loaded_threshold_mib:
                print(f"[setup] GR00T model loaded (gpu_mem={used} MiB)")
                break
            sleep(5)
        else:
            raise RuntimeError(
                "GR00T model didn't reach load threshold within 180 s. "
                "Check `docker logs <container>` for stderr."
            )

    if args.policy == "groot":
        # Client-side `data_config="libero_panda"` — this is the registered
        # key in `strands_robots.policies.groot.DATA_CONFIG_MAP` that tells
        # the local `Gr00tPolicy` how to format LIBERO observations into the
        # GR00T-N1.7 input layout. Note this is *separate from* the server's
        # `--embodiment-tag libero_sim` (an alias of `LIBERO_PANDA` per the
        # checkpoint's `embodiment_id.json`); the two sides happen to mean
        # the same thing but the strings are not interchangeable.
        # `groot_version="n1.7"` is required when the client doesn't have
        # the upstream `gr00t` package installed (auto-detection only works
        # when it does); without it the client serializes 4D video and the
        # N1.7 server rejects with "must be (B, T, H, W, C), got (B, H, W, C)".
        policy_kwargs = {
            "policy_provider": "groot",
            "policy_config": {
                "host": "localhost",
                "port": args.port,
                "data_config": "libero_panda",
                "groot_version": "n1.7",
            },
        }
    else:
        policy_kwargs = {"policy_provider": "mock"}

    # Round-43 (#168) — when ``--engine=libero_offscreen_render``, route
    # through the new SimEngine backend that delegates to upstream's
    # ``OffScreenRenderEnv``. Bypasses our procedural-scene + custom-OSC
    # path in favour of robosuite's training-distribution-equivalent
    # physics+render. ``Simulation`` (the AgentTool) wraps
    # ``MuJoCoSimEngine`` directly and is the legacy default; both
    # backends implement the same SimEngine ABC so the rest of this
    # script doesn't care which one is in use.
    if args.engine == "libero_offscreen_render":
        sim = create_simulation("libero_offscreen_render")
    else:
        sim = Simulation(tool_name="libero_sim", mesh=False)
    try:
        sim.create_world()
        # Pre-add a Panda named ``robot`` so:
        #   1. evaluate_benchmark's pre-flight check (`No robots in sim`)
        #      passes BEFORE on_episode_start runs scene loading.
        #   2. The resolved-name `evaluate_benchmark` picks up here
        #      survives the rename that LIBERO scene MJCFs do — the
        #      scenes ship a Franka Panda named `robot` (LIBERO/RoboSuite
        #      convention), so picking the same name client-side keeps
        #      the resolved robot stable across `on_episode_start`.
        sim.add_robot("robot", data_config="panda")

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
        # Pick the camera to record from. The LIBERO scene auto-loaded by
        # `LiberoAdapter` (per `strands-labs/robots#165`) supplies cameras
        # named `image` (third-person agentview) and `wrist_image`
        # (gripper view). Without LIBERO loaded — e.g. on `--policy=mock`
        # paths that hit the scene-gen ImportError fallback — only the
        # world's `default` camera exists.
        recording_camera = "image" if args.policy == "groot" else "default"
        recording_cameras = (
            ["image", "wrist_image"] if args.policy == "groot" else ["default"]
        )

        # Pre-warm the scene so `image` actually exists at recording-start
        # time. `start_cameras_recording` looks up the camera by name in
        # the live model and resolving fails if the scene hasn't been
        # loaded yet — but `on_episode_start` (where scene-load happens)
        # only runs *inside* `evaluate_benchmark`. We force the
        # auto-generation + load here so the camera is registered before
        # the recorder starts; subsequent per-episode reloads in the eval
        # loop reuse the cached scene_path so the camera name stays
        # stable across them.
        #
        # Round 43 (#168) — only relevant on the legacy ``mujoco`` engine
        # path. The ``libero_offscreen_render`` engine constructs its
        # own ``OffScreenRenderEnv`` lazily inside
        # ``LiberoAdapter._on_episode_start_offscreen`` (the round-43
        # fast-path) and doesn't need a separate scene load step here.
        if args.policy == "groot" and args.engine == "mujoco":
            from strands_robots.simulation.benchmark import get_benchmark
            import random as _random

            spec = get_benchmark(args.task)
            if spec.scene_path is None and getattr(spec, "_auto_generate_scene", False):
                generated = spec._generate_scene_from_bddl()
                if generated:
                    spec.scene_path = generated
            if spec.scene_path:
                sim.load_scene(spec.scene_path)
                # Prewarm BEFORE the redundant-Panda check below, so
                # prewarm's _register_default_robot wraps the
                # scene-supplied Panda first → list_robots() returns
                # ['robot'] → the if-check below is False → no
                # redundant add_robot recompile that would change
                # model.nq away from the LIBERO width init_states[0]
                # is sized for (#168 round 18 finding).
                if hasattr(spec, "prewarm"):
                    spec.prewarm(sim)
                # Defensive fallback for non-LIBERO benchmarks that
                # don't expose `prewarm` and don't ship a Panda in
                # the loaded scene MJCF.
                if "robot" not in sim.list_robots():
                    sim.add_robot("robot", data_config="panda")

        # Camera recording is mujoco-engine specific. The
        # ``libero_offscreen_render`` engine has no ``start_cameras_recording``
        # — its observations come from upstream's ``OffScreenRenderEnv``
        # which doesn't expose a per-call recorder. Skip recording on that
        # backend; rollout MP4s from the ``mujoco`` engine remain the
        # canonical artefact for visual debugging.
        recording_active = args.engine == "mujoco"
        if recording_active:
            sim.start_cameras_recording(
                cameras=recording_cameras, output_dir=video_dir, name=rec_name
            )
        try:
            t0 = time.time()
            result = sim.evaluate_benchmark(
                benchmark_name=args.task,
                # robot_name omitted on purpose — `LiberoAdapter`'s scene
                # auto-generation loads a scene that names its Panda
                # ``robot`` (LIBERO/RoboSuite convention), so any
                # specific name we pre-resolve here is gone after
                # `on_episode_start`. `evaluate_benchmark` auto-picks
                # when there's only one robot, which is the LIBERO case.
                n_episodes=args.n_episodes,
                seed=args.seed,
                **policy_kwargs,
            )
            wall_time = time.time() - t0
        finally:
            if recording_active:
                sim.stop_cameras_recording()

        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        success_rate = json_payload["success_rate"]
        if recording_active:
            video_path = os.path.join(video_dir, f"{rec_name}__{recording_camera}.mp4")
        else:
            video_path = "(none — libero_offscreen_render engine doesn't record)"

        # Two grep-stable lines for R15 to subprocess-and-parse. Keep the
        # exact format (`policy=`, `task=`, `success_rate=`, `wall_time=`,
        # `videos=`) stable across rebases / refactors.
        print(f"benchmark_name={args.task}")
        print(
            f"engine={args.engine}  policy={args.policy}  task={args.task}  "
            f"success_rate={success_rate:.2f}  "
            f"wall_time={wall_time:.1f}s  videos={video_path}"
        )
    finally:
        sim.destroy()
        # Tear down the GR00T inference container if we brought it up.
        if server_handle is not None:
            from strands_robots.tools import gr00t_inference

            gr00t_inference(
                action="lifecycle", lifecycle="teardown", container_name=args.container
            )


if __name__ == "__main__":
    main()
