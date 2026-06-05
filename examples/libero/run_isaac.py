#!/usr/bin/env python3
"""LIBERO on the Isaac Sim backend (``strands-robots-sim[isaac]``).

Companion to ``run_mujoco.py``: same CLI shape, same two grep-stable
output lines, same ``evaluate_benchmark(...)`` driver. Differs in
backend choice (``IsaacSimulation`` instead of MuJoCo's
``Simulation``), procedural Panda construction (Isaac builds the
robot on its own USD stage rather than loading a LIBERO MJCF), and
an explicit ``add_camera(...)`` call (Isaac doesn't auto-attach
viewport cameras the way MuJoCo does -- the camera prim has to land
on the stage before ``render`` / recorder pulls from it).

This file is **draft / scaffolding** as of 2026-06: the ``--policy
mock`` path runs end-to-end against the procedural Panda that ships
in main today, but the resulting ``success_rate`` is structurally
``0.0`` until two pieces of unmerged Phase-2 wiring land. Both are
explicitly documented inline at their call sites:

* `#61 (add_camera Phase 2) <https://github.com/strands-labs/robots-sim/pull/61>`_
  — wires :meth:`IsaacSimulation.add_camera` to actually create the
  RTX camera prim so ``render`` returns non-blank frames. Without
  this, the GR00T policy's ``video.image`` observation key would be
  empty zero-arrays, so ``--policy=groot`` cannot reach
  ``success_rate>0``. ``--policy=mock`` doesn't read images and
  therefore tolerates the missing camera, but you'll notice the
  printed video path goes to a placeholder file.
* `#14 (procedural-robot articulation Phase 2) <https://github.com/strands-labs/robots-sim/issues/14>`_
  — the procedural ``add_robot("panda")`` branch currently leaves
  ``_RobotState.articulation`` as ``None``, so
  ``get_observation`` returns ``{}`` and ``send_action`` silently
  no-ops on procedural robots.
  `PR #63 <https://github.com/strands-labs/robots-sim/pull/63>`_
  /
  `PR #64 <https://github.com/strands-labs/robots-sim/pull/64>`_
  wired the USD- / URDF-loaded paths but the procedural path is its
  own slice on #14.

Once both land, the same script produces meaningful ``success_rate``
values without any code change here. The CLI surface is fixed now so
shell wrappers / matrix-driver scripts can be written today against
the eventual end-to-end behaviour.

Tracks `#15 <https://github.com/strands-labs/robots-sim/issues/15>`_
(R8 — example file). Filename is ``run_isaac.py`` rather than the
issue's pre-rescope ``libero_isaac.py`` to match the post-rescope
``examples/libero/run_<backend>.py`` layout that ``run_mujoco.py``
established (see ``examples/libero_example.py:126``).

Why no ``run_isaac_agent.py`` yet
---------------------------------
``run_mujoco_agent.py`` wraps the ``Simulation`` AgentTool's full
64-action enum so the agent can pick ``evaluate_benchmark`` from
natural language. The Isaac-side equivalent of that AgentTool surface
hasn't been built yet -- ``IsaacSimulation`` currently exposes the
``SimEngine`` ABC directly (good for programmatic flows) but is not
yet wrapped as an AgentTool. Once that wrapper exists (likely a
sibling-repo follow-up), a ``run_isaac_agent.py`` can mirror
``run_mujoco_agent.py``'s shape against it.

Usage
-----
::

    # 1) Smoke test, GPU + Isaac Sim 5.x required (see is_available()
    #    error path for setup hints):
    python examples/libero/run_isaac.py --policy mock --n-episodes 5

    # 2) Real LIBERO eval against `nvidia/GR00T-N1.7-LIBERO`. Same
    #    GR00T-inference-container lifecycle as `run_mujoco.py`. Only
    #    meaningful once #14's procedural-robot articulation Phase 2
    #    wiring lands; until then `success_rate` will be 0 because
    #    `get_observation` returns `{}` for procedural Panda.
    python examples/libero/run_isaac.py --policy groot --port 8000 --n-episodes 50

Requires
--------
``pip install 'strands-robots-sim[isaac]' 'strands-robots[benchmark-libero]'``
plus a working Isaac Sim 5.x install on the host (RTX GPU, Ubuntu
22.04+, CUDA 12+). On a non-Isaac host the script exits early with a
diagnostic from :meth:`IsaacSimulation.is_available` rather than
crashing on the first ``omni.*`` import.

Verification status
-------------------
Not yet run end-to-end against a real Isaac Sim install -- needs the
nightly GPU runner from
`#17 <https://github.com/strands-labs/robots-sim/issues/17>`_ /
`PR #59 <https://github.com/strands-labs/robots-sim/pull/59>`_
provisioned. CLI / control-flow / lint validation pass against the
current main on a CPU-only dev box (Isaac is *not* importable; the
``is_available()`` short-circuit in :func:`main` exits cleanly).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time

from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation


def _date_dir(date_root: str = "rollouts") -> str:
    """Return a date-stamped subdirectory of ``date_root``, creating it.

    Mirrors :func:`run_mujoco._date_dir` so the post-eval video
    aggregation script (R15 backend matrix) finds artifacts under the
    same convention regardless of backend.
    """
    out = os.path.join(date_root, _dt.date.today().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    return out


def _suite_for_task(task: str) -> str:
    """Auto-derive a LIBERO suite name from a benchmark task ID.

    Identical to :func:`run_mujoco._suite_for_task` -- LIBERO task IDs
    follow the same ``libero-<suite>-<task_stem>`` convention regardless
    of which backend is going to evaluate them.

    >>> _suite_for_task("libero-spatial-pick_up_the_red_cube")
    'libero_spatial'
    >>> _suite_for_task("libero-10-LIVING_ROOM_SCENE5_...")
    'libero_10'
    """
    parts = task.split("-", 2)
    if len(parts) < 3 or parts[0] != "libero":
        raise ValueError(
            f"--task must look like 'libero-<suite>-<task_stem>', got {task!r}. "
            "See `load_libero_suite` for registered names."
        )
    return f"libero_{parts[1]}"


def _build_parser() -> argparse.ArgumentParser:
    """Mirror :func:`run_mujoco.main`'s parser surface.

    Argument names / defaults / choices are kept identical to the
    MuJoCo file so a matrix-driver shell wrapper that supplies the
    same flags works against both backends (the only difference is
    which ``run_<backend>.py`` is invoked).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["mock", "groot"], default="mock")
    p.add_argument("--port", type=int, default=8000, help="GR00T inference port (only used with --policy=groot)")
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
        help="(--policy=groot only) Bring up the GR00T inference service before the eval.",
    )
    p.add_argument(
        "--no-auto-server",
        dest="auto_server",
        action="store_false",
        help="(--policy=groot only) Don't manage the inference service.",
    )
    p.add_argument(
        "--image",
        default="gr00t:latest",
        help="(--auto-server only) Docker image tag of the GR00T container.",
    )
    p.add_argument(
        "--container",
        default="gr00t-libero-isaac",
        help="(--auto-server only) Docker container name to (re)use. Defaults to "
        "the Isaac-specific name so Isaac and MuJoCo eval runs don't clobber "
        "each other's containers when run side-by-side on the same host.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="(--auto-server only) Where to cache the HF checkpoint. "
        "Default: `~/.cache/strands_robots/checkpoints/`.",
    )
    return p


def _orchestrate_groot_server(args: argparse.Namespace, suite: str) -> dict | None:
    """Bring up the GR00T inference container if ``--policy=groot``.

    Same lifecycle shape as :func:`run_mujoco.main`'s setup block:
    `gr00t_inference(action='lifecycle', lifecycle='full', ...)` →
    poll GPU memory until the model loads → return a handle the
    teardown path picks up. Pulled into its own function so the Isaac
    main is shorter and the MuJoCo-vs-Isaac comparison stays focused
    on the simulation-side differences.

    Returns ``None`` if the server doesn't need to be brought up
    (i.e. ``--policy=mock`` or ``--no-auto-server``).
    """
    if args.policy != "groot" or not args.auto_server:
        return None

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
        checkpoint_path=f"/data/checkpoints/{suite}",
        embodiment_tag="libero_sim",
        protocol="n1.7",
        use_sim_policy_wrapper=True,
        port=args.port,
    )
    if result.get("status") != "success":
        raise RuntimeError(f"gr00t_inference lifecycle=full failed: {result}")
    print(f"[setup] {result.get('message')}")

    # Wait for model load. Same heuristic as run_mujoco.py: GR00T-N1.7
    # is ~6 GB on the L4; cross 10 GiB GPU mem to consider it loaded.
    import subprocess
    from time import monotonic, sleep

    deadline = monotonic() + 180
    loaded_threshold_mib = 10_000
    while monotonic() < deadline:
        try:
            used = int(
                subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
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
            "GR00T model didn't reach load threshold within 180 s. Check `docker logs <container>` for stderr."
        )

    return result


def _resolve_task(suite: str, requested_task: str) -> str:
    """Register the LIBERO suite and resolve ``requested_task``.

    Identical fallback-to-first-registered semantics as the MuJoCo
    file: the spec's default placeholder
    ``libero-spatial-pick_up_the_red_cube`` isn't an actual LIBERO
    task name; if the user passes the default and it doesn't resolve,
    fall back to the first registered task with a clear note.
    Explicitly-supplied unknown tasks still error loudly.
    """
    from strands_robots.benchmarks.libero import load_libero_suite

    registered = load_libero_suite(suite)
    if not registered:
        raise RuntimeError(
            f"load_libero_suite({suite!r}) registered 0 tasks. "
            "Apply upstream fix from strands-labs/robots#147 if it isn't merged."
        )
    if requested_task in registered:
        return requested_task
    if requested_task == "libero-spatial-pick_up_the_red_cube":
        fallback = next(iter(registered))
        print(
            f"NOTE: default --task {requested_task!r} isn't in real LIBERO "
            f"(it's the spec's aspirational placeholder); falling back "
            f"to first registered task {fallback!r}."
        )
        return fallback
    raise RuntimeError(f"--task {requested_task!r} is not in the {suite} suite. Available: {sorted(registered)[:3]}…")


def main() -> None:
    args = _build_parser().parse_args()
    suite = _suite_for_task(args.task)

    # Fail-fast on hosts without Isaac Sim. The is_available probe is
    # cheap (it only does importlib.util.find_spec on omni.isaac.kit;
    # zero omni.* modules land in sys.modules) so we run it before
    # touching the GR00T container or any benchmark side effects --
    # a misconfigured host should exit with a structured error
    # before a docker pull starts.
    available, reason = IsaacSimulation.is_available()
    if not available:
        raise RuntimeError(
            f"Isaac Sim is not available on this host: {reason}. "
            "Install Isaac Sim 5.x via the Omniverse Launcher / Isaac Lab / NGC "
            "Docker image and ensure `omni.isaac.kit` is importable in this "
            "Python environment."
        )

    # Bring up GR00T container (idempotent; no-op for --policy=mock).
    server_handle = _orchestrate_groot_server(args, suite)

    if args.policy == "groot":
        # Same client-side data_config + groot_version as run_mujoco.py
        # — see that file for the rationale on why both strings are
        # required (data_config tells the local client how to format
        # observations; groot_version forces 5D video serialization
        # for the N1.7 server).
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

    # Construct the Isaac sim. headless=True avoids opening a Kit
    # viewport (the GR00T eval doesn't need an interactive GUI). The
    # IsaacConfig dataclass is a pure-Python construct (no omni.*
    # imports), so this constructor is cheap and runs on a non-Isaac
    # host — the actual SimulationApp boot happens inside create_world().
    sim = IsaacSimulation(IsaacConfig(headless=True, num_envs=1))
    try:
        result = sim.create_world()
        if result.get("status") != "success":
            raise RuntimeError(f"create_world failed: {result}")

        # Procedural Panda — same robot shape as the LIBERO MJCF the
        # MuJoCo file's pre-warm path loads. ``data_config="panda"``
        # routes through the procedural builder shipped in PR #46
        # (`isaac/procedural.py:_build_panda`); no URDF/USD asset on
        # disk required.
        #
        # Phase-1 caveat: ``_RobotState.articulation`` is ``None`` for
        # procedural robots until #14's procedural-articulation slice
        # lands. ``evaluate_benchmark`` will still loop, but
        # ``get_observation`` returns ``{}`` and ``send_action``
        # silently no-ops on procedural robots, so ``success_rate`` is
        # 0 by construction. See module docstring for the gating PRs.
        result = sim.add_robot(name="robot", data_config="panda")
        if result.get("status") != "success":
            raise RuntimeError(f"add_robot failed: {result}")

        # Phase-2 camera: Isaac doesn't auto-attach viewport cameras
        # the way MuJoCo's mjData does. Add an explicit RTX camera at
        # the same over-the-shoulder vantage that the LIBERO ``image``
        # camera uses on MuJoCo (`agentview` ≈ [2, 0, 1.5] looking at
        # origin).
        #
        # Once #61 (add_camera Phase 2) merges, this constructs an
        # actual ``omni.isaac.sensor.Camera`` and ``render()`` returns
        # non-blank frames keyed off it. Pre-#61, this call silently
        # registers the path in the in-Python registry and ``render``
        # returns blank frames -- enough for ``--policy=mock`` (which
        # doesn't read images), insufficient for ``--policy=groot``.
        result = sim.add_camera(
            name="image",
            position=[2.0, 0.0, 1.5],
            target=[0.0, 0.0, 0.5],
            fov=60.0,
        )
        if result.get("status") != "success":
            raise RuntimeError(f"add_camera failed: {result}")

        args.task = _resolve_task(suite, args.task)

        # Filename convention matches run_mujoco.py so the matrix
        # driver's video discovery glob (`rollouts/*/*--task=*.mp4`)
        # picks up Isaac and MuJoCo runs uniformly.
        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        rec_name = (
            f"{ts}--task={args.task}--n_eps={args.n_episodes}"
            f"--seed={args.seed}--policy={args.policy}--backend=isaac"
        )
        video_dir = _date_dir()
        # Pre-#61, ``render()`` returns blank frames -- the recorder
        # would write all-black MP4s. Document the placeholder path
        # so post-eval analysis tools see the filename without
        # mistaking it for a successful capture. Once #61 lands +
        # the sibling-repo recorder integrates an Isaac-aware path
        # (separate slice), this becomes a real video filename.
        video_path = os.path.join(video_dir, f"{rec_name}__image.mp4.placeholder")

        t0 = time.time()
        result = sim.evaluate_benchmark(
            benchmark_name=args.task,
            # robot_name omitted for the same reason run_mujoco.py
            # omits it: the benchmark's on_episode_start may rename /
            # reload the robot; ``evaluate_benchmark`` auto-picks the
            # single robot when there's only one.
            n_episodes=args.n_episodes,
            seed=args.seed,
            **policy_kwargs,
        )
        wall_time = time.time() - t0

        if result.get("status") != "success":
            raise RuntimeError(f"evaluate_benchmark failed: {result}")

        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        success_rate = json_payload["success_rate"]

        # Two grep-stable lines for the R15 matrix script -- exact
        # same format as run_mujoco.py's so subprocess-and-parse
        # consumers don't have to special-case the Isaac backend.
        # ``backend=isaac`` is the discriminator on the second line.
        print(f"benchmark_name={args.task}")
        print(
            f"policy={args.policy}  task={args.task}  "
            f"success_rate={success_rate:.2f}  "
            f"wall_time={wall_time:.1f}s  videos={video_path}  backend=isaac"
        )
    finally:
        sim.destroy()
        if server_handle is not None:
            from strands_robots.tools import gr00t_inference

            gr00t_inference(action="lifecycle", lifecycle="teardown", container_name=args.container)


if __name__ == "__main__":
    main()
