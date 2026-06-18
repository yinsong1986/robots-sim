#!/usr/bin/env python3
"""LIBERO on the Isaac Sim backend (``strands-robots-sim[isaac]``).

Companion to ``run_mujoco.py``: same CLI shape, same two grep-stable
output lines, same ``evaluate_benchmark(...)`` driver. Differs in
backend choice (``IsaacSimulation`` instead of MuJoCo's
``Simulation``), real-asset robot loading (loads the bundled Franka
Panda USD via ``add_robot(usd_path=...)`` rather than a LIBERO MJCF),
and an explicit ``add_camera(...)`` call (Isaac doesn't auto-attach
viewport cameras the way MuJoCo does -- the camera prim has to land
on the stage before ``render`` / recorder pulls from it).

Robot asset
-----------
By default this loads Isaac Sim's bundled **Franka Panda USD**,
resolved from the assets root
(``get_assets_root_path()/Isaac/Robots/Franka/franka.usd`` -- reachable
over HTTPS from the Omniverse CDN, no local Nucleus required). Override
with ``--robot-usd PATH`` or ``--robot-urdf PATH`` to load your own
asset.

This deliberately loads a **real** robot, not the procedural builder
(``add_robot(data_config="panda")``): the procedural Panda is a
kinematically approximate stick-figure (right joint count, wrong link
geometry / masses / joint origins) -- fine for lifecycle smoke tests,
useless for a LIBERO manipulation policy whose end-effector targets
depend on correct kinematics. Loading the real Franka USD constructs
a true ``omni.isaac.core.articulations.Articulation`` whose joints are
observable via ``get_observation`` and actuatable via ``send_action``
(GPU-validated: 9 DoF -- 7 arm + 2 fingers).

Dependency status (as of 2026-06)
---------------------------------
The real-asset robot load rides on the USD- / URDF-loaded
``add_robot`` paths:

* `PR #63 <https://github.com/strands-labs/robots-sim/pull/63>`_ /
  `PR #70 <https://github.com/strands-labs/robots-sim/pull/70>`_ --
  ``add_robot(usd_path=...)`` Articulation construction (the default
  path this script uses). **Merged.**
* `PR #64 <https://github.com/strands-labs/robots-sim/pull/64>`_ /
  `PR #70 <https://github.com/strands-labs/robots-sim/pull/70>`_ --
  ``add_robot(urdf_path=...)`` for the ``--robot-urdf`` override. The
  URDF importer module path differs across releases; the backend
  tries the modern ``isaacsim.asset.importer.urdf`` first and falls
  back to the legacy ``omni.importer.urdf`` for pre-4.5 builds.

The camera / video path rides on:

* `PR #61 (add_camera Phase 2) <https://github.com/strands-labs/robots-sim/pull/61>`_
  + `PR #62 (render frame-path) <https://github.com/strands-labs/robots-sim/pull/62>`_
  -- both merged. ``--policy=mock`` works fully (doesn't read images);
  ``--policy=groot`` will read from the camera handle.

Tracks `#15 <https://github.com/strands-labs/robots-sim/issues/15>`_
(R8 — example file). Filename is ``run_isaac.py`` rather than the
issue's pre-rescope ``libero_isaac.py`` to match the post-rescope
``examples/libero/run_<backend>.py`` layout that ``run_mujoco.py``
established (see ``examples/libero_example.py:126``).

Usage
-----
::

    # 1) Smoke test, GPU + Isaac Sim required (see is_available()
    #    error path for setup hints). Loads the default Franka USD:
    python examples/libero/run_isaac.py --policy mock --n-episodes 5

    # 1b) Bring your own robot asset:
    python examples/libero/run_isaac.py --policy mock --robot-usd /path/to/robot.usd
    python examples/libero/run_isaac.py --policy mock --robot-urdf /path/to/robot.urdf

    # 2) Real LIBERO eval against `nvidia/GR00T-N1.7-LIBERO`. Same
    #    GR00T-inference-container lifecycle as `run_mujoco.py`.
    python examples/libero/run_isaac.py --policy groot --port 8000 --n-episodes 50

Requires
--------
``pip install 'strands-robots-sim[isaac]' 'strands-robots[benchmark-libero]'``
plus a working Isaac Sim 6.0+ install on the host (RTX GPU, Ubuntu
22.04+, CUDA 12+). ``strands-robots-sim`` requires Python 3.12 (the
interpreter bundled by Isaac Sim 6.0). On a non-Isaac host the script exits early with a
diagnostic from :meth:`IsaacSimulation.is_available` rather than
crashing on the first ``omni.*`` / ``isaacsim.*`` import.

Verification status (as of 2026-06)
-----------------------------------
Target runtime is the Isaac Sim 6.0 NGC docker image
(``nvcr.io/nvidia/isaac-sim:6.0``, Python 3.12, RTX/L4 GPU, headless),
matching the ``isaacsim>=6.0`` / ``requires-python>=3.12`` migration.
The end-to-end lifecycle was previously validated on the Isaac Sim 4.5
image (``nvcr.io/nvidia/isaac-sim:4.5.0``); the dual-path
``isaacsim.*`` / ``omni.isaac.*`` imports in
``strands_robots_sim/isaac/simulation.py`` keep the same code path
working on both. The
script runs past ``IsaacSimulation.is_available`` → ``create_world``
→ ``add_robot`` (real Franka USD over the Omniverse CDN) →
``add_camera`` → physics ``step``. ``--policy=mock --n-episodes=5``
exercises the full lifecycle except for ``evaluate_benchmark``, which
additionally depends on the LIBERO benchmark suite being importable
inside Isaac's bundled Python (``strands-robots`` interpreter
constraint -- see `#71 <https://github.com/strands-labs/robots-sim/issues/71>`_).
On a non-Isaac host (no GPU, no Omniverse) ``is_available()`` still
short-circuits cleanly with the install-hint reason string, so this
file remains safe to import / lint on CPU-only CI runners.
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
        "--robot-usd",
        default=None,
        help="Path / URL to a USD robot asset to load via add_robot(usd_path=...). "
        "Default: Isaac Sim's bundled Franka Panda resolved from the assets root "
        "(`get_assets_root_path()/Isaac/Robots/Franka/franka.usd`). Mutually "
        "exclusive with --robot-urdf.",
    )
    p.add_argument(
        "--robot-urdf",
        default=None,
        help="Path to a URDF robot asset to load via add_robot(urdf_path=...). "
        "Mutually exclusive with --robot-usd. Converted to USD on import via "
        "the Isaac URDF importer.",
    )
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


def _resolve_hf_token() -> str:
    """Resolve a HuggingFace token for the gated GR00T checkpoint download.

    Prefers the ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) environment
    variable -- CI / container environments typically inject the token that
    way and don't have the ``~/.cache/huggingface/token`` file that
    ``huggingface-cli login`` writes. Falls back to that file for interactive
    dev boxes. Raises if neither is present.
    """
    from pathlib import Path

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    hf_token_path = Path("~/.cache/huggingface/token").expanduser()
    if hf_token_path.is_file():
        return hf_token_path.read_text().strip()
    raise RuntimeError(
        "--policy groot needs an HF token (Cosmos-Reason2-2B is gated). "
        "Set the HF_TOKEN env var (preferred for CI), or run "
        "`huggingface-cli login` to write ~/.cache/huggingface/token, then retry."
    )


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

    from strands_robots.tools import gr00t_inference

    hf_token = _resolve_hf_token()
    result = gr00t_inference(
        action="lifecycle",
        lifecycle="full",
        image_name=args.image,
        hf_repo="nvidia/GR00T-N1.7-LIBERO",
        hf_subfolder=suite,
        hf_local_dir=args.checkpoint_dir,
        container_name=args.container,
        hf_token=hf_token,
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


def _resolve_robot_asset(args: argparse.Namespace) -> "tuple[str | None, str | None]":
    """Resolve which robot asset to load → ``(usd_path, urdf_path)``.

    Precedence:

    1. ``--robot-urdf`` if given → ``(None, urdf)``.
    2. ``--robot-usd`` if given → ``(usd, None)``.
    3. Default → the bundled Franka Panda USD resolved from Isaac Sim's
       assets root (``get_assets_root_path()/Isaac/Robots/Franka/franka.usd``).
       Reachable over HTTPS from the public Omniverse CDN even without a
       local Nucleus server, so the example runs out-of-the-box on any
       Isaac Sim install with internet.

    Why a *real* asset (not the procedural builder): the procedural
    ``add_robot(data_config="panda")`` path produces a kinematically
    approximate stick-figure (correct joint count, wrong link geometry /
    masses / joint origins) -- fine for lifecycle smoke tests, useless
    for a LIBERO manipulation policy. Loading the real Franka USD gives
    the correct kinematics a GR00T / LIBERO policy expects.

    ``get_assets_root_path`` is imported lazily (only resolvable after
    ``create_world`` has booted ``SimulationApp``), so this is called
    *after* ``sim.create_world()`` in :func:`main`. Tries the modern
    ``isaacsim.storage.native`` namespace first (Isaac Sim 4.5+ supported
    path) and falls back to the legacy ``omni.isaac.nucleus`` shim --
    matches the dual-path policy in ``strands_robots_sim/isaac/simulation.py``.
    """
    if args.robot_urdf is not None:
        return None, args.robot_urdf
    if args.robot_usd is not None:
        return args.robot_usd, None
    try:
        from isaacsim.storage.native import (  # type: ignore[import-not-found]
            get_assets_root_path,
        )
    except ImportError:
        from omni.isaac.nucleus import (  # type: ignore[import-not-found]
            get_assets_root_path,
        )

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError(
            "Could not resolve the Isaac Sim assets root for the default Franka USD. "
            "Pass --robot-usd / --robot-urdf with an explicit asset path, or configure "
            "a Nucleus server / internet access for the Omniverse CDN."
        )
    return f"{assets_root}/Isaac/Robots/Franka/franka.usd", None


def main() -> None:
    args = _build_parser().parse_args()
    if args.robot_usd is not None and args.robot_urdf is not None:
        raise SystemExit("--robot-usd and --robot-urdf are mutually exclusive; pass at most one.")
    suite = _suite_for_task(args.task)

    # Fail-fast on hosts without Isaac Sim. The is_available probe is
    # cheap (it only does importlib.util.find_spec on omni.isaac.kit /
    # isaacsim; zero omni.* / isaacsim.* modules land in sys.modules)
    # so we run it before touching the GR00T container or any benchmark
    # side effects -- a misconfigured host should exit with a structured
    # error before a docker pull starts.
    available, reason = IsaacSimulation.is_available()
    if not available:
        raise RuntimeError(
            f"Isaac Sim is not available on this host: {reason}. "
            "Install Isaac Sim 6.0+ via the Omniverse Launcher / Isaac Lab / NGC "
            "Docker image and ensure `isaacsim` (6.0+, Python 3.12) or the legacy "
            "`omni.isaac.kit` is importable in this Python environment."
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

        # Load a *real* robot asset (default: Isaac's bundled Franka
        # Panda USD; override via --robot-usd / --robot-urdf). This
        # routes through add_robot's usd_path / urdf_path branch, which
        # constructs a real ``omni.isaac.core.articulations.Articulation``
        # (joints observable via get_observation, actuatable via
        # send_action) -- see PR #63 (USD) / PR #64 (URDF).
        #
        # Name "robot" is deliberately NOT a procedural alias (so the
        # usd_path / urdf_path branch is taken, not procedural lookup)
        # and matches the LIBERO/RoboSuite convention for the Franka.
        robot_usd, robot_urdf = _resolve_robot_asset(args)
        if robot_urdf is not None:
            print(f"[setup] loading robot from URDF: {robot_urdf}")
            result = sim.add_robot(name="robot", urdf_path=robot_urdf)
        else:
            print(f"[setup] loading robot from USD: {robot_usd}")
            result = sim.add_robot(name="robot", usd_path=robot_usd)
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

        # Keep the CLI-requested task distinct from the resolved one. The
        # default placeholder transparently falls back to the first registered
        # task, but the grep-stable line below must echo what the caller passed
        # (``requested_task``) so the R15 matrix driver can replay a run from
        # its recorded output -- re-running with the *resolved* value would
        # change behaviour (the fallback wouldn't fire). ``resolved_task`` is
        # what actually executes and what the filename / benchmark call use.
        requested_task = args.task
        resolved_task = _resolve_task(suite, args.task)
        args.task = resolved_task

        # Filename convention matches run_mujoco.py so the matrix
        # driver's video discovery glob (`rollouts/*/*--task=*.mp4`)
        # picks up Isaac and MuJoCo runs uniformly.
        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        rec_name = (
            f"{ts}--task={resolved_task}--n_eps={args.n_episodes}"
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
            benchmark_name=resolved_task,
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
        # ``task=`` echoes the CLI-REQUESTED task so the run is replayable
        # from this line; ``resolved_task=`` records what actually ran when
        # the aspirational-placeholder fallback rewrote it (identical to
        # ``task=`` in the common case). ``backend=isaac`` is the discriminator.
        print(f"benchmark_name={requested_task}")
        print(
            f"policy={args.policy}  task={requested_task}  "
            f"resolved_task={resolved_task}  "
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
