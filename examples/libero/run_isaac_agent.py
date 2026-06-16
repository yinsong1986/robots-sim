#!/usr/bin/env python3
"""LIBERO on Isaac Sim, driven by a Strands ``Agent`` in natural language.

Companion to ``run_mujoco_agent.py``: same prompt-to-tool-pick agent
shape, same one-shot ``evaluate_benchmark`` invocation, same GR00T
container lifecycle. Differs in backend choice
(``IsaacSimulation``) and the tool-wrapping shape: where the MuJoCo
file passes the ``Simulation`` instance directly to ``Agent(tools=...)``
because ``MuJoCoSimEngine`` inherits from ``AgentTool`` and exposes
its 64-action enum, this Isaac file wraps :meth:`IsaacSimulation.evaluate_benchmark`
in a single ``@tool``-decorated function and passes that to the agent.

Status (as of 2026-06)
----------------------
Validated end-to-end on the canonical Isaac Sim 4.5 NGC docker image
(``nvcr.io/nvidia/isaac-sim:4.5.0``, RTX/L4 GPU, headless): the script
runs past ``IsaacSimulation.is_available`` → ``create_world`` →
``add_robot`` (real Franka USD over the Omniverse CDN) → ``add_camera``,
and the agent invokes ``evaluate_isaac_benchmark`` on a one-tool
surface. ``evaluate_benchmark`` additionally depends on the LIBERO
benchmark suite being importable inside Isaac's bundled Python
(``strands-robots`` interpreter constraint -- see
`#71 <https://github.com/strands-labs/robots-sim/issues/71>`_); on a
host without that wired up, the agent's tool call surfaces the
``ImportError`` in its summary rather than crashing the process.

The earlier ``--policy=groot`` ``success_rate=0.0`` warning predates
PRs `#61 <https://github.com/strands-labs/robots-sim/pull/61>`_ /
`#70 <https://github.com/strands-labs/robots-sim/pull/70>`_ landing the
camera + Articulation Phase-2 wiring; with both merged on ``main``,
the procedural / real-asset robot load + camera frames are now wired
correctly. End-to-end ``--policy=groot`` numbers still need a host
that satisfies all of: Isaac Sim 4.5+, libero (BDDL files),
``strands-robots`` under Isaac's Python (#71), and a GR00T inference
container.

Why a single-tool wrapper instead of full enum
-----------------------------------------------
``run_mujoco_agent.py``'s preamble notes that earlier sessions
"reduced the agent's tool surface from the full 64-action ``Simulation``
enum to a single wrapper" and judged that **worse** for MuJoCo
because: (a) MuJoCo already has the AgentTool wiring, (b) the full
enum exercises the natural-language → action-pick → kwarg-fill flow
the agent demo is meant to showcase. The Isaac case is different:

- ``IsaacSimulation`` does NOT yet inherit ``AgentTool``; it extends
  the ``SimEngine`` ABC alone. Wiring it up to mirror MuJoCo's
  ``MuJoCoSimEngine(PhysicsMixin, RenderingMixin, RecordingMixin,
  RandomizationMixin, SimEngine, AgentTool)`` is its own (substantial)
  refactor on `#14 <https://github.com/strands-labs/robots-sim/issues/14>`_'s
  Phase 3 list.
- Until that refactor lands, the only way to give the agent a tool
  is the ``@tool`` decorator on a hand-written wrapper. The wrapper
  necessarily collapses the full surface -- the ``Simulation`` tool's
  64-action JSON enum has no ``IsaacSimulation`` equivalent yet.

So the agent here picks ``evaluate_isaac_benchmark`` from a 1-tool
surface, fills its kwargs from prompt context, runs, and summarises.
The agent demo still exercises the prompt → tool-call → kwarg-fill
path; it's just a degenerate "1-of-1 pick". Once the Isaac AgentTool
wrapper exists in either ``strands-robots-sim`` or ``strands-robots``,
this file should be migrated to the full-enum shape (single-line
diff: drop ``@tool`` wrapper + change ``Agent(tools=[...])`` arg) so
the agent regains the natural-language action-pick demo value.

What the script handles deterministically (NOT the agent)
---------------------------------------------------------
Same partition as ``run_mujoco_agent.py``. The agent owns ONE
decision (invoking ``evaluate_isaac_benchmark`` with kwargs filled
from prompt context); the script owns:

* ``is_available()`` short-circuit on a non-Isaac host -- the cheap
  ``importlib.util.find_spec("omni.isaac.kit") | find_spec("isaacsim")``
  probe runs before the GR00T container side effects so a misconfigured
  matrix run exits with a structured ``RuntimeError`` rather than wasting
  30 s on a docker pull.
* GR00T inference container lifecycle (start, wait-for-load, teardown
  on exit) via ``gr00t_inference(action='lifecycle', ...)`` -- same
  block as ``run_mujoco_agent.py``, container name defaulted to
  ``gr00t-libero-isaac`` so Isaac and MuJoCo eval runs don't collide
  on the same host.
* Procedural Panda + RTX camera setup. Isaac doesn't auto-attach
  viewport cameras the way MuJoCo's ``mjData`` does, so the camera
  prim has to land on the stage before any ``render`` / recorder
  pulls from it.
* MP4 recording is intentionally skipped here (the Isaac-side
  recorder integration is a separate slice). The agent's printed
  summary documents the absence rather than crashing on a
  ``start_cameras_recording`` that doesn't exist on the Isaac side.

Owned by the agent: the single ``evaluate_isaac_benchmark(...)`` call
with ``benchmark_name`` + ``n_episodes`` + ``seed`` + ``policy_provider`` +
``policy_config`` filled from natural language, plus the natural-language
summary at the end.

Usage
-----
::

    # 1) Smoke test (mock policy; no GPU / Docker needed beyond Isaac
    #    Sim itself):
    python examples/libero/run_isaac_agent.py --policy mock --n-episodes 5

    # 2) Real run against `nvidia/GR00T-N1.7-LIBERO`. Script auto-
    #    orchestrates the GR00T inference container (idempotent). Pre-
    #    condition: HF token at `~/.cache/huggingface/token` (gated
    #    Cosmos-Reason2-2B backbone) + Docker + an NVIDIA GPU + Isaac
    #    Sim 4.5+ installed.
    python examples/libero/run_isaac_agent.py --policy groot --port 8000 --n-episodes 5

Requires
--------
- ``pip install 'strands-robots-sim[isaac]' 'strands-robots[benchmark-libero]' strands-agents``
- A configured LLM provider for Strands. Default is Anthropic Claude
  via AWS Bedrock -- see https://strandsagents.com/ for setup. Without
  one the ``Agent(...)`` call below raises an authentication /
  configuration error pointing at the SDK setup docs.
- Isaac Sim 4.5+ installed via Omniverse Launcher / Isaac Lab / NGC
  Docker image. Pure-Python ``pip install`` doesn't suffice.
- For ``--policy=groot``: Docker + an NVIDIA GPU + ~30 GB free disk
  for the GR00T checkpoint (cached across re-runs).

Notes
-----
- Output is non-deterministic by design (LLM-generated summary); the
  R15 backend matrix consumes ``run_isaac.py`` (sibling file) for
  grep-stable numbers.
- No video recording in this scaffold (Isaac-side recorder
  integration is a separate slice). Once that lands, this file gains
  a ``start_cameras_recording`` block matching the MuJoCo agent's
  shape.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time
from typing import Any

from strands import Agent, tool
from strands_robots.benchmarks.libero import load_libero_suite
from strands_robots.tools import gr00t_inference

from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

# Module-level handle for the @tool-wrapped function below.
# The wrapper has to access ``_sim`` from outer scope because
# ``@tool``'s OpenAPI-schema generator inspects the function's
# signature and would surface ``sim: IsaacSimulation`` to the LLM as
# a JSON-castable parameter -- which it isn't. Keeping ``_sim`` as a
# module attribute is the cleanest stopgap until the Isaac AgentTool
# wrapper lands and the indirection goes away.
_sim: IsaacSimulation | None = None


def _date_dir(date_root: str = "rollouts") -> str:
    out = os.path.join(date_root, _dt.date.today().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    return out


def _suite_for_task(task: str) -> str:
    """Auto-derive a LIBERO suite name from a benchmark task ID.

    Same shape as ``run_mujoco_agent.py``'s helper; see that file for
    the canonical doctest examples.
    """
    parts = task.split("-", 2)
    if len(parts) < 3 or parts[0] != "libero":
        raise ValueError(
            f"--task must look like 'libero-<suite>-<task_stem>', got {task!r}. "
            "See `load_libero_suite` for registered names."
        )
    return f"libero_{parts[1]}"


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


def _bring_up_gr00t_server(args: argparse.Namespace, suite: str) -> dict | None:
    """Start the GR00T inference container and block until model loads.

    Mirrors the lifecycle block in ``run_mujoco_agent.py`` so the Isaac
    agent file has identical "real-eval" plumbing. Returns the
    lifecycle handle (or ``None`` if ``--policy=mock`` /
    ``--no-auto-server``).
    """
    if args.policy != "groot" or not args.auto_server:
        return None

    import subprocess
    from time import monotonic, sleep

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

    deadline = monotonic() + 180
    loaded_threshold_mib = 10_000
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
        "GR00T model didn't reach load threshold within 180 s. Check `docker logs <container>` for stderr."
    )


@tool(
    name="evaluate_isaac_benchmark",
    description=(
        "Run a registered LIBERO benchmark on the configured Isaac Sim "
        "environment. The world, robot, and camera have already been "
        "set up by the calling script -- this tool only invokes the "
        "evaluation loop. Returns a status dict whose JSON payload "
        "carries `success_rate`, `avg_reward`, `avg_steps`, plus per-"
        "episode cumulative reward."
    ),
)
def evaluate_isaac_benchmark(
    benchmark_name: str,
    n_episodes: int = 10,
    seed: int = 42,
    policy_provider: str = "mock",
    policy_config: dict[str, Any] | None = None,
    instruction: str = "",
) -> dict[str, Any]:
    """Tool wrapper around :meth:`IsaacSimulation.evaluate_benchmark`.

    Stopgap until ``IsaacSimulation`` itself becomes a Strands
    ``AgentTool`` (Phase-3 work on #14). Forwards a fixed-shape
    subset of ``evaluate_benchmark``'s kwargs -- the ones the agent
    needs to fill from prompt context -- onto the module-scoped
    ``_sim`` instance configured in :func:`main`.

    Parameters
    ----------
    benchmark_name : str
        Registered LIBERO task name, e.g.
        ``"libero-spatial-pick_up_the_red_cube"``.
    n_episodes : int
        Number of episodes to roll out. Default 10.
    seed : int
        Master RNG seed for per-episode reproducibility. Default 42.
    policy_provider : str
        Strands policy registry key. ``"mock"`` (default) or
        ``"groot"``.
    policy_config : dict, optional
        Provider-specific kwargs. For ``policy_provider="groot"``,
        carries ``host`` / ``port`` / ``data_config`` /
        ``groot_version``.
    instruction : str
        Optional natural-language instruction forwarded to the policy.

    Returns
    -------
    dict
        Standard ``{"status", "content": [...]}`` envelope. On success,
        ``content[0]["json"]`` carries ``success_rate``, ``avg_reward``,
        ``avg_steps``, and per-episode cumulative reward.
    """
    if _sim is None:
        return {
            "status": "error",
            "content": [{"text": "evaluate_isaac_benchmark: _sim is not initialised. main() must run first."}],
        }
    return _sim.evaluate_benchmark(
        benchmark_name=benchmark_name,
        n_episodes=n_episodes,
        seed=seed,
        policy_provider=policy_provider,
        policy_config=policy_config,
        instruction=instruction,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Mirror ``run_mujoco_agent.py``'s parser surface.

    Argument names / defaults are kept identical to the MuJoCo file
    (and to ``run_isaac.py``) so a matrix-driver shell wrapper that
    supplies the same flags works against any of the three.
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
        help="USD robot asset for add_robot(usd_path=...). Default: bundled Franka "
        "Panda from the assets root. Mutually exclusive with --robot-urdf.",
    )
    p.add_argument(
        "--robot-urdf",
        default=None,
        help="URDF robot asset for add_robot(urdf_path=...). Mutually exclusive with --robot-usd.",
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
        help="(--auto-server only) Docker container name to (re)use. Defaults "
        "to the Isaac-specific name so Isaac and MuJoCo eval runs don't "
        "collide on the same host.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="(--auto-server only) Where to cache the HF checkpoint. "
        "Default: `~/.cache/strands_robots/checkpoints/`.",
    )
    return p


def _resolve_robot_asset(args: argparse.Namespace) -> "tuple[str | None, str | None]":
    """Resolve which robot asset to load → ``(usd_path, urdf_path)``.

    Same contract as ``run_isaac.py._resolve_robot_asset``: ``--robot-urdf``
    > ``--robot-usd`` > default Franka Panda USD from the assets root
    (``get_assets_root_path()/Isaac/Robots/Franka/franka.usd``). Loads a
    *real* robot rather than the procedural stick-figure (see
    ``run_isaac.py`` for the rationale). ``get_assets_root_path`` is
    imported lazily (only resolvable after ``create_world``) and tries
    the modern ``isaacsim.storage.native`` namespace first, falling back
    to the legacy ``omni.isaac.nucleus`` shim -- matches the dual-path
    policy in ``strands_robots_sim/isaac/simulation.py``.
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
    global _sim

    args = _build_parser().parse_args()
    if args.robot_usd is not None and args.robot_urdf is not None:
        raise SystemExit("--robot-usd and --robot-urdf are mutually exclusive; pass at most one.")
    suite = _suite_for_task(args.task)

    # Fail-fast on hosts without Isaac Sim. Same probe as run_isaac.py
    # -- runs before the GR00T container side effects so a CPU-only
    # host exits cleanly without wasting docker bandwidth.
    available, reason = IsaacSimulation.is_available()
    if not available:
        raise RuntimeError(
            f"Isaac Sim is not available on this host: {reason}. "
            "Install Isaac Sim 4.5+ via the Omniverse Launcher / Isaac Lab / NGC "
            "Docker image and ensure `omni.isaac.kit` (legacy) or `isaacsim` "
            "(4.5+ supported) is importable in this Python environment."
        )

    # Build the policy-config phrase the agent will paste into its
    # ``evaluate_isaac_benchmark`` call. Constructed deterministically
    # here so the agent doesn't have to invent dict literals from the
    # prompt -- mirrors run_mujoco_agent.py's policy_phrase pattern.
    if args.policy == "groot":
        policy_phrase = (
            f"with `policy_provider='groot'` and `policy_config={{'host': 'localhost', "
            f"'port': {args.port}, 'data_config': 'libero_panda', "
            f"'groot_version': 'n1.7'}}`"
        )
    else:
        policy_phrase = "with `policy_provider='mock'`"

    server_handle = _bring_up_gr00t_server(args, suite)

    _sim = IsaacSimulation(IsaacConfig(headless=True, num_envs=1))
    try:
        result = _sim.create_world()
        if result.get("status") != "success":
            raise RuntimeError(f"create_world failed: {result}")

        # Load a *real* robot asset (default: bundled Franka Panda USD;
        # override via --robot-usd / --robot-urdf). Routes through
        # add_robot's usd_path / urdf_path branch (real Articulation,
        # observable joints) rather than the procedural builder, which
        # produces a kinematically-approximate stick-figure unusable for
        # LIBERO. See run_isaac.py's _resolve_robot_asset docstring.
        robot_usd, robot_urdf = _resolve_robot_asset(args)
        if robot_urdf is not None:
            print(f"[setup] loading robot from URDF: {robot_urdf}")
            result = _sim.add_robot(name="robot", urdf_path=robot_urdf)
        else:
            print(f"[setup] loading robot from USD: {robot_usd}")
            result = _sim.add_robot(name="robot", usd_path=robot_usd)
        if result.get("status") != "success":
            raise RuntimeError(f"add_robot failed: {result}")

        # Phase-2 RTX camera at the same over-the-shoulder vantage
        # LIBERO's `agentview` uses on MuJoCo. Pre-#61 the call
        # silently registers a stub camera; post-#61 it's a real
        # `omni.isaac.sensor.Camera`.
        result = _sim.add_camera(name="image", position=[2.0, 0.0, 1.5], target=[0.0, 0.0, 0.5], fov=60.0)
        if result.get("status") != "success":
            raise RuntimeError(f"add_camera failed: {result}")

        # Resolve the LIBERO task. Same default-aspirational fallback
        # as run_mujoco_agent.py / run_isaac.py. Keep the CLI-requested
        # task distinct from the resolved one so the [agent-eval] line
        # below echoes what the caller passed (replayable) while the
        # actual eval / filename use what really ran.
        requested_task = args.task
        registered = load_libero_suite(suite)
        if not registered:
            raise RuntimeError(
                f"load_libero_suite({suite!r}) registered 0 tasks. "
                "Apply upstream fix from strands-labs/robots#147 if it isn't merged."
            )
        if args.task not in registered:
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
                    f"--task {args.task!r} is not in the {suite} suite. Available: {sorted(registered)[:3]}…"
                )
        resolved_task = args.task

        # NB: no `start_cameras_recording` here -- the Isaac-side
        # recorder integration is a separate slice. Once that lands,
        # this block gains a `_sim.start_cameras_recording(...)` /
        # `_sim.stop_cameras_recording()` pair matching the MuJoCo
        # agent's shape.

        # Hand the agent a 1-tool surface (`evaluate_isaac_benchmark`)
        # and a prompt that fills the eval kwargs from --task /
        # --n-episodes / --seed / --policy. The "1-of-1 tool pick" is
        # the degenerate shape that lets us ship the agent demo today
        # against an IsaacSimulation that doesn't yet inherit
        # AgentTool. See module docstring for migration plan.
        agent = Agent(tools=[evaluate_isaac_benchmark])
        t0 = time.time()
        result = agent(
            f"Make exactly one tool call: invoke `evaluate_isaac_benchmark` "
            f"with `benchmark_name='{resolved_task}'`, "
            f"`n_episodes={args.n_episodes}`, `seed={args.seed}`, "
            f"{policy_phrase}. Do not call any other action -- the world, "
            f"robot, and camera have already been set up. When the call "
            f"returns, parse the `success_rate` field from the JSON "
            f"payload and report it as a percentage of the {args.n_episodes} "
            f"episodes."
        )
        wall_time = time.time() - t0
        print(result)

        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        rec_name = (
            f"{ts}--task={resolved_task}--n_eps={args.n_episodes}"
            f"--seed={args.seed}--policy={args.policy}--backend=isaac--agent"
        )
        # Placeholder until the Isaac recorder integration lands; the
        # filename's still useful for matrix-driver bookkeeping even
        # when no MP4 is produced.
        video_path = os.path.join(_date_dir(), f"{rec_name}__image.mp4.placeholder")
        # Echo the CLI-requested task (replayable) plus the resolved one.
        print(
            f"[agent-eval] policy={args.policy} task={requested_task} "
            f"resolved_task={resolved_task} wall_time={wall_time:.1f}s videos={video_path}"
        )
    finally:
        try:
            if _sim is not None:
                _sim.destroy()
        except Exception:
            pass
        _sim = None
        if server_handle is not None:
            gr00t_inference(action="lifecycle", lifecycle="teardown", container_name=args.container)


if __name__ == "__main__":
    main()
