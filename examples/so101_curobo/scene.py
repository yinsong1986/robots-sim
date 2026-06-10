# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene factory for the SO-101 cuRobo synthetic-data demo (issue #67).

Builds a tabletop pick-and-place world: an SO-101 6-DoF arm, a small red cube
to grasp, a "bin" placement target, and a camera trio. The demo is
**backend-agnostic** by design (the executor/collector speak the ``SimEngine``
surface), but today the only runtime present on most boxes is the **MuJoCo**
backend, which already loads a real SO-101. :func:`make_sim` returns a MuJoCo
``Simulation`` by default and lazily attempts the Isaac backend when requested
(``create_simulation("isaac")``), degrading with a clear message if the Isaac
Sim runtime isn't installed.

See ``README.md`` for how this maps onto the issue's T1-T10 task breakdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("so101_curobo.scene")

# SO-101 is the canonical robot; the rest are progressively-more-available
# fallbacks with the same "small arm + gripper" shape so the pick-place script
# still makes sense. (The MuJoCo SO-101 asset doesn't resolve on every box.)
ROBOT_CONFIG_CANDIDATES = ["so101", "so100", "so_arm100", "panda"]

# Workspace layout (metres, world frame; arm base at origin).
DEFAULT_CUBE_POSITION = [0.20, 0.20, 0.025]
DEFAULT_CUBE_HALF = [0.022, 0.022, 0.022]
DEFAULT_CUBE_COLOR = [0.85, 0.10, 0.10, 1.0]
DEFAULT_PLACE_POSITION = [-0.20, 0.22, 0.0]  # the "bin" drop target
DEFAULT_BIN_HALF = [0.05, 0.05, 0.012]
DEFAULT_BIN_COLOR = [0.15, 0.55, 0.20, 1.0]


@dataclass
class SceneInfo:
    """What :func:`build_pick_place_scene` actually created."""

    robot_name: str
    robot_config: str
    joint_names: List[str]
    gripper_joint: Optional[str]
    cube_name: str
    cube_position: List[float]
    place_position: List[float]
    cameras: List[str] = field(default_factory=list)
    backend: str = "mujoco"

    def pretty(self) -> str:
        return (
            f"{self.robot_config} arm '{self.robot_name}' ({len(self.joint_names)} joints) "
            f"+ red cube at {[round(x, 2) for x in self.cube_position]} "
            f"+ bin at {[round(x, 2) for x in self.place_position]}; "
            f"cameras={self.cameras}; backend={self.backend}"
        )


def make_sim(backend: str = "mujoco", **isaac_kwargs: Any):
    """Return a ``SimEngine``-style simulation for the requested backend.

    ``mujoco`` (default) uses ``strands_robots.simulation.Simulation`` and loads
    a real SO-101. ``isaac`` lazily tries ``create_simulation("isaac")``; if the
    Isaac Sim runtime isn't installed this raises a clear, actionable error
    instead of a cryptic ImportError (the demo's app catches it and falls back
    to MuJoCo so the planning + collection loop is still demonstrable).
    """
    backend = (backend or "mujoco").lower()
    if backend in ("mujoco", "mj"):
        from strands_robots.simulation import Simulation

        return Simulation(tool_name="sim", mesh=False)

    if backend in ("isaac", "isaacsim", "isaac_sim"):
        try:
            from strands_robots.simulation import create_simulation
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Isaac backend requested but create_simulation() is unavailable. "
                "Use backend='mujoco' (default), or land the backend registration "
                "(issue #67 T1) so create_simulation('isaac') resolves."
            ) from exc
        try:
            return create_simulation(
                "isaac",
                render_mode=isaac_kwargs.pop("render_mode", "rtx_realtime"),
                headless=isaac_kwargs.pop("headless", True),
                **isaac_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - runtime missing / not wired
            raise RuntimeError(
                f"Could not create the Isaac Sim backend ({type(exc).__name__}: {exc}). "
                "The Isaac Sim runtime (~30 GB: Omniverse / nvcr.io/nvidia/isaac-sim) "
                "and backend registration (#67 T1) are required. Falling back to "
                "MuJoCo is recommended on boxes without it."
            ) from exc

    raise ValueError(f"Unknown backend {backend!r}. Use 'mujoco' or 'isaac'.")


def _status(result: Any) -> str:
    return str(result.get("status", "unknown")) if isinstance(result, dict) else "unknown"


def _add_robot_with_fallback(sim, name: str, candidates: List[str]) -> str:
    errors = []
    for cfg in candidates:
        if _status(sim.add_robot(name=name, data_config=cfg, position=[0.0, 0.0, 0.0])) == "success":
            if cfg != candidates[0]:
                logger.warning("Robot %r unavailable; fell back to %r.", candidates[0], cfg)
            return cfg
        errors.append(cfg)
    raise RuntimeError(
        f"Could not load any SO-101-class arm. Tried {candidates}. "
        "Install a MuJoCo Menagerie SO-101/SO-100 model or pass a resolvable config."
    )


def _erect_arm(sim, robot_name: str) -> bool:
    """Stand the arm in its model's ``home`` keyframe (zero pose sprawls flat).

    Sets qpos + actuator targets so the pose holds when stepped. MuJoCo-only;
    a no-op (returns False) on backends without ``mj_model``/``mj_data`` or a
    home keyframe.
    """
    try:
        import mujoco

        m = getattr(sim, "mj_model", None)
        d = getattr(sim, "mj_data", None)
        if m is None or d is None:
            return False
        key_id = next(
            (k for k in range(m.nkey) if "home" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or "").lower()),
            -1,
        )
        if key_id < 0:
            return False
        kq = m.key_qpos[key_id]
        ns = f"{robot_name}/"
        hinge_slide = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        for j in range(m.njnt):
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if jn.startswith(ns) and m.jnt_type[j] in hinge_slide:
                d.qpos[m.jnt_qposadr[j]] = kq[m.jnt_qposadr[j]]
        for a in range(m.nu):
            jid = int(m.actuator_trnid[a, 0])
            if jid >= 0 and m.jnt_type[jid] in hinge_slide:
                d.ctrl[a] = kq[m.jnt_qposadr[jid]]
        mujoco.mj_forward(m, d)
        return True
    except Exception:  # noqa: BLE001 - non-fatal pose nicety
        logger.debug("Could not set home pose.", exc_info=True)
        return False


def build_pick_place_scene(
    sim,
    cube_position: Optional[List[float]] = None,
    place_position: Optional[List[float]] = None,
    robot_candidates: Optional[List[str]] = None,
    add_bin: bool = True,
    camera_size: tuple[int, int] = (320, 240),
    backend: str = "mujoco",
) -> SceneInfo:
    """Populate ``sim`` with the SO-101 pick-and-place world. Returns a SceneInfo.

    Assumes a fresh ``sim`` (``create_world`` is called here).
    """
    cube_position = list(cube_position or DEFAULT_CUBE_POSITION)
    place_position = list(place_position or DEFAULT_PLACE_POSITION)
    candidates = robot_candidates or ROBOT_CONFIG_CANDIDATES
    cw, ch = camera_size

    cw_res = sim.create_world(timestep=0.002, gravity=[0.0, 0.0, -9.81], ground_plane=True)
    if _status(cw_res) != "success":
        raise RuntimeError(f"create_world failed: {cw_res}")

    robot_config = _add_robot_with_fallback(sim, name="arm", candidates=candidates)

    sim.add_object(
        name="cube",
        shape="box",
        position=cube_position,
        size=DEFAULT_CUBE_HALF,
        color=DEFAULT_CUBE_COLOR,
        mass=0.04,
    )
    if add_bin:
        r = sim.add_object(
            name="bin",
            shape="box",
            position=[place_position[0], place_position[1], DEFAULT_BIN_HALF[2]],
            size=DEFAULT_BIN_HALF,
            color=DEFAULT_BIN_COLOR,
            mass=1.0,
            is_static=True,
        )
        if _status(r) != "success":
            logger.info("bin marker not added (non-fatal): %s", r)

    cams = []
    for name, pos, tgt, fov in (
        ("front", [0.05, -0.95, 0.45], [0.05, 0.05, 0.18], 58.0),
        ("topdown", [0.05, 0.05, 1.25], [0.05, 0.05, 0.0], 62.0),
        ("oblique", [0.75, -0.7, 0.55], [0.05, 0.05, 0.18], 55.0),
    ):
        if _status(sim.add_camera(name=name, position=pos, target=tgt, fov=fov, width=cw, height=ch)) == "success":
            cams.append(name)

    _erect_arm(sim, robot_name="arm")
    sim.step(20)  # settle into the home pose

    jn = list(sim.robot_joint_names("arm"))
    gripper = jn[-1] if jn else None  # SO-101/SO-100: last joint is the gripper jaw
    info = SceneInfo(
        robot_name="arm",
        robot_config=robot_config,
        joint_names=jn,
        gripper_joint=gripper,
        cube_name="cube",
        cube_position=cube_position,
        place_position=place_position,
        cameras=cams,
        backend=backend,
    )
    logger.info("SO-101 cuRobo scene ready: %s", info.pretty())
    return info
