"""Default Isaac scene for the 3DGS hybrid-render demo.

A real Franka Panda (loaded from Isaac's bundled USD, *not* the
procedural stick-figure -- see ``examples/libero/run_isaac.py`` for
why) plus a small red cube on the ground, and an over-the-shoulder
RTX camera. The robot + cube are the RTX foreground the compositor
z-composites over the captured-real 3DGS background.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SceneBuild:
    """Summary of what ``build_default_scene`` actually loaded."""

    robot_name: str
    robot_joint_count: int
    camera_name: str
    object_names: list[str]


def _default_franka_usd(sim: "object") -> str:
    """Resolve Isaac's bundled Franka Panda USD from the assets root.

    Reachable over HTTPS from the Omniverse CDN (no local Nucleus
    required). Same default as ``examples/libero/run_isaac.py``.
    """
    from omni.isaac.nucleus import get_assets_root_path  # type: ignore[import-not-found]

    root = get_assets_root_path()
    if not root:
        raise RuntimeError(
            "Could not resolve the Isaac assets root for the default Franka USD. " "Pass robot_usd=... explicitly."
        )
    return f"{root}/Isaac/Robots/Franka/franka.usd"


def build_default_scene(
    sim: "object",
    robot_usd: "str | None" = None,
    camera_name: str = "front",
    camera_position: "list[float] | None" = None,
    camera_target: "list[float] | None" = None,
    camera_width: int = 640,
    camera_height: int = 480,
) -> SceneBuild:
    """Build the demo scene on a fresh ``IsaacSimulation``.

    Steps: ``create_world`` -> load a real Franka USD via
    ``add_robot(usd_path=...)`` (real Articulation, observable joints)
    -> add a red cube -> add the RTX camera the compositor renders
    from. Verifies each step's status so the caller never composites
    an empty stage.

    Parameters
    ----------
    sim : IsaacSimulation
        Fresh instance (``create_world`` is called here).
    robot_usd : str, optional
        Override the robot asset. Default: bundled Franka Panda USD.
    camera_name : str
        Name for the RTX camera (the compositor renders this).
    camera_position, camera_target : list[float], optional
        Camera placement. Defaults frame the arm over-the-shoulder.

    Returns
    -------
    SceneBuild
        What loaded (robot name + joint count, camera, objects).
    """
    cw = sim.create_world()
    if cw.get("status") != "success":
        raise RuntimeError(f"create_world failed: {cw}")

    usd = robot_usd or _default_franka_usd(sim)
    # Name "robot" is not a procedural alias, so the usd_path branch is
    # taken (real Articulation), not the procedural builder.
    rr = sim.add_robot(name="robot", usd_path=usd)
    if rr.get("status") != "success":
        raise RuntimeError(f"add_robot(usd_path={usd!r}) failed: {rr}")
    robot_info = rr.get("content", [{}])[0].get("json", {})
    joint_count = int(robot_info.get("joint_count") or 0)

    # A small red cube on the ground, in front of the arm -- a second
    # RTX foreground element so the composite shows depth ordering
    # between two objects + the background.
    obj_names: list[str] = []
    co = sim.add_object(
        name="cube",
        shape="box",
        position=[0.4, 0.0, 0.03],
        size=[0.05, 0.05, 0.05],
        color=[1.0, 0.0, 0.0],
        mass=0.1,
    )
    if co.get("status") == "success":
        obj_names.append("cube")
    else:
        logger.warning("add_object(cube) failed (non-fatal): %s", co)

    pos = camera_position or [1.6, -1.6, 1.2]
    tgt = camera_target or [0.0, 0.0, 0.3]
    ca = sim.add_camera(
        name=camera_name,
        position=pos,
        target=tgt,
        width=camera_width,
        height=camera_height,
        fov=60.0,
    )
    if ca.get("status") != "success":
        raise RuntimeError(f"add_camera({camera_name!r}) failed: {ca}")

    # Settle physics so the first composited frame isn't mid-drop.
    sim.step(20)

    build = SceneBuild(
        robot_name="robot",
        robot_joint_count=joint_count,
        camera_name=camera_name,
        object_names=obj_names,
    )
    logger.info(
        "Scene built: robot=%s (%d joints), camera=%s, objects=%s",
        build.robot_name,
        build.robot_joint_count,
        build.camera_name,
        build.object_names,
    )
    return build
