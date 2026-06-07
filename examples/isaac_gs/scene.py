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


# Hero camera presets (pos, target) framing the Franka on the 3DGS backdrop.
#
# CRITICAL: a captured 3DGS room only renders from viewpoints *inside* its
# shell -- it's reconstructed from photos taken within the room, so gaussians
# carry color/opacity only for inward-facing surfaces. The default tabletop
# scene spans ~x[-2,1.8] y[-1.6,0.6] z[-0.8,1.6] (centroid ≈ origin), so eyes
# must sit INSIDE that ~2 m volume or the splat renders empty (the grey fill).
# Earlier presets sat 3-5 m out (outside the shell) and the room came back
# blank. These eyes are ~1.5-2 m from the arm, inside the room, looking at the
# Franka so the photoreal kitchen reads behind it. (A true top-down isn't
# possible -- the capture has no ceiling -- so the third angle is a reverse
# view rather than an overhead one.)
CAMERA_PRESETS: "dict[str, tuple[list[float], list[float]]]" = {
    "oblique": ([1.5, -1.0, 1.1], [0.0, 0.0, 0.4]),
    "front": ([1.2, -1.2, 1.0], [0.0, 0.0, 0.4]),
    "reverse": ([-1.4, -1.0, 1.1], [0.0, 0.0, 0.4]),
}


def _add_lighting(sim: "object") -> None:
    """Add explicit key + dome lights to the stage.

    The digital-twin composite uses ``ground_plane=False`` (so the
    background provides the floor, not occluded by a sim ground). But
    Isaac's default lighting is part of the default-ground-plane
    subtree, so without it the robot renders as an unlit black
    silhouette. Author a distant key light + a dome fill light directly
    so lighting is independent of the (absent) floor.

    No-op if ``pxr`` / the stage aren't available.
    """
    try:
        import omni.usd  # type: ignore[import-not-found]
        from pxr import Gf, Sdf, UsdGeom, UsdLux  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    key.CreateIntensityAttr(3000.0)
    key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 10.0, 0.0))

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(800.0)
    logger.info("Added key + dome lights (ground-plane lighting unavailable with ground_plane=False)")


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
    # No ground plane in the digital-twin composite: the background
    # (captured-real 3DGS scene / panorama) is the visible floor, so a
    # sim ground plane would occlude it everywhere. The Franka USD is
    # fixed-base (stays up without one); the cube is static (below) so
    # it doesn't fall through. Lighting is added explicitly via
    # _add_lighting since Isaac's default light rides with the ground
    # plane we're omitting.
    cw = sim.create_world(ground_plane=False)
    if cw.get("status") != "success":
        raise RuntimeError(f"create_world failed: {cw}")
    _add_lighting(sim)

    usd = robot_usd or _default_franka_usd(sim)
    # Name "robot" is not a procedural alias, so the usd_path branch is
    # taken (real Articulation), not the procedural builder.
    rr = sim.add_robot(name="robot", usd_path=usd)
    if rr.get("status") != "success":
        raise RuntimeError(f"add_robot(usd_path={usd!r}) failed: {rr}")
    robot_info = rr.get("content", [{}])[0].get("json", {})
    joint_count = int(robot_info.get("joint_count") or 0)

    # A small red cube in front of the arm -- a second RTX foreground
    # element so the composite shows depth ordering between two objects
    # + the background. Static (is_static=True) so it doesn't fall
    # through the absent ground plane.
    obj_names: list[str] = []
    co = sim.add_object(
        name="cube",
        shape="box",
        position=[0.4, 0.0, 0.4],
        size=[0.05, 0.05, 0.05],
        color=[1.0, 0.0, 0.0],
        mass=0.1,
        is_static=True,
    )
    if co.get("status") == "success":
        obj_names.append("cube")
    else:
        logger.warning("add_object(cube) failed (non-fatal): %s", co)

    # Default camera == the "front" CAMERA_PRESETS pose: an INSIDE-the-room
    # eye (see the CAMERA_PRESETS note on why eyes must sit inside the ~2 m
    # 3DGS shell). The app skips re-adding "front" since this creates it, so
    # this default must stay in sync with CAMERA_PRESETS["front"]; render_demo
    # also renders this camera.
    pos = camera_position or [1.2, -1.2, 1.0]
    tgt = camera_target or [0.0, 0.0, 0.4]
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


def add_preset_cameras(
    sim: "object",
    width: int = 640,
    height: int = 480,
    presets: "dict[str, tuple[list[float], list[float]]] | None" = None,
) -> "list[str]":
    """Add the hero camera presets (``CAMERA_PRESETS``) to a built scene.

    Used by the Gradio app so the camera dropdown can switch angles
    without re-adding cameras per render. Skips any preset name already
    present. Returns the list of camera names available.
    """
    presets = presets or CAMERA_PRESETS
    added: list[str] = []
    for name, (pos, tgt) in presets.items():
        if name in sim._cameras:
            added.append(name)
            continue
        r = sim.add_camera(name=name, position=list(pos), target=list(tgt), width=width, height=height, fov=60.0)
        if r.get("status") == "success":
            added.append(name)
        else:
            logger.warning("add_camera(%s) failed: %s", name, r)
    return added
