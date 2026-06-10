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


# Hero camera presets (pos, target, fov_deg) framing the Franka in the 3DGS
# room. Adapted from the MuJoCo-GS demo's authored cameras (the tabletop scene
# + its skybox alignment were tuned together with those), then pulled back
# ~1.3x and aimed higher (target z~0.3-0.4 rather than the workspace floor) so
# the WHOLE arm fits with margin -- the Franka's default pose stands tall, and
# a low aim clipped the top. Eyes stay INSIDE the ~2 m captured shell (z < the
# ~1.6 m ceiling) so the splat still fills the frame; aiming higher captures
# the arm without grazing the unobserved ceiling.
CAMERA_PRESETS: "dict[str, tuple[list[float], list[float], float]]" = {
    "oblique": ([0.96, -0.93, 0.88], [0.05, 0.05, 0.40], 55.0),
    "front": ([0.05, -1.25, 0.75], [0.05, 0.05, 0.40], 58.0),
    # A high, slightly-offset eye (a perfectly vertical look-at is degenerate
    # for a +Z-up roll axis); kept under the ~1.6 m shell ceiling.
    "topdown": ([0.05, -0.61, 1.56], [0.05, 0.05, 0.30], 62.0),
}

# SO-101 (SO-ARM100) tabletop arm: a much smaller (~0.4 m) robot than the
# Franka, imported from the MuJoCo Menagerie MJCF (the URDF import doesn't
# render in RTX). The display USD holds the upright "home" pose via joint
# drives (centre ~[0,-0.12,0.12]). It renders empty at close range, so instead
# of moving in we keep the eye ~1.1 m back and use a NARROW fov (~30 deg) to
# zoom -- that enlarges the small arm to a centred hero shot while the 3DGS
# room still fills the frame.
SO101_CAMERA_PRESETS: "dict[str, tuple[list[float], list[float], float]]" = {
    "oblique": ([0.8, -0.9, 0.58], [0.0, -0.12, 0.12], 30.0),
    "front": ([0.0, -1.15, 0.46], [0.0, -0.12, 0.12], 30.0),
    "topdown": ([0.12, -0.85, 0.9], [0.0, -0.12, 0.10], 33.0),
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
    key.CreateIntensityAttr(1500.0)
    key.CreateAngleAttr(1.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))  # slightly warm, kitchen-ish
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 10.0, 0.0))

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(450.0)
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
    camera_fov: float = 58.0,
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

    # Default camera == the "front" CAMERA_PRESETS pose. The app skips
    # re-adding "front" since this creates it, so this default must stay in
    # sync with CAMERA_PRESETS["front"]; render_demo renders it too.
    pos = camera_position or [0.05, -1.25, 0.75]
    tgt = camera_target or [0.05, 0.05, 0.40]
    ca = sim.add_camera(
        name=camera_name,
        position=pos,
        target=tgt,
        width=camera_width,
        height=camera_height,
        fov=camera_fov,
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
    for name, (pos, tgt, fov) in presets.items():
        if name in sim._cameras:
            added.append(name)
            continue
        r = sim.add_camera(name=name, position=list(pos), target=list(tgt), width=width, height=height, fov=float(fov))
        if r.get("status") == "success":
            added.append(name)
        else:
            logger.warning("add_camera(%s) failed: %s", name, r)
    return added
