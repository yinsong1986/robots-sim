"""Background resolution for the isaac_gs demo -- defaults to a real 3DGS scene.

Mirrors ``mujoco_gs``'s default: the demo composites the robot against a
real 3D Gaussian Splatting scene (the ``tabletop`` preset from
MuJoCo-GS-Web) when ``gsplat`` is available, and falls back to the
procedural panorama otherwise so it still runs with zero ML deps.

All the heavy lifting (preset download + cache, ``.spz`` loading,
skybox alignment, the ``gsplat`` rasterizer) is **reused verbatim**
from ``examples.mujoco_gs.backgrounds``; this module only picks which
renderer to construct from the CLI / UI options.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("isaac_gs.background")

# Default 3DGS scene: MuJoCo-GS-Web's purpose-built tabletop room. Open
# floor, clean from every angle, has curated skybox alignment metadata
# (``GSPLAT_SKYBOX_ALIGN["tabletop"]``) so it sits behind the robot
# correctly without per-camera tuning.
DEFAULT_GS_SCENE = "tabletop (indoor room)"


def resolve_background(
    gsplat_ply: Optional[str] = None,
    gsplat_scene: Optional[str] = None,
    panorama: Optional[str] = None,
    prefer_gs: bool = True,
):
    """Pick a ``BackgroundRenderer`` from the demo's background options.

    Precedence:

    1. ``gsplat_ply`` -> live 3DGS skybox from that ``.ply`` / ``.spz``.
    2. ``panorama`` -> ``PanoramaBackground`` from that image.
    3. default (``prefer_gs``) -> the ``gsplat_scene`` preset (or
       :data:`DEFAULT_GS_SCENE`), downloaded + skybox-aligned. If
       ``gsplat`` isn't importable (no CUDA build) or the scene fails to
       load, **fall back to the procedural panorama** with a logged
       note -- so the demo always renders something.

    Returns a renderer satisfying ``mujoco_gs.backgrounds.BackgroundRenderer``.
    """
    from examples.mujoco_gs.backgrounds import GsplatBackground, PanoramaBackground

    if gsplat_ply:
        logger.info("background: live 3DGS skybox from %s", gsplat_ply)
        return GsplatBackground(ply_path=gsplat_ply, skybox=True)

    if panorama:
        logger.info("background: panorama image %s", panorama)
        return PanoramaBackground(image_path=panorama)

    if not prefer_gs:
        return PanoramaBackground()

    scene = gsplat_scene or DEFAULT_GS_SCENE
    try:
        import gsplat  # noqa: F401  -- probe the CUDA rasterizer is importable

        from examples.mujoco_gs.backgrounds import download_gsplat_scene, gsplat_skybox_align_for

        logger.info("background: downloading + loading default 3DGS scene %r ...", scene)
        ply = download_gsplat_scene(scene)
        align = gsplat_skybox_align_for(scene)
        bg = GsplatBackground(ply_path=str(ply), skybox=True, **align)
        logger.info("background: live 3DGS skybox %r%s", scene, "" if align else " (uncurated alignment)")
        return bg
    except Exception as exc:  # noqa: BLE001 - any failure falls back to panorama
        logger.warning(
            "background: 3DGS scene %r unavailable (%s: %s); falling back to procedural panorama. "
            "Install `gsplat` (CUDA) for the real captured scene.",
            scene,
            type(exc).__name__,
            exc,
        )
        return PanoramaBackground()
