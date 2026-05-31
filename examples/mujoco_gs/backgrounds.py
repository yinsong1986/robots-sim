# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Photoreal background renderers for the MuJoCo-GS hybrid demo.

A :class:`BackgroundRenderer` maps a camera (intrinsics + world-from-camera
pose + image size) to an ``(rgb, depth)`` pair that the compositor blends
behind MuJoCo's foreground render. Two implementations ship with the example:

* :class:`PanoramaBackground` — equirectangular HDRI / panorama lookup. No ML
  deps. Treats the panorama as a sphere at infinity (``depth = +inf``), so
  every MuJoCo pixel "wins" the depth test. If no image path is given a
  procedural kitchen-ish gradient is generated so the demo runs anywhere.

* :class:`GsplatBackground` — true 3D Gaussian Splatting via the ``gsplat``
  library, behind a soft import. Requires ``pip install gsplat`` (CUDA-only
  in practice). Produces real per-pixel depth, so foreground objects
  correctly occlude / are occluded by GS geometry.

Both implement the :class:`BackgroundRenderer` protocol — drop-in pluggable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np

from .camera_utils import CameraParams

logger = logging.getLogger(__name__)


class BackgroundRenderer(Protocol):
    """Render a photoreal background at a given camera pose.

    Implementations should be deterministic for a fixed camera and idempotent
    across calls (the compositor will call this every frame).
    """

    name: str

    def render(self, cam: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(rgb_uint8, depth_metric_float32)`` for ``cam``.

        Args:
            cam: pinhole camera parameters at the desired image size.

        Returns:
            ``rgb`` as ``(H, W, 3) uint8`` and ``depth`` as ``(H, W) float32``
            in meters. Pixels at "infinity" should report ``depth = np.inf``
            (or any value larger than ``cam.zfar``) so the compositor's depth
            test always picks the foreground.
        """
        ...


# --------------------------------------------------------------------------- #
# Panorama (equirectangular) background
# --------------------------------------------------------------------------- #


class PanoramaBackground:
    """Equirectangular sky/scene panorama as the photoreal background.

    The panorama is interpreted as the inside of a unit sphere centred on the
    camera. For each output pixel we cast a ray from the camera into the
    world, normalise it, and look up the corresponding texel using
    :math:`(\\theta, \\phi)` spherical coords. Depth is fixed at ``cam.zfar``
    so MuJoCo geometry always occludes correctly.

    This is *not* a 3DGS scene — there is no parallax as the camera moves —
    but it ships zero ML deps and gives the demo a believable photoreal
    backdrop on day 0. Swap in :class:`GsplatBackground` once you have a
    trained scene.

    Args:
        image_path: optional path to an equirectangular ``.jpg``/``.png``.
            If ``None`` a procedural gradient (sky + warm-tone "kitchen wall"
            + floor) is generated so the example runs without external assets.
        rotation_deg: yaw rotation of the panorama in degrees (rotates around
            the world +Z axis). Useful for aligning the panorama with the
            scene without re-rendering it.
    """

    name = "panorama"

    def __init__(
        self,
        image_path: Optional[str | Path] = None,
        rotation_deg: float = 0.0,
    ) -> None:
        self._image_path = Path(image_path) if image_path else None
        self._rotation_rad = float(np.deg2rad(rotation_deg))
        self._panorama: Optional[np.ndarray] = None

    # ----- panorama loading ----- #

    def _ensure_panorama(self) -> np.ndarray:
        if self._panorama is not None:
            return self._panorama
        if self._image_path is not None and self._image_path.exists():
            try:
                from PIL import Image
            except ImportError as e:  # pragma: no cover
                raise ImportError("Pillow is required to load panorama images.") from e
            pano = np.array(Image.open(self._image_path).convert("RGB"))
            logger.info("PanoramaBackground: loaded %s (%s)", self._image_path, pano.shape)
        else:
            if self._image_path is not None:
                logger.warning(
                    "Panorama path %s does not exist — falling back to procedural panorama.",
                    self._image_path,
                )
            pano = _make_procedural_kitchen_panorama(width=2048, height=1024)
        self._panorama = pano
        return pano

    # ----- BackgroundRenderer interface ----- #

    def render(self, cam: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
        pano = self._ensure_panorama()  # (Hp, Wp, 3) uint8
        H, W = cam.height, cam.width

        # Per-pixel camera-frame direction. MuJoCo / OpenGL convention:
        # +X right, +Y up, -Z forward.
        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        Kinv = np.linalg.inv(cam.K)
        homo = np.stack([u, v, np.ones_like(u)], axis=-1)  # (H, W, 3)
        dirs_cam = homo @ Kinv.T  # (H, W, 3) — image-plane rays
        # Flip Z to match GL's "-Z forward" convention.
        dirs_cam[..., 2] *= -1.0
        dirs_cam[..., 1] *= -1.0  # image v grows down → world up flips
        dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True) + 1e-12

        # World-frame directions = R_world_cam @ dirs_cam.
        R = cam.T_world_cam[:3, :3]
        dirs_world = dirs_cam @ R.T  # (H, W, 3)

        # Optional yaw rotation around world +Z (lets users spin the pano
        # without recomputing the texture).
        if self._rotation_rad != 0.0:
            c, s = np.cos(self._rotation_rad), np.sin(self._rotation_rad)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            dirs_world = dirs_world @ Rz.T

        # Spherical mapping. World convention: +Z up, atan2 in XY plane.
        x, y, z = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]
        theta = np.arctan2(y, x)  # in [-pi, pi]   azimuth
        phi = np.arcsin(np.clip(z, -1.0, 1.0))  # in [-pi/2, pi/2]   elevation

        Hp, Wp, _ = pano.shape
        # Equirectangular UV: u in [0, 1] left→right (theta), v in [0, 1] top→bottom (phi).
        uu = (theta + np.pi) / (2.0 * np.pi)
        vv = 0.5 - phi / np.pi
        # Bilinear lookup.
        rgb = _bilinear_sample(pano, uu, vv)
        depth = np.full((H, W), cam.zfar, dtype=np.float32)
        return rgb, depth


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample an equirectangular image at normalised ``(u, v)``.

    ``u`` wraps around the seam; ``v`` is clamped at the poles.
    """
    Hp, Wp, _ = image.shape
    # Wrap u, clamp v.
    u = np.mod(u, 1.0)
    v = np.clip(v, 0.0, 1.0)
    fx = u * (Wp - 1)
    fy = v * (Hp - 1)
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    x1 = (x0 + 1) % Wp
    y1 = np.clip(y0 + 1, 0, Hp - 1)
    wx = fx - x0
    wy = fy - y0

    img = image.astype(np.float32)
    p00 = img[y0, x0]
    p01 = img[y0, x1]
    p10 = img[y1, x0]
    p11 = img[y1, x1]
    out = (
        p00 * ((1 - wx) * (1 - wy))[..., None]
        + p01 * (wx * (1 - wy))[..., None]
        + p10 * ((1 - wx) * wy)[..., None]
        + p11 * (wx * wy)[..., None]
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def _make_procedural_kitchen_panorama(width: int = 2048, height: int = 1024) -> np.ndarray:
    """Generate a procedural "warm kitchen" equirectangular panorama.

    No external assets required. The vertical layout mirrors a typical indoor
    panorama: blue ceiling at the top, warm wall band in the middle, parquet
    floor at the bottom. We add a soft horizontal gradient and a light source
    to give the cube a sense of room context.
    """
    rng = np.random.default_rng(seed=42)

    # Vertical bands (top → bottom).
    img = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        t = y / (height - 1)  # 0 at top, 1 at bottom
        if t < 0.30:
            # Ceiling: cool off-white.
            base = np.array([235.0, 240.0, 245.0])
        elif t < 0.55:
            # Upper wall: warm beige.
            base = np.array([218.0, 198.0, 168.0])
        elif t < 0.78:
            # Lower wall / cabinet line.
            base = np.array([198.0, 170.0, 140.0])
        else:
            # Floor: parquet brown.
            base = np.array([130.0, 90.0, 60.0])
        img[y, :, :] = base[None, :]

    # Horizontal soft "window light" lobe.
    x = np.linspace(0, 2 * np.pi, width, endpoint=False)
    light = 1.0 + 0.10 * np.cos(x - np.pi / 2)  # brighter on one wall
    img *= light[None, :, None]

    # Faint vertical falloff towards floor (ambient occlusion-ish).
    falloff = np.linspace(1.05, 0.85, height)[:, None, None]
    img *= falloff

    # Speckle / noise so the texture isn't dead-flat (helps you tell the
    # background apart from the MuJoCo render).
    img += rng.normal(0, 4.0, size=img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Optional 3D Gaussian Splatting background (soft dep on `gsplat`)
# --------------------------------------------------------------------------- #


class GsplatBackground:
    """Real 3D Gaussian Splatting background using the `gsplat` library.

    This is the upgrade path from :class:`PanoramaBackground` once you have a
    trained 3DGS scene (e.g. exported from Nerfstudio, Polycam, or World Labs
    Marble as a ``.ply``). The MuJoCo-GS-Web demo uses ``.spz`` — that's a
    sparkjs-only format, so for the Python side we recommend re-exporting to
    ``.ply`` (Nerfstudio supports both).

    Install:

        pip install '.[gsplat]'    # from this directory's pyproject extras

    or

        pip install gsplat torch

    Usage:

        bg = GsplatBackground(ply_path="scenes/kitchen.ply")
        compositor = HybridCompositor(sim, background=bg)

    Notes:
        * ``gsplat`` rasterises in *batch* (B, H, W, 3); we run B=1 every
          frame and convert to numpy on the way out, so this is not as fast as
          a JS-side sparkjs viewer but plenty fine for an offline demo.
        * Depth is read from gsplat's accumulated alpha-weighted Z, then
          divided through to give metric depth. Empty pixels (no Gaussians
          along the ray) report ``cam.zfar`` so the compositor falls through
          to whatever fallback you pass.
        * You'll likely want to align the GS scene to your MuJoCo world frame
          via the ``transform`` kwarg (4x4 SE(3) ``world_from_gs``).
    """

    name = "gsplat"

    def __init__(
        self,
        ply_path: str | Path,
        device: str = "cuda",
        transform: Optional[np.ndarray] = None,
    ) -> None:
        self._ply_path = Path(ply_path)
        self._device = device
        self._transform = np.asarray(transform, dtype=np.float64) if transform is not None else np.eye(4)
        self._splats: Optional[dict] = None  # lazily loaded

    # ----- lazy load ----- #

    def _load(self) -> None:
        try:
            import torch  # noqa: F401  — required at runtime
            from gsplat import rasterization  # noqa: F401  — sanity import
        except ImportError as e:
            raise ImportError(
                "gsplat / torch not installed. Run `pip install '.[gsplat]'` "
                "(see examples/mujoco_gs/README.md) or fall back to "
                "PanoramaBackground."
            ) from e
        if not self._ply_path.exists():
            raise FileNotFoundError(f"Gaussian Splat .ply not found: {self._ply_path}")
        self._splats = _load_ply_splats(self._ply_path, device=self._device)

    # ----- BackgroundRenderer interface ----- #

    def render(self, cam: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
        if self._splats is None:
            self._load()
        import torch
        from gsplat import rasterization

        s = self._splats  # type: ignore[assignment]
        assert s is not None  # for type checker

        # Build view matrix: gsplat expects camera_from_world (so we invert).
        T_cam_world = np.linalg.inv(cam.T_world_cam @ self._transform)
        viewmat = torch.from_numpy(T_cam_world).float().unsqueeze(0).to(self._device)
        K = torch.from_numpy(cam.K).float().unsqueeze(0).to(self._device)

        rgb, depth, _ = rasterization(
            means=s["means"],
            quats=s["quats"],
            scales=s["scales"],
            opacities=s["opacities"],
            colors=s["colors"],
            viewmats=viewmat,
            Ks=K,
            width=cam.width,
            height=cam.height,
            near_plane=cam.znear,
            far_plane=cam.zfar,
            render_mode="RGB+D",
        )
        rgb_np = (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        depth_np = depth[0, ..., 0].cpu().numpy().astype(np.float32)
        # Pixels with no contribution come back as 0; promote to zfar so they
        # lose the depth test against any MuJoCo geometry.
        depth_np = np.where(depth_np <= cam.znear, cam.zfar, depth_np)
        return rgb_np, depth_np


def _load_ply_splats(ply_path: Path, device: str) -> dict:
    """Minimal Gaussian-splat .ply loader.

    Supports the standard 3DGS PLY layout (means as ``x y z``, scales as
    ``scale_0 scale_1 scale_2`` in log-space, rotations as ``rot_0..rot_3``
    quaternions, opacity as ``opacity``, and SH DC color as ``f_dc_0..2``).
    Higher-order spherical harmonics are dropped — for a backdrop demo the
    DC term is fine.
    """
    try:
        import torch
        from plyfile import PlyData
    except ImportError as e:  # pragma: no cover
        raise ImportError("plyfile is required for .ply loading. Run `pip install plyfile`.") from e

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data

    means = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1)
    opac = np.array(v["opacity"])
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1)

    # SH DC → linear RGB (see https://github.com/graphdeco-inria/gaussian-splatting).
    SH_C0 = 0.28209479177387814
    colors = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0)
    # Sigmoid opacity, exp scale.
    opac = 1.0 / (1.0 + np.exp(-opac))
    scales = np.exp(scales)

    def to_t(a: np.ndarray, dt: "torch.dtype" = torch.float32) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(a)).to(dt).to(device)

    return {
        "means": to_t(means),
        "scales": to_t(scales),
        "quats": to_t(quats),
        "opacities": to_t(opac),
        "colors": to_t(colors),
    }
