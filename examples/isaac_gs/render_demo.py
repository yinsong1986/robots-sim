#!/usr/bin/env python3
"""Render an Isaac RTX robot composited into a 3DGS / panorama background.

Entry point for the Isaac-Sim + 3D Gaussian Splatting hybrid-render
example -- the digital-twin companion to ``examples/mujoco_gs``.

Builds the default scene (real Franka + red cube + RTX camera),
renders one or more depth-composited frames (Isaac RTX foreground
z-composited over a captured-real 3DGS scene, or the procedural
panorama by default), and writes them to disk.

This ships a **render-stills / short-clip** entry point rather than a
live Gradio view (like ``mujoco_gs/app.py``): Isaac's RTX renderer
isn't real-time-cheap the way MuJoCo's offscreen renderer is, and the
SimulationApp boot is heavyweight (~200 s), so a render-and-save shape
is the honest fit. A live-view / agent-driven variant can layer on
once the per-frame RTX cost is budgeted.

Usage
-----
::

    # Procedural panorama background (zero ML deps), default Franka:
    python -m examples.isaac_gs.render_demo --frames 1 --out rollouts/isaac_gs

    # Real captured 3DGS background (requires gsplat + a .ply):
    python -m examples.isaac_gs.render_demo --gsplat-ply /path/to/kitchen.ply

    # Sweep a joint across frames to show the arm moving on the backdrop:
    python -m examples.isaac_gs.render_demo --frames 12 --wave

Requires
--------
``pip install 'strands-robots-sim[isaac]'`` + a working Isaac Sim
install (RTX GPU). For the real-3DGS path: ``pip install gsplat`` +
a ``.ply`` capture. The procedural panorama path needs neither.

Depends at runtime on PR #61 (add_camera) + PR #62 (render frame-path);
see the package docstring.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--frames", type=int, default=1, help="Number of composited frames to render.")
    p.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: rollouts/<date>/isaac_gs.",
    )
    p.add_argument(
        "--gsplat-ply",
        default=None,
        help="Path to a .ply / .spz 3DGS capture for the background. "
        "Requires `pip install gsplat`. Overrides the default preset scene.",
    )
    p.add_argument(
        "--gsplat-scene",
        default=None,
        help="Named built-in 3DGS preset for the background (e.g. "
        "'tabletop (indoor room)'). Default: the tabletop preset, "
        "auto-downloaded + skybox-aligned, when gsplat is installed.",
    )
    p.add_argument(
        "--robot-usd",
        default=None,
        help="Override the robot asset USD. Default: bundled Franka Panda.",
    )
    p.add_argument(
        "--panorama",
        default=None,
        help="Path to an equirectangular panorama image for the background "
        "(used by PanoramaBackground when no --gsplat-ply is given).",
    )
    p.add_argument(
        "--wave",
        action="store_true",
        help="Sweep the arm's first joint across frames so the composite "
        "shows the robot moving on the backdrop (needs --frames > 1).",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p


def _date_out(out: "str | None") -> str:
    if out:
        os.makedirs(out, exist_ok=True)
        return out
    d = os.path.join("rollouts", _dt.date.today().strftime("%Y_%m_%d"), "isaac_gs")
    os.makedirs(d, exist_ok=True)
    return d


def _make_background(args: argparse.Namespace):
    """Construct the background renderer from CLI args.

    Defaults to the real 3DGS ``tabletop`` scene (falls back to the
    procedural panorama if gsplat isn't installed) -- see
    ``examples.isaac_gs.background.resolve_background``.
    """
    from examples.isaac_gs.background import resolve_background

    return resolve_background(
        gsplat_ply=args.gsplat_ply,
        gsplat_scene=args.gsplat_scene,
        panorama=args.panorama,
    )


def _save_png(path: str, rgb) -> None:
    """Write an (H, W, 3) uint8 array to PNG without a hard PIL dep at import."""
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except ImportError:
        # Fallback: numpy .npy so the demo still produces output without PIL.
        import numpy as np

        np.save(path.replace(".png", ".npy"), rgb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()

    from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

    # Fail-fast on non-Isaac hosts (cheap probe, no omni import).
    available, reason = IsaacSimulation.is_available()
    if not available:
        raise RuntimeError(
            f"Isaac Sim is not available on this host: {reason}. " "Install Isaac Sim (RTX GPU) and the [isaac] extra."
        )

    from examples.isaac_gs.compositor import IsaacHybridCompositor
    from examples.isaac_gs.scene import build_default_scene

    out_dir = _date_out(args.out)
    # rtx_realtime so render() takes the RTX frame path (not headless blanks).
    sim = IsaacSimulation(IsaacConfig(headless=True, num_envs=1, render_mode="rtx_realtime"))
    try:
        build = build_default_scene(
            sim,
            robot_usd=args.robot_usd,
            camera_name="front",
            camera_width=args.width,
            camera_height=args.height,
        )
        print(f"[scene] robot={build.robot_name} joints={build.robot_joint_count} objects={build.object_names}")

        compositor = IsaacHybridCompositor(sim, background=_make_background(args))

        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        for i in range(max(1, args.frames)):
            if args.wave and build.robot_joint_count > 0:
                # Sweep the first joint to show motion on the backdrop.
                import math

                jn = sim.robot_joint_names(build.robot_name)
                if jn:
                    angle = 0.6 * math.sin(2.0 * math.pi * i / max(1, args.frames))
                    sim.send_action({jn[0]: angle}, robot_name=build.robot_name)
                    sim.step(5)

            frame = compositor.render(camera_name=build.camera_name)
            path = os.path.join(out_dir, f"{ts}--isaac_gs--frame{i:03d}.png")
            _save_png(path, frame.rgb)
            fg_px = int(frame.mask.sum())
            print(f"[frame {i}] saved {path}  foreground_px={fg_px}  ({frame.rgb.shape[1]}x{frame.rgb.shape[0]})")

        # Grep-stable summary line.
        print(f"isaac_gs  frames={args.frames}  robot={build.robot_name}  out={out_dir}  backend=isaac")
    finally:
        sim.destroy()


if __name__ == "__main__":
    # Isaac's SimulationApp installs a fast-shutdown path (``simulation_app
    # .close()`` / ``os._exit``-style teardown) that can swallow a non-zero
    # process exit even when ``main`` raised -- so a failed
    # ``build_default_scene`` would otherwise exit 0 and hide the failure
    # from CI / scripts (see strands-labs/robots-sim#110). Catch any
    # exception here, log it, and force a non-zero exit *after* the
    # SimulationApp teardown via ``os._exit`` so the status survives.
    try:
        main()
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("render_demo failed")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
