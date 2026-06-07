#!/usr/bin/env python3
"""Gradio web app for the Isaac Sim + 3DGS hybrid-render demo.

Browser-accessible companion to the ``render_demo`` CLI -- the Isaac
analogue of ``examples/mujoco_gs/app.py``, but **render-on-demand**
rather than a live MJPEG stream:

    +------------------------------------+ +---------------------------+
    |  Composite (Isaac RTX + 3DGS)      | |  Controls                 |
    |                                    | |  [Camera ▼] oblique       |
    |   <rendered Franka on backdrop>    | |  [Background ▼] panorama  |
    |                                    | |  [Upload .ply ]           |
    |                                    | |  [Render]  [Wave + render]|
    +------------------------------------+ |  status: ...              |
                                           +---------------------------+

Why render-on-demand, not live: Isaac's RTX renderer isn't
real-time-cheap like MuJoCo's offscreen path, and the ``SimulationApp``
boot is heavyweight (~200 s). So the app boots the sim **once** at
first render and keeps it alive, then re-renders on a button click.

Thread model: Isaac's RTX render context is thread-affine (like
MuJoCo's EGL). Gradio runs callbacks on worker threads, so **all** sim
/ render calls are funnelled through a single dedicated worker thread
(one ``ThreadPoolExecutor(max_workers=1)``); callbacks submit work and
block for the result, keeping the public handlers synchronous.

Run::

    python -m examples.isaac_gs.app --server-port 7862
    # open http://127.0.0.1:7862  (7860/7861 are the mujoco_gs apps)

Needs a working Isaac Sim (RTX GPU) + the Phase-2 wiring (#61 / #62 /
#63); see the package docstring. The render handler is also importable
(``render_once``) so it can be exercised headlessly for validation
without a browser.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

# Make importable both as `python -m examples.isaac_gs.app` and
# `python examples/isaac_gs/app.py`.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent.parent))

logger = logging.getLogger("isaac_gs.app")


class IsaacGsApp:
    """Owns the persistent Isaac sim + compositor behind the Gradio UI.

    All sim / render work runs on a single executor thread (Isaac's RTX
    context is thread-affine). The sim is booted lazily on the first
    render so app construction (and ``--help``) doesn't pay the ~200 s
    SimulationApp boot.
    """

    def __init__(
        self,
        default_camera: str = "oblique",
        panorama_path: Optional[str] = None,
        gsplat_ply: Optional[str] = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.default_camera = default_camera
        self.panorama_path = panorama_path
        self.gsplat_ply = gsplat_ply
        self.width = int(width)
        self.height = int(height)

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="isaac_gs")
        self._lock = threading.Lock()
        self._sim = None
        self._compositor = None
        self._build = None
        self._cameras: list[str] = []
        self._booted = False

    # --- sim lifecycle (runs on the executor thread) --------------------

    def _ensure_booted(self) -> None:
        """Boot SimulationApp + build the scene once. Executor-thread only."""
        if self._booted:
            return
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        available, reason = IsaacSimulation.is_available()
        if not available:
            raise RuntimeError(f"Isaac Sim not available: {reason}")

        from examples.isaac_gs.compositor import IsaacHybridCompositor
        from examples.isaac_gs.scene import add_preset_cameras, build_default_scene

        logger.info("Booting IsaacSimulation (first render; ~200 s)...")
        sim = IsaacSimulation(IsaacConfig(headless=True, num_envs=1, render_mode="rtx_realtime"))
        self._build = build_default_scene(sim, camera_width=self.width, camera_height=self.height)
        self._cameras = add_preset_cameras(sim, width=self.width, height=self.height)
        self._compositor = IsaacHybridCompositor(sim, background=self._make_background())
        self._sim = sim
        self._booted = True
        logger.info("Scene ready: cameras=%s robot=%s", self._cameras, self._build.robot_name)

    def _make_background(self):
        if self.gsplat_ply:
            from examples.mujoco_gs.backgrounds import GsplatBackground

            return GsplatBackground(ply_path=self.gsplat_ply)
        from examples.mujoco_gs.backgrounds import PanoramaBackground

        if self.panorama_path:
            return PanoramaBackground(panorama_path=self.panorama_path)
        return PanoramaBackground()

    def _render_on_thread(self, camera: str, wave: bool) -> "tuple[np.ndarray, str]":
        """Render one composite on the executor thread. Returns (rgb, status)."""
        self._ensure_booted()
        camera = camera if camera in self._cameras else (self._cameras[0] if self._cameras else "front")
        if wave and self._build and self._build.robot_joint_count > 0:
            import math
            import time as _t

            jn = self._sim.robot_joint_names(self._build.robot_name)
            if jn:
                angle = 0.6 * math.sin(_t.time())
                self._sim.send_action({jn[0]: angle}, robot_name=self._build.robot_name)
                self._sim.step(5)
        frame = self._compositor.render(camera_name=camera)
        fg_px = int(frame.mask.sum())
        status = f"camera={camera} foreground_px={fg_px} size={frame.rgb.shape[1]}x{frame.rgb.shape[0]}"
        return frame.rgb, status

    # --- public handlers (called from Gradio worker threads) ------------

    def render_once(self, camera: str = "oblique", wave: bool = False) -> "tuple[np.ndarray, str]":
        """Submit a render to the executor thread and block for the result."""
        with self._lock:
            fut = self._executor.submit(self._render_on_thread, camera, wave)
        return fut.result()

    def set_background(self, gsplat_ply: Optional[str], panorama_path: Optional[str]) -> str:
        """Swap the compositor background. Applied on the executor thread."""
        self.gsplat_ply = gsplat_ply or None
        self.panorama_path = panorama_path or None

        def _apply():
            if self._compositor is not None:
                self._compositor.background = self._make_background()
                self._compositor._bg_cache.clear()
            return "background updated"

        with self._lock:
            fut = self._executor.submit(_apply)
        return fut.result()

    def shutdown(self) -> None:
        def _destroy():
            if self._sim is not None:
                self._sim.destroy()

        try:
            self._executor.submit(_destroy).result(timeout=60)
        except Exception:  # noqa: BLE001
            pass
        self._executor.shutdown(wait=False)


def build_ui(app: IsaacGsApp):
    """Construct the Gradio Blocks UI bound to an :class:`IsaacGsApp`."""
    import gradio as gr

    from examples.isaac_gs.scene import CAMERA_PRESETS

    with gr.Blocks(title="Isaac Sim + 3DGS hybrid render") as demo:
        gr.Markdown(
            "# Isaac Sim + 3DGS hybrid render\n"
            "An RTX-rendered **simulated** Franka composited into a photoreal "
            "background (procedural panorama by default; upload a `.ply` for a "
            "real captured 3DGS scene — the digital-twin use case). "
            "Render-on-demand: the first render boots Isaac Sim (~200 s)."
        )
        with gr.Row():
            with gr.Column(scale=3):
                preview = gr.Image(label="Composite (Isaac RTX + 3DGS)", height=480)
            with gr.Column(scale=1):
                camera_dd = gr.Dropdown(
                    choices=list(CAMERA_PRESETS.keys()),
                    value=app.default_camera if app.default_camera in CAMERA_PRESETS else "oblique",
                    label="Camera",
                )
                ply_upload = gr.File(label="3DGS background (.ply) — optional", file_types=[".ply"])
                apply_bg_btn = gr.Button("Apply background")
                render_btn = gr.Button("Render", variant="primary")
                wave_btn = gr.Button("Wave + render")
                status = gr.Textbox(label="status", interactive=False)

        def on_render(cam):
            rgb, st = app.render_once(camera=cam, wave=False)
            return rgb, st

        def on_wave(cam):
            rgb, st = app.render_once(camera=cam, wave=True)
            return rgb, st

        def on_apply_bg(ply_file):
            ply_path = ply_file.name if ply_file is not None else None
            return app.set_background(gsplat_ply=ply_path, panorama_path=app.panorama_path)

        render_btn.click(on_render, inputs=[camera_dd], outputs=[preview, status])
        wave_btn.click(on_wave, inputs=[camera_dd], outputs=[preview, status])
        apply_bg_btn.click(on_apply_bg, inputs=[ply_upload], outputs=[status])

    return demo


def main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--panorama", default=None, help="Equirectangular panorama image for the background.")
    parser.add_argument("--gsplat-ply", default=None, help="3DGS .ply for the background (needs gsplat).")
    parser.add_argument("--camera", default="oblique", help="Initial camera preset (oblique / front / topdown).")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862, help="Default 7862 (7860/7861 are mujoco_gs).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.environ.setdefault("MUJOCO_GL", "egl")  # harmless; some deps probe it

    app = IsaacGsApp(
        default_camera=args.camera,
        panorama_path=args.panorama,
        gsplat_ply=args.gsplat_ply,
        width=args.width,
        height=args.height,
    )
    demo = build_ui(app)
    try:
        demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
