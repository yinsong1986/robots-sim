#!/usr/bin/env python3
"""Gradio web app for the Isaac Sim + 3DGS hybrid-render demo.

Browser-accessible companion to the ``render_demo`` CLI -- the Isaac
analogue of ``examples/mujoco_gs/app.py``, but **render-on-demand**
rather than a live MJPEG stream:

    +------------------------------------+ +---------------------------+
    |  Composite (Isaac RTX + 3DGS)      | |  Controls                 |
    |                                    | |  [Camera v] oblique       |
    |   <rendered Franka on backdrop>    | |  [Upload .ply ]           |
    |                                    | |  [Render]  [Wave + render]|
    +------------------------------------+ |  status: ...              |
                                           +---------------------------+

Why render-on-demand, not live: Isaac's RTX renderer isn't
real-time-cheap like MuJoCo's offscreen path, and the ``SimulationApp``
boot is heavyweight (~200 s). So the app boots the sim once at startup
and re-renders on a button click.

Threading model -- the important bit. Isaac's ``SimulationApp`` must be
created on the **main thread** (it installs SIGINT handlers, and
``signal.signal`` only works on the main thread), and its RTX render
context is thread-affine. But Gradio serves callbacks on worker
threads. So we invert control flow:

* The **main thread** owns Isaac: :meth:`IsaacGsApp.boot` creates
  ``SimulationApp`` + builds the scene, then :meth:`serve_forever`
  drains a render-request queue (executing each render on the main
  thread).
* **Gradio** is launched non-blocking (``prevent_thread_lock=True``) so
  it serves in background threads; its callbacks enqueue a
  :class:`_RenderRequest` and block on its event, keeping the handler
  synchronous while the render runs on the main thread.

Run::

    python -m examples.isaac_gs.app --server-port 7862
    # open http://127.0.0.1:7862  (7860/7861 are the mujoco_gs apps)

Needs a working Isaac Sim (RTX GPU) + the Phase-2 wiring (#61 / #62 /
#63). The render handler is importable (``boot`` + ``render_once``) so
it can be exercised headlessly for validation without a browser.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent.parent))

logger = logging.getLogger("isaac_gs.app")


class _RenderRequest:
    """A render job marshalled from a Gradio worker to the main thread."""

    __slots__ = ("camera", "wave", "done", "result", "error")

    def __init__(self, camera: str, wave: bool) -> None:
        self.camera = camera
        self.wave = wave
        self.done = threading.Event()
        self.result: "tuple[np.ndarray, str] | None" = None
        self.error: "Exception | None" = None


class IsaacGsApp:
    """Owns the persistent Isaac sim + compositor behind the Gradio UI.

    See the module docstring for the (important) threading model: Isaac
    on the main thread, Gradio in background threads, renders marshalled
    to the main thread via a queue.
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

        self._queue: "queue.Queue[_RenderRequest]" = queue.Queue()
        self._bg_lock = threading.Lock()
        self._pending_bg: "tuple[str | None, str | None] | None" = None
        self._sim = None
        self._compositor = None
        self._build = None
        self._cameras: list[str] = []
        self._stop = threading.Event()
        self._serving = False

    # --- main-thread lifecycle ------------------------------------------

    def boot(self) -> None:
        """Create SimulationApp + build the scene. **Main thread only.**"""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        available, reason = IsaacSimulation.is_available()
        if not available:
            raise RuntimeError(f"Isaac Sim not available: {reason}")

        from examples.isaac_gs.compositor import IsaacHybridCompositor
        from examples.isaac_gs.scene import add_preset_cameras, build_default_scene

        logger.info("Booting IsaacSimulation on the main thread (~200 s)...")
        sim = IsaacSimulation(IsaacConfig(headless=True, num_envs=1, render_mode="rtx_realtime"))
        self._build = build_default_scene(sim, camera_width=self.width, camera_height=self.height)
        self._cameras = add_preset_cameras(sim, width=self.width, height=self.height)
        self._compositor = IsaacHybridCompositor(sim, background=self._make_background())
        self._sim = sim
        logger.info("Scene ready: cameras=%s robot=%s", self._cameras, self._build.robot_name)

    def serve_forever(self, poll: float = 0.05) -> None:
        """Main-thread loop: drain render requests. Runs after :meth:`boot`."""
        self._serving = True
        try:
            while not self._stop.is_set():
                try:
                    req = self._queue.get(timeout=poll)
                except queue.Empty:
                    continue
                try:
                    self._apply_pending_background()
                    req.result = self._render_on_main(req.camera, req.wave)
                except Exception as exc:  # noqa: BLE001 - surfaced to caller
                    req.error = exc
                finally:
                    req.done.set()
        finally:
            self._serving = False

    def stop(self) -> None:
        self._stop.set()

    def _render_on_main(self, camera: str, wave: bool) -> "tuple[np.ndarray, str]":
        camera = camera if camera in self._cameras else (self._cameras[0] if self._cameras else "front")
        if wave and self._build and self._build.robot_joint_count > 0:
            import math

            jn = self._sim.robot_joint_names(self._build.robot_name)
            if jn:
                angle = 0.6 * math.sin(time.time())
                self._sim.send_action({jn[0]: angle}, robot_name=self._build.robot_name)
                self._sim.step(5)
        frame = self._compositor.render(camera_name=camera)
        fg_px = int(frame.mask.sum())
        status = f"camera={camera} foreground_px={fg_px} size={frame.rgb.shape[1]}x{frame.rgb.shape[0]}"
        return frame.rgb, status

    def _apply_pending_background(self) -> None:
        with self._bg_lock:
            pending = self._pending_bg
            self._pending_bg = None
        if pending is None:
            return
        self.gsplat_ply, self.panorama_path = pending
        if self._compositor is not None:
            self._compositor.background = self._make_background()
            self._compositor._bg_cache.clear()

    def _make_background(self):
        if self.gsplat_ply:
            from examples.mujoco_gs.backgrounds import GsplatBackground

            return GsplatBackground(ply_path=self.gsplat_ply)
        from examples.mujoco_gs.backgrounds import PanoramaBackground

        if self.panorama_path:
            return PanoramaBackground(panorama_path=self.panorama_path)
        return PanoramaBackground()

    # --- public handlers (called from Gradio worker threads) ------------

    def render_once(
        self, camera: str = "oblique", wave: bool = False, timeout: float = 600.0
    ) -> "tuple[np.ndarray, str]":
        """Enqueue a render for the main thread and block for the result.

        When no :meth:`serve_forever` loop is active (e.g. a headless
        test that called :meth:`boot` on the main thread itself), renders
        inline on the caller's thread instead.
        """
        if not self._serving:
            self._apply_pending_background()
            return self._render_on_main(camera, wave)
        req = _RenderRequest(camera, wave)
        self._queue.put(req)
        if not req.done.wait(timeout=timeout):
            raise TimeoutError(f"render timed out after {timeout}s")
        if req.error is not None:
            raise req.error
        assert req.result is not None
        return req.result

    def set_background(self, gsplat_ply: Optional[str], panorama_path: Optional[str]) -> str:
        """Queue a background swap; applied on the main thread before the next render."""
        with self._bg_lock:
            self._pending_bg = (gsplat_ply or None, panorama_path or None)
        return "background queued — applies on next render"

    def shutdown(self) -> None:
        self.stop()
        if self._sim is not None:
            try:
                self._sim.destroy()
            except Exception:  # noqa: BLE001
                pass


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
            "Render-on-demand: Isaac Sim boots once at startup (~200 s)."
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
            return app.render_once(camera=cam, wave=False)

        def on_wave(cam):
            return app.render_once(camera=cam, wave=True)

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
    os.environ.setdefault("MUJOCO_GL", "egl")

    app = IsaacGsApp(
        default_camera=args.camera,
        panorama_path=args.panorama,
        gsplat_ply=args.gsplat_ply,
        width=args.width,
        height=args.height,
    )
    demo = build_ui(app)

    # Boot Isaac on the MAIN thread (SimulationApp requires it), THEN
    # launch Gradio non-blocking so it serves in background threads while
    # the main thread runs the Isaac render loop.
    app.boot()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        prevent_thread_lock=True,
    )
    print(f"[isaac_gs] UI at http://{args.server_name}:{args.server_port} — Ctrl-C to stop", flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        try:
            demo.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
