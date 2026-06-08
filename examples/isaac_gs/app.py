#!/usr/bin/env python3
"""Gradio web app for the Isaac Sim + 3DGS hybrid-render demo.

Browser-accessible companion to the ``render_demo`` CLI -- the Isaac
analogue of ``examples/mujoco_gs/app.py``: a hands-free **live MJPEG
stream** of the composite plus on-demand full-res stills.

    +------------------------------------+ +---------------------------+
    |  ● Live view (Isaac RTX + 3DGS)    | |  Controls                 |
    |   <streamed Franka on backdrop>    | |  [Camera v] oblique       |
    +------------------------------------+ |  [Background v] tabletop  |
    |  Composite still (full-res)        | |  [Upload .ply ]           |
    |   <rendered Franka on backdrop>    | |  [Render still] [Wave]    |
    +------------------------------------+ |  status: ...              |
                                           +---------------------------+

Live vs still: Isaac's RTX renderer isn't real-time-cheap like MuJoCo's
offscreen path, and the ``SimulationApp`` boot is heavyweight (~200 s).
So the app boots the sim once at startup; the ``/live`` MJPEG route then
re-renders continuously (a few fps) for a hands-free view, while the
buttons grab a full-res still on demand. Both go through the same
main-thread render queue.

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

# Background dropdown sentinels (non-scene choices).
PANORAMA_CHOICE = "procedural panorama"
UPLOAD_CHOICE = "uploaded .ply"

# Live MJPEG stream settings (the Isaac analogue of mujoco_gs's live view).
# Isaac's RTX render isn't real-time, so the effective rate is whatever the
# render achieves (a few fps); LIVE_FPS just caps the busy-loop. The stream is
# downscaled to keep a remote `gradio.live` share tunnel smooth.
LIVE_W, LIVE_H, LIVE_FPS = 480, 360, 10


def _live_img_html(camera: str) -> str:
    """A visible, labeled panel with an ``<img>`` pulling the MJPEG live stream.

    Mirrors ``examples.mujoco_gs.app``: the browser renders
    ``multipart/x-mixed-replace`` JPEG frames as they arrive (one long-lived
    HTTP response), which is far more proxy/share-tunnel friendly than Gradio's
    buffered SSE queue. The ``<img>`` is wrapped in a fixed-aspect box so the
    panel is visible before the first frame, with an ``onerror`` fallback.
    """
    bust = int(time.time() * 1000)
    return f"""
<div style="border:2px solid #4a90d9; border-radius:8px; padding:6px; background:#0b0b0b;
            max-width:{LIVE_W + 16}px; margin:0 auto;">
  <div style="color:#9cc; font-size:13px; margin-bottom:4px;">
    <span style="color:#e33;">&#9679;</span> Live view &mdash; {camera} (Isaac RTX + 3DGS, near real-time MJPEG)
  </div>
  <img src="/live?camera={camera}&t={bust}"
       style="width:100%; aspect-ratio:{LIVE_W} / {LIVE_H}; height:auto; display:block;
              object-fit:contain; background:#000; border-radius:6px;"
       alt="connecting to live stream…"
       onerror="this.alt='live stream did not load (Isaac may still be booting) — use Render below';" />
</div>
"""


def _mjpeg_frames(app: "IsaacGsApp", camera: str):
    """Yield ``multipart/x-mixed-replace`` JPEG chunks of the live composite.

    Each frame is rendered on Isaac's **main thread** via ``app.render_once``
    (the same render queue the buttons use), keeping the RTX context
    thread-affine. We only drive renders once the main-thread serve loop is
    live -- rendering Isaac off the main thread (pre-serve) is unsafe -- and
    block-render one frame at a time, so the queue never backs up.
    """
    import io

    from PIL import Image

    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    frame_dt = 1.0 / float(LIVE_FPS)
    try:
        while True:
            t0 = time.time()
            if getattr(app, "_serving", False):
                try:
                    rgb, _status = app.render_once(camera=camera)
                    im = Image.fromarray(np.asarray(rgb)[:, :, :3].astype(np.uint8))
                    if im.size != (LIVE_W, LIVE_H):
                        im = im.resize((LIVE_W, LIVE_H))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=75)
                    yield boundary + buf.getvalue() + b"\r\n"
                except Exception as exc:  # noqa: BLE001 - a dropped frame must not kill the stream
                    logger.debug("live frame error: %s", exc)
                    time.sleep(0.2)
            else:
                time.sleep(0.2)
            elapsed = time.time() - t0
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except GeneratorExit:  # client disconnected
        return


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
        gsplat_scene: Optional[str] = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.default_camera = default_camera
        self.panorama_path = panorama_path
        self.gsplat_ply = gsplat_ply
        self.gsplat_scene = gsplat_scene
        self.width = int(width)
        self.height = int(height)

        self._queue: "queue.Queue[_RenderRequest]" = queue.Queue()
        self._bg_lock = threading.Lock()
        self._pending_bg: "dict | None" = None
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
        self._warmup_cameras()
        logger.info("Scene ready: cameras=%s robot=%s", self._cameras, self._build.robot_name)

    def _warmup_cameras(self, steps: int = 30) -> None:
        """Prime each camera's RTX render product before first real render.

        Cameras added after the initial ``sim.step`` (the preset cameras)
        haven't produced a frame yet, so ``get_rgba()`` can come back
        malformed (empty / 1-D) until the render product is triggered.
        Stepping the world + a guarded throwaway render per camera warms
        them so the first user render returns a well-formed frame.
        """
        if self._sim is None:
            return
        self._sim.step(steps)
        for cam in self._cameras:
            for _ in range(3):
                try:
                    self._sim.render(camera_name=cam)
                except Exception:  # noqa: BLE001 - warmup is best-effort
                    pass
                self._sim.step(2)

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
        # pending is a dict of resolve_background kwargs.
        self.gsplat_ply = pending.get("gsplat_ply")
        self.gsplat_scene = pending.get("gsplat_scene")
        self.panorama_path = pending.get("panorama")
        self._prefer_gs = pending.get("prefer_gs", True)
        if self._compositor is not None:
            self._compositor.background = self._make_background()
            self._compositor._bg_cache.clear()

    def _make_background(self):
        from examples.isaac_gs.background import resolve_background

        return resolve_background(
            gsplat_ply=self.gsplat_ply,
            gsplat_scene=self.gsplat_scene,
            panorama=self.panorama_path,
            prefer_gs=getattr(self, "_prefer_gs", True),
        )

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

    def set_background(self, choice: str, ply_upload: Optional[str] = None) -> str:
        """Queue a background swap; applied on the main thread before the next render.

        ``choice`` is a UI dropdown value: a 3DGS preset scene name, the
        ``PANORAMA_CHOICE`` sentinel, or the ``UPLOAD_CHOICE`` sentinel
        (in which case ``ply_upload`` is the uploaded ``.ply`` path).
        """
        if ply_upload and choice == UPLOAD_CHOICE:
            kwargs = {"gsplat_ply": ply_upload}
            msg = f"background queued: uploaded 3DGS {ply_upload}"
        elif choice == PANORAMA_CHOICE:
            kwargs = {"prefer_gs": False}
            msg = "background queued: procedural panorama"
        else:
            kwargs = {"gsplat_scene": choice}
            msg = f"background queued: 3DGS scene {choice!r}"
        with self._bg_lock:
            self._pending_bg = kwargs
        return msg + " — applies on next render"

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

    # Background choices: the curated 3DGS skybox presets + procedural
    # panorama + an uploaded-.ply sentinel.
    try:
        from examples.mujoco_gs.backgrounds import gsplat_skybox_scene_names

        gs_scenes = gsplat_skybox_scene_names()
    except Exception:  # noqa: BLE001
        gs_scenes = []
    bg_choices = gs_scenes + [PANORAMA_CHOICE, UPLOAD_CHOICE]
    # Default the dropdown to the configured default GS scene if it's a
    # known skybox preset, else the first available scene, else panorama.
    from examples.isaac_gs.background import DEFAULT_GS_SCENE

    default_bg = DEFAULT_GS_SCENE if DEFAULT_GS_SCENE in gs_scenes else (gs_scenes[0] if gs_scenes else PANORAMA_CHOICE)

    with gr.Blocks(title="Isaac Sim + 3DGS hybrid render") as demo:
        gr.Markdown(
            "# Isaac Sim + 3DGS hybrid render\n"
            "An RTX-rendered **simulated** Franka composited into a photoreal "
            "background — by default a real captured **3D Gaussian Splatting** "
            "scene (the digital-twin use case), falling back to a procedural "
            "panorama if `gsplat` isn't installed. "
            "A **live MJPEG view** streams the composite hands-free (Isaac RTX "
            "isn't real-time, so it runs at a few fps); the buttons grab a "
            "full-res still on demand. Isaac Sim boots once at startup (~200 s)."
        )
        initial_camera = app.default_camera if app.default_camera in CAMERA_PRESETS else "oblique"
        with gr.Row():
            with gr.Column(scale=3):
                # Live MJPEG view (hands-free, near real-time) + a full-res
                # still you can grab on demand with the buttons below.
                live_view = gr.HTML(_live_img_html(initial_camera))
                preview = gr.Image(label="Composite still (Isaac RTX + 3DGS)", height=480)
            with gr.Column(scale=1):
                camera_dd = gr.Dropdown(
                    choices=list(CAMERA_PRESETS.keys()),
                    value=initial_camera,
                    label="Camera",
                )
                bg_dd = gr.Dropdown(choices=bg_choices, value=default_bg, label="Background")
                ply_upload = gr.File(label="3DGS .ply upload (with 'uploaded .ply')", file_types=[".ply"])
                apply_bg_btn = gr.Button("Apply background")
                render_btn = gr.Button("Render still", variant="primary")
                wave_btn = gr.Button("Wave + render")
                status = gr.Textbox(label="status", interactive=False)

        def on_render(cam):
            return app.render_once(camera=cam, wave=False)

        def on_wave(cam):
            return app.render_once(camera=cam, wave=True)

        def on_apply_bg(choice, ply_file):
            ply_path = ply_file.name if ply_file is not None else None
            return app.set_background(choice=choice, ply_upload=ply_path)

        render_btn.click(on_render, inputs=[camera_dd], outputs=[preview, status])
        wave_btn.click(on_wave, inputs=[camera_dd], outputs=[preview, status])
        apply_bg_btn.click(on_apply_bg, inputs=[bg_dd, ply_upload], outputs=[status])
        # Point the live MJPEG <img> at the newly selected camera.
        camera_dd.change(lambda cam: _live_img_html(cam), inputs=[camera_dd], outputs=[live_view])

    return demo


def main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--panorama", default=None, help="Equirectangular panorama image for the background.")
    parser.add_argument("--gsplat-ply", default=None, help="3DGS .ply for the background (needs gsplat).")
    parser.add_argument(
        "--gsplat-scene",
        default=None,
        help="Named built-in 3DGS preset (default: the tabletop scene when gsplat is installed).",
    )
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
        gsplat_scene=args.gsplat_scene,
        width=args.width,
        height=args.height,
    )
    demo = build_ui(app)

    # MJPEG live-stream route. Streams near-real-time JPEG frames over a single
    # long-lived HTTP response (the <img> in the UI renders them incrementally),
    # bypassing Gradio's buffered SSE queue, which a port-forward / share proxy
    # coalesces into an end-of-turn burst. Each frame is rendered on Isaac's
    # main thread via the same render queue the buttons use (see _mjpeg_frames).
    from fastapi.responses import StreamingResponse

    def _live_route(camera: str = "oblique"):
        return StreamingResponse(
            _mjpeg_frames(app, camera),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
        )

    # Mount /live *before* the server accepts requests so an early hit can't
    # 404. Gradio rebuilds `demo.app` inside launch(), discarding routes added
    # beforehand, so we wrap `App.create_app` to attach the route at app-creation
    # time (keeps `demo.launch(share=...)` intact). Falls back to a post-launch
    # mount if Gradio's internals differ.
    mounted_pre_serve = False
    try:
        import gradio.routes as _gr_routes

        _orig_create_app = _gr_routes.App.create_app

        def _create_app_with_live(*a, **k):
            fastapi_app = _orig_create_app(*a, **k)
            fastapi_app.add_api_route("/live", _live_route, methods=["GET"])
            return fastapi_app

        _gr_routes.App.create_app = staticmethod(_create_app_with_live)
        mounted_pre_serve = True
    except Exception:  # noqa: BLE001 - fall back to post-launch mount
        logger.debug("Could not hook App.create_app; mounting /live post-launch.", exc_info=True)

    # Boot Isaac on the MAIN thread (SimulationApp requires it), THEN
    # launch Gradio non-blocking so it serves in background threads while
    # the main thread runs the Isaac render loop.
    app.boot()
    demo.queue(default_concurrency_limit=1)  # Gradio events serialized; /live is separate.
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        prevent_thread_lock=True,
    )
    if not mounted_pre_serve:
        demo.app.add_api_route("/live", _live_route, methods=["GET"])
    logger.info("MJPEG live stream mounted at /live")
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
