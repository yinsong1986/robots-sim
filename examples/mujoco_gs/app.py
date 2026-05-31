# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gradio chat + live preview UI for the MuJoCo-GS demo.

Two-column layout that mirrors the MuJoCo-GS-Web vibe but is fully Python:

    +----------------------------+ +-------------------------------------+
    |  Live composite (RGB)      | |  Strands Agent chat                 |
    |  (MuJoCo + 3DGS / pano)    | |                                     |
    |                            | |  user > make the arm wave           |
    |                            | |  agent > done — showing front view  |
    |  [Preview camera ▼]        | |                                     |
    |  [Background ▼]            | |  user > switch to topdown           |
    |  [Render now] [Reset]      | |                                     |
    +----------------------------+ +-------------------------------------+

* The chat panel goes through :class:`MujocoGsAgent` — every Strands tool
  call (real ``Simulation`` actions, e.g. ``run_policy`` / ``render``) is
  logged in the Gradio event stream.
* The "Render now" button calls the compositor outside the agent, so the
  user can poke the scene by hand without burning agent tokens.
* The "Background" dropdown hot-swaps between the procedural panorama,
  a user-supplied panorama image, and (if installed) a 3DGS ``.ply`` —
  the same plug-in pattern as the in-browser MuJoCo-GS-Web ``.spz`` upload.

Run:

    python -m examples.mujoco_gs.app
    # or
    python examples/mujoco_gs/app.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Make the example importable both as `python -m examples.mujoco_gs.app` and
# as `python examples/mujoco_gs/app.py`.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent.parent))

from examples.mujoco_gs import agent as agent_mod  # noqa: E402
from examples.mujoco_gs.backgrounds import (  # noqa: E402
    GsplatBackground,
    PanoramaBackground,
)

logger = logging.getLogger("mujoco_gs.app")

# Live MJPEG stream settings.
LIVE_W, LIVE_H, LIVE_FPS = 480, 360, 15


def _live_img_html(camera: str) -> str:
    """A visible, labeled panel with an <img> pulling the MJPEG live stream.

    The browser renders ``multipart/x-mixed-replace`` JPEG frames as they
    arrive — a single long-lived HTTP response that streams near-real-time and
    is far more proxy-friendly than Gradio's buffered SSE queue.

    We wrap the <img> in a fixed-height bordered box so the panel is visible
    even before the first frame arrives (a bare <img> collapses to zero
    height), and add an ``onerror`` fallback so a failed stream shows a
    message instead of nothing. ``gr.HTML`` injects markup via innerHTML and
    does not run inline <script>, so the stream URL is set directly on the
    ``src`` attribute (root-absolute ``/live`` — the app is served at root).
    """
    import time as _t

    bust = int(_t.time() * 1000)
    return f"""
<div style="border:2px solid #4a90d9; border-radius:8px; padding:6px; background:#0b0b0b;
            max-width:{LIVE_W + 16}px; margin:0 auto;">
  <div style="color:#9cc; font-size:13px; margin-bottom:4px;">
    🔴 Live view — {camera} (near real-time MJPEG)
  </div>
  <img src="/live?camera={camera}&t={bust}"
       style="width:100%; aspect-ratio:{LIVE_W} / {LIVE_H}; height:auto; display:block;
              object-fit:contain; background:#000; border-radius:6px;"
       alt="connecting to live stream…"
       onerror="this.alt='live stream did not load (connection may be buffering) - use the clip below';" />
</div>
"""


def _mjpeg_frames(holder, camera: str):
    """Yield ``multipart/x-mixed-replace`` JPEG chunks of the live composite."""
    import time as _t

    import cv2

    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    frame_dt = 1.0 / float(LIVE_FPS)
    try:
        while True:
            t0 = _t.time()
            try:
                rgb = holder.render_now(camera_name=camera, width=LIVE_W, height=LIVE_H)
                ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield boundary + buf.tobytes() + b"\r\n"
            except Exception as e:  # pragma: no cover
                logger.debug("live frame error: %s", e)
            elapsed = _t.time() - t0
            if elapsed < frame_dt:
                _t.sleep(frame_dt - elapsed)
    except GeneratorExit:  # client disconnected
        return


# --------------------------------------------------------------------------- #
# Build the Gradio interface
# --------------------------------------------------------------------------- #


def build_app(
    panorama_path: Optional[str] = None,
    gsplat_ply: Optional[str] = None,
    model_id: Optional[str] = None,
    initial_camera: str = "front",
):
    try:
        import gradio as gr
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "gradio is required for the UI. Run `pip install gradio` " "(see examples/mujoco_gs/README.md)."
        ) from e

    logger.info("Building MujocoGsAgent (panorama=%s, gsplat_ply=%s, model=%s)", panorama_path, gsplat_ply, model_id)
    holder = agent_mod.build(
        panorama_path=panorama_path,
        gsplat_ply=gsplat_ply,
        model_id=model_id,
    )

    # ------ callbacks ------ #

    def on_render(camera: str) -> np.ndarray:
        return holder.render_now(camera_name=camera)

    def on_reset() -> Tuple[List, np.ndarray, str]:
        holder.reset_scene()
        frame = holder.render_now(camera_name=initial_camera)
        return [], frame, "Scene reset."

    def on_background_change(
        choice: str,
        panorama_upload: Optional[str],
        ply_upload: Optional[str],
        rotation_deg: float,
    ) -> str:
        try:
            if choice == "Procedural panorama":
                bg = PanoramaBackground(rotation_deg=rotation_deg)
            elif choice == "Custom panorama":
                if not panorama_upload:
                    return "⚠ Upload a panorama image first."
                bg = PanoramaBackground(image_path=panorama_upload, rotation_deg=rotation_deg)
            elif choice == "3D Gaussian Splat (.ply)":
                if not ply_upload:
                    return "⚠ Upload a .ply file first."
                bg = GsplatBackground(ply_path=ply_upload)
            else:
                return f"Unknown background: {choice}"
            holder.set_background(bg)
            return f"Background → {bg.name} (ok)"
        except ImportError as e:
            return f"⚠ {e}"
        except Exception as e:  # pragma: no cover
            logger.exception("Background swap failed.")
            return f"⚠ {type(e).__name__}: {e}"

    # Live-stream frame size: small enough to push smoothly through a remote
    # `gradio.live` share tunnel (full-res 640×480 PNGs at >10 fps saturate
    # the tunnel and the preview looks frozen). The final frame is full-res.

    def on_chat(user_message: str, chat_history: list, camera: str):
        """Run one agent turn; show the reply + a freshly composited still.

        The agent drives the scene with real `Simulation` actions (e.g.
        `run_policy` for motion); the live MJPEG view above shows the motion
        as it happens, and this returns the composited still afterwards.

        Yields (msg_box, chat_history, preview).
        """
        import gradio as gr

        if not user_message.strip():
            yield "", chat_history, gr.update()
            return

        base = chat_history + [{"role": "user", "content": user_message}]
        working = base + [{"role": "assistant", "content": "_…working (watch the live view)…_"}]
        yield "", working, gr.update()

        result: dict = {}

        def _run():
            try:
                result["out"] = holder.chat(user_message)
            except Exception as e:  # pragma: no cover
                logger.exception("Agent turn failed.")
                result["err"] = f"⚠ Agent error: {type(e).__name__}: {e}"

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        while th.is_alive():
            time.sleep(0.1)
        th.join()

        if "err" in result:
            text, frame = result["err"], None
        else:
            text, frame = result.get("out", ("(no reply)", None))
        if frame is None:
            try:
                frame = holder.render_now(camera_name=camera)
            except Exception:  # pragma: no cover
                frame = None
        final = base + [{"role": "assistant", "content": text or "(no reply)"}]
        yield "", final, frame

    # ------ layout ------ #

    title = "MuJoCo-GS — strands-robots hybrid render demo"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            f"# {title}\n"
            "MuJoCo physics + photoreal background, composited with depth-aware "
            "occlusion. Inspired by "
            "[MuJoCo-GS-Web](https://vector-wangel.github.io/MuJoCo-GS-Web/), "
            "implemented on top of the `strands-robots` `Simulation` AgentTool.\n\n"
            "Type natural-language commands on the right (e.g. *“have the arm wave”*, "
            "*“move the cube to the left”*, *“switch to topdown view and render”*). "
            "The Strands agent will translate them into `Simulation` actions and "
            "show the composited frame on the left."
        )

        with gr.Row():
            # Left column — preview + scene controls.
            with gr.Column(scale=5):
                live_view = gr.HTML(_live_img_html(initial_camera))
                preview = gr.Image(
                    value=holder.render_now(camera_name=initial_camera),
                    label="Composite preview (still)",
                    type="numpy",
                    interactive=False,
                    height=360,
                )
                with gr.Row():
                    camera_dd = gr.Dropdown(
                        choices=["front", "topdown", "oblique", "default"],
                        value=initial_camera,
                        label="Preview camera",
                    )
                    render_btn = gr.Button("Render now", variant="primary")
                    reset_btn = gr.Button("Reset scene")

                with gr.Accordion("Background", open=False):
                    bg_choice = gr.Radio(
                        choices=[
                            "Procedural panorama",
                            "Custom panorama",
                            "3D Gaussian Splat (.ply)",
                        ],
                        value="Procedural panorama",
                        label="Background renderer",
                    )
                    rotation = gr.Slider(
                        minimum=-180,
                        maximum=180,
                        value=0,
                        step=5,
                        label="Panorama yaw (deg)",
                    )
                    panorama_file = gr.File(
                        label="Equirectangular panorama (.jpg / .png)",
                        file_types=["image"],
                        type="filepath",
                    )
                    ply_file = gr.File(
                        label="3DGS .ply (requires `pip install '.[gsplat]'`)",
                        file_types=[".ply"],
                        type="filepath",
                    )
                    apply_bg_btn = gr.Button("Apply background")
                    bg_status = gr.Markdown("Background → panorama (procedural)")

            # Right column — chat.
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    label="Strands Agent",
                    height=520,
                )
                msg_box = gr.Textbox(
                    label="Message",
                    placeholder="e.g. 'have the arm wave then render the front view'",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear chat")

        # ------ wiring ------ #
        render_btn.click(on_render, inputs=[camera_dd], outputs=[preview])
        reset_btn.click(on_reset, outputs=[chatbot, preview, bg_status])
        # Point the live MJPEG <img> at the newly selected camera.
        camera_dd.change(lambda cam: _live_img_html(cam), inputs=[camera_dd], outputs=[live_view])
        apply_bg_btn.click(
            on_background_change,
            inputs=[bg_choice, panorama_file, ply_file, rotation],
            outputs=[bg_status],
        ).then(on_render, inputs=[camera_dd], outputs=[preview])

        # Chat: send button + Enter key both submit.
        send_btn.click(
            on_chat,
            inputs=[msg_box, chatbot, camera_dd],
            outputs=[msg_box, chatbot, preview],
        )
        msg_box.submit(
            on_chat,
            inputs=[msg_box, chatbot, camera_dd],
            outputs=[msg_box, chatbot, preview],
        )
        clear_btn.click(lambda: [], outputs=[chatbot])

    return demo, holder


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--panorama",
        type=str,
        default=None,
        help="Path to an equirectangular panorama image. If omitted a " "procedural kitchen-ish panorama is generated.",
    )
    parser.add_argument(
        "--gsplat-ply",
        type=str,
        default=None,
        help="Path to a 3DGS .ply file. Requires `pip install '.[gsplat]'`. " "Overrides --panorama.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Strands model id (e.g. an Anthropic / Bedrock model). If " "unset Strands picks a default.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="front",
        help="Initial preview camera (default: front).",
    )
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (off by default).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Most Linux servers default to no display — pick a headless GL backend.
    os.environ.setdefault("MUJOCO_GL", "egl")

    demo, holder = build_app(
        panorama_path=args.panorama,
        gsplat_ply=args.gsplat_ply,
        model_id=args.model,
        initial_camera=args.camera,
    )
    demo.queue(default_concurrency_limit=1)  # Gradio events serialized; /live is separate.
    launch_kwargs: dict = {
        "server_name": args.server_name,
        "server_port": args.server_port,
        "share": args.share,
        "prevent_thread_lock": True,  # so we can mount the /live route, then block
    }
    # Gradio 6+ moved `theme` from `Blocks(...)` to `launch(...)`. Older
    # versions don't take it as a launch kwarg, so we feature-test.
    try:
        import gradio as gr  # noqa: F401  — already imported by build_app

        if hasattr(gr, "themes"):
            launch_kwargs["theme"] = gr.themes.Soft()
    except Exception:  # pragma: no cover
        pass
    demo.launch(**launch_kwargs)

    # Mount the MJPEG live-stream route on Gradio's FastAPI app. This streams
    # near-real-time JPEG frames over a single long-lived HTTP response (an
    # <img> in the UI renders them incrementally) — bypassing Gradio's
    # buffered SSE queue, which a port-forwarding / share proxy coalesces into
    # an end-of-turn burst.
    from fastapi.responses import StreamingResponse

    def _live_route(camera: str = "front"):
        return StreamingResponse(
            _mjpeg_frames(holder, camera),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
        )

    demo.app.add_api_route("/live", _live_route, methods=["GET"])
    logger.info("MJPEG live stream mounted at /live")

    # Block the main thread (launch used prevent_thread_lock=True).
    try:
        demo.block_thread()
    except (KeyboardInterrupt, AttributeError):  # pragma: no cover
        import time as _t

        while True:
            _t.sleep(3600)


if __name__ == "__main__":
    main()
