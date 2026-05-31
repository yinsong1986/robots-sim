# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strands Agent driving the MuJoCo + 3DGS hybrid demo.

Wires a :class:`strands_robots.simulation.Simulation` (MuJoCo backend) to a
:class:`HybridCompositor` (panorama or gsplat background) and exposes both
through a Strands :class:`Agent`. The agent has access to:

* the upstream ``Simulation`` AgentTool (58 actions: world composition,
  physics, run_policy, recording, …)
* a custom ``hybrid_render`` tool that returns a *composited* PNG
  (foreground MuJoCo + photoreal background) as the agent's reply image.

Designed to be driven from :mod:`app` (Gradio chat) or from a plain REPL —
:meth:`MujocoGsAgent.chat` is the single user-facing entry point.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from .backgrounds import BackgroundRenderer, GsplatBackground, PanoramaBackground
from .compositor import HybridCompositor
from .scene import SCENE_DESCRIPTION, build_default_scene, make_scene_description

if TYPE_CHECKING:
    from strands_robots.simulation import Simulation

logger = logging.getLogger(__name__)


def make_system_prompt(scene_description: str) -> str:
    """Build the agent system prompt around a concrete scene description.

    Takes the *actual* scene description (with the robot config that really
    loaded) so the agent is never misinformed about what's in the world.
    """
    return f"""\
You are the orchestrator for a MuJoCo + 3D Gaussian Splatting (3DGS) hybrid
robotics demo. You control a small simulated arm and you can render
photoreal frames that show the arm against a 3DGS background scene — exactly
like the MuJoCo-GS-Web browser demo, but driven from Python through the
strands-robots Simulation AgentTool.

>>> THE SCENE IS ALREADY FULLY BUILT AND READY. <<<
The world, the `arm` robot, the `cube`, and the cameras below already exist.
You do NOT need to set anything up. NEVER call these actions — they will
error or wipe the demo, and make you loop:
  - `create_world`, `load_scene`   (world already exists)
  - `add_robot`, `add_object`, `add_camera`   (everything already added)
  - `reset`, `destroy`             (these wipe the running demo)
If a `Simulation` call returns an error, do NOT try to "fix" it by rebuilding
the scene or adding robots/objects. Just report the error briefly and stop.

{scene_description}

How to drive the scene (allowed actions only):
  * Move / wave / "do a demo": call the `animate` tool ONCE
    (`animate(kind="wave"|"nod"|"reach"|"stir", camera_name="front")`). It
    plays a smooth real-time motion the user can WATCH in the live preview,
    then returns the final frame. Prefer this over `run_policy` for any
    motion request — `run_policy` only jitters randomly and shows no live
    motion.
  * Pose the arm precisely: `set_joint_positions(robot_name="arm",
    positions={...})`.
  * Move the cube: `move_object(name="cube", position=[x, y, z])`.
  * Other physics: `apply_force`, `step`, `get_state`, `get_body_state`.
  * To show a still frame WITHOUT moving anything, call `hybrid_render`
    (NOT `Simulation.render`). Default `camera_name="front"` unless the user
    asks for "topdown" or "oblique".

Keep every turn SHORT: usually ONE tool call (`animate` for motion, or
`hybrid_render` for a still), then a one- or two-sentence reply. Do not
explore or verify with extra calls — trust the scene description above. Be
concise.
"""


# Default prompt (used only if a caller builds an agent without a live scene).
SYSTEM_PROMPT = make_system_prompt(SCENE_DESCRIPTION)


@dataclass
class MujocoGsAgent:
    """High-level wrapper holding the sim, compositor, and Strands agent.

    Construct via :func:`build`, not directly — it owns lifecycle resources
    (a MuJoCo simulation and a thread-pool executor) that should be created
    in one place.

    Attributes:
        sim: the underlying ``Simulation`` (MuJoCo backend).
        compositor: hybrid renderer.
        agent: Strands ``Agent`` with the Simulation tool and the
            ``hybrid_render`` tool registered.
        history: list of (user_msg, assistant_msg, last_frame_or_None)
            tuples — handy to drive a Gradio chat panel.
    """

    sim: "Simulation"
    compositor: HybridCompositor
    agent: Any  # strands.Agent — typed Any to keep import optional at module load
    history: List[Tuple[str, str, Optional[np.ndarray]]] = field(default_factory=list)

    # ----- user-facing API ----- #

    def chat(self, message: str):
        """Send a message to the agent and return ``(text, frame, video)``.

        ``frame`` is the most-recent still composited during the turn;
        ``video`` is the path to an MP4 if the agent recorded a motion (via
        the ``animate`` tool), else ``None``. This default implementation is
        replaced per-instance by :func:`_wire_frame_holder` so it can read the
        tool closures; it is kept as a sensible fallback.
        """
        result = self.agent(message)
        text = _extract_agent_text(result)
        return text, None, None

    def render_now(
        self,
        camera_name: str = "front",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> np.ndarray:
        """Render a composite frame *without* going through the agent.

        Useful for the Gradio "live preview" panel that polls outside of
        chat turns. ``width``/``height`` let the live stream use a smaller,
        bandwidth-friendly frame over a remote share tunnel.
        """
        return self.compositor.render(camera_name=camera_name, width=width, height=height).rgb

    def stream_motion(
        self,
        kind: str = "wave",
        camera_name: str = "front",
        duration_s: float = 0.8,
        fps: int = 12,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """Yield composited frames of a scripted arm motion, in real time.

        This is the "watch the arm actually move" path. It scripts a smooth
        joint trajectory directly (single-threaded: set joints → step →
        render → yield), so the caller — e.g. a Gradio generator callback —
        receives a live stream of frames rather than a single final pose.

        Args:
            kind: motion preset. One of ``"wave"``, ``"nod"``, ``"reach"``,
                ``"stir"``. Unknown values fall back to ``"wave"``.
            camera_name: camera to render from.
            duration_s: wall-clock length of the motion.
            fps: target frames per second (also the trajectory resolution).
            width: optional render width (smaller = faster, for a snappier
                live stream). Defaults to the compositor's default width.
            height: optional render height.

        Yields:
            ``(H, W, 3) uint8`` composited frames.
        """
        import time as _time

        n_frames = max(1, int(duration_s * fps))
        traj = _build_joint_trajectory(self.sim, kind=kind, n_frames=n_frames, duration_s=duration_s)
        frame_dt = 1.0 / float(fps)

        for i, positions in enumerate(traj):
            t0 = _time.time()
            if positions is not None:
                try:
                    self.sim.set_joint_positions(positions=positions, robot_name="arm")
                except Exception as e:  # pragma: no cover
                    logger.warning("set_joint_positions failed mid-motion: %s", e)
            self.sim.step(2)
            frame = self.compositor.render(camera_name=camera_name, width=width, height=height).rgb
            yield frame
            # Pace to ~real time so the motion looks natural in the browser.
            elapsed = _time.time() - t0
            if elapsed < frame_dt:
                _time.sleep(frame_dt - elapsed)

    def set_background(self, background: BackgroundRenderer) -> None:
        """Hot-swap the background (UI dropdown, etc.)."""
        self.compositor.set_background(background)

    def record_motion(
        self,
        kind: str = "wave",
        camera_name: str = "front",
        duration_s: float = 4.0,
        fps: int = 20,
        width: Optional[int] = None,
        height: Optional[int] = None,
        realtime: bool = True,
    ) -> str:
        """Render a scripted arm motion to an MP4 file and return its path.

        Records a compact H.264 clip (a single file transfers reliably and
        plays back smoothly on the client). When ``realtime`` is True the sim
        is stepped at wall-clock pace so the ``/live`` MJPEG view also shows
        the motion at natural speed as it happens; when False, frames are
        rendered as fast as possible (the MP4 fps metadata still gives correct
        playback) for a quick, no-live-view export.

        Args:
            kind: motion preset — "wave", "nod", "reach", "stir".
            camera_name: camera to render from.
            duration_s: playback length of the clip.
            fps: playback frame rate (also trajectory resolution).
            width/height: render size (smaller = smaller file).
            realtime: pace stepping to wall-clock so the live view animates.

        Returns:
            Absolute path to a ``.mp4`` file.
        """
        import time as _time

        n_frames = max(2, int(duration_s * fps))
        traj = _build_joint_trajectory(self.sim, kind=kind, n_frames=n_frames, duration_s=duration_s)
        step_dt = duration_s / max(1, n_frames)
        frames = []
        for positions in traj:
            t0 = _time.time()
            if positions is not None:
                try:
                    self.sim.set_joint_positions(positions=positions, robot_name="arm")
                except Exception as e:  # pragma: no cover
                    logger.warning("record_motion set_joint_positions failed: %s", e)
            self.sim.step(2)
            frames.append(self.compositor.render(camera_name=camera_name, width=width, height=height).rgb)
            if realtime:
                elapsed = _time.time() - t0
                if elapsed < step_dt:
                    _time.sleep(step_dt - elapsed)
        return _encode_mp4(frames, fps=fps)

    def reset_scene(self) -> None:
        """Tear down and rebuild the default scene."""
        try:
            self.sim.destroy()
        except Exception as e:  # pragma: no cover  — best-effort cleanup
            logger.warning("Simulation.destroy() raised: %s", e)
        build_default_scene(self.sim)
        # Cached renderers/backgrounds reference the old model — drop them.
        self.compositor.clear_caches()
        self.history.clear()

    def close(self) -> None:
        try:
            self.compositor.close()
        except Exception:  # pragma: no cover
            pass
        try:
            self.sim.destroy()
        except Exception:  # pragma: no cover
            pass


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def build(
    background: Optional[BackgroundRenderer] = None,
    panorama_path: Optional[str] = None,
    gsplat_ply: Optional[str] = None,
    model_id: Optional[str] = None,
) -> MujocoGsAgent:
    """Construct a fully wired :class:`MujocoGsAgent`.

    Args:
        background: pre-built renderer. If ``None``, one is chosen from the
            other args (``gsplat_ply`` > ``panorama_path`` > procedural).
        panorama_path: path to an equirectangular ``.jpg``/``.png``.
        gsplat_ply: path to a 3DGS ``.ply`` (requires ``gsplat`` extra).
        model_id: optional Strands model id (e.g. an Anthropic / Bedrock
            model). If ``None``, defaults to whatever Strands is configured
            with.

    Returns:
        A ready-to-chat :class:`MujocoGsAgent`.
    """
    # Lazy imports so the module stays importable for static analysis even
    # when strands deps aren't installed yet.
    try:
        from strands import Agent, tool
    except ImportError as e:  # pragma: no cover
        raise ImportError("strands-agents is required. Run `pip install strands-agents`.") from e
    try:
        from strands_robots.simulation import Simulation
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "strands-robots[sim-mujoco] is required. " "Run `pip install 'strands-robots[sim-mujoco]'`."
        ) from e

    # 1. Pick / build the background.
    if background is None:
        if gsplat_ply:
            background = GsplatBackground(ply_path=gsplat_ply)
        else:
            background = PanoramaBackground(image_path=panorama_path)

    # 2. Build sim + compositor + scene. Capture the robot config that
    #    actually loaded so the system prompt reflects reality.
    sim = Simulation(tool_name="sim", mesh=False)
    scene_summary = build_default_scene(sim)
    compositor = HybridCompositor(sim, background=background)
    scene_description = make_scene_description(
        robot_config=scene_summary.get("robot_config", "so101"),
        robot_name=scene_summary.get("robot_name", "arm"),
    )

    # 3. The custom `hybrid_render` tool. Closed over `compositor` and a
    #    holder that lets `MujocoGsAgent.chat` recover the last rendered
    #    frame (Strands itself doesn't propagate image bytes back to us out
    #    of band — closures are the simplest fix).
    last_frame_holder: dict = {"frame": None, "video": None}

    @tool
    def hybrid_render(
        camera_name: str = "front",
        width: int = 640,
        height: int = 480,
    ) -> dict:
        """Render a photoreal hybrid frame (MuJoCo foreground + 3DGS background).

        Always prefer this over `Simulation.render` when showing the user
        a frame — it composites the photoreal background behind the MuJoCo
        geometry with proper depth occlusion, which is the whole point of
        the demo.

        Args:
            camera_name: existing camera name. One of "front" (hero shot,
                default), "topdown", "oblique", or any other camera the user
                added with `Simulation.add_camera`.
            width: image width in pixels (default 640).
            height: image height in pixels (default 480).

        Returns:
            An AgentTool result containing the composited PNG image plus a
            small JSON summary (foreground pixel count, depth range).
        """
        frame = compositor.render(camera_name=camera_name, width=int(width), height=int(height))
        last_frame_holder["frame"] = frame.rgb
        png_bytes = _png_bytes(frame.rgb)
        fg_frac = float(frame.foreground_mask.mean())
        depth_min = float(frame.depth.min())
        depth_max = float(frame.depth.max())
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Hybrid render {width}x{height} from camera "
                        f"'{camera_name}' — {fg_frac*100:.1f}% MuJoCo foreground, "
                        f"depth {depth_min:.2f}–{depth_max:.2f} m"
                    )
                },
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": png_bytes},
                    }
                },
                {
                    "json": {
                        "camera": camera_name,
                        "width": width,
                        "height": height,
                        "foreground_fraction": fg_frac,
                        "depth_min_m": depth_min,
                        "depth_max_m": depth_max,
                        "background": compositor.background.name,
                    }
                },
            ],
        }

    @tool
    def animate(kind: str = "wave", camera_name: str = "front", duration_s: float = 4.0) -> dict:
        """Make the arm move and record a video the user can watch.

        Use this for any "make the arm move / wave / reach / do a demo"
        request. It renders the motion to a short MP4 clip (shown autoplaying
        in the UI's video panel) plus a final still frame. Prefer this over
        `run_policy` for demo motions — it produces a clean, legible
        trajectory (a real wave / reach) instead of random jitter.

        Args:
            kind: motion preset — "wave" (greeting waggle, default), "nod",
                "reach" (extend forward), or "stir" (circular).
            camera_name: camera to render from.
            duration_s: playback length of the clip in seconds (0.3–15).
                Default 4 s. Pass a larger value (e.g. 8) for a longer demo or
                a smaller one (e.g. 0.6) for a quick flick.

        Returns:
            An AgentTool result with the final composited PNG. The video clip
            is surfaced to the UI separately.
        """
        dur = float(max(0.3, min(15.0, duration_s)))
        fps = 20
        # Step the motion in REAL TIME so the /live MJPEG view shows it at
        # natural speed, while also collecting frames (reduced res) for the
        # MP4 clip. The live view renders independently at ~15 fps and picks
        # up each step as it happens.
        import time as _time

        traj = _build_joint_trajectory(sim, kind=kind, n_frames=max(2, int(dur * fps)), duration_s=dur)
        step_dt = dur / max(1, len(traj))
        frames = []
        for positions in traj:
            t0 = _time.time()
            if positions is not None:
                try:
                    sim.set_joint_positions(positions=positions, robot_name="arm")
                except Exception as e:  # pragma: no cover
                    logger.warning("animate set_joint_positions failed: %s", e)
            sim.step(2)
            frames.append(compositor.render(camera_name=camera_name, width=480, height=360).rgb)
            elapsed = _time.time() - t0
            if elapsed < step_dt:
                _time.sleep(step_dt - elapsed)
        video_path = _encode_mp4(frames, fps=fps)
        last_frame_holder["video"] = video_path

        frame = compositor.render(camera_name=camera_name)
        last_frame_holder["frame"] = frame.rgb
        return {
            "status": "success",
            "content": [
                {"text": f"Recorded a {dur:.1f}s '{kind}' motion from camera '{camera_name}'."},
                {"image": {"format": "png", "source": {"bytes": _png_bytes(frame.rgb)}}},
                {"json": {"motion": kind, "duration_s": dur, "camera": camera_name, "video": video_path}},
            ],
        }

    # 4. Build the Strands agent.
    agent_kwargs: dict = {
        "tools": [sim, hybrid_render, animate],
        "system_prompt": make_system_prompt(scene_description),
    }
    if model_id is not None:
        agent_kwargs["model"] = model_id
    agent = Agent(**agent_kwargs)

    holder = MujocoGsAgent(sim=sim, compositor=compositor, agent=agent)
    # Patch in the closure so MujocoGsAgent.chat can read it.
    holder._last_frame_holder = last_frame_holder  # type: ignore[attr-defined]
    holder._wire_frame_holder()
    return holder


# Wire it up on the dataclass without making it a constructor arg.
def _wire_frame_holder(self: MujocoGsAgent) -> None:  # type: ignore[override]
    holder: dict = getattr(self, "_last_frame_holder", {"frame": None, "video": None})

    def chat(self_inner: MujocoGsAgent, message: str):
        holder["frame"] = None
        holder["video"] = None
        result = self_inner.agent(message)
        text = _extract_agent_text(result)
        frame = holder["frame"]
        video = holder.get("video")
        self_inner.history.append((message, text, frame))
        return text, frame, video

    # Per-instance override.
    self.chat = chat.__get__(self, MujocoGsAgent)  # type: ignore[assignment]

    # Per-instance override.
    self.chat = chat.__get__(self, MujocoGsAgent)  # type: ignore[assignment]


MujocoGsAgent._wire_frame_holder = _wire_frame_holder  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _encode_mp4(frames: list, fps: int = 20) -> str:
    """Encode a list of ``(H, W, 3) uint8`` frames to an H.264 MP4.

    Returns the path to a temp ``.mp4``. A compact video transfers reliably
    through buffering proxies / share tunnels and plays back smoothly on the
    client — unlike streaming many per-frame image updates.
    """
    import tempfile

    try:
        import imageio
    except ImportError as e:  # pragma: no cover
        raise ImportError("imageio (with imageio-ffmpeg) is required to record motion clips.") from e

    if not frames:
        raise ValueError("no frames to encode")
    # libx264 needs even dimensions; macro_block_size=8 pads as needed.
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="mujoco_gs_")
    os.close(fd)
    imageio.mimsave(path, frames, fps=int(fps), codec="libx264", quality=7, macro_block_size=8)
    return path


def _arm_joints(sim) -> list:
    """Return ``[(name, qpos_addr, (lo, hi)), ...]`` for the arm's hinge joints.

    Hinge joints are MuJoCo type 3 (``mjJNT_HINGE``). We select those whose
    name belongs to the ``arm`` robot (namespaced as ``arm/<joint>``), which
    excludes the free-floating ``cube_joint``.
    """
    import mujoco

    m = sim.mj_model
    d = sim.mj_data
    out = []
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name or int(m.jnt_type[j]) != 3:  # 3 == hinge
            continue
        if "/" in name and not name.startswith("arm/"):
            continue
        if "/" not in name and "arm" not in name.lower():
            # In single-robot scenes joints may be un-namespaced; keep hinges.
            pass
        qadr = int(m.jnt_qposadr[j])
        lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
        if lo >= hi:  # unlimited joint
            lo, hi = -3.14, 3.14
        out.append((name, qadr, (lo, hi), float(d.qpos[qadr])))
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _build_joint_trajectory(sim, kind: str = "wave", n_frames: int = 48, duration_s: float = 4.0) -> list:
    """Build a list of per-frame ``{joint_name: angle}`` dicts for a motion.

    Scripts smooth, watchable trajectories using whatever hinge joints the
    arm exposes (works for SO-100/SO-101 by joint-name suffix; degrades
    gracefully to "oscillate the first couple of joints" for other arms).

    Args:
        sim: the simulation (read for joint names / ranges / home pose).
        kind: ``"wave"``, ``"nod"``, ``"reach"``, or ``"stir"``.
        n_frames: number of trajectory samples.
        duration_s: wall-clock length of the motion. Oscillation cycle counts
            scale with this so a longer wave waves *more times* (at a natural
            ~1.2 Hz) rather than just slower.

    Returns:
        A list of length ``n_frames``; each entry is a dict mapping joint
        name → target angle (radians).
    """
    joints = _arm_joints(sim)
    if not joints:
        return [None] * n_frames

    by_suffix = {name.split("/")[-1].lower(): (name, qa, rng, home) for name, qa, rng, home in joints}
    home = {name: h for name, _, _, h in joints}

    def pick(*suffixes):
        for s in suffixes:
            if s in by_suffix:
                return by_suffix[s]
        return None

    rot = pick("rotation", "joint1", "waist")
    pitch = pick("pitch", "shoulder", "joint2")
    elbow = pick("elbow", "joint3")
    wroll = pick("wrist_roll", "joint6", "joint7")
    wpitch = pick("wrist_pitch", "joint5")

    import math

    # Oscillation cycles scale with duration (~1.2 Hz) so a longer motion
    # waves more times rather than slowing down. Floor of 2 keeps even a
    # sub-second wave a recognisable back-and-forth.
    n_cycles = max(2, int(round(1.2 * float(duration_s))))

    traj = []
    for i in range(n_frames):
        p = i / max(1, n_frames - 1)  # 0..1
        # Smoothstep ease for the "lift into ready pose" portion. Lift quickly
        # (first ~15%) so most of the motion is the actual wave/oscillation.
        ramp = min(1.0, p / 0.15)
        ease = ramp * ramp * (3 - 2 * ramp)
        pose = dict(home)

        if kind == "nod" and wpitch:
            name, qa, (lo, hi), h = wpitch
            pose[name] = _clamp(h + 0.6 * math.sin(2 * math.pi * n_cycles * p), lo, hi)
        elif kind == "reach" and pitch and elbow:
            pn, _, (plo, phi), ph = pitch
            en, _, (elo, ehi), eh = elbow
            pose[pn] = _clamp(ph + ease * (-1.0), plo, phi)
            pose[en] = _clamp(eh + ease * (1.4), elo, ehi)
            if rot:
                rn, _, (rlo, rhi), rh = rot
                pose[rn] = _clamp(rh + ease * 0.5 * math.sin(2 * math.pi * max(1, n_cycles // 3) * p), rlo, rhi)
        elif kind == "stir" and rot and wroll:
            rn, _, (rlo, rhi), rh = rot
            wn, _, (wlo, whi), wh = wroll
            pose[rn] = _clamp(rh + ease * 0.5 * math.cos(2 * math.pi * n_cycles * p), rlo, rhi)
            pose[wn] = _clamp(wh + ease * 0.8 * math.sin(2 * math.pi * n_cycles * p), wlo, whi)
        else:  # "wave" (default): lift the arm, then waggle the wrist/base
            if pitch:
                pn, _, (plo, phi), ph = pitch
                pose[pn] = _clamp(ph + ease * (-0.9), plo, phi)
            if elbow:
                en, _, (elo, ehi), eh = elbow
                pose[en] = _clamp(eh + ease * (1.3), elo, ehi)
            waggle = ease * 0.9 * math.sin(2 * math.pi * n_cycles * p)
            if wroll:
                wn, _, (wlo, whi), wh = wroll
                pose[wn] = _clamp(wh + waggle, wlo, whi)
            elif rot:
                rn, _, (rlo, rhi), rh = rot
                pose[rn] = _clamp(rh + waggle, rlo, rhi)

        traj.append(pose)
    return traj


def _extract_agent_text(result: Any) -> str:
    """Best-effort extraction of the assistant's text reply from a Strands
    ``Agent.__call__`` result.

    Strands' agent return shape evolved across versions; this tries the most
    common shapes and falls back to ``str(result)``.
    """
    if result is None:
        return ""
    # Strands AgentResult.message.content -> list of {"text": ...} dicts
    msg = getattr(result, "message", None)
    if msg is not None:
        content = getattr(msg, "content", None) or msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            if texts:
                return "\n".join(t for t in texts if t)
    # Plain string return
    if isinstance(result, str):
        return result
    return str(result)
