# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic GR00T-on-LIBERO runner for the MuJoCo-GS demo.

A real NVIDIA GR00T vision-language-action policy drives a Franka **Panda**
through a **LIBERO** task, invoked **agentically**: a Strands ``Agent`` picks
``evaluate_benchmark`` off the ``Simulation`` tool's action surface and fills
its kwargs from a natural-language instruction (the pattern from
``examples/libero/run_mujoco_agent.py``). Meanwhile a background thread renders
the scene through the :class:`HybridCompositor` into a JPEG buffer so the UI's
``/live`` MJPEG route shows the arm in near-real-time, and collects frames for
a recorded clip.

Getting a *successful* episode (the recipe verified against
``nvidia/GR00T-N1.7-LIBERO``):

* **Match the task suite to the served checkpoint.** The bundled container
  serves ``/data/checkpoints/libero_10``, so run ``libero_10`` tasks. Running a
  different suite against it is an embodiment/skill mismatch and scores ~0.
* **Don't cap ``max_steps``.** LIBERO-Long episodes need ~500 steps; capping
  them truncates before completion. We use the adapter's default.
* **Pre-warm the scene** (generate BDDL scene → ``load_scene`` → ``prewarm``)
  so the ``image``/``wrist_image`` cameras exist before inference/recording.
* **Let ``evaluate_benchmark`` auto-pick the robot** (omit ``robot_name``) and
  use its **default ``action_horizon``** — the LIBERO scene renames its Panda
  to ``robot`` on episode start.

Concurrency note: only the eval thread (sim's internal camera renders for the
policy) and the compositor's single render thread touch GL; the ``/live`` route
just re-serves ``latest_jpeg``. The LIBERO ``viz_option`` (collision-geom / site
markers hidden) is applied by the compositor, so the arm renders clean.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

import numpy as np

logger = logging.getLogger("mujoco_gs.groot_libero")

# Suite whose checkpoint the bundled GR00T container serves. Override if your
# server hosts a different `libero_<suite>/` sub-checkpoint.
DEFAULT_SUITE = "libero_10"


class GrootLiberoRunner:
    """Agentic driver for real-GR00T-on-LIBERO episodes used by the UI."""

    def __init__(
        self,
        suite: str = DEFAULT_SUITE,
        host: str = "127.0.0.1",
        port: int = 8000,
        data_config: str = "libero_panda",
        groot_version: str = "n1.7",
        render_width: int = 512,
        render_height: int = 384,
        model_id: Optional[str] = None,
    ) -> None:
        self.suite = suite
        self.host = host
        self.port = port
        self.data_config = data_config
        self.groot_version = groot_version
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.model_id = model_id

        self._adapters: Optional[dict] = None
        self._task_label_to_name: dict = {}
        self.latest_jpeg: Optional[bytes] = None
        self.latest_rgb: Optional[np.ndarray] = None
        self.running = False
        self._lock = threading.Lock()

    # ----- task discovery ----- #

    def available_tasks(self) -> List[str]:
        self._ensure_suite_loaded()
        return list(self._task_label_to_name.keys())

    def _ensure_suite_loaded(self) -> None:
        if self._adapters is not None:
            return
        from strands_robots.benchmarks.libero import load_libero_suite

        logger.info("Loading LIBERO suite %r…", self.suite)
        # NOTE: no max_steps cap — LIBERO-Long needs the full default budget.
        self._adapters = load_libero_suite(self.suite)
        for full_name in self._adapters:
            label = full_name.split("-", 2)[-1].replace("_", " ")
            self._task_label_to_name[label] = full_name
        logger.info("Loaded %d LIBERO tasks.", len(self._task_label_to_name))

    def default_task_label(self) -> Optional[str]:
        """A task likely to succeed with the default (libero_10) checkpoint."""
        labels = self.available_tasks()
        if not labels:
            return None
        # Prefer the verified white-mug task if present.
        for lab in labels:
            if "white mug" in lab and "left plate" in lab:
                return lab
        return labels[0]

    # ----- run one episode, agentically ----- #

    def run(self, task_label: str, n_episodes: int = 3, seed: int = 42, on_progress=None) -> dict:
        """Agentically run ``n_episodes`` of ``task_label`` under GR00T.

        A Strands ``Agent`` invokes ``evaluate_benchmark``; a background thread
        streams the composited live view and collects clip frames.

        Returns ``{task, instruction, success_rate, n_episodes, video,
        agent_summary, error?}``.
        """
        self._ensure_suite_loaded()
        task = self._task_label_to_name.get(task_label)
        if task is None:
            return {"error": f"Unknown task: {task_label!r}"}
        instruction = self._adapters[task].problem.language

        import cv2

        from examples.mujoco_gs.backgrounds import PanoramaBackground
        from examples.mujoco_gs.compositor import HybridCompositor

        try:
            from strands import Agent
            from strands_robots.benchmarks.libero import load_libero_suite  # noqa: F401 (already loaded)
            from strands_robots.simulation import Simulation
            from strands_robots.simulation.benchmark import get_benchmark
        except ImportError as e:  # pragma: no cover
            return {"error": f"missing deps: {e}"}

        sim = Simulation(tool_name="libero_sim", mesh=False)
        compositor = HybridCompositor(sim, background=PanoramaBackground())
        frames: List[np.ndarray] = []
        stop = threading.Event()

        def live_loop():
            # Render the scene through the compositor at ~15 fps into the live
            # buffer (+ collect clip frames). Reads mj_data while the eval
            # steps it on another thread — safe (read-during-step) and the
            # LIBERO viz_option keeps the arm clean.
            dt = 1.0 / 15.0
            while not stop.is_set():
                t0 = time.time()
                try:
                    rgb = compositor.render(camera_name="image", width=self.render_width, height=self.render_height).rgb
                    frames.append(rgb)
                    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        with self._lock:
                            self.latest_jpeg = buf.tobytes()
                            self.latest_rgb = rgb
                    if on_progress:
                        on_progress(len(frames))
                except Exception as e:  # pragma: no cover
                    logger.debug("live render skipped: %s", e)
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        self.running = True
        live_thread = None
        try:
            sim.create_world()
            sim.add_robot("robot", data_config="panda")

            # Pre-warm: generate + load the LIBERO scene and register its
            # cameras/Panda BEFORE eval, so the live render + policy see them.
            spec = get_benchmark(task)
            if spec.scene_path is None and getattr(spec, "_auto_generate_scene", False):
                generated = spec._generate_scene_from_bddl()
                if generated:
                    spec.scene_path = generated
            if spec.scene_path:
                sim.load_scene(spec.scene_path)
                if hasattr(spec, "prewarm"):
                    spec.prewarm(sim)
                if "robot" not in sim.list_robots():
                    sim.add_robot("robot", data_config="panda")
            compositor.clear_caches()  # new model → drop stale renderer/cam caches

            # Start the live view now that cameras exist.
            live_thread = threading.Thread(target=live_loop, daemon=True)
            live_thread.start()

            # --- agentic invocation: let the Agent pick evaluate_benchmark. ---
            agent_kwargs = {"tools": [sim]}
            if self.model_id:
                agent_kwargs["model"] = self.model_id
            agent = Agent(**agent_kwargs)
            policy_phrase = (
                f"using the GR00T policy with `policy_provider='groot'` and "
                f"`policy_config={{'host': '{self.host}', 'port': {self.port}, "
                f"'data_config': '{self.data_config}', 'groot_version': '{self.groot_version}'}}`"
            )
            prompt = (
                f"Make exactly one tool call: invoke the `libero_sim` tool with "
                f"`action='evaluate_benchmark'`, `benchmark_name='{task}'`, "
                f"`n_episodes={n_episodes}`, `seed={seed}`, {policy_phrase}. "
                f"Do not call any other action — the world, robot, and scene are "
                f"already set up. When the call returns, parse the `success_rate` "
                f"field from the JSON payload and report it as a percentage of "
                f"the {n_episodes} episodes, plus a one-sentence description of "
                f"the task: '{instruction}'."
            )
            agent_result = agent(prompt)
            agent_summary = _extract_text(agent_result)
            success_rate = _extract_success_rate(agent)
        except Exception as e:  # pragma: no cover
            logger.exception("Agentic GR00T run failed.")
            return {"error": f"{type(e).__name__}: {e}", "task": task, "instruction": instruction}
        finally:
            self.running = False
            stop.set()
            if live_thread is not None:
                live_thread.join(timeout=5)

        video = _encode_mp4(frames) if frames else None
        try:
            compositor.close()
            sim.destroy()
        except Exception:  # pragma: no cover
            pass

        return {
            "task": task,
            "instruction": instruction,
            "success_rate": success_rate,
            "n_episodes": n_episodes,
            "video": video,
            "agent_summary": agent_summary,
            "n_frames": len(frames),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _extract_text(result) -> str:
    if result is None:
        return ""
    msg = getattr(result, "message", None)
    if msg is not None:
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            if texts:
                return "\n".join(t for t in texts if t)
    return str(result)


def _extract_success_rate(agent) -> Optional[float]:
    """Find the evaluate_benchmark toolResult's ``success_rate`` in the
    agent's message history."""
    try:
        for msg in agent.messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or "toolResult" not in block:
                    continue
                for c in block["toolResult"].get("content", []):
                    if isinstance(c, dict) and "json" in c:
                        payload = c["json"]
                        if isinstance(payload, dict) and "success_rate" in payload:
                            return float(payload["success_rate"])
    except Exception:  # pragma: no cover
        pass
    return None


def _encode_mp4(frames: List[np.ndarray], fps: int = 20) -> Optional[str]:
    import os
    import tempfile

    try:
        import imageio
    except ImportError:  # pragma: no cover
        return None
    if not frames:
        return None
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="groot_libero_")
    os.close(fd)
    imageio.mimsave(path, frames, fps=int(fps), codec="libx264", quality=7, macro_block_size=8)
    return path
