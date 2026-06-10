# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orchestration for the SO-101 cuRobo synthetic-data demo (issue #67).

``SO101CuroboDemo`` ties the four pieces together — simulation backend
(:func:`make_sim`), scene (:func:`build_pick_place_scene`), motion planner
(:func:`make_planner`), and the LeRobot collector — behind a small API used by
both the Strands agent (:mod:`agent`) and the Gradio app (:mod:`app`):

    plan_and_execute(task) -> plan a trajectory, execute it, record one episode
    record_dataset(n)      -> generate N episodes into one LeRobot dataset
    render(camera)         -> an RGB frame for the UI
    describe()             -> a human-readable status (backend, planner, deps)

All simulation access is serialized with a lock so the Gradio worker threads
don't race on the sim.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from .collector import LeRobotDataCollector, lerobot_available
from .planner import CUROBO_AVAILABLE, make_planner
from .scene import build_pick_place_scene, make_sim

logger = logging.getLogger("so101_curobo.controller")


class _FallbackPlanner:
    """Wrap a primary planner; fall back to ScriptedPlanner if it raises.

    Keeps the demo robust when cuRobo is installed but a specific pick-place
    pose is infeasible for the 5-DOF SO-101 (a known calibration gap -- see
    README #67 T5): the arm still moves and an episode is still recorded.
    """

    def __init__(self, primary):
        self.primary = primary
        self.name = getattr(primary, "name", "planner")
        self._scripted = None

    def plan_pick_place(self, **kwargs):
        try:
            return self.primary.plan_pick_place(**kwargs)
        except Exception as exc:  # noqa: BLE001 - infeasible/unreachable -> fallback
            if self._scripted is None:
                from .planner import ScriptedPlanner

                self._scripted = ScriptedPlanner()
            logger.warning(
                "%s planning failed (%s); using scripted fallback.", getattr(self.primary, "name", "?"), str(exc)[:140]
            )
            self.name = f"scripted(fallback from {getattr(self.primary, 'name', '?')})"
            scripted_keys = ("joint_names", "start_q", "gripper_joint", "cube_xy", "place_xy", "steps_per_phase")
            return self._scripted.plan_pick_place(**{k: v for k, v in kwargs.items() if k in scripted_keys})


class SO101CuroboDemo:
    """Backend-agnostic SO-101 pick-and-place + synthetic-data controller."""

    def __init__(
        self,
        backend: str = "mujoco",
        repo_id: str = "local/so101_curobo_pickplace",
        root: Optional[str] = None,
        prefer_planner: str = "auto",
        fps: int = 20,
        camera_size: tuple = (320, 240),
        record_images: bool = True,
        planner_kwargs: Optional[dict] = None,
    ):
        self.backend = backend
        self.repo_id = repo_id
        self.root = root
        self.prefer_planner = prefer_planner
        self.fps = fps
        self.camera_size = camera_size
        self.record_images = record_images
        # Extra kwargs forwarded to make_planner (e.g. cuRobo urdf_path/asset_path).
        self.planner_kwargs = dict(planner_kwargs or {})

        self._lock = threading.RLock()
        self.sim = None
        self.scene = None
        self.planner = None
        self.collector: Optional[LeRobotDataCollector] = None
        self.current_camera = "front"
        self._built = False
        self._backend_note = ""

    # --- lifecycle ----------------------------------------------------------

    def build(self) -> "SO101CuroboDemo":
        """Create the sim + scene + planner + collector. Falls back MuJoCo<-Isaac."""
        with self._lock:
            backend = self.backend
            try:
                self.sim = make_sim(backend=backend)
            except Exception as exc:  # noqa: BLE001 - Isaac runtime missing, etc.
                if backend != "mujoco":
                    logger.warning("Backend %r unavailable (%s); falling back to MuJoCo.", backend, exc)
                    self._backend_note = f"{backend} unavailable ({exc}); using MuJoCo."
                    backend = "mujoco"
                    self.sim = make_sim(backend="mujoco")
                else:
                    raise
            self.backend = backend
            self.scene = build_pick_place_scene(self.sim, camera_size=self.camera_size, backend=backend)
            self.planner = _FallbackPlanner(
                make_planner(prefer=self.prefer_planner, robot_cfg="so101", **self.planner_kwargs)
            )
            self.collector = LeRobotDataCollector(
                self.sim,
                self.scene,
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.root,
                cameras=self.scene.cameras,
                record_images=self.record_images,
            )
            if self.scene.cameras:
                self.current_camera = self.scene.cameras[0]
            self._built = True
            logger.info("Demo built: %s | planner=%s", self.scene.pretty(), self.planner.name)
            return self

    def _require(self):
        if not self._built:
            raise RuntimeError("Demo not built yet — call build() first.")

    # --- actions ------------------------------------------------------------

    def _plan(self):
        start_q = [
            float(self.sim.get_observation(self.scene.robot_name, skip_images=True)[j]) for j in self.scene.joint_names
        ]
        return self.planner.plan_pick_place(
            joint_names=self.scene.joint_names,
            start_q=start_q,
            gripper_joint=self.scene.gripper_joint,
            cube_xy=self.scene.cube_position[:2],
            place_xy=self.scene.place_position[:2],
        )

    def plan_and_execute(self, task: str = "pick up the red cube and place it in the bin", n_substeps: int = 5) -> str:
        """Plan a pick-and-place, execute it, and record one LeRobot episode."""
        with self._lock:
            self._require()
            try:
                traj = self._plan()
            except RuntimeError as exc:  # cuRobo not wired -> actionable message
                return f"Planner unavailable: {exc}"
            if not self.collector.available():
                # Still execute (move the arm) even if we can't record a dataset.
                robot = self.scene.robot_name
                for wp in traj.waypoints:
                    self.sim.send_action(wp, robot_name=robot, n_substeps=n_substeps)
                return (
                    f"Executed a {self.planner.name} pick-and-place ({len(traj)} waypoints). "
                    f"Dataset NOT recorded: {LeRobotDataCollector.__module__}: lerobot missing."
                )
            res = self.collector.record_episode(traj, task=task, n_substeps=n_substeps)
            return (
                f"Planned ({self.planner.name}) + executed + recorded 1 episode: "
                f"{res.frames} frames, success={res.success} (cube moved={res.cube_moved}, "
                f"displacement={res.displacement:.3f} m). Dataset: {self.repo_id}."
            )

    def record_dataset(
        self,
        n_episodes: int = 5,
        task: str = "pick up the red cube and place it in the bin",
        randomize: bool = True,
        n_substeps: int = 5,
        on_episode=None,
    ) -> Dict[str, Any]:
        with self._lock:
            self._require()
            return self.collector.record_dataset(
                self.planner,
                n_episodes=n_episodes,
                task=task,
                randomize=randomize,
                n_substeps=n_substeps,
                on_episode=on_episode,
            )

    def render(self, camera: Optional[str] = None) -> Optional[np.ndarray]:
        """Return an RGB frame from ``camera`` (defaults to current)."""
        with self._lock:
            self._require()
            cam = camera or self.current_camera
            try:
                obs = self.sim.get_observation(self.scene.robot_name)
                img = obs.get(cam)
                if img is not None and hasattr(img, "shape"):
                    return np.asarray(img)[:, :, :3]
            except Exception:  # noqa: BLE001 - rendering needs GL/EGL
                logger.debug("render failed for %s", cam, exc_info=True)
            return None

    def set_camera(self, camera: str) -> str:
        with self._lock:
            if self.scene and camera in self.scene.cameras:
                self.current_camera = camera
                return f"Camera set to {camera}."
            return f"Unknown camera {camera!r}. Options: {self.scene.cameras if self.scene else []}."

    def describe(self) -> str:
        bits: List[str] = []
        if self.scene:
            bits.append(self.scene.pretty())
        bits.append(f"planner={self.planner.name if self.planner else '?'}")
        bits.append(f"cuRobo={'available' if CUROBO_AVAILABLE else 'NOT installed (scripted fallback)'}")
        bits.append(f"lerobot={'available' if lerobot_available() else 'NOT installed'}")
        if self._backend_note:
            bits.append(self._backend_note)
        return " | ".join(bits)

    def close(self):
        with self._lock:
            try:
                if self.sim is not None and hasattr(self.sim, "destroy"):
                    self.sim.destroy()
            except Exception:  # noqa: BLE001
                pass
