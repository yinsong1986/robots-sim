# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LeRobot data collection for the SO-101 cuRobo demo (issue #67 T7).

Streams a planned :class:`~examples.so101_curobo.planner.JointTrajectory` into
the simulation (``send_action`` per waypoint) while recording each control step
as a LeRobot frame (state + action + camera images) via the tested
``strands_robots.dataset_recorder.DatasetRecorder``. Writes a v3.0 LeRobot
dataset (parquet + per-camera video), with a programmatic success check and a
resumable multi-episode loop over (optionally) randomized scenes.

Validated recipe (matches ``Simulation.stop_recording``):
    create() -> add_frame()* -> save_episode() -> finalize()
then reload locally with ``load_lerobot_episode`` (no Hub round-trip).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("so101_curobo.collector")

LEROBOT_INSTALL_HINT = (
    "LeRobot dataset writing requires the lerobot extra: "
    "pip install 'strands-robots[lerobot]'  (or `pip install lerobot`)."
)


def lerobot_available() -> bool:
    try:
        from strands_robots.dataset_recorder import has_lerobot_dataset

        return bool(has_lerobot_dataset())
    except Exception:  # noqa: BLE001
        return False


@dataclass
class EpisodeResult:
    success: bool
    cube_moved: bool
    placed: bool
    frames: int
    displacement: float
    place_distance: float
    phases: int


def _object_position(sim, name: str) -> Optional[List[float]]:
    """Best-effort world position of object ``name`` (MuJoCo via mj_data; else None)."""
    try:
        import mujoco

        m = getattr(sim, "mj_model", None)
        d = getattr(sim, "mj_data", None)
        if m is not None and d is not None:
            for b in range(m.nbody):
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if name == bn or bn.endswith(f"/{name}") or (name in bn and "cube" in name):
                    return [float(x) for x in d.xpos[b]]
    except Exception:  # noqa: BLE001
        logger.debug("mj_data object read failed for %s", name, exc_info=True)
    return None


class LeRobotDataCollector:
    """Execute trajectories and record LeRobot episodes (issue #67 T7)."""

    def __init__(
        self,
        sim,
        scene_info,
        repo_id: str = "local/so101_curobo_pickplace",
        fps: int = 20,
        root: Optional[str] = None,
        cameras: Optional[Sequence[str]] = None,
        place_radius: float = 0.10,
        move_threshold: float = 0.03,
        record_images: bool = True,
    ):
        self.sim = sim
        self.scene = scene_info
        self.repo_id = repo_id
        self.fps = fps
        self.root = root
        self.cameras = list(cameras) if cameras else list(scene_info.cameras)
        self.place_radius = place_radius
        self.move_threshold = move_threshold
        # When False, record state+action only (no camera frames) -> no GL/EGL
        # needed. Used by the CI smoke path on CPU-only boxes.
        self.record_images = record_images

    # --- recording lifecycle ------------------------------------------------

    @staticmethod
    def available() -> bool:
        return lerobot_available()

    def _new_recorder(self, task: str):
        if not self.available():
            raise RuntimeError(LEROBOT_INSTALL_HINT)
        import os
        import shutil

        from strands_robots.dataset_recorder import DatasetRecorder

        # LeRobotDataset.create() requires a non-existent root (mkdir exist_ok=False).
        # Clear a prior dataset dir (or an empty placeholder, e.g. from mkdtemp) so
        # "regenerate" works; only touch dirs that look like a dataset or are empty.
        if self.root and os.path.isdir(self.root):
            if os.path.isdir(os.path.join(self.root, "meta")) or not os.listdir(self.root):
                shutil.rmtree(self.root, ignore_errors=True)

        # Per-camera (height, width) so the schema matches what render returns.
        cam_dims: Dict[str, tuple] = {}
        if self.record_images:
            obs = self.sim.get_observation(self.scene.robot_name)
            for cam in self.cameras:
                img = obs.get(cam)
                if img is not None and hasattr(img, "shape") and len(img.shape) >= 2:
                    cam_dims[cam] = (int(img.shape[0]), int(img.shape[1]))
        return DatasetRecorder.create(
            repo_id=self.repo_id,
            fps=self.fps,
            robot_type=self.scene.robot_config,
            joint_names=self.scene.joint_names,
            camera_keys=list(cam_dims.keys()),
            camera_dims=cam_dims,
            task=task,
            root=self.root,
        )

    def _execute_and_record(self, trajectory, task: str, recorder, n_substeps: int) -> int:
        """Stream the trajectory; record one frame per waypoint. Returns frame count."""
        robot = self.scene.robot_name
        frames = 0
        for wp in trajectory.waypoints:
            self.sim.send_action(wp, robot_name=robot, n_substeps=n_substeps)
            obs = self.sim.get_observation(robot, skip_images=not self.record_images)
            recorder.add_frame(observation=obs, action=wp, task=task)
            frames += 1
        return frames

    def _assess(self, cube_start: Optional[List[float]]) -> EpisodeResult:
        cube_now = _object_position(self.sim, self.scene.cube_name)
        disp = place_d = 0.0
        moved = placed = False
        if cube_now is not None:
            if cube_start is not None:
                disp = math.dist(cube_now, cube_start)
                moved = disp > self.move_threshold
            place_d = math.dist(cube_now[:2], self.scene.place_position[:2])
            placed = place_d < self.place_radius
        return EpisodeResult(
            success=placed,
            cube_moved=moved,
            placed=placed,
            frames=0,
            displacement=disp,
            place_distance=place_d,
            phases=0,
        )

    def record_episode(self, trajectory, task: str, n_substeps: int = 5, recorder=None) -> EpisodeResult:
        """Record one episode. If ``recorder`` is None, creates+finalizes a 1-episode dataset."""
        own = recorder is None
        recorder = recorder or self._new_recorder(task)
        cube_start = _object_position(self.sim, self.scene.cube_name)
        try:
            frames = self._execute_and_record(trajectory, task, recorder, n_substeps)
            recorder.save_episode()
            result = self._assess(cube_start)
            result.frames = frames
            result.phases = len(set(trajectory.phases))
            logger.info(
                "episode recorded: frames=%d success=%s moved=%s disp=%.3f place_d=%.3f",
                frames,
                result.success,
                result.cube_moved,
                result.displacement,
                result.place_distance,
            )
            return result
        finally:
            if own:
                recorder.finalize()

    def record_dataset(
        self,
        planner,
        n_episodes: int = 5,
        task: str = "pick up the red cube and place it in the bin",
        n_substeps: int = 5,
        randomize: bool = True,
        seed: Optional[int] = None,
        rebuild_scene=None,
        on_episode=None,
    ) -> Dict[str, Any]:
        """Record ``n_episodes`` into one dataset (append + single finalize).

        ``rebuild_scene(seed)`` (optional) re-randomizes the world between
        episodes; if absent we jitter via ``sim.randomize`` when ``randomize``.
        ``on_episode(i, result)`` is an optional progress callback (UI/agent).
        """
        if not self.available():
            return {"status": "error", "message": LEROBOT_INSTALL_HINT}

        import random as _random

        rng = _random.Random(seed)
        recorder = self._new_recorder(task)
        results: List[EpisodeResult] = []
        try:
            for i in range(n_episodes):
                if randomize:
                    try:
                        self.sim.randomize(
                            randomize_colors=True,
                            randomize_lighting=True,
                            randomize_positions=False,
                            seed=rng.randint(0, 10_000),
                        )
                    except Exception:  # noqa: BLE001 - randomize is best-effort
                        logger.debug("randomize failed (non-fatal)", exc_info=True)
                start_q = [float(self.sim.get_observation(self.scene.robot_name)[j]) for j in self.scene.joint_names]
                traj = planner.plan_pick_place(
                    joint_names=self.scene.joint_names,
                    start_q=start_q,
                    gripper_joint=self.scene.gripper_joint,
                    cube_xy=self.scene.cube_position[:2],
                    place_xy=self.scene.place_position[:2],
                )
                res = self.record_episode(traj, task=task, n_substeps=n_substeps, recorder=recorder)
                results.append(res)
                if on_episode:
                    try:
                        on_episode(i, res)
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            recorder.finalize()

        n_ok = sum(1 for r in results if r.success)
        return {
            "status": "success",
            "repo_id": self.repo_id,
            "root": getattr(recorder, "root", self.root),
            "episodes": len(results),
            "successes": n_ok,
            "success_rate": (n_ok / len(results)) if results else 0.0,
            "total_frames": sum(r.frames for r in results),
            "planner": getattr(planner, "name", "?"),
        }

    def load_back(self, episode: int = 0):
        """Reload a recorded episode locally (no Hub). Returns (dataset, start, length)."""
        from strands_robots.dataset_recorder import load_lerobot_episode

        return load_lerobot_episode(repo_id=self.repo_id, episode=episode, root=self.root)
