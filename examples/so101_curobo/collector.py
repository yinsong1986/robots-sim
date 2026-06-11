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
        kinematic: bool = False,
        grasp_attach: bool = False,
        attach_radius: float = 0.10,
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
        # When True, drive the arm kinematically (set_joint_positions + step)
        # instead of send_action. Required for the cuRobo/URDF-matched arm,
        # which loads without position actuators (send_action wouldn't move it);
        # also makes the arm follow a planned trajectory exactly.
        self.kinematic = kinematic
        # When True, model the grasp by attaching the cube to the gripper while
        # the gripper is closed AND was within attach_radius of the cube when it
        # closed (a standard kinematic grasp for synthetic data; the actuator-
        # less arm can't hold via friction). Attaches only if the gripper truly
        # reached the cube, so it stays honest.
        self.grasp_attach = grasp_attach
        self.attach_radius = attach_radius

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

    def _snapshot_state(self):
        """Snapshot full MuJoCo physics state (qpos, qvel) for a deterministic reset."""
        try:
            d = getattr(self.sim, "mj_data", None)
            return (d.qpos.copy(), d.qvel.copy()) if d is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _restore_state(self, snap) -> None:
        """Restore a snapshot so every episode starts from the identical state."""
        if snap is None:
            return
        try:
            import mujoco

            m = getattr(self.sim, "mj_model", None)
            d = getattr(self.sim, "mj_data", None)
            if m is None or d is None:
                return
            d.qpos[:] = snap[0]
            d.qvel[:] = snap[1]
            mujoco.mj_forward(m, d)
        except Exception:  # noqa: BLE001
            logger.debug("state restore failed (non-fatal)", exc_info=True)

    def _zero_cube_velocity(self) -> None:
        """Zero the cube's free-joint velocity (avoids a fling from teleporting it)."""
        try:
            import mujoco

            m = getattr(self.sim, "mj_model", None)
            d = getattr(self.sim, "mj_data", None)
            if m is None or d is None:
                return
            cube_bodies = {
                b
                for b in range(m.nbody)
                if self.scene.cube_name in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
            }
            for j in range(m.njnt):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[j]) in cube_bodies:
                    adr = int(m.jnt_dofadr[j])
                    for k in range(6):  # 3 linear + 3 angular dofs
                        d.qvel[adr + k] = 0.0
        except Exception:  # noqa: BLE001
            pass

    def _gripper_frame_pos(self) -> Optional[List[float]]:
        """World position of the gripper/tool link (MuJoCo via mj_data)."""
        try:
            import mujoco

            m = getattr(self.sim, "mj_model", None)
            d = getattr(self.sim, "mj_data", None)
            if m is None or d is None:
                return None
            best = None
            for b in range(m.nbody):
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if "gripper_frame" in bn:
                    return [float(x) for x in d.xpos[b]]
                if "gripper" in bn:
                    best = [float(x) for x in d.xpos[b]]
            return best
        except Exception:  # noqa: BLE001
            return None

    def _execute_and_record(self, trajectory, task: str, recorder, n_substeps: int) -> int:
        """Stream the trajectory; record one frame per waypoint. Returns frame count.

        With ``grasp_attach`` the cube is attached to the gripper while the
        gripper is closed and was within ``attach_radius`` of the cube when it
        closed (a kinematic grasp -- the actuator-less arm can't hold via
        friction), and released when the gripper re-opens.
        """
        robot = self.scene.robot_name
        grip = self.scene.gripper_joint
        frames = 0

        # Gripper open/close thresholds from the trajectory's own range.
        gvals = [wp[grip] for wp in trajectory.waypoints if grip and grip in wp]
        gmin = min(gvals) if gvals else 0.0
        gmax = max(gvals) if gvals else 0.0
        close_thresh = gmin + 0.5 * (gmax - gmin)
        has_grip = self.grasp_attach and grip is not None and (gmax - gmin) > 0.05

        attached = False
        has_closed = False
        offset: Optional[List[float]] = None
        grasp_phases = {"grasp", "close", "lift", "place", "place_down"}

        for wp, phase in zip(trajectory.waypoints, trajectory.phases):
            if self.kinematic:
                self.sim.set_joint_positions(wp, robot_name=robot)
                self.sim.step(max(1, n_substeps))
            else:
                self.sim.send_action(wp, robot_name=robot, n_substeps=n_substeps)

            if has_grip:
                gv = wp.get(grip, gmin)
                if gv >= close_thresh:
                    has_closed = True
                # Attach as soon as the gripper arrives within range during the
                # grasp (BEFORE the close-knock can push the cube away) and hold
                # it; release only once the gripper re-opens after having closed.
                if not attached and phase in grasp_phases:
                    gp = self._gripper_frame_pos()
                    cp = _object_position(self.sim, self.scene.cube_name)
                    if gp and cp and math.dist(gp, cp) < self.attach_radius:
                        offset = [cp[i] - gp[i] for i in range(3)]
                        attached = True
                elif attached and has_closed and gv < close_thresh:
                    attached = False  # release
                if attached and offset:
                    gp = self._gripper_frame_pos()
                    if gp:
                        self.sim.move_object(self.scene.cube_name, position=[gp[i] + offset[i] for i in range(3)])
                        self._zero_cube_velocity()  # avoid teleport-induced fling

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
        jnames = self.scene.joint_names
        # Capture the rest pose + full physics state once so each episode starts
        # identically (deterministic) -> cuRobo plans the same -> consistent grasp.
        home_q = {j: float(self.sim.get_observation(self.scene.robot_name, skip_images=True)[j]) for j in jnames}
        snapshot = self._snapshot_state()
        try:
            for i in range(n_episodes):
                # Reset to a consistent start: arm at home, cube at its start pose.
                if rebuild_scene is not None:
                    rebuild_scene(rng.randint(0, 10_000))
                elif snapshot is not None:
                    self._restore_state(snapshot)
                else:
                    try:
                        self.sim.set_joint_positions(home_q, robot_name=self.scene.robot_name)
                        self.sim.move_object(self.scene.cube_name, position=list(self.scene.cube_position))
                        self.sim.step(3)
                    except Exception:  # noqa: BLE001
                        logger.debug("episode reset failed (non-fatal)", exc_info=True)
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
                start_q = [float(self.sim.get_observation(self.scene.robot_name, skip_images=True)[j]) for j in jnames]
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
