# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Motion planning for the SO-101 pick-and-place demo (issue #67).

Two planners, one interface — both emit a backend-agnostic
:class:`JointTrajectory` (a list of joint-target waypoints) that the collector
streams via ``send_action``:

* :class:`CuroboMotionPlanner` — the real, collision-aware path (issue #67
  T4/T5). It **lazy-imports cuRobo**; cuRobo isn't installed on most boxes (and
  the latest release wants driver >= 580 vs our 550), so every entry point
  raises a clear, actionable install hint rather than a cryptic ImportError.
* :class:`ScriptedPlanner` — the dependency-free fallback (#67 graceful
  degradation). It interpolates a joint-space pick -> lift -> place -> release
  demonstration so the planning + collection loop is fully demonstrable on the
  MuJoCo backend today, without cuRobo/IK.

:func:`make_planner` returns cuRobo when available, else the scripted planner.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("so101_curobo.planner")

CUROBO_INSTALL_HINT = (
    "cuRobo is not installed. VALIDATED install on driver 550 / CUDA 12.4 / L4 "
    "(issue #67 T3 -- the docs' driver>=580 is conservative; CUDA 12.x kernels "
    "run on a 12.4 driver):\n"
    "  export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9\n"
    "  python -m venv --system-site-packages .venv && source .venv/bin/activate\n"
    "  pip install -U pip setuptools wheel ninja\n"
    "  git clone --depth 1 https://github.com/NVlabs/curobo && cd curobo\n"
    "  sed -i '/Topic :: Scientific.Engineering :: Robotics/d' pyproject.toml\n"
    "  pip install -e . --no-build-isolation\n"
    "  pip install 'cuda-core[cu12]'   # runtime kernel backend (required by the\n"
    "                                  #  refactored cuRobo; no precompile needed)\n"
    "Until installed, the demo uses the ScriptedPlanner fallback."
)


def curobo_available() -> bool:
    """True if the ``curobo`` package is importable."""
    try:
        return importlib.util.find_spec("curobo") is not None
    except Exception:  # noqa: BLE001
        return False


CUROBO_AVAILABLE = curobo_available()


@dataclass
class JointTrajectory:
    """A planned joint-space trajectory: dense ``{joint_name: target}`` waypoints."""

    joint_names: List[str]
    waypoints: List[Dict[str, float]]
    phases: List[str]
    planner: str = "scripted"
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.waypoints)

    def phase_of(self, i: int) -> str:
        return self.phases[i] if 0 <= i < len(self.phases) else ""


def _merge(base: Dict[str, float], updates: Dict[str, float]) -> Dict[str, float]:
    out = dict(base)
    for k, v in updates.items():
        if k is not None:
            out[k] = v
    return out


class ScriptedPlanner:
    """Dependency-free joint-space pick-and-place demonstration.

    No IK: it sweeps the arm through scripted keyframes (aim -> reach -> grasp
    -> lift -> traverse -> lower -> release) using heuristic joint deltas from
    the home pose, then linearly interpolates. Produces coherent, recordable
    motion; it is **not** guaranteed to achieve a physical grasp (that's what
    the cuRobo path is for). Success is judged empirically by the collector.
    """

    name = "scripted"

    def __init__(self, gripper_open: float = 0.0, gripper_close: float = 0.9, steps_per_phase: int = 8):
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.steps_per_phase = steps_per_phase

    def plan_pick_place(
        self,
        joint_names: Sequence[str],
        start_q: Sequence[float],
        gripper_joint: Optional[str] = None,
        cube_xy: Optional[Sequence[float]] = None,
        place_xy: Optional[Sequence[float]] = None,
        steps_per_phase: Optional[int] = None,
    ) -> JointTrajectory:
        jn = list(joint_names)
        n = len(jn)
        steps = steps_per_phase or self.steps_per_phase
        home = {j: float(q) for j, q in zip(jn, start_q)}

        base = jn[0] if n > 0 else None
        lift = jn[1] if n > 1 else None
        elbow = jn[2] if n > 2 else None
        wristf = jn[3] if n > 3 else None
        grip = gripper_joint or (jn[5] if n > 5 else (jn[-1] if jn else None))

        OPEN, CLOSE = self.gripper_open, self.gripper_close

        def aim(xy) -> float:
            if not (base and xy):
                return home.get(base, 0.0) if base else 0.0
            return float(math.atan2(xy[1], xy[0]))

        reach = dict(home)
        if base:
            reach[base] = aim(cube_xy)
        if lift:
            reach[lift] = home[lift] - 0.8
        if elbow:
            reach[elbow] = home[elbow] + 1.0
        if wristf:
            reach[wristf] = home[wristf] + 0.5
        if grip:
            reach[grip] = OPEN

        liftq = _merge(reach, {grip: CLOSE})
        if lift:
            liftq[lift] = home[lift] - 0.2
        if elbow:
            liftq[elbow] = home[elbow] + 0.4

        traverse = _merge(liftq, {base: aim(place_xy)})
        lower = dict(traverse)
        if lift:
            lower[lift] = home[lift] - 0.6
        if elbow:
            lower[elbow] = home[elbow] + 0.8

        keyframes = [
            ("home", _merge(home, {grip: OPEN})),
            ("aim_pick", _merge(home, {base: aim(cube_xy), grip: OPEN})),
            ("reach", reach),
            ("grasp", _merge(reach, {grip: CLOSE})),
            ("lift", liftq),
            ("traverse", traverse),
            ("lower", lower),
            ("release", _merge(lower, {grip: OPEN})),
            ("retreat", _merge(home, {base: aim(place_xy), grip: OPEN})),
        ]

        waypoints: List[Dict[str, float]] = []
        phases: List[str] = []
        for (_, q0), (p1, q1) in zip(keyframes[:-1], keyframes[1:]):
            for s in range(1, steps + 1):
                t = s / steps
                waypoints.append({j: (1.0 - t) * q0.get(j, home[j]) + t * q1.get(j, home[j]) for j in jn})
                phases.append(p1)

        return JointTrajectory(
            joint_names=jn,
            waypoints=waypoints,
            phases=phases,
            planner=self.name,
            meta={"gripper_joint": grip, "cube_xy": list(cube_xy or []), "place_xy": list(place_xy or [])},
        )


class CuroboMotionPlanner:
    """cuRobo collision-aware motion planner for the SO-101 (issue #67 T4/T5).

    Builds the SO-101 kinematic model from a URDF with cuRobo's new
    ``RobotBuilder`` (auto-derives the 5-DOF arm chain to the tool frame,
    excluding the gripper jaw), then plans EEF pose-to-pose trajectories with
    ``MotionPlanner`` and chains them into reach -> grasp -> lift -> place ->
    release. The arm is collision-aware; the gripper is scripted. Validated on
    driver 550 / CUDA 12.4 / L4 (see CUROBO_INSTALL_HINT).

    Lazy: imports cuRobo only in :meth:`_ensure`. Needs an SO-101 URDF
    (``urdf_path`` or env ``SO101_URDF``, plus meshes via ``asset_path`` /
    ``SO101_ASSET``); otherwise raises an actionable error so callers fall back
    to :class:`ScriptedPlanner`.
    """

    name = "curobo"

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        asset_path: str = "",
        tool_frame: str = "gripper_frame_link",
        self_collision: bool = False,
        device: str = "cuda",
        grasp_quaternion: Optional[Sequence[float]] = None,
        gripper_open: float = 0.0,
        gripper_close: float = 0.9,
        **_ignored,
    ):
        import os

        self.urdf_path = urdf_path or os.environ.get("SO101_URDF")
        self.asset_path = asset_path or os.environ.get("SO101_ASSET", "")
        self.tool_frame = tool_frame
        self.self_collision = self_collision
        self.device = device
        self.grasp_quaternion = list(grasp_quaternion) if grasp_quaternion else None
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self._planner = None
        self._arm_joint_names: List[str] = []
        self._default_quat: Optional[List[float]] = None

    @staticmethod
    def available() -> bool:
        return curobo_available()

    def _ensure(self):
        if not self.available():
            raise RuntimeError(CUROBO_INSTALL_HINT)
        if self._planner is not None:
            return
        import os
        import tempfile

        import yaml

        if not self.urdf_path or not os.path.exists(self.urdf_path):
            raise RuntimeError(
                "CuroboMotionPlanner needs an SO-101 URDF. Pass urdf_path=... or set "
                "SO101_URDF (+ SO101_ASSET for meshes). See README (#67 T2/T4)."
            )
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.robot_builder import RobotBuilder
        from curobo.types import JointState

        builder = RobotBuilder(urdf_path=self.urdf_path, asset_path=self.asset_path, tool_frames=[self.tool_frame])
        tmp_yml = os.path.join(tempfile.mkdtemp(prefix="curobo_so101_"), "so101_curobo.yml")
        builder.save(builder.build(), tmp_yml, include_cspace=True)
        robot = yaml.safe_load(open(tmp_yml))
        self._planner = MotionPlanner(
            MotionPlannerCfg.create(robot=robot, self_collision_check=self.self_collision, use_cuda_graph=False)
        )
        self._arm_joint_names = list(self._planner.joint_names)
        # Capture the home EE orientation as the default (guaranteed-achievable)
        # goal orientation when no grasp_quaternion is supplied. A true top-down
        # grasp orientation can be passed via grasp_quaternion once calibrated.
        q0 = self._planner.default_joint_state.position
        q0 = q0.unsqueeze(0) if q0.dim() == 1 else q0
        st = self._planner.compute_kinematics(JointState.from_position(q0, joint_names=self._arm_joint_names))
        tp = st.tool_poses
        if isinstance(tp, dict):
            tp = tp.get(self.tool_frame, list(tp.values())[0])
        self._default_quat = [float(x) for x in tp.quaternion.reshape(-1, 4)[0].detach().cpu().tolist()]
        logger.info("cuRobo SO-101 planner ready: arm joints=%s", self._arm_joint_names)

    def _plan_segment(self, start_arm_q, goal_xyz, goal_quat) -> List[List[float]]:
        import torch
        from curobo.types import GoalToolPose, JointState

        jn = self._arm_joint_names
        start = JointState.from_position(
            torch.tensor([start_arm_q], dtype=torch.float32, device=self.device), joint_names=jn
        )
        pos = torch.tensor(goal_xyz, dtype=torch.float32, device=self.device).reshape(1, 1, 1, 1, 3)
        quat = torch.tensor(goal_quat, dtype=torch.float32, device=self.device).reshape(1, 1, 1, 1, 4)
        goal = GoalToolPose(tool_frames=[self.tool_frame], position=pos, quaternion=quat)
        res = self._planner.plan_pose(goal, start)
        if res is None or not bool(res.success.any()):
            raise RuntimeError(f"cuRobo could not reach {[round(float(x), 3) for x in goal_xyz]}")
        return res.get_interpolated_plan().position.reshape(-1, len(jn)).detach().cpu().tolist()

    def plan_pick_place(
        self,
        joint_names: Sequence[str],
        start_q: Sequence[float],
        gripper_joint: Optional[str] = None,
        cube_xy: Optional[Sequence[float]] = None,
        place_xy: Optional[Sequence[float]] = None,
        table_z: float = 0.0,
        approach: float = 0.10,
        grasp_z: float = 0.03,
        hold_steps: int = 5,
        **_ignored,
    ) -> JointTrajectory:
        """Collision-aware reach -> grasp -> lift -> place -> release.

        cuRobo plans the arm DOFs (mapped to ``joint_names`` by kinematic order);
        the trailing gripper joint is opened/closed per phase. Raises
        ``RuntimeError`` (caught by the caller -> ScriptedPlanner fallback) if a
        segment is unreachable.
        """
        self._ensure()
        jn = list(joint_names)
        n_arm = len(self._arm_joint_names)
        if len(jn) < n_arm:
            raise RuntimeError(f"scene has {len(jn)} joints; SO-101 arm needs {n_arm}.")
        arm_names = jn[:n_arm]
        grip = gripper_joint or (jn[-1] if len(jn) > n_arm else None)
        start_arm = [float(x) for x in start_q[:n_arm]]
        quat = self.grasp_quaternion or self._default_quat
        cx, cy = (list(cube_xy) if cube_xy else [0.2, 0.2])[:2]
        px, py = (list(place_xy) if place_xy else [-0.2, 0.2])[:2]
        OPEN, CLOSE = self.gripper_open, self.gripper_close
        home = {j: float(q) for j, q in zip(jn, start_q)}

        waypoints: List[Dict[str, float]] = []
        phases: List[str] = []
        cur_arm = start_arm

        def emit(arm_q, gripval, phase):
            wp = dict(home)
            for i, a in enumerate(arm_names):
                wp[a] = float(arm_q[i])
            if grip is not None:
                wp[grip] = gripval
            waypoints.append(wp)
            phases.append(phase)

        # (phase, goal_xyz | None for an in-place gripper event, gripper value)
        segments = [
            ("reach", [cx, cy, table_z + approach], OPEN),
            ("grasp", [cx, cy, table_z + grasp_z], OPEN),
            ("close", None, CLOSE),
            ("lift", [cx, cy, table_z + approach], CLOSE),
            ("place", [px, py, table_z + approach], CLOSE),
            ("release", None, OPEN),
        ]
        for phase, xyz, gripval in segments:
            if xyz is None:
                for _ in range(hold_steps):
                    emit(cur_arm, gripval, phase)
                continue
            seg = self._plan_segment(cur_arm, xyz, quat)
            for arm_q in seg:
                emit(arm_q, gripval, phase)
            cur_arm = seg[-1]

        return JointTrajectory(
            joint_names=jn,
            waypoints=waypoints,
            phases=phases,
            planner="curobo",
            meta={
                "arm_joints": arm_names,
                "gripper_joint": grip,
                "tool_frame": self.tool_frame,
                "cube_xy": [cx, cy],
                "place_xy": [px, py],
            },
        )


_SCRIPTED_KEYS = ("gripper_open", "gripper_close", "steps_per_phase")
_CUROBO_KEYS = (
    "urdf_path",
    "asset_path",
    "tool_frame",
    "self_collision",
    "device",
    "grasp_quaternion",
    "gripper_open",
    "gripper_close",
)


def _curobo_usable(kwargs: dict) -> bool:
    """cuRobo is usable only if installed AND an SO-101 URDF is resolvable."""
    import os

    if not CUROBO_AVAILABLE:
        return False
    return bool(kwargs.get("urdf_path") or os.environ.get("SO101_URDF"))


def make_planner(prefer: str = "auto", robot_cfg: str = "so101", **kwargs):
    """Return the best available planner.

    ``prefer``: ``"auto"`` (cuRobo if installed *and* a URDF is resolvable, else
    scripted), ``"curobo"`` (force cuRobo — raises later if unusable), or
    ``"scripted"``.
    """
    prefer = (prefer or "auto").lower()

    def _scripted():
        return ScriptedPlanner(**{k: v for k, v in kwargs.items() if k in _SCRIPTED_KEYS})

    if prefer == "scripted":
        return _scripted()
    if prefer == "curobo":
        if not CUROBO_AVAILABLE:
            logger.warning(
                "cuRobo requested but not installed; using ScriptedPlanner. %s", CUROBO_INSTALL_HINT.splitlines()[0]
            )
            return _scripted()
        return CuroboMotionPlanner(**{k: v for k, v in kwargs.items() if k in _CUROBO_KEYS})
    # auto
    if _curobo_usable(kwargs):
        return CuroboMotionPlanner(**{k: v for k, v in kwargs.items() if k in _CUROBO_KEYS})
    return _scripted()
