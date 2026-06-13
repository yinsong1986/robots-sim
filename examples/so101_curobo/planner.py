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


def _approach_angle_deg(quat_wxyz: Sequence[float]) -> float:
    """Angle in degrees between the tool approach axis and straight down (-Z).

    The SO-101 ``gripper_frame_link`` local +x axis is the grasp approach
    direction (FK-verified: at the home config the home quaternion
    [0.707, 0, 0.707, 0] maps local +x to world -Z). For a unit quaternion
    (w, x, y, z) the world z-component of local +x is the rotation-matrix entry
    R[2, 0] = 2 (x z - w y); since local +x is a unit vector, its angle from the
    straight-down axis (0, 0, -1) is ``acos(-R[2, 0])``. Returns 0 for a perfect
    top-down approach, 90 for a horizontal (sideways) approach.
    """
    w, x, y, z = quat_wxyz
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    lx_z = 2.0 * (x * z - w * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, -lx_z))))


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
        base_sign: float = 1.0,
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
            # Base-pan angle to face a world XY target. ``base_sign`` flips the
            # joint sign for robots whose shoulder_pan rotates opposite to world
            # +Z (the SO-101 URDF on Isaac: commanding +pan swings the arm to
            # -Y, so a +Y target needs a negative pan). MuJoCo's model uses the
            # default +1 convention. See controller._plan.
            if not (base and xy):
                return home.get(base, 0.0) if base else 0.0
            return float(base_sign * math.atan2(xy[1], xy[0]))

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
        position_only: bool = True,
        top_down_grasp: bool = True,
        top_down_weight: float = 0.05,
        orientation_tolerance: float = 1.6,
        top_down_attempts: int = 6,
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
        # The SO-101 is a 5-DOF arm and cannot achieve arbitrary 6-DOF
        # orientations, so a fully-constrained pose goal is usually infeasible.
        # Default to POSITION-ONLY tracking (orientation is left free) so the
        # arm can actually reach tabletop pick/place positions.
        self.position_only = position_only
        # Top-down grasp (issue #67 T5 "further-tuning path"): a *soft* downward
        # orientation bias on the pick/approach/lift segments so the gripper
        # descends onto the cube near-vertically instead of grazing it sideways.
        # Strict vertical is infeasible for this 5-DOF arm and a high weight
        # makes IK finicky; calibration on the SO-101 URDF (driver 550 / L4)
        # found rpy weight ~0.05 + a relaxed orientation success tolerance
        # (~1.6 rad) + best-of-N attempts yields a consistently near-vertical
        # (<= ~13 deg from straight down) grasp at 8/8 cube targets, vs ~84 deg
        # (horizontal, high-variance) with free orientation. The place segments
        # stay position-only (the bin pose is not vertical-reachable, and a
        # top-down drop into a bin is not required). Unreachable oriented solves
        # fall back to position-only, so this never regresses below the
        # position-only path.
        self.top_down_grasp = top_down_grasp
        self.top_down_weight = float(top_down_weight)
        self.orientation_tolerance = float(orientation_tolerance)
        self.top_down_attempts = max(1, int(top_down_attempts))
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
        # When top-down grasping is on, relax the *success* orientation tolerance
        # so the solver accepts the most-vertical pose the 5-DOF arm can reach
        # (strict vertical is infeasible). Position-only segments set the
        # orientation weight to 0, so this relaxation does not affect them
        # (verified: free-orientation reach is identical at tol 0.05 vs 1.6).
        ori_tol = self.orientation_tolerance if self.top_down_grasp else 0.05
        self._planner = MotionPlanner(
            MotionPlannerCfg.create(
                robot=robot,
                self_collision_check=self.self_collision,
                use_cuda_graph=False,
                orientation_tolerance=ori_tol,
            )
        )
        self._arm_joint_names = list(self._planner.joint_names)
        # 5-DOF arm: default to POSITION-ONLY tracking so tabletop targets are
        # reachable (orientation free; a fully-constrained 6-DOF goal is
        # infeasible). This is also the per-segment fallback and the criteria
        # used for the place segments; the pick/approach/lift segments override
        # it with a soft top-down orientation bias in :meth:`_plan_segment`.
        if self.position_only or self.top_down_grasp:
            from curobo.types import ToolPoseCriteria

            self._planner.update_tool_pose_criteria(
                {self.tool_frame: ToolPoseCriteria.track_position(xyz=[1.0, 1.0, 1.0])}
            )
        # Capture the home EE orientation as the default goal orientation. On the
        # SO-101 the home quaternion is [0.707, 0, 0.707, 0] (wxyz) -> the tool's
        # local +x axis (the grasp approach axis) points along world -Z (straight
        # down), so it doubles as the top-down grasp orientation. Override via
        # grasp_quaternion if a different approach is wanted.
        q0 = self._planner.default_joint_state.position
        q0 = q0.unsqueeze(0) if q0.dim() == 1 else q0
        st = self._planner.compute_kinematics(JointState.from_position(q0, joint_names=self._arm_joint_names))
        tp = st.tool_poses
        if isinstance(tp, dict):
            tp = tp.get(self.tool_frame, list(tp.values())[0])
        self._default_quat = [float(x) for x in tp.quaternion.reshape(-1, 4)[0].detach().cpu().tolist()]
        logger.info("cuRobo SO-101 planner ready: arm joints=%s", self._arm_joint_names)

    def _plan_segment(self, start_arm_q, goal_xyz, goal_quat, orient: bool = False) -> List[List[float]]:
        import torch
        from curobo.types import GoalToolPose, JointState, ToolPoseCriteria

        jn = self._arm_joint_names
        start = JointState.from_position(
            torch.tensor([start_arm_q], dtype=torch.float32, device=self.device), joint_names=jn
        )
        pos = torch.tensor(goal_xyz, dtype=torch.float32, device=self.device).reshape(1, 1, 1, 1, 3)
        quat = torch.tensor(goal_quat, dtype=torch.float32, device=self.device).reshape(1, 1, 1, 1, 4)
        goal = GoalToolPose(tool_frames=[self.tool_frame], position=pos, quaternion=quat)

        def _solve():
            res = self._planner.plan_pose(goal, start)
            if res is None or not bool(res.success.any()):
                return None
            return res.get_interpolated_plan().position.reshape(-1, len(jn)).detach().cpu().tolist()

        # Top-down segments: bias orientation toward the (downward) goal quat with
        # a soft weight, and keep the most-vertical of N attempts (a single solve
        # varies ~10-86 deg from vertical due to cuRobo nondeterminism; best-of-N
        # reliably lands <= ~13 deg). Restore position-only afterwards so the
        # place segments and any fallback are unconstrained in orientation.
        if orient and self.top_down_grasp:
            w = self.top_down_weight
            self._planner.update_tool_pose_criteria(
                {self.tool_frame: ToolPoseCriteria.track_position_and_orientation(xyz=[1.0, 1.0, 1.0], rpy=[w, w, w])}
            )
            best_traj, best_err = None, None
            try:
                for _ in range(self.top_down_attempts):
                    traj = _solve()
                    if traj is None:
                        continue
                    err = self._approach_deg(traj[-1])
                    if best_err is None or err < best_err:
                        best_traj, best_err = traj, err
            finally:
                self._planner.update_tool_pose_criteria(
                    {self.tool_frame: ToolPoseCriteria.track_position(xyz=[1.0, 1.0, 1.0])}
                )
            if best_traj is not None:
                logger.info(
                    "cuRobo top-down reach %s: %.1f deg from vertical (best of %d)",
                    [round(float(x), 3) for x in goal_xyz],
                    best_err,
                    self.top_down_attempts,
                )
                return best_traj
            logger.info(
                "cuRobo top-down unreachable at %s; falling back to position-only",
                [round(float(x), 3) for x in goal_xyz],
            )

        traj = _solve()
        if traj is None:
            raise RuntimeError(f"cuRobo could not reach {[round(float(x), 3) for x in goal_xyz]}")
        return traj

    def _approach_deg(self, arm_q) -> float:
        """Angle (deg) of the tool approach axis from straight down (world -Z).

        The SO-101 ``gripper_frame_link`` local +x is the grasp approach axis
        (FK-verified: at home it maps to world -Z). 0 deg == perfectly top-down,
        90 deg == horizontal.
        """
        import torch
        from curobo.types import JointState

        q = torch.tensor([arm_q], dtype=torch.float32, device=self.device)
        st = self._planner.compute_kinematics(JointState.from_position(q, joint_names=self._arm_joint_names))
        tp = st.tool_poses
        if isinstance(tp, dict):
            tp = tp.get(self.tool_frame, list(tp.values())[0])
        quat = [float(v) for v in tp.quaternion.reshape(-1, 4)[0].detach().cpu().tolist()]
        return _approach_angle_deg(quat)

    def plan_pick_place(
        self,
        joint_names: Sequence[str],
        start_q: Sequence[float],
        gripper_joint: Optional[str] = None,
        cube_xy: Optional[Sequence[float]] = None,
        place_xy: Optional[Sequence[float]] = None,
        table_z: float = 0.0,
        approach: float = 0.12,
        grasp_z: float = 0.05,
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

        # (phase, goal_xyz | None for an in-place gripper event, gripper value,
        #  orient): pick/approach/lift descend onto the cube near-vertically
        #  (top-down); the place segments stay position-only (the bin pose is not
        #  vertical-reachable and a top-down drop is unnecessary).
        segments = [
            ("reach", [cx, cy, table_z + approach], OPEN, True),
            ("grasp", [cx, cy, table_z + grasp_z], OPEN, True),
            ("close", None, CLOSE, False),
            ("lift", [cx, cy, table_z + approach], CLOSE, True),
            ("place", [px, py, table_z + approach], CLOSE, False),
            ("place_down", [px, py, table_z + grasp_z + 0.02], CLOSE, False),
            ("release", None, OPEN, False),
        ]
        for phase, xyz, gripval, orient in segments:
            if xyz is None:
                for _ in range(hold_steps):
                    emit(cur_arm, gripval, phase)
                continue
            seg = self._plan_segment(cur_arm, xyz, quat, orient=orient)
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
    "top_down_grasp",
    "top_down_weight",
    "orientation_tolerance",
    "top_down_attempts",
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
