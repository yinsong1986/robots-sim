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
    "cuRobo is not installed (issue #67 T3/T4). Install it out-of-band:\n"
    "  git clone https://github.com/NVlabs/curobo && cd curobo\n"
    "  uv venv --python 3.11 && source .venv/bin/activate\n"
    "  uv pip install .[cu12-torch]\n"
    "Requires an NVIDIA GPU (> Turing) and driver >= 580.65.06 for the latest "
    "release (our box is 550.x; a pinned older cuRobo may work — see T3). "
    "Until then the demo uses the ScriptedPlanner fallback."
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
    """cuRobo ``MotionGen`` wrapper (issue #67 T5) — collision-aware planning.

    Lazy-imports cuRobo. Until cuRobo + a SO-101 robot config (T4) are present,
    planning entry points raise :class:`RuntimeError` with :data:`CUROBO_INSTALL_HINT`
    so callers can fall back to :class:`ScriptedPlanner`.
    """

    name = "curobo"

    def __init__(self, robot_cfg: str = "so101", world_collision=None, device: str = "cuda"):
        self.robot_cfg = robot_cfg
        self.world_collision = world_collision
        self.device = device
        self._motion_gen = None

    @staticmethod
    def available() -> bool:
        return curobo_available()

    def _ensure(self):
        if not self.available():
            raise RuntimeError(CUROBO_INSTALL_HINT)
        if self._motion_gen is None:
            # T5: build MotionGenConfig from the SO-101 robot YAML (T4) + the
            # sim collision world, warm up, and cache. Intentionally not wired
            # to a stub: it requires the real cuRobo runtime + SO-101 config.
            raise RuntimeError(
                "cuRobo is importable but the SO-101 MotionGen setup (issue #67 "
                "T4 robot config + T5 world sync) is not wired yet. Use the "
                "ScriptedPlanner fallback, or complete T4/T5 to enable real "
                "collision-aware planning."
            )

    def plan_pick_place(self, *args, **kwargs) -> JointTrajectory:
        self._ensure()  # raises with an actionable message today
        raise NotImplementedError  # pragma: no cover

    def plan_to_pose(self, *args, **kwargs) -> JointTrajectory:
        self._ensure()
        raise NotImplementedError  # pragma: no cover


def make_planner(prefer: str = "auto", robot_cfg: str = "so101", **kwargs):
    """Return the best available planner.

    ``prefer``: ``"auto"`` (cuRobo if importable, else scripted), ``"curobo"``
    (force cuRobo — raises later if unusable), or ``"scripted"``.
    """
    prefer = (prefer or "auto").lower()
    if prefer == "scripted":
        return ScriptedPlanner(
            **{k: v for k, v in kwargs.items() if k in ("gripper_open", "gripper_close", "steps_per_phase")}
        )
    if prefer == "curobo" or (prefer == "auto" and CUROBO_AVAILABLE):
        if CUROBO_AVAILABLE:
            return CuroboMotionPlanner(
                robot_cfg=robot_cfg, **{k: v for k, v in kwargs.items() if k in ("world_collision", "device")}
            )
        logger.warning(
            "cuRobo requested but unavailable; using ScriptedPlanner. %s", CUROBO_INSTALL_HINT.splitlines()[0]
        )
    return ScriptedPlanner(
        **{k: v for k, v in kwargs.items() if k in ("gripper_open", "gripper_close", "steps_per_phase")}
    )
