"""Procedural robot builders for Isaac Sim (USD prim API).

Mirrors strands_robots_sim.newton.procedural but uses USD/Isaac APIs
to construct robots purely from code -- no binary asset files required.

Supported procedural robots:
    - so100: SO-100 6-DOF tabletop arm
    - panda: Franka Emika Panda 7-DOF
    - unitree_g1: Unitree G1 humanoid (21-DOF, simplified)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JointDef:
    """Definition of a single joint in a procedural robot."""

    name: str
    joint_type: str = "revolute"  # revolute, prismatic, fixed
    parent_body: int = 0
    child_body: int = 1
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    limit_lower: float = -3.14159
    limit_upper: float = 3.14159
    damping: float = 0.1
    stiffness: float = 0.0
    armature: float = 0.01


@dataclass
class BodyDef:
    """Definition of a single body/link in a procedural robot."""

    name: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 1.0
    shape: str = "box"  # box, sphere, capsule, cylinder
    shape_size: tuple[float, ...] = (0.05, 0.05, 0.05)


@dataclass
class ProceduralRobot:
    """Complete procedural robot definition."""

    name: str
    bodies: list[BodyDef] = field(default_factory=list)
    joints: list[JointDef] = field(default_factory=list)
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def num_joints(self) -> int:
        """Number of actuated (non-fixed) joints."""
        return sum(1 for j in self.joints if j.joint_type != "fixed")

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of actuated joint names."""
        return [j.name for j in self.joints if j.joint_type != "fixed"]


def _build_so100() -> ProceduralRobot:
    """Build SO-100 6-DOF tabletop arm procedurally."""
    bodies = [
        BodyDef(name="base_link", position=(0.0, 0.0, 0.0), mass=2.0, shape="cylinder", shape_size=(0.05, 0.03)),
        BodyDef(name="shoulder_link", position=(0.0, 0.0, 0.05), mass=0.5, shape="box", shape_size=(0.04, 0.04, 0.08)),
        BodyDef(name="upper_arm_link", position=(0.0, 0.0, 0.15), mass=0.3, shape="box", shape_size=(0.03, 0.03, 0.1)),
        BodyDef(name="forearm_link", position=(0.0, 0.0, 0.25), mass=0.2, shape="box", shape_size=(0.025, 0.025, 0.08)),
        BodyDef(name="wrist_link", position=(0.0, 0.0, 0.33), mass=0.1, shape="box", shape_size=(0.02, 0.02, 0.04)),
        BodyDef(name="hand_link", position=(0.0, 0.0, 0.37), mass=0.1, shape="box", shape_size=(0.02, 0.04, 0.02)),
        BodyDef(name="gripper_link", position=(0.0, 0.0, 0.39), mass=0.05, shape="box", shape_size=(0.01, 0.05, 0.02)),
    ]

    joints = [
        JointDef(name="shoulder_pan", parent_body=0, child_body=1, axis=(0, 0, 1), limit_lower=-3.14, limit_upper=3.14),
        JointDef(
            name="shoulder_lift", parent_body=1, child_body=2, axis=(0, 1, 0), limit_lower=-1.57, limit_upper=1.57
        ),
        JointDef(name="elbow_flex", parent_body=2, child_body=3, axis=(0, 1, 0), limit_lower=-2.35, limit_upper=2.35),
        JointDef(name="wrist_flex", parent_body=3, child_body=4, axis=(0, 1, 0), limit_lower=-1.57, limit_upper=1.57),
        JointDef(name="wrist_roll", parent_body=4, child_body=5, axis=(0, 0, 1), limit_lower=-3.14, limit_upper=3.14),
        JointDef(
            name="gripper",
            parent_body=5,
            child_body=6,
            joint_type="prismatic",
            axis=(0, 1, 0),
            limit_lower=0.0,
            limit_upper=0.04,
        ),
    ]

    return ProceduralRobot(name="so100", bodies=bodies, joints=joints)


def _build_panda() -> ProceduralRobot:
    """Build Franka Emika Panda 7-DOF arm procedurally."""
    bodies = [
        BodyDef(name="panda_link0", position=(0.0, 0.0, 0.0), mass=4.0, shape="cylinder", shape_size=(0.06, 0.05)),
        BodyDef(name="panda_link1", position=(0.0, 0.0, 0.333), mass=3.0, shape="capsule", shape_size=(0.04, 0.06)),
        BodyDef(name="panda_link2", position=(0.0, 0.0, 0.333), mass=3.0, shape="capsule", shape_size=(0.04, 0.06)),
        BodyDef(name="panda_link3", position=(0.0, 0.0, 0.649), mass=2.5, shape="capsule", shape_size=(0.035, 0.05)),
        BodyDef(name="panda_link4", position=(0.0, 0.0, 0.649), mass=2.5, shape="capsule", shape_size=(0.035, 0.05)),
        BodyDef(name="panda_link5", position=(0.0, 0.0, 1.033), mass=2.0, shape="capsule", shape_size=(0.03, 0.04)),
        BodyDef(name="panda_link6", position=(0.0, 0.0, 1.033), mass=1.5, shape="capsule", shape_size=(0.03, 0.035)),
        BodyDef(name="panda_link7", position=(0.0, 0.0, 1.143), mass=0.5, shape="cylinder", shape_size=(0.04, 0.02)),
    ]

    joints = [
        JointDef(
            name="panda_joint1", parent_body=0, child_body=1, axis=(0, 0, 1), limit_lower=-2.8973, limit_upper=2.8973
        ),
        JointDef(
            name="panda_joint2", parent_body=1, child_body=2, axis=(0, 1, 0), limit_lower=-1.7628, limit_upper=1.7628
        ),
        JointDef(
            name="panda_joint3", parent_body=2, child_body=3, axis=(0, 0, 1), limit_lower=-2.8973, limit_upper=2.8973
        ),
        JointDef(
            name="panda_joint4", parent_body=3, child_body=4, axis=(0, -1, 0), limit_lower=-3.0718, limit_upper=-0.0698
        ),
        JointDef(
            name="panda_joint5", parent_body=4, child_body=5, axis=(0, 0, 1), limit_lower=-2.8973, limit_upper=2.8973
        ),
        JointDef(
            name="panda_joint6", parent_body=5, child_body=6, axis=(0, 1, 0), limit_lower=-0.0175, limit_upper=3.7525
        ),
        JointDef(
            name="panda_joint7", parent_body=6, child_body=7, axis=(0, 0, 1), limit_lower=-2.8973, limit_upper=2.8973
        ),
    ]

    return ProceduralRobot(name="panda", bodies=bodies, joints=joints)


def _build_unitree_g1() -> ProceduralRobot:
    """Build Unitree G1 humanoid (simplified 21-DOF) procedurally."""
    bodies = [
        BodyDef(name="pelvis", position=(0.0, 0.0, 0.85), mass=10.0, shape="box", shape_size=(0.15, 0.1, 0.15)),
        BodyDef(name="torso", position=(0.0, 0.0, 1.1), mass=8.0, shape="box", shape_size=(0.12, 0.08, 0.3)),
        BodyDef(name="head", position=(0.0, 0.0, 1.4), mass=2.0, shape="sphere", shape_size=(0.08,)),
        # Left leg
        BodyDef(name="l_hip", position=(-0.08, 0.0, 0.75), mass=2.0, shape="sphere", shape_size=(0.05,)),
        BodyDef(name="l_thigh", position=(-0.08, 0.0, 0.5), mass=3.0, shape="capsule", shape_size=(0.04, 0.15)),
        BodyDef(name="l_shin", position=(-0.08, 0.0, 0.25), mass=2.0, shape="capsule", shape_size=(0.035, 0.15)),
        BodyDef(name="l_foot", position=(-0.08, 0.0, 0.05), mass=1.0, shape="box", shape_size=(0.1, 0.06, 0.03)),
        # Right leg
        BodyDef(name="r_hip", position=(0.08, 0.0, 0.75), mass=2.0, shape="sphere", shape_size=(0.05,)),
        BodyDef(name="r_thigh", position=(0.08, 0.0, 0.5), mass=3.0, shape="capsule", shape_size=(0.04, 0.15)),
        BodyDef(name="r_shin", position=(0.08, 0.0, 0.25), mass=2.0, shape="capsule", shape_size=(0.035, 0.15)),
        BodyDef(name="r_foot", position=(0.08, 0.0, 0.05), mass=1.0, shape="box", shape_size=(0.1, 0.06, 0.03)),
        # Left arm
        BodyDef(name="l_shoulder", position=(-0.2, 0.0, 1.2), mass=1.5, shape="sphere", shape_size=(0.04,)),
        BodyDef(name="l_upper_arm", position=(-0.35, 0.0, 1.2), mass=1.5, shape="capsule", shape_size=(0.03, 0.12)),
        BodyDef(name="l_forearm", position=(-0.55, 0.0, 1.2), mass=1.0, shape="capsule", shape_size=(0.025, 0.1)),
        # Right arm
        BodyDef(name="r_shoulder", position=(0.2, 0.0, 1.2), mass=1.5, shape="sphere", shape_size=(0.04,)),
        BodyDef(name="r_upper_arm", position=(0.35, 0.0, 1.2), mass=1.5, shape="capsule", shape_size=(0.03, 0.12)),
        BodyDef(name="r_forearm", position=(0.55, 0.0, 1.2), mass=1.0, shape="capsule", shape_size=(0.025, 0.1)),
    ]

    # Simplified joint set (21 DOF total: 1 torso + 6 left leg + 6 right leg + 4 left arm + 4 right arm).
    # NOTE: this kinematic graph contains duplicate (parent, child) edges on each leg/arm
    # (e.g. l_hip_roll and l_hip_pitch both map bodies 3 -> 4). A real USD/MuJoCo articulation
    # builder requires a tree where each non-root link has exactly one inbound joint, so this
    # topology will need intermediate massless link bodies before Phase 2 wires up the actual
    # USD prim chain. Tracked as Phase-2 work; the Phase-1 skeleton does not instantiate the
    # articulation, so the duplicate-edge defect is dormant on this branch.
    joints = [
        # Torso
        JointDef(name="torso_yaw", parent_body=0, child_body=1, axis=(0, 0, 1), limit_lower=-1.0, limit_upper=1.0),
        # Left leg (6 DOF)
        JointDef(name="l_hip_yaw", parent_body=0, child_body=3, axis=(0, 0, 1), limit_lower=-0.5, limit_upper=0.5),
        JointDef(name="l_hip_roll", parent_body=3, child_body=4, axis=(1, 0, 0), limit_lower=-0.5, limit_upper=0.5),
        JointDef(name="l_hip_pitch", parent_body=3, child_body=4, axis=(0, 1, 0), limit_lower=-1.5, limit_upper=0.5),
        JointDef(name="l_knee", parent_body=4, child_body=5, axis=(0, 1, 0), limit_lower=-0.1, limit_upper=2.5),
        JointDef(name="l_ankle_pitch", parent_body=5, child_body=6, axis=(0, 1, 0), limit_lower=-0.8, limit_upper=0.5),
        JointDef(name="l_ankle_roll", parent_body=5, child_body=6, axis=(1, 0, 0), limit_lower=-0.3, limit_upper=0.3),
        # Right leg (6 DOF)
        JointDef(name="r_hip_yaw", parent_body=0, child_body=7, axis=(0, 0, 1), limit_lower=-0.5, limit_upper=0.5),
        JointDef(name="r_hip_roll", parent_body=7, child_body=8, axis=(1, 0, 0), limit_lower=-0.5, limit_upper=0.5),
        JointDef(name="r_hip_pitch", parent_body=7, child_body=8, axis=(0, 1, 0), limit_lower=-1.5, limit_upper=0.5),
        JointDef(name="r_knee", parent_body=8, child_body=9, axis=(0, 1, 0), limit_lower=-0.1, limit_upper=2.5),
        JointDef(name="r_ankle_pitch", parent_body=9, child_body=10, axis=(0, 1, 0), limit_lower=-0.8, limit_upper=0.5),
        JointDef(name="r_ankle_roll", parent_body=9, child_body=10, axis=(1, 0, 0), limit_lower=-0.3, limit_upper=0.3),
        # Left arm (4 DOF simplified)
        JointDef(
            name="l_shoulder_pitch", parent_body=1, child_body=11, axis=(0, 1, 0), limit_lower=-3.14, limit_upper=1.0
        ),
        JointDef(
            name="l_shoulder_roll", parent_body=11, child_body=12, axis=(1, 0, 0), limit_lower=-0.3, limit_upper=3.14
        ),
        JointDef(
            name="l_shoulder_yaw", parent_body=12, child_body=13, axis=(0, 0, 1), limit_lower=-1.5, limit_upper=1.5
        ),
        JointDef(name="l_elbow", parent_body=12, child_body=13, axis=(0, 1, 0), limit_lower=-2.5, limit_upper=0.0),
        # Right arm (4 DOF simplified)
        JointDef(
            name="r_shoulder_pitch", parent_body=1, child_body=14, axis=(0, 1, 0), limit_lower=-3.14, limit_upper=1.0
        ),
        JointDef(
            name="r_shoulder_roll", parent_body=14, child_body=15, axis=(1, 0, 0), limit_lower=-3.14, limit_upper=0.3
        ),
        JointDef(
            name="r_shoulder_yaw", parent_body=15, child_body=16, axis=(0, 0, 1), limit_lower=-1.5, limit_upper=1.5
        ),
        JointDef(name="r_elbow", parent_body=15, child_body=16, axis=(0, 1, 0), limit_lower=-2.5, limit_upper=0.0),
    ]

    return ProceduralRobot(name="unitree_g1", bodies=bodies, joints=joints, base_position=(0.0, 0.0, 0.85))


# Registry of procedural robots
_PROCEDURAL_REGISTRY: dict[str, ProceduralRobot] = {}

# Aliases for common names
_ALIASES: dict[str, str] = {
    "so100": "so100",
    "so-100": "so100",
    "so_100": "so100",
    "so101": "so100",
    "franka": "panda",
    "franka_panda": "panda",
    "panda": "panda",
    "unitree_g1": "unitree_g1",
    "g1": "unitree_g1",
}


def get_procedural_robot(name: str) -> ProceduralRobot | None:
    """Look up a procedural robot by name or alias.

    Parameters
    ----------
    name : str
        Robot name or alias.

    Returns
    -------
    ProceduralRobot or None
        The robot definition, or None if not found.
    """
    # Lazy-build registry
    if not _PROCEDURAL_REGISTRY:
        _PROCEDURAL_REGISTRY["so100"] = _build_so100()
        _PROCEDURAL_REGISTRY["panda"] = _build_panda()
        _PROCEDURAL_REGISTRY["unitree_g1"] = _build_unitree_g1()

    canonical = _ALIASES.get(name.lower(), name.lower())
    return _PROCEDURAL_REGISTRY.get(canonical)


def list_procedural_robots() -> list[str]:
    """Return list of available procedural robot names."""
    return ["so100", "panda", "unitree_g1"]
