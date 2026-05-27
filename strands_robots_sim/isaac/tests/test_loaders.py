"""Tests for ``strands_robots_sim.isaac.loaders``.

Acceptance-criteria coverage (from issue #50):

* ``load_urdf(path) -> ProceduralRobot`` round-trips a Panda-style URDF.
* ``load_mjcf(path) -> ProceduralRobot`` round-trips a LIBERO-style MJCF.
* ``load_usd(path) -> ProceduralRobot`` round-trips a USD asset (gated
  behind the ``[isaac]`` extra → skipped when ``pxr`` is unavailable).
* Parse failure raises explicit :class:`ValueError` with file path +
  offending element — closes the #33 class of "phantom robot" silent-zero
  bugs.
* DOF count, joint names, joint types, and body parents match the
  procedural builders for at least one robot per format (parity test).

Test fixtures are written into the pytest ``tmp_path`` per test, so this
suite is hermetic — no binary assets in the repo (out-of-scope per the
issue).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from strands_robots_sim.isaac import loaders
from strands_robots_sim.isaac.procedural import (
    ProceduralRobot,
    _build_panda,
    _build_so100,
)

# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------


PANDA_URDF = """<?xml version="1.0"?>
<robot name="panda">
  <link name="panda_link0">
    <inertial><mass value="4.0"/></inertial>
    <collision><geometry><cylinder radius="0.06" length="0.05"/></geometry></collision>
  </link>
  <link name="panda_link1">
    <inertial><mass value="3.0"/></inertial>
    <collision><geometry><box size="0.04 0.04 0.06"/></geometry></collision>
  </link>
  <link name="panda_link2">
    <inertial><mass value="3.0"/></inertial>
  </link>
  <link name="panda_link3">
    <inertial><mass value="2.5"/></inertial>
  </link>
  <link name="panda_link4">
    <inertial><mass value="2.5"/></inertial>
  </link>
  <link name="panda_link5">
    <inertial><mass value="2.0"/></inertial>
  </link>
  <link name="panda_link6">
    <inertial><mass value="1.5"/></inertial>
  </link>
  <link name="panda_link7">
    <inertial><mass value="0.5"/></inertial>
  </link>

  <joint name="panda_joint1" type="revolute">
    <parent link="panda_link0"/>
    <child link="panda_link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.8973" upper="2.8973" effort="87" velocity="2.175"/>
  </joint>
  <joint name="panda_joint2" type="revolute">
    <parent link="panda_link1"/>
    <child link="panda_link2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.7628" upper="1.7628" effort="87" velocity="2.175"/>
  </joint>
  <joint name="panda_joint3" type="revolute">
    <parent link="panda_link2"/>
    <child link="panda_link3"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.8973" upper="2.8973" effort="87" velocity="2.175"/>
  </joint>
  <joint name="panda_joint4" type="revolute">
    <parent link="panda_link3"/>
    <child link="panda_link4"/>
    <axis xyz="0 -1 0"/>
    <limit lower="-3.0718" upper="-0.0698" effort="87" velocity="2.175"/>
  </joint>
  <joint name="panda_joint5" type="revolute">
    <parent link="panda_link4"/>
    <child link="panda_link5"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.8973" upper="2.8973" effort="12" velocity="2.61"/>
  </joint>
  <joint name="panda_joint6" type="revolute">
    <parent link="panda_link5"/>
    <child link="panda_link6"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.0175" upper="3.7525" effort="12" velocity="2.61"/>
  </joint>
  <joint name="panda_joint7" type="revolute">
    <parent link="panda_link6"/>
    <child link="panda_link7"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.8973" upper="2.8973" effort="12" velocity="2.61"/>
  </joint>
</robot>
"""


SO100_URDF = """<?xml version="1.0"?>
<robot name="so100">
  <link name="base_link"><inertial><mass value="2.0"/></inertial></link>
  <link name="shoulder_link"><inertial><mass value="0.5"/></inertial></link>
  <link name="upper_arm_link"><inertial><mass value="0.3"/></inertial></link>
  <link name="forearm_link"><inertial><mass value="0.2"/></inertial></link>
  <link name="wrist_link"><inertial><mass value="0.1"/></inertial></link>
  <link name="hand_link"><inertial><mass value="0.1"/></inertial></link>
  <link name="gripper_link"><inertial><mass value="0.05"/></inertial></link>

  <joint name="shoulder_pan" type="revolute">
    <parent link="base_link"/><child link="shoulder_link"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/>
  </joint>
  <joint name="shoulder_lift" type="revolute">
    <parent link="shoulder_link"/><child link="upper_arm_link"/>
    <axis xyz="0 1 0"/><limit lower="-1.57" upper="1.57"/>
  </joint>
  <joint name="elbow_flex" type="revolute">
    <parent link="upper_arm_link"/><child link="forearm_link"/>
    <axis xyz="0 1 0"/><limit lower="-2.35" upper="2.35"/>
  </joint>
  <joint name="wrist_flex" type="revolute">
    <parent link="forearm_link"/><child link="wrist_link"/>
    <axis xyz="0 1 0"/><limit lower="-1.57" upper="1.57"/>
  </joint>
  <joint name="wrist_roll" type="revolute">
    <parent link="wrist_link"/><child link="hand_link"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/>
  </joint>
  <joint name="gripper" type="prismatic">
    <parent link="hand_link"/><child link="gripper_link"/>
    <axis xyz="0 1 0"/><limit lower="0.0" upper="0.04"/>
  </joint>
</robot>
"""


# Minimal LIBERO-style MJCF: a 2-body chain (so100 shoulder + upper arm).
LIBERO_MJCF = """<?xml version="1.0"?>
<mujoco model="so100_lite">
  <worldbody>
    <body name="base_link" pos="0 0 0">
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.001 0.001 0.001"/>
      <geom type="cylinder" size="0.05 0.03"/>
      <body name="shoulder_link" pos="0 0 0.05">
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.0001 0.0001 0.0001"/>
        <geom type="box" size="0.04 0.04 0.08"/>
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <body name="upper_arm_link" pos="0 0 0.1">
          <inertial pos="0 0 0" mass="0.3" diaginertia="0.0001 0.0001 0.0001"/>
          <geom type="box" size="0.03 0.03 0.1"/>
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# URDF loader tests
# ---------------------------------------------------------------------------


class TestLoadUrdf:
    """Acceptance: ``loaders.load_urdf(path) -> ProceduralRobot`` round-trips Panda."""

    def test_load_urdf_round_trips_panda(self, tmp_path: Path) -> None:
        urdf_path = tmp_path / "panda.urdf"
        urdf_path.write_text(PANDA_URDF)

        robot = loaders.load_urdf(str(urdf_path))

        assert isinstance(robot, ProceduralRobot)
        assert robot.name == "panda"
        assert len(robot.bodies) == 8
        # 7 revolute joints (DOF count parity with _build_panda).
        assert robot.num_joints == 7

    def test_load_urdf_panda_parity_with_procedural(self, tmp_path: Path) -> None:
        """Acceptance: DOF count, joint names, joint types, body parents
        match the procedural builder for Panda."""
        urdf_path = tmp_path / "panda.urdf"
        urdf_path.write_text(PANDA_URDF)
        loaded = loaders.load_urdf(str(urdf_path))
        proc = _build_panda()

        # DOF count parity
        assert loaded.num_joints == proc.num_joints, "DOF count diverges"
        # Joint names parity (same order in both)
        assert loaded.joint_names == proc.joint_names, "joint names diverge"
        # Joint types parity
        loaded_types = [j.joint_type for j in loaded.joints]
        proc_types = [j.joint_type for j in proc.joints]
        assert loaded_types == proc_types, "joint types diverge"
        # Body parents parity (parent_body / child_body indices identical)
        loaded_edges = [(j.parent_body, j.child_body) for j in loaded.joints]
        proc_edges = [(j.parent_body, j.child_body) for j in proc.joints]
        assert loaded_edges == proc_edges, "body parent/child edges diverge"

    def test_load_urdf_so100_parity_with_procedural(self, tmp_path: Path) -> None:
        """Mixed revolute+prismatic parity case (so100's gripper is prismatic)."""
        urdf_path = tmp_path / "so100.urdf"
        urdf_path.write_text(SO100_URDF)
        loaded = loaders.load_urdf(str(urdf_path))
        proc = _build_so100()

        assert loaded.num_joints == proc.num_joints
        assert loaded.joint_names == proc.joint_names
        loaded_types = [j.joint_type for j in loaded.joints]
        proc_types = [j.joint_type for j in proc.joints]
        assert loaded_types == proc_types, "prismatic gripper joint should preserve type"

    def test_load_urdf_extracts_axis_and_limits(self, tmp_path: Path) -> None:
        urdf_path = tmp_path / "panda.urdf"
        urdf_path.write_text(PANDA_URDF)
        robot = loaders.load_urdf(str(urdf_path))

        joint1 = robot.joints[0]
        assert joint1.name == "panda_joint1"
        assert joint1.axis == (0.0, 0.0, 1.0)
        assert joint1.limit_lower == pytest.approx(-2.8973)
        assert joint1.limit_upper == pytest.approx(2.8973)

        joint4 = robot.joints[3]
        assert joint4.axis == (0.0, -1.0, 0.0)


class TestUrdfFailureModes:
    """Acceptance: parse failure raises explicit ValueError with file path —
    closes #33 (no silent ``joint_count=0`` phantom robots)."""

    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="URDF loader: file not found"):
            loaders.load_urdf(str(tmp_path / "nope.urdf"))

    def test_malformed_xml_raises_valueerror_with_path(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.urdf"
        bad.write_text("<robot><not-closing>")
        with pytest.raises(ValueError, match="malformed XML"):
            loaders.load_urdf(str(bad))

    def test_wrong_root_tag_raises_valueerror(self, tmp_path: Path) -> None:
        wrong = tmp_path / "wrong.urdf"
        wrong.write_text("<?xml version='1.0'?><scene><link name='a'/></scene>")
        with pytest.raises(ValueError, match="root element must be <robot>"):
            loaders.load_urdf(str(wrong))

    def test_zero_links_raises_valueerror(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.urdf"
        empty.write_text("<?xml version='1.0'?><robot name='ghost'></robot>")
        with pytest.raises(ValueError, match="zero <link>"):
            loaders.load_urdf(str(empty))

    def test_unknown_joint_type_raises_valueerror_with_offending_element(self, tmp_path: Path) -> None:
        bad = tmp_path / "weird.urdf"
        bad.write_text(
            "<?xml version='1.0'?><robot name='r'>"
            "<link name='a'/><link name='b'/>"
            "<joint name='j1' type='wibble'>"
            "<parent link='a'/><child link='b'/>"
            "</joint>"
            "</robot>"
        )
        with pytest.raises(ValueError, match="unknown joint type"):
            loaders.load_urdf(str(bad))

    def test_joint_referencing_unknown_link_raises_valueerror(self, tmp_path: Path) -> None:
        bad = tmp_path / "danglingref.urdf"
        bad.write_text(
            "<?xml version='1.0'?><robot name='r'>"
            "<link name='a'/><link name='b'/>"
            "<joint name='j1' type='revolute'>"
            "<parent link='a'/><child link='c'/>"
            "<axis xyz='0 0 1'/><limit lower='-1' upper='1'/>"
            "</joint>"
            "</robot>"
        )
        with pytest.raises(ValueError, match="unknown child link"):
            loaders.load_urdf(str(bad))


# ---------------------------------------------------------------------------
# MJCF loader tests
# ---------------------------------------------------------------------------


class TestLoadMjcf:
    """Acceptance: ``loaders.load_mjcf(path) -> ProceduralRobot`` round-trips
    a LIBERO-style MJCF scene."""

    def test_load_mjcf_round_trips_libero_like_scene(self, tmp_path: Path) -> None:
        mjcf_path = tmp_path / "scene.xml"
        mjcf_path.write_text(LIBERO_MJCF)

        robot = loaders.load_mjcf(str(mjcf_path))

        assert isinstance(robot, ProceduralRobot)
        assert robot.name == "so100_lite"
        # Synthetic "world" + 3 declared bodies.
        assert len(robot.bodies) == 4
        assert robot.bodies[0].name == "world"
        # Two hinge joints → two actuated DOF.
        assert robot.num_joints == 2
        assert robot.joint_names == ["shoulder_pan", "shoulder_lift"]

    def test_load_mjcf_joint_types_mapped(self, tmp_path: Path) -> None:
        mjcf_path = tmp_path / "scene.xml"
        mjcf_path.write_text(LIBERO_MJCF)
        robot = loaders.load_mjcf(str(mjcf_path))
        types = [j.joint_type for j in robot.joints]
        assert types == ["revolute", "revolute"], "hinge → revolute mapping"

    def test_load_mjcf_parent_child_indices_consistent(self, tmp_path: Path) -> None:
        """Parent-body should be the synthetic world (idx 0) for the root
        body; the second joint connects shoulder_link → upper_arm_link."""
        mjcf_path = tmp_path / "scene.xml"
        mjcf_path.write_text(LIBERO_MJCF)
        robot = loaders.load_mjcf(str(mjcf_path))

        # Body indices: 0=world, 1=base_link, 2=shoulder_link, 3=upper_arm_link
        # First joint is on shoulder_link (parent: base_link=1 → child=2)
        assert robot.joints[0].parent_body == 1
        assert robot.joints[0].child_body == 2
        # Second joint on upper_arm_link (parent: shoulder_link=2 → child=3)
        assert robot.joints[1].parent_body == 2
        assert robot.joints[1].child_body == 3

    def test_load_mjcf_extracts_axis_and_limits(self, tmp_path: Path) -> None:
        mjcf_path = tmp_path / "scene.xml"
        mjcf_path.write_text(LIBERO_MJCF)
        robot = loaders.load_mjcf(str(mjcf_path))

        assert robot.joints[0].axis == (0.0, 0.0, 1.0)
        assert robot.joints[0].limit_lower == pytest.approx(-3.14)
        assert robot.joints[0].limit_upper == pytest.approx(3.14)
        assert robot.joints[1].axis == (0.0, 1.0, 0.0)


class TestMjcfFailureModes:
    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="MJCF loader"):
            loaders.load_mjcf(str(tmp_path / "missing.xml"))

    def test_wrong_root_tag_raises_valueerror(self, tmp_path: Path) -> None:
        wrong = tmp_path / "wrong.xml"
        wrong.write_text("<?xml version='1.0'?><scene/>")
        with pytest.raises(ValueError, match="root element must be <mujoco>"):
            loaders.load_mjcf(str(wrong))

    def test_no_worldbody_raises_valueerror(self, tmp_path: Path) -> None:
        bad = tmp_path / "noworld.xml"
        bad.write_text("<?xml version='1.0'?><mujoco model='x'></mujoco>")
        with pytest.raises(ValueError, match="no <worldbody>"):
            loaders.load_mjcf(str(bad))

    def test_empty_worldbody_raises_valueerror(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.xml"
        empty.write_text("<?xml version='1.0'?><mujoco model='x'><worldbody/></mujoco>")
        with pytest.raises(ValueError, match="no <body>"):
            loaders.load_mjcf(str(empty))

    def test_unknown_joint_type_raises_valueerror(self, tmp_path: Path) -> None:
        bad = tmp_path / "weird.xml"
        bad.write_text(
            "<?xml version='1.0'?><mujoco model='m'>"
            "<worldbody><body name='b'><joint name='j' type='wibble'/></body></worldbody>"
            "</mujoco>"
        )
        with pytest.raises(ValueError, match="unknown joint type"):
            loaders.load_mjcf(str(bad))

    def test_malformed_xml_raises_valueerror_with_path(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.xml"
        bad.write_text("<mujoco><not-closing>")
        with pytest.raises(ValueError, match="malformed XML"):
            loaders.load_mjcf(str(bad))


# ---------------------------------------------------------------------------
# USD loader tests
# ---------------------------------------------------------------------------


_HAS_PXR = importlib.util.find_spec("pxr") is not None


@pytest.mark.skipif(not _HAS_PXR, reason="USD loader requires Pixar USD ([isaac] extra)")
class TestLoadUsd:
    """Acceptance: ``loaders.load_usd(path) -> ProceduralRobot`` round-trips a USD
    asset (gated behind the ``[isaac]`` extra; skipped when unavailable)."""

    @staticmethod
    def _build_two_body_revolute_stage(out_path: str) -> None:
        """Create a minimal USD stage: two rigid bodies connected by a revolute joint."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore

        stage = Usd.Stage.CreateNew(out_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # Two cube prims with RigidBodyAPI applied.
        for body_path in ("/World/base", "/World/link1"):
            prim = UsdGeom.Cube.Define(stage, body_path).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.GetMassAttr().Set(1.5 if body_path.endswith("base") else 0.7)

        # Revolute joint between them, axis Z, limits ±1.0 rad.
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/joint1")
        joint.GetBody0Rel().SetTargets([Sdf.Path("/World/base")])
        joint.GetBody1Rel().SetTargets([Sdf.Path("/World/link1")])
        joint.GetAxisAttr().Set("Z")
        joint.GetLowerLimitAttr().Set(-1.0)
        joint.GetUpperLimitAttr().Set(1.0)

        stage.GetRootLayer().Save()

    def test_load_usd_round_trips_two_body_revolute(self, tmp_path: Path) -> None:
        usd_path = tmp_path / "stage.usda"
        self._build_two_body_revolute_stage(str(usd_path))

        robot = loaders.load_usd(str(usd_path))

        assert isinstance(robot, ProceduralRobot)
        assert len(robot.bodies) == 2
        assert robot.num_joints == 1
        joint = robot.joints[0]
        assert joint.joint_type == "revolute"
        assert joint.axis == (0.0, 0.0, 1.0)
        assert joint.limit_lower == pytest.approx(-1.0)
        assert joint.limit_upper == pytest.approx(1.0)

    def test_load_usd_extracts_mass_from_mass_api(self, tmp_path: Path) -> None:
        usd_path = tmp_path / "stage.usda"
        self._build_two_body_revolute_stage(str(usd_path))
        robot = loaders.load_usd(str(usd_path))
        masses = sorted(b.mass for b in robot.bodies)
        # Authored masses 0.7 and 1.5.
        assert masses == pytest.approx([0.7, 1.5])


class TestUsdFailureModes:
    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="USD loader"):
            loaders.load_usd(str(tmp_path / "missing.usda"))

    @pytest.mark.skipif(_HAS_PXR, reason="exercise the no-pxr path only when pxr is unavailable")
    def test_load_usd_without_pxr_raises_importerror(self, tmp_path: Path) -> None:
        # Touch a real path so we hit the lazy import (not the file-not-found guard).
        path = tmp_path / "stage.usda"
        path.write_text("dummy")
        with pytest.raises(ImportError, match=r"strands-robots-sim\[isaac\]"):
            loaders.load_usd(str(path))

    @pytest.mark.skipif(not _HAS_PXR, reason="needs pxr to construct the malformed stage scenario")
    def test_load_usd_zero_rigid_bodies_raises_valueerror(self, tmp_path: Path) -> None:
        from pxr import Usd, UsdGeom  # type: ignore

        usd_path = tmp_path / "empty.usda"
        stage = Usd.Stage.CreateNew(str(usd_path))
        # A pure-geometry prim with no RigidBodyAPI applied.
        UsdGeom.Cube.Define(stage, "/World/decorative")
        stage.GetRootLayer().Save()

        with pytest.raises(ValueError, match="zero rigid bodies"):
            loaders.load_usd(str(usd_path))

    @pytest.mark.skipif(not _HAS_PXR, reason="needs pxr to construct the malformed stage scenario")
    def test_load_usd_joint_dangling_body_ref_raises_valueerror(self, tmp_path: Path) -> None:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore

        usd_path = tmp_path / "dangling.usda"
        stage = Usd.Stage.CreateNew(str(usd_path))
        prim = UsdGeom.Cube.Define(stage, "/World/base").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        # The joint references body0 OK, but body1 is a non-rigid prim.
        not_rigid = UsdGeom.Cube.Define(stage, "/World/decorative").GetPrim()  # noqa: F841
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/joint")
        joint.GetBody0Rel().SetTargets([Sdf.Path("/World/base")])
        joint.GetBody1Rel().SetTargets([Sdf.Path("/World/decorative")])
        stage.GetRootLayer().Save()

        with pytest.raises(ValueError, match="not a rigid body"):
            loaders.load_usd(str(usd_path))


# ---------------------------------------------------------------------------
# Cross-format invariants
# ---------------------------------------------------------------------------


class TestLoadersAreImportable:
    """The loaders module must be importable on a clean environment with
    no Pixar USD / mujoco / urdfpy installed (issue's recommendation for
    keeping procedural builders as zero-dep fallback)."""

    def test_loaders_module_imports_with_only_stdlib(self) -> None:
        # If this import succeeded above (file-level), we already passed.
        assert hasattr(loaders, "load_urdf")
        assert hasattr(loaders, "load_mjcf")
        assert hasattr(loaders, "load_usd")

    def test_load_urdf_does_not_require_pxr(self, tmp_path: Path) -> None:
        """URDF + MJCF parsers must work without pxr; they only need stdlib."""
        urdf_path = tmp_path / "p.urdf"
        urdf_path.write_text(PANDA_URDF)
        # Just verify the call succeeds; the implementation must not have
        # imported pxr at module level (already verified above by importing
        # ``loaders`` itself in this test file).
        robot = loaders.load_urdf(str(urdf_path))
        assert robot.num_joints == 7


# ---------------------------------------------------------------------------
# Real-asset parity: robosuite robots that strands-robots' LIBERO adapter
# consumes via MJCF must round-trip through ``load_mjcf`` cleanly.
#
# Closes cagataycali's review on PR #51 asking for "tests for verifying the
# robots we have in strands-robots to smoothly maps into isaac".
#
# The strands-robots LIBERO adapter uses robosuite's bundled MJCF assets
# (``robosuite/models/assets/robots/<name>/robot.xml``) for the seven
# robot embodiments LIBERO ships against: baxter / iiwa / jaco / kinova3
# / panda / sawyer / ur5e. Loading each via ``load_mjcf`` proves the
# loader handles the wire format the rest of the strands ecosystem
# already produces -- not just the synthetic fixtures above.
#
# The ``robosuite`` package is an optional / heavy dep (pulls mujoco +
# numpy etc.); these tests skip when it's not on PYTHONPATH, mirroring
# the ``pxr`` import-guard pattern on ``TestLoadUsd``.
# ---------------------------------------------------------------------------


_ROBOSUITE_ROBOT_EMBODIMENTS = {
    "panda": {"min_joints": 7, "max_joints": 7},  # Franka Panda 7-DOF arm
    "iiwa": {"min_joints": 7, "max_joints": 7},  # KUKA LBR iiwa 7-DOF
    "kinova3": {"min_joints": 7, "max_joints": 7},  # Kinova Gen3 7-DOF
    "jaco": {"min_joints": 7, "max_joints": 7},  # Kinova Jaco 7-DOF
    "sawyer": {"min_joints": 7, "max_joints": 7},  # Rethink Sawyer 7-DOF
    "ur5e": {"min_joints": 6, "max_joints": 6},  # Universal Robots UR5e 6-DOF
    "baxter": {"min_joints": 14, "max_joints": 14},  # Rethink Baxter dual 7-DOF arms
}


def _robosuite_robot_xml_path(robot_name: str) -> Path | None:
    """Return the path to the robosuite-bundled MJCF for ``robot_name``.

    Returns ``None`` if ``robosuite`` isn't installed, so each test can
    skip individually without a session-level fixture.
    """
    try:
        import robosuite
    except ImportError:
        return None
    rs_root = Path(robosuite.__file__).parent
    candidate = rs_root / "models" / "assets" / "robots" / robot_name / "robot.xml"
    return candidate if candidate.is_file() else None


_HAS_ROBOSUITE = _robosuite_robot_xml_path("panda") is not None


@pytest.mark.skipif(
    not _HAS_ROBOSUITE,
    reason="robosuite not installed; real-asset parity tests are gated on the optional dep",
)
class TestRobosuiteMjcfParity:
    """Pin: every robot embodiment LIBERO consumes via robosuite's bundled
    MJCFs must round-trip through ``load_mjcf`` and produce a sensible
    ProceduralRobot with the documented joint count.

    Locking the joint counts catches two failure modes at once:

    1. **Loader regression** (e.g. revolute / prismatic / fixed joint
       handling drifts under future refactors) — would surface as a
       ``num_joints`` mismatch on a known robot.
    2. **strands-robots upstream change** (e.g. robosuite ships a
       different MJCF schema or renames the asset) — would surface as
       a missing-file skip in CI rather than a silent zero-joints
       phantom robot (the #33-class failure).
    """

    @pytest.mark.parametrize("robot_name", sorted(_ROBOSUITE_ROBOT_EMBODIMENTS))
    def test_robosuite_robot_loads_cleanly(self, robot_name: str) -> None:
        """Each robosuite-bundled MJCF must load without raising."""
        xml_path = _robosuite_robot_xml_path(robot_name)
        if xml_path is None:
            pytest.skip(f"robosuite asset for {robot_name!r} not present (custom robosuite install?)")

        robot = loaders.load_mjcf(str(xml_path))

        assert robot.name, f"loaded {robot_name!r} robot has empty name (MJCF model attribute missing?)"
        assert len(robot.bodies) > 0, f"loaded {robot_name!r} robot has zero bodies"
        # Every body referenced by a joint must be a real body in the dataclass.
        body_count = len(robot.bodies)
        for j in robot.joints:
            assert 0 <= j.parent_body < body_count, (
                f"{robot_name}: joint {j.name!r} parent_body={j.parent_body} out of range "
                f"(robot has {body_count} bodies)"
            )
            assert 0 <= j.child_body < body_count, (
                f"{robot_name}: joint {j.name!r} child_body={j.child_body} out of range "
                f"(robot has {body_count} bodies)"
            )

    @pytest.mark.parametrize("robot_name", sorted(_ROBOSUITE_ROBOT_EMBODIMENTS))
    def test_robosuite_robot_joint_count_matches_spec(self, robot_name: str) -> None:
        """Loaded joint count must match the documented DOF for the embodiment.

        Locking these prevents silent regressions where a loader change
        drops a joint type or duplicates one. Spec values are based on
        each robot's published kinematic spec (from robosuite's docs).
        """
        xml_path = _robosuite_robot_xml_path(robot_name)
        if xml_path is None:
            pytest.skip(f"robosuite asset for {robot_name!r} not present")

        spec = _ROBOSUITE_ROBOT_EMBODIMENTS[robot_name]
        robot = loaders.load_mjcf(str(xml_path))

        assert spec["min_joints"] <= robot.num_joints <= spec["max_joints"], (
            f"{robot_name}: loaded {robot.num_joints} joints, expected "
            f"{spec['min_joints']}-{spec['max_joints']}. Either the loader regressed "
            f"or robosuite's MJCF for this robot changed; verify the spec table above "
            f"against the upstream asset."
        )

    @pytest.mark.parametrize("robot_name", sorted(_ROBOSUITE_ROBOT_EMBODIMENTS))
    def test_robosuite_robot_joints_are_revolute(self, robot_name: str) -> None:
        """Every robosuite arm uses revolute joints; the loader must reflect that.

        All seven robots are revolute-only manipulators (no prismatic
        actuation in the standard arm bodies). If the loader misclassifies
        a hinge as prismatic or fixed, this test catches it.
        """
        xml_path = _robosuite_robot_xml_path(robot_name)
        if xml_path is None:
            pytest.skip(f"robosuite asset for {robot_name!r} not present")

        robot = loaders.load_mjcf(str(xml_path))
        joint_types = {j.joint_type for j in robot.joints if j.joint_type != "fixed"}
        assert joint_types == {"revolute"}, (
            f"{robot_name}: expected all actuated joints to be revolute; got {sorted(joint_types)}. "
            f"The loader's hinge -> revolute mapping may be misclassifying joints."
        )

    def test_all_embodiments_at_least_load(self) -> None:
        """Sanity check: every robosuite robot in the spec table loads.

        If this fails on a particular robot, the parametrized tests above
        will give a more specific diagnostic. This test exists to make
        the failure visible at a glance even when the parametrized cases
        scroll off-screen.
        """
        failures = []
        for robot_name in _ROBOSUITE_ROBOT_EMBODIMENTS:
            xml_path = _robosuite_robot_xml_path(robot_name)
            if xml_path is None:
                continue
            try:
                loaders.load_mjcf(str(xml_path))
            except Exception as exc:  # noqa: BLE001 - aggregate failures
                failures.append(f"  {robot_name}: {type(exc).__name__}: {exc}")

        assert not failures, (
            "robosuite-bundled MJCFs failed to load:\n"
            + "\n".join(failures)
            + "\nstrands-robots' LIBERO adapter consumes these directly; a regression here "
            "would silently break the matrix's mujoco baseline for any of the affected robots."
        )
