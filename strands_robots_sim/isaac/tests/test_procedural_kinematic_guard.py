"""Phase-1 fail-fast guard pin: ``_validate_kinematic_tree``.

cagataycali's review on PR #46 (procedural USD builders) flagged that
the documented duplicate-edge defect on G1 is silently returned by
``_build_unitree_g1`` and would only surface as a cryptic articulation
error when Phase 2 wires up USD instantiation. The guard is opt-in via
``STRANDS_ISAAC_VALIDATE_KINEMATICS`` so it does not break Phase-1
callers (which never instantiate the articulation), but it MUST fire on
the known defect when Phase-2 development flips the env var on.

This test pins three properties:

1. Default behaviour: ``get_procedural_robot("unitree_g1")`` returns a
   robot regardless of the topology defect (the env var defaults off).
2. When ``STRANDS_ISAAC_VALIDATE_KINEMATICS=1`` is set, building G1
   raises ``ValueError`` mentioning duplicate edges + the offending
   joint names (so Phase 2 sees the defect at builder time with full
   diagnostic context).
3. SO-100 and Panda still build cleanly under the same env var --
   their topologies are already valid trees, so the guard must be a
   no-op on them.

The guard intentionally lives in ``procedural.py``; pinning the
behaviour here means a future refactor that drops the guard or relaxes
its trigger logic will fail this test rather than silently regressing.
"""

from __future__ import annotations

import importlib
import os

import pytest


def _reload_procedural():
    """Re-import ``procedural`` with whatever env state is currently set.

    The module's lazy registry caches built robots on first call, so
    we reload it between tests to make the env-var check authoritative
    on each build.
    """
    import strands_robots_sim.isaac.procedural as procedural

    return importlib.reload(procedural)


class TestKinematicGuard:
    """Pin: ``_validate_kinematic_tree`` opt-in semantics + G1 trigger."""

    def test_g1_builds_by_default_when_guard_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the env var, G1 builds fine despite the documented defect."""
        monkeypatch.delenv("STRANDS_ISAAC_VALIDATE_KINEMATICS", raising=False)
        procedural = _reload_procedural()
        robot = procedural.get_procedural_robot("unitree_g1")
        assert robot is not None
        assert robot.name == "unitree_g1"
        assert robot.num_joints == 21

    @pytest.mark.parametrize("env_value", ["1", "true", "yes", "TRUE", "Yes"])
    def test_g1_raises_when_guard_enabled(self, monkeypatch: pytest.MonkeyPatch, env_value: str) -> None:
        """With the env var on, building G1 raises with the duplicate edges named."""
        monkeypatch.setenv("STRANDS_ISAAC_VALIDATE_KINEMATICS", env_value)
        procedural = _reload_procedural()
        with pytest.raises(ValueError) as excinfo:
            procedural._build_unitree_g1()
        msg = str(excinfo.value)
        assert "unitree_g1" in msg
        assert "duplicate parent->child body edges" in msg
        # At least one of the documented duplicate-edge joint pairs must
        # be named in the diagnostic so Phase-2 callers know which joints
        # to split.
        assert "l_hip_roll" in msg or "l_hip_pitch" in msg

    @pytest.mark.parametrize("env_value", ["", "0", "false", "no", "off"])
    def test_g1_guard_off_for_non_truthy_env_values(self, monkeypatch: pytest.MonkeyPatch, env_value: str) -> None:
        """Only ``1``/``true``/``yes`` enable the guard; everything else is a no-op."""
        monkeypatch.setenv("STRANDS_ISAAC_VALIDATE_KINEMATICS", env_value)
        procedural = _reload_procedural()
        # Should not raise.
        robot = procedural._build_unitree_g1()
        assert robot.name == "unitree_g1"

    def test_so100_and_panda_pass_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SO-100 and Panda are valid trees; guard must accept them when enabled."""
        monkeypatch.setenv("STRANDS_ISAAC_VALIDATE_KINEMATICS", "1")
        procedural = _reload_procedural()
        # Both must build without raising; the guard is wired into the end of
        # each ``_build_*`` so a regression that tightens the check too far
        # would surface here.
        so100 = procedural._build_so100()
        panda = procedural._build_panda()
        assert so100.name == "so100"
        assert panda.name == "panda"


def teardown_module(module: object) -> None:
    """Restore the module to its default state for downstream tests."""
    os.environ.pop("STRANDS_ISAAC_VALIDATE_KINEMATICS", None)
    _reload_procedural()
