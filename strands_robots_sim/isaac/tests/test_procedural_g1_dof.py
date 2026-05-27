"""Documentation honesty pin: G1 DOF count.

R2 review on #31 surfaced that ``procedural.py``'s G1 docstring / inline
comment claimed "29 DOF" but the actual ``g1()`` joint set is 21 (1 torso
+ 6 left leg + 6 right leg + 4 left arm + 4 right arm). ``__init__.py``
and ``docs/backends/isaac.md`` already advertised 21-DOF; only
``procedural.py`` was stale. Pinned here so the comment / docstring don't
drift back out of sync with the code under future refactors.

Companion pin for the ``docs/backends/isaac.md`` Phase 1 banner lives in
``test_phase1_doc_banner.py`` (lands in PR-5 alongside the docs file
itself).

The kinematic-tree topology invariant (each non-root link has exactly
one inbound joint; G1 splits its 2-DOF compound joints through six
intermediate massless link bodies) is pinned in this same PR's companion
file ``test_procedural_kinematic_guard.py``. Split by concern: this
file pins doc/comment honesty against a stale literal value, that file
pins the invariant the literal value is asserting against. Both run on
every CI invocation; neither is deferred.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROCEDURAL_PY = Path(__file__).resolve().parent.parent / "procedural.py"


class TestG1DOFCount:
    """Pin: G1 doc-string / inline-comment DOF count must match the joint set."""

    def test_g1_actual_joint_count_is_21(self) -> None:
        """The shipped G1 procedural definition has exactly 21 joints."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        robot = get_procedural_robot("unitree_g1")
        assert robot.num_joints == 21, (
            f"unitree_g1 has {robot.num_joints} joints; if this changes, the "
            f"DOF count in procedural.py docstrings/comments and "
            f"docs/backends/isaac.md must be updated together."
        )

    def test_g1_module_docstring_advertises_21_not_29(self) -> None:
        """Module docstring must not claim 29-DOF (stale -- actual is 21)."""
        text = _PROCEDURAL_PY.read_text(encoding="utf-8")
        # Look only at the module-level docstring (everything before the first
        # `from __future__` import, which is the canonical top-of-module
        # marker for this file).
        head = text.split("from __future__", 1)[0]
        assert "29-DOF" not in head and "29 DOF" not in head, (
            "procedural.py module docstring still claims 29-DOF for G1; the "
            "actual joint count is 21 (verified by test_g1_actual_joint_count_is_21)."
        )
        assert "21-DOF" in head or "21 DOF" in head, (
            "procedural.py module docstring no longer mentions the actual "
            "21-DOF count for G1; documentation must stay in sync with code."
        )

    def test_g1_builder_docstring_advertises_21_not_29(self) -> None:
        """``_build_unitree_g1`` docstring must not claim 29-DOF."""
        text = _PROCEDURAL_PY.read_text(encoding="utf-8")
        # Find the def / docstring window for _build_unitree_g1 specifically.
        match = re.search(
            r"def\s+_build_unitree_g1\b.*?(?=\ndef |\Z)",
            text,
            flags=re.DOTALL,
        )
        assert match is not None, "could not locate _build_unitree_g1 in procedural.py"
        body = match.group(0)
        assert "29-DOF" not in body and "29 DOF" not in body, (
            "_build_unitree_g1 still mentions 29-DOF; the actual joint count "
            "is 21 (verified by test_g1_actual_joint_count_is_21)."
        )
        assert "21-DOF" in body or "21 DOF" in body, "_build_unitree_g1 no longer documents its actual 21-DOF count."
