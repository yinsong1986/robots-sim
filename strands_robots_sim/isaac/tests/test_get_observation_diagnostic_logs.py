"""Pin tests for get_observation diagnostic-log surface.

The Phase 1 ``get_observation`` returns a plain ``dict[str, Any]`` and an
empty dict is indistinguishable across four failure modes:

1. ``world not yet created``
2. ``robot_name=None`` with multiple robots present (ambiguous)
3. ``robot_name`` not in the registered robots (typo / not-yet-added)
4. robot present but ``Articulation`` handle not initialised

The return-shape contract is preserved (callers consume positions keyed by
joint name), but each silent-``{}`` mode now emits a log line at an
appropriate level so operators can distinguish the conditions in logs.

These tests fail on pre-fix code (no log lines emitted) and pass after the
fix.  They also pin the level discipline (DEBUG for expected pre-init, WARNING
for diagnostically-meaningful operator-attention conditions).
"""

from __future__ import annotations

import logging

# -----------------------------------------------------------------------------
# Helper: build a simulation with N fake robots without touching Isaac Sim.
# -----------------------------------------------------------------------------


def _make_sim_with_fake_robots(n_robots: int):
    """Create an IsaacSimulation with ``n_robots`` registered fake robots.

    Bypasses the real World/USD machinery -- we set ``_world_created = True``
    and stuff ``_RobotState`` placeholders into ``_robots`` directly.
    """
    from strands_robots_sim.isaac.simulation import IsaacSimulation, _RobotState

    sim = IsaacSimulation()
    sim._world_created = True

    for i in range(n_robots):
        name = f"robot{i}"
        sim._robots[name] = _RobotState(
            name=name,
            prim_path=f"/World/{name}",
            joint_names=["j0"],
            articulation=None,  # Phase 1 stub
        )

    return sim


# -----------------------------------------------------------------------------
# Failure mode 1: world not yet created -> DEBUG.
# -----------------------------------------------------------------------------


class TestWorldNotCreatedLogsDebug:
    """``get_observation`` before ``create_world()`` should DEBUG-log.

    DEBUG (not WARNING) because feature-detection callers probe before init.
    """

    def test_world_not_created_logs_debug(self, caplog):
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        # _world_created is False by construction.

        with caplog.at_level(logging.DEBUG, logger="strands_robots_sim.isaac.simulation"):
            result = sim.get_observation("nonexistent")

        assert result == {}
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(
            "world not yet created" in r.getMessage() for r in debug_records
        ), f"Expected DEBUG log with 'world not yet created'; got {[r.getMessage() for r in debug_records]}"

    def test_world_not_created_does_not_log_warning(self, caplog):
        """Pre-init probe must not pollute WARNING; many callers feature-detect."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        with caplog.at_level(logging.WARNING, logger="strands_robots_sim.isaac.simulation"):
            sim.get_observation("nonexistent")

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert (
            warning_records == []
        ), f"Pre-init probe should not emit WARNING; got {[r.getMessage() for r in warning_records]}"


# -----------------------------------------------------------------------------
# Failure mode 2: robot_name=None with multiple robots -> WARNING.
# -----------------------------------------------------------------------------


class TestAmbiguousRobotNameLogsWarning:
    """``get_observation(None)`` with N>1 robots must WARNING-log."""

    def test_ambiguous_robot_name_logs_warning(self, caplog):
        sim = _make_sim_with_fake_robots(n_robots=3)

        with caplog.at_level(logging.WARNING, logger="strands_robots_sim.isaac.simulation"):
            result = sim.get_observation(robot_name=None)

        assert result == {}
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "ambiguous" in r.getMessage() for r in warning_records
        ), f"Expected WARNING with 'ambiguous'; got {[r.getMessage() for r in warning_records]}"

    def test_ambiguous_warning_lists_known_robots(self, caplog):
        """The warning must surface the robot-name set so operators can disambiguate."""
        sim = _make_sim_with_fake_robots(n_robots=3)

        with caplog.at_level(logging.WARNING, logger="strands_robots_sim.isaac.simulation"):
            sim.get_observation(robot_name=None)

        msg = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        for name in ("robot0", "robot1", "robot2"):
            assert name in msg, f"Ambiguous-WARNING must list {name}; got {msg!r}"


# -----------------------------------------------------------------------------
# Failure mode 3: unknown robot_name -> WARNING.
# -----------------------------------------------------------------------------


class TestUnknownRobotNameLogsWarning:
    """``get_observation('typo')`` must WARNING-log with a known-set hint."""

    def test_unknown_robot_name_logs_warning(self, caplog):
        sim = _make_sim_with_fake_robots(n_robots=2)

        with caplog.at_level(logging.WARNING, logger="strands_robots_sim.isaac.simulation"):
            result = sim.get_observation(robot_name="rob0t0")  # typo

        assert result == {}
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "unknown robot" in r.getMessage() for r in warning_records
        ), f"Expected WARNING with 'unknown robot'; got {[r.getMessage() for r in warning_records]}"

    def test_unknown_warning_includes_typo_value(self, caplog):
        """The typo value itself must appear so operators can grep their callsite."""
        sim = _make_sim_with_fake_robots(n_robots=2)

        with caplog.at_level(logging.WARNING, logger="strands_robots_sim.isaac.simulation"):
            sim.get_observation(robot_name="rob0t0")

        msg = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert "rob0t0" in msg, f"Unknown-WARNING must echo the typo; got {msg!r}"

    def test_unknown_warning_lists_known_robots(self, caplog):
        """The warning must list known robot names so operators can fix the typo."""
        sim = _make_sim_with_fake_robots(n_robots=2)

        with caplog.at_level(logging.WARNING, logger="strands_robots_sim.isaac.simulation"):
            sim.get_observation(robot_name="rob0t0")

        msg = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert "robot0" in msg and "robot1" in msg, f"Unknown-WARNING must list known names; got {msg!r}"


# -----------------------------------------------------------------------------
# Failure mode 4: single robot, articulation None -> empty dict, no spurious logs.
# -----------------------------------------------------------------------------


class TestSingleRobotPhase1Stub:
    """One robot, articulation=None (Phase 1 stub) -> {} with no WARNING."""

    def test_single_robot_articulation_none_returns_empty(self, caplog):
        """The Phase 1 silent-success path is documented; should not WARNING."""
        sim = _make_sim_with_fake_robots(n_robots=1)

        with caplog.at_level(logging.WARNING, logger="strands_robots_sim.isaac.simulation"):
            result = sim.get_observation()  # auto-resolve

        assert result == {}
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            "Single-robot articulation-None is a documented Phase 1 surface; "
            "should not emit WARNING (would create alert noise on every "
            f"observation tick).  Got: {[r.getMessage() for r in warning_records]}"
        )


# -----------------------------------------------------------------------------
# Docstring contract pin: the return-shape decision is recorded in the docstring
# so future contributors don't accidentally "fix" the dict shape and break the
# joint-positions-keyed-by-name contract.
# -----------------------------------------------------------------------------


class TestDocstringRecordsReturnShapeDecision:
    """The four silent-``{}`` modes must be enumerated in the docstring."""

    def test_docstring_enumerates_four_silent_modes(self):
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        doc = IsaacSimulation.get_observation.__doc__
        assert doc is not None
        # Each mode named explicitly so a future doc-rewrite still has to
        # account for them.
        assert "world not yet created" in doc
        assert "ambiguous" in doc
        assert "unknown robot_name" in doc
        assert "Articulation handle not yet initialised" in doc
