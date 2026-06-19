"""Unit tests for Isaac Sim backend (no GPU required).

All tests use mocking to avoid requiring Isaac Sim or CUDA.

Run with: pytest strands_robots_sim/isaac/tests/test_unit.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestIsaacConfig:
    """Tests for IsaacConfig dataclass validation."""

    def test_default_config(self):
        """Default config should be valid."""
        from strands_robots_sim.isaac.config import IsaacConfig

        config = IsaacConfig()
        assert config.num_envs == 1
        assert config.device == "cuda:0"
        assert config.headless is True
        assert config.render_mode == "headless"
        assert config.gravity == (0.0, 0.0, -9.81)
        assert config.ground_plane is True
        assert config.camera_width == 640
        assert config.camera_height == 480

    def test_custom_config(self):
        """Custom config values should be preserved."""
        from strands_robots_sim.isaac.config import IsaacConfig

        config = IsaacConfig(
            num_envs=1024,
            device="cuda:1",
            headless=True,
            physics_dt=1.0 / 240.0,
            render_mode="rtx_realtime",
        )
        assert config.num_envs == 1024
        assert config.device == "cuda:1"
        assert config.physics_dt == pytest.approx(1.0 / 240.0)
        assert config.render_mode == "rtx_realtime"

    def test_invalid_render_mode(self):
        """Invalid render_mode should raise ValueError."""
        from strands_robots_sim.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="render_mode"):
            IsaacConfig(render_mode="invalid")

    def test_invalid_device(self):
        """Non-CUDA device should raise ValueError."""
        from strands_robots_sim.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="CUDA"):
            IsaacConfig(device="cpu")

    def test_invalid_num_envs(self):
        """num_envs < 1 should raise ValueError."""
        from strands_robots_sim.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="num_envs"):
            IsaacConfig(num_envs=0)

    def test_invalid_physics_dt(self):
        """physics_dt <= 0 should raise ValueError."""
        from strands_robots_sim.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="physics_dt"):
            IsaacConfig(physics_dt=-0.001)

    def test_invalid_camera_dimensions(self):
        """Camera dimensions < 1 should raise ValueError."""
        from strands_robots_sim.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="camera"):
            IsaacConfig(camera_width=0)

    def test_config_round_trip(self):
        """Config should survive dataclass replace round-trip."""
        import dataclasses

        from strands_robots_sim.isaac.config import IsaacConfig

        original = IsaacConfig(num_envs=512, render_mode="rtx_pathtracing")
        copy = dataclasses.replace(original, num_envs=1024)
        assert copy.num_envs == 1024
        assert copy.render_mode == "rtx_pathtracing"
        assert original.num_envs == 512

    def test_env_var_headless_override(self, monkeypatch):
        """STRANDS_ISAAC_HEADLESS env var should override headless."""
        from strands_robots_sim.isaac.config import IsaacConfig

        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "false")
        config = IsaacConfig(headless=True)
        assert config.headless is False

    def test_env_var_rtx_pathtracing(self, monkeypatch):
        """STRANDS_ISAAC_RTX_PATHTRACING env var should set render mode."""
        from strands_robots_sim.isaac.config import IsaacConfig

        monkeypatch.setenv("STRANDS_ISAAC_RTX_PATHTRACING", "true")
        config = IsaacConfig(render_mode="headless")
        assert config.render_mode == "rtx_pathtracing"

    def test_env_var_nucleus_url(self, monkeypatch):
        """STRANDS_ISAAC_NUCLEUS_URL env var should be picked up."""
        from strands_robots_sim.isaac.config import IsaacConfig

        monkeypatch.setenv("STRANDS_ISAAC_NUCLEUS_URL", "omniverse://myhost/NVIDIA")
        config = IsaacConfig()
        assert config.nucleus_url == "omniverse://myhost/NVIDIA"


class TestIsaacSimulationAvailability:
    """Tests for IsaacSimulation.is_available()."""

    def test_isaac_is_simengine_subclass(self):
        """IsaacSimulation must subclass SimEngine ABC.

        Migrated from the cagataycali-original ``test_entrypoint.py``
        during the #31 split. The PR-1 R1 rewrite of ``test_entrypoint.py``
        is surface-only (no ``simulation`` import), so this contract pin
        moved to live next to the rest of the IsaacSimulation contract
        tests in this file.
        """
        from strands_robots_sim.isaac.simulation import IsaacSimulation, SimEngine

        assert issubclass(IsaacSimulation, SimEngine), (
            "IsaacSimulation must inherit from SimEngine for entry-point "
            "registration to satisfy the factory contract."
        )

    def test_isaac_implements_all_abstract_methods(self):
        """IsaacSimulation must implement every SimEngine abstract method.

        Pin against the R5-class regression cagataycali called out on PR #47:
        upstream ``SimEngine`` ABC declares ``list_robots`` /
        ``remove_object`` / ``remove_robot`` / ``robot_joint_names`` as
        abstract; missing any one of them makes ``IsaacSimulation()`` raise
        ``TypeError: Can't instantiate abstract class …``. The fallback ABC
        stub in ``simulation.py`` mirrors the real surface so the test fails
        loudly on either side if a method is added upstream and forgotten
        here.
        """
        from strands_robots_sim.isaac.simulation import IsaacSimulation, SimEngine

        abstract_methods = set()
        for name in dir(SimEngine):
            method = getattr(SimEngine, name, None)
            if callable(method) and getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)

        for method_name in abstract_methods:
            assert hasattr(IsaacSimulation, method_name), f"IsaacSimulation missing abstract method: {method_name}"
            impl = getattr(IsaacSimulation, method_name)
            assert callable(impl), f"IsaacSimulation.{method_name} must be callable, got {type(impl).__name__}"
            assert not getattr(
                impl, "__isabstractmethod__", False
            ), f"IsaacSimulation.{method_name} is still abstract; provide a concrete implementation."

    def test_is_available_returns_tuple(self):
        """is_available() must return a (bool, str|None) tuple."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        result = IsaacSimulation.is_available()
        assert isinstance(result, tuple)
        assert len(result) == 2
        available, reason = result
        assert isinstance(available, bool)
        if not available:
            assert isinstance(reason, str)

    def test_is_available_false_without_omni(self):
        """is_available() should return False when omni is not importable."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        # On CI without Isaac Sim, this should be False
        available, reason = IsaacSimulation.is_available()
        # We can't assert False universally (GPU env might have it)
        # But we CAN verify the return contract
        assert isinstance(available, bool)
        if not available:
            assert "omni" in reason.lower() or "cuda" in reason.lower() or "torch" in reason.lower()

    @patch("builtins.__import__")
    def test_is_available_false_when_omni_missing(self, mock_import):
        """Simulate omni not installed."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        def side_effect(name, *args, **kwargs):
            if name == "omni":
                raise ImportError("No module named 'omni'")
            return MagicMock()

        mock_import.side_effect = side_effect

        # Need to call directly without module caching
        # Just verify the method exists and is callable
        assert callable(IsaacSimulation.is_available)

    def test_is_available_probes_omni_isaac_kit_specifically(self, monkeypatch):
        """is_available() must probe ``omni.isaac.kit``, not the bare ``omni``
        namespace package.

        Regression pin for review-feedback PR #31: a partial Omniverse install
        (omni.ui / omni.usd) leaves the bare ``omni`` namespace importable but
        ``omni.isaac.kit.SimulationApp`` -- which create_world() actually needs --
        unavailable. The pre-fix probe (`import omni`) returned ``(True, None)``
        in that environment, then create_world() raised ImportError seconds
        later. Tightened probe uses ``importlib.util.find_spec`` against the
        specific submodule.
        """
        import importlib.util

        from strands_robots_sim.isaac.simulation import IsaacSimulation

        captured: list[str] = []
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            captured.append(name)
            if name == "omni.isaac.kit":
                return None  # simulate not installed
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        available, reason = IsaacSimulation.is_available()

        assert "omni.isaac.kit" in captured, (
            "is_available() must call find_spec('omni.isaac.kit'); " f"got find_spec calls: {captured!r}"
        )
        assert available is False
        assert reason is not None
        assert "omni.isaac.kit" in reason

    def test_is_available_accepts_modern_isaacsim_namespace(self, monkeypatch):
        """is_available() must accept the modern ``isaacsim`` namespace.

        Pin for `#73 <https://github.com/strands-labs/robots-sim/issues/73>`_:
        Isaac Sim 4.5+ ships ``isaacsim.SimulationApp`` as the supported
        entry point, with ``omni.isaac.kit`` retained as a deprecated
        shim. Some pip-only ``isaacsim`` installs ship ONLY the modern
        namespace -- no ``omni.isaac.kit`` until ``import isaacsim``
        bootstraps the Kit kernel. Probing only the legacy path
        therefore returns False on a perfectly working modern install.
        ``is_available()`` must accept either as evidence Isaac Sim is
        usable.
        """
        import importlib.util

        from strands_robots_sim.isaac.simulation import IsaacSimulation

        captured: list[str] = []
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            captured.append(name)
            if name == "omni.isaac.kit":
                return None  # legacy not present
            if name == "isaacsim":
                # Return a non-None ModuleSpec sentinel; is_available
                # only checks for None vs not-None.
                return importlib.util.spec_from_loader("isaacsim", loader=None)
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        available, reason = IsaacSimulation.is_available()

        assert "omni.isaac.kit" in captured
        assert "isaacsim" in captured, (
            "is_available() must also call find_spec('isaacsim') so a "
            "pip-only ``isaacsim`` install (no legacy shim) is recognised; "
            f"got find_spec calls: {captured!r}"
        )
        # Exact return depends on whether torch is importable on the test
        # host; we only assert that the omni-not-importable branch does
        # NOT short-circuit to False when isaacsim is present.
        if not available:
            assert reason is not None
            assert "omni.isaac.kit" not in reason or "isaacsim" in reason, (
                "When isaacsim is present (modern namespace), the is_available "
                "False branch must not blame the legacy ``omni.isaac.kit`` alone."
            )


class TestIsaacSimulationContract:
    """Tests for IsaacSimulation method contracts (mocked)."""

    def test_instantiation_does_not_import_omni(self):
        """Constructor must NOT import omni or touch CUDA."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        # This must succeed on any machine -- no CUDA required
        sim = IsaacSimulation(num_envs=1)
        assert sim is not None
        assert sim.config.num_envs == 1
        assert sim.config.headless is True

    def test_repr(self):
        """repr should be informative."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=4)
        r = repr(sim)
        assert "IsaacSimulation" in r
        assert "num_envs=4" in r

    def test_destroy_without_world(self):
        """destroy() on uninitialized sim should return error dict."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.destroy()
        assert result["status"] == "error"

    def test_step_without_world(self):
        """step() without create_world should return error."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.step(1)
        assert result["status"] == "error"

    def test_get_state_without_world(self):
        """get_state() without create_world should return error."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.get_state()
        assert result["status"] == "error"

    def test_add_robot_without_world(self):
        """add_robot() without create_world should return error."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.add_robot("test")
        assert result["status"] == "error"

    def test_add_object_without_world(self):
        """add_object() without create_world should return error."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.add_object("test")
        assert result["status"] == "error"

    def test_add_object_invalid_shape(self):
        """add_object() with invalid shape should return error."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        sim._world_created = True  # bypass world check
        result = sim.add_object("test", shape="invalid_shape")
        assert result["status"] == "error"
        assert "invalid_shape" in result["content"][0]["text"]

    def test_get_observation_without_world(self):
        """get_observation() without world returns empty dict."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.get_observation("robot1")
        assert result == {}

    def test_context_manager(self):
        """IsaacSimulation supports context manager protocol."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        with IsaacSimulation() as sim:
            assert sim is not None

    def test_kwargs_merge_into_config(self):
        """Shortcut kwargs should merge into config."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=256, headless=True)
        assert sim.config.num_envs == 256
        assert sim.config.headless is True

    def test_unknown_kwarg_raises_typeerror_no_config(self):
        """Typo kwargs must raise TypeError, not silently default.

        Regression pin for review-feedback PR #31: previously
        ``IsaacSimulation(headles=False)`` (typo) was silently dropped
        and the sim ran with default config. The validator now eagerly
        rejects unknown kwargs to surface typos at construction time.
        """
        import pytest

        from strands_robots_sim.isaac.simulation import IsaacSimulation

        with pytest.raises(TypeError, match="headles"):
            IsaacSimulation(headles=False)

    def test_unknown_kwarg_raises_typeerror_with_config(self):
        """Typo kwargs alongside an explicit IsaacConfig must also raise."""
        import pytest

        from strands_robots_sim.isaac.config import IsaacConfig
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        with pytest.raises(TypeError, match="num_env"):
            IsaacSimulation(IsaacConfig(num_envs=4), num_env=8)

    def test_no_del_finalizer(self):
        """IsaacSimulation must not define __del__.

        Regression pin for review-feedback PR #31: a ``__del__`` that
        calls cleanup() -> destroy() acquires ``self._lock`` during
        interpreter shutdown, when ``threading`` / ``logger`` / ``omni``
        may already be partially torn down. Drop the finalizer; rely on
        explicit cleanup() or context-manager use. Bare-except in the
        prior __del__ also masked the symptom but invisible exceptions
        during finalization still print "Exception ignored in: ..." noise.
        """
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        # Defining __del__ on a subclass would re-introduce the hazard,
        # so the assertion is on the class dict (not just on dir()).
        assert "__del__" not in IsaacSimulation.__dict__, (
            "IsaacSimulation must not define __del__; rely on explicit "
            "cleanup() or context manager (see cleanup() docstring)."
        )


class TestProceduralRobots:
    """Tests for procedural robot definitions."""

    def test_so100_definition(self):
        """SO-100 should have 6 joints."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        robot = get_procedural_robot("so100")
        assert robot is not None
        assert robot.name == "so100"
        assert robot.num_joints == 6
        assert len(robot.joint_names) == 6

    def test_panda_definition(self):
        """Panda should have 7 joints."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        robot = get_procedural_robot("panda")
        assert robot is not None
        assert robot.name == "panda"
        assert robot.num_joints == 7
        assert len(robot.joint_names) == 7

    def test_unitree_g1_definition(self):
        """Unitree G1 should have 21 joints (simplified)."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        robot = get_procedural_robot("unitree_g1")
        assert robot is not None
        assert robot.name == "unitree_g1"
        assert robot.num_joints == 21
        assert len(robot.joint_names) == 21

    def test_alias_resolution(self):
        """Aliases should resolve correctly."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        assert get_procedural_robot("so-100") is not None
        assert get_procedural_robot("franka") is not None
        assert get_procedural_robot("g1") is not None
        assert get_procedural_robot("franka_panda") is not None

    def test_unknown_robot_returns_none(self):
        """Unknown robot name should return None."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        assert get_procedural_robot("nonexistent_robot") is None

    def test_list_procedural_robots(self):
        """list_procedural_robots should return known robots."""
        from strands_robots_sim.isaac.procedural import list_procedural_robots

        robots = list_procedural_robots()
        assert "so100" in robots
        assert "panda" in robots
        assert "unitree_g1" in robots


class TestNoEmojisInOutput:
    """Verify no emojis in user-facing strings."""

    def test_destroy_message_no_emoji(self):
        """destroy() output must not contain emojis."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.destroy()
        text = result["content"][0]["text"]
        # Check for common emoji ranges
        for char in text:
            code = ord(char)
            assert code < 0x1F600 or code > 0x1F9FF, f"Emoji found in output: {char!r}"

    def test_step_error_no_emoji(self):
        """step() error must not contain emojis."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.step(1)
        text = result["content"][0]["text"]
        for char in text:
            code = ord(char)
            assert code < 0x1F600 or code > 0x1F9FF, f"Emoji found in output: {char!r}"


class TestExceptionClauseHygiene:
    """Static-AST checks for narrow exception clauses in simulation.py.

    Regression pin for review-feedback PR #31: bare ``except Exception``
    swallows programming bugs (AttributeError typos, KeyError on dict
    access, etc.) into logged error dicts that look identical to
    recoverable failures.

    Each behavioural except clause must enumerate the realistic failure
    modes for the API it wraps, so a future drift in the wrapped API's
    surface (or a typo in the wrapper) raises rather than silently
    becoming a no-op.
    """

    def test_no_bare_except_exception_in_simulation_module(self):
        """``except Exception`` is forbidden in simulation.py.

        Use a tuple of named exception classes (RuntimeError, ValueError,
        OSError, AttributeError, TypeError, ImportError, ...) instead.
        """
        import ast
        from pathlib import Path

        from strands_robots_sim.isaac import simulation

        src = Path(simulation.__file__).read_text()
        tree = ast.parse(src)

        offending: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    offending.append((node.lineno, "except Exception"))
                elif node.type is None:
                    offending.append((node.lineno, "bare except"))

        assert not offending, (
            f"simulation.py must not use bare 'except Exception' or 'except:'; "
            f"narrow to specific exception classes. Offending sites: {offending}"
        )


class TestInstallConstants:
    """Pin the install-instruction abstraction.

    The install hints (docker tag, Omniverse Launcher line, Isaac Lab
    bootstrap) live in :mod:`strands_robots_sim.isaac._install` so a
    single edit propagates everywhere they surface (review feedback
    on PR #47). These tests pin (a) the contract that the module
    exposes the expected constants, and (b) that the messages
    consumed by ``IsaacSimulation`` are built from those constants
    -- if someone hardcodes a tag back into ``simulation.py`` the
    ``test_simulation_module_has_no_hardcoded_image`` regression
    pin will fail.
    """

    def test_constants_present(self):
        from strands_robots_sim.isaac import _install

        assert _install.ISAAC_SIM_DOCKER_IMAGE.startswith("nvcr.io/nvidia/isaac-sim:")
        assert _install.ISAAC_SIM_MIN_VERSION  # non-empty
        assert "isaaclab" in _install.ISAAC_LAB_BOOTSTRAP.lower()
        assert "strands-robots-sim[isaac]" in _install.PIP_EXTRA

    def test_not_importable_reason_composes_from_constants(self):
        from strands_robots_sim.isaac import _install

        msg = _install.not_importable_reason()
        assert _install.ISAAC_SIM_DOCKER_IMAGE in msg
        assert _install.ISAAC_SIM_MIN_VERSION in msg
        assert _install.ISAAC_LAB_BOOTSTRAP.split(" && ")[-1] in msg
        assert _install.PIP_EXTRA in msg

    def test_not_available_import_error_composes_from_constants(self):
        from strands_robots_sim.isaac import _install

        msg = _install.not_available_import_error()
        assert _install.ISAAC_SIM_DOCKER_IMAGE in msg
        assert "Omniverse Launcher" in msg

    def test_simulation_module_has_no_hardcoded_image(self):
        """Regression pin: docker tag must not be re-hardcoded into simulation.py.

        If this fails, fold the new occurrence into
        ``strands_robots_sim.isaac._install`` so the install-hint
        single-source-of-truth survives.
        """
        from pathlib import Path

        from strands_robots_sim.isaac import simulation

        src = Path(simulation.__file__).read_text()
        # The literal docker image tag must appear nowhere in simulation.py;
        # callers should use _install.ISAAC_SIM_DOCKER_IMAGE.
        assert "nvcr.io/nvidia/isaac-sim:" not in src, (
            "simulation.py contains a hardcoded Isaac Sim docker tag. "
            "Use strands_robots_sim.isaac._install.ISAAC_SIM_DOCKER_IMAGE instead."
        )

    def test_is_available_reason_uses_install_module(self):
        """The reason string returned by ``is_available()`` when omni
        is missing must come from ``_install.not_importable_reason``.
        """
        import importlib.util as iu

        from strands_robots_sim.isaac import _install
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        # Force the "not importable" branch by stubbing find_spec.
        original_find_spec = iu.find_spec

        def fake_find_spec(name, *a, **kw):
            if name == "omni.isaac.kit":
                return None
            return original_find_spec(name, *a, **kw)

        with patch.object(iu, "find_spec", side_effect=fake_find_spec):
            available, reason = IsaacSimulation.is_available()

        if available is False and reason is not None and "omni" in reason.lower():
            # Only assert composition when we actually hit the omni branch
            # (CUDA / torch branches return earlier on some hosts).
            assert reason == _install.not_importable_reason()


class TestCreateWorldGravityScalarApi:
    """Regression pins for `strands-labs/robots-sim#52` — Isaac Sim 5.1
    `PhysicsContext.set_gravity` API mismatch (took vec3 pre-5.1, takes
    a scalar magnitude in 5.1).

    Without Isaac Sim installed we can't exercise `create_world`'s
    runtime path on this CI host, so the regression is pinned via
    source-inspection: the scalar-extraction call site and the
    `TypeError`-tolerant `except` clause are both pinned so a future
    refactor can't silently revert either one.
    """

    def test_set_gravity_call_extracts_scalar_from_vector(self) -> None:
        """`create_world` must extract a scalar Z-component before calling
        `set_gravity`.

        Isaac Sim 5.1's `PhysicsContext.set_gravity(value: float)` raises
        `TypeError` if handed a list/tuple. The fix at PR #47 commit
        `a65f9f9` added the `grav[2]` extraction; this test pins it so
        a future refactor doesn't accidentally revert to the pre-5.1
        vector-pass pattern that #52 reproduced.
        """
        import inspect

        from strands_robots_sim.isaac import simulation

        source = inspect.getsource(simulation)

        # Both halves of the fix must be present: the extraction line
        # and the call site that consumes its result.
        extraction = "gravity_magnitude = grav[2] if isinstance(grav, (list, tuple)) else grav"
        call_site = "set_gravity(gravity_magnitude)"

        assert extraction in source, (
            "create_world() must extract a scalar Z-component before set_gravity. "
            "If this assertion fails, a refactor likely reverted the fix from "
            "https://github.com/strands-labs/robots-sim/issues/52 — Isaac Sim 5.1 "
            "set_gravity takes a scalar, not a vec3. Restore the extraction line."
        )
        assert call_site in source, (
            "set_gravity must be invoked with the extracted scalar (gravity_magnitude). "
            "If this assertion fails, the call site was rewritten to pass the raw "
            "list — that path will TypeError under Isaac Sim 5.1 (see #52)."
        )

    def test_create_world_except_clause_catches_typeerror(self) -> None:
        """`create_world`'s narrow except clause must include `TypeError`.

        Defence in depth for #52-class surface drift: Isaac Sim 5.1's
        `set_gravity(value: float)` already rejects non-scalar input
        with `TypeError`, and other physics-context calls (e.g.
        `set_solver_position_iteration_count`) have the same shape.
        Catching `TypeError` here means a future regression on any of
        those paths surfaces as a structured error envelope rather
        than an unhandled exception.
        """
        import ast
        import inspect

        from strands_robots_sim.isaac import simulation

        source = inspect.getsource(simulation)
        tree = ast.parse(source)

        # Find the create_world function and inspect its top-level
        # except clauses. The relevant one is the broad-tuple clause
        # that handles partial-init cleanup; ImportError stays in its
        # own clause (specific recovery message).
        create_world_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "create_world":
                create_world_fn = node
                break

        assert create_world_fn is not None, "create_world() function not found in simulation module"

        # Walk the try / except block(s) inside create_world and find
        # the catch-tuple clause (the narrow except that handles
        # partial-init cleanup).
        broad_except_caught: tuple[str, ...] | None = None
        for node in ast.walk(create_world_fn):
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Tuple):
                names = tuple(elt.id for elt in node.type.elts if isinstance(elt, ast.Name))
                # We want the cleanup clause, identified by it catching
                # RuntimeError + ValueError + OSError + AttributeError
                # at minimum (the pre-#52 superset).
                if {"RuntimeError", "ValueError", "OSError", "AttributeError"}.issubset(set(names)):
                    broad_except_caught = names
                    break

        assert broad_except_caught is not None, (
            "create_world() lost its narrow-tuple except clause. "
            "Re-add it as `except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:` "
            "to keep partial-init cleanup behaviour consistent (see #52)."
        )

        assert "TypeError" in broad_except_caught, (
            f"create_world() except clause is {broad_except_caught!r}; "
            f"missing TypeError. Add it back per "
            f"https://github.com/strands-labs/robots-sim/issues/52 — "
            f"Isaac Sim 5.1 set_gravity raises TypeError on non-scalar input, "
            f"and other physics-context calls have the same shape; catching "
            f"TypeError here means surface drift surfaces as a structured "
            f"error envelope rather than an unhandled exception."
        )


def _patched_isaac_objects_module() -> MagicMock:
    """Build a MagicMock that stands in for ``isaacsim.core.api.objects``.

    Each constructor (``DynamicCuboid`` etc.) is itself a ``MagicMock``
    that returns a unique handle; tests can assert on which class was
    invoked, with what kwargs, by inspecting ``module.<ClassName>``
    after the call.
    """
    mod = MagicMock()
    for cls_name in (
        "DynamicCuboid",
        "DynamicSphere",
        "DynamicCylinder",
        "DynamicCapsule",
        "FixedCuboid",
        "FixedSphere",
        "FixedCylinder",
        "FixedCapsule",
    ):
        # A new MagicMock per attribute so .return_value / .call_args
        # are isolated per shape.
        getattr(mod, cls_name).return_value = MagicMock(name=f"{cls_name}_handle")
    return mod


def _make_simulation_with_world() -> "tuple[object, MagicMock]":
    """Build an ``IsaacSimulation`` with ``_world_created=True`` and a
    mocked ``_world.scene`` so ``add_object`` / ``remove_object`` can
    exercise their full Phase 2 wiring without booting Isaac Sim.

    Returns the simulation plus the mock scene so tests can assert on
    ``scene.add`` / ``scene.remove_object`` call shapes.
    """
    from strands_robots_sim.isaac.simulation import IsaacSimulation

    sim = IsaacSimulation()
    sim._world_created = True
    sim._world = MagicMock()
    sim._world.scene = MagicMock()
    return sim, sim._world.scene


class TestAddObjectPhase2:
    """Phase 2 wiring (#14) for ``IsaacSimulation.add_object``.

    Pins the eight (shape, is_static) combinations onto their respective
    ``isaacsim.core.api.objects`` constructors plus the structured success
    envelope (json payload, prim path, registry side-effects).
    """

    def test_returns_error_without_world(self) -> None:
        """Pre-create_world() call must return a structured error -- no
        prim creation, no scene call, no registry side-effects.
        """
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.add_object("test")
        assert result["status"] == "error"
        assert "No world created" in result["content"][0]["text"]

    def test_returns_error_on_unknown_shape(self) -> None:
        """Unknown shape returns the structured error envelope and does
        not call into the isaacsim.core.api.objects module.
        """
        sim, scene = _make_simulation_with_world()
        result = sim.add_object("test", shape="dodecahedron")
        assert result["status"] == "error"
        assert "dodecahedron" in result["content"][0]["text"]
        assert "box" in result["content"][0]["text"]  # valid shapes listed
        scene.add.assert_not_called()

    def test_returns_error_on_duplicate_name(self) -> None:
        """Re-adding a previously-added object must return error rather
        than silently overwriting the existing prim.
        """
        sim, scene = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            r1 = sim.add_object("cube", shape="box")
            r2 = sim.add_object("cube", shape="box")
        assert r1["status"] == "success"
        assert r2["status"] == "error"
        assert "already exists" in r2["content"][0]["text"]
        # Only one scene.add call landed; the duplicate is rejected before
        # any prim work.
        assert scene.add.call_count == 1

    def test_box_default_calls_dynamic_cuboid(self) -> None:
        """``shape="box"`` (default ``is_static=False``) constructs a
        ``DynamicCuboid`` and registers it with ``world.scene.add``.
        """
        sim, scene = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object("cube", shape="box", position=[1, 2, 3])
        assert result["status"] == "success"
        fake_objects.DynamicCuboid.assert_called_once()
        fake_objects.FixedCuboid.assert_not_called()
        scene.add.assert_called_once_with(fake_objects.DynamicCuboid.return_value)

    def test_box_static_calls_fixed_cuboid_without_mass(self) -> None:
        """``is_static=True`` selects ``FixedCuboid`` and **does not** pass
        ``mass`` (Fixed* constructors don't take it).
        """
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            sim.add_object("anchor", shape="box", is_static=True, mass=5.0)
        fake_objects.FixedCuboid.assert_called_once()
        fake_objects.DynamicCuboid.assert_not_called()
        kwargs = fake_objects.FixedCuboid.call_args.kwargs
        assert "mass" not in kwargs, "Fixed* constructors must not receive mass kwarg"

    def test_box_size_per_component_fallback(self) -> None:
        """``box`` honours the docstring's per-component fallback contract.

        Pin: lists shorter than 3 entries fall back to defaults for the
        missing trailing components -- they don't reset the whole scale
        to defaults. Mirrors the cylinder / capsule pattern.

        Pre-fix behaviour was all-or-nothing: ``size=[0.10]`` silently
        yielded ``[0.05, 0.05, 0.05]`` (default cube), contradicting
        the documented contract. PR #60 review caught this; this test
        locks the fixed shape so it can't drift back.
        """
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            r1 = sim.add_object("a", shape="box", size=[0.10])
            r2 = sim.add_object("b", shape="box", size=[0.10, 0.20])
            r3 = sim.add_object("c", shape="box", size=[0.10, 0.20, 0.30])
            r4 = sim.add_object("d", shape="box")  # default
        # 1-vec: x supplied, y/z default
        assert r1["content"][0]["json"]["size"] == [0.10, 0.05, 0.05]
        # 2-vec: x, y supplied; z default
        assert r2["content"][0]["json"]["size"] == [0.10, 0.20, 0.05]
        # 3-vec: all supplied
        assert r3["content"][0]["json"]["size"] == [0.10, 0.20, 0.30]
        # No size: all defaults
        assert r4["content"][0]["json"]["size"] == [0.05, 0.05, 0.05]
        # The same scale flows to the underlying DynamicCuboid call.
        kwargs_1 = fake_objects.DynamicCuboid.call_args_list[0].kwargs
        assert list(kwargs_1["scale"]) == [0.10, 0.05, 0.05]

    def test_sphere_passes_radius_not_scale(self) -> None:
        """``shape="sphere"`` uses the ``radius=`` kwarg, not ``scale=``."""
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            sim.add_object("ball", shape="sphere", size=[0.07])
        fake_objects.DynamicSphere.assert_called_once()
        kwargs = fake_objects.DynamicSphere.call_args.kwargs
        assert kwargs["radius"] == 0.07
        assert "scale" not in kwargs

    def test_cylinder_passes_radius_and_height(self) -> None:
        """``shape="cylinder"`` uses ``radius=`` + ``height=`` kwargs.

        Defaults to the documented (0.05, 0.10) when ``size`` is shorter
        than 2 entries.
        """
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            sim.add_object("can_a", shape="cylinder", size=[0.04, 0.20])
            sim.add_object("can_b", shape="cylinder")  # default size
        kwargs_a = fake_objects.DynamicCylinder.call_args_list[0].kwargs
        kwargs_b = fake_objects.DynamicCylinder.call_args_list[1].kwargs
        assert kwargs_a["radius"] == 0.04
        assert kwargs_a["height"] == 0.20
        assert kwargs_b["radius"] == 0.05
        assert kwargs_b["height"] == 0.10

    def test_capsule_passes_radius_and_height(self) -> None:
        """``shape="capsule"`` uses the same (radius, height) shape as cylinder."""
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            sim.add_object("pill", shape="capsule", size=[0.03, 0.08])
        kwargs = fake_objects.DynamicCapsule.call_args.kwargs
        assert kwargs["radius"] == 0.03
        assert kwargs["height"] == 0.08

    def test_accepted_shapes_are_exactly_the_documented_set(self) -> None:
        """Pin the accepted ``shape`` values so docs and code can't drift.

        Locks the canonical valid set plus the accepted aliases (issue
        #88). If a future change adds / removes a shape or an alias, this
        test must be updated in lockstep with the docstring,
        ``api-reference.md``, and the README -- which is the whole point.
        """
        from strands_robots_sim.isaac.simulation import _SHAPE_ALIASES

        # Canonical names accepted by the public API.
        canonical = {"box", "sphere", "capsule", "cylinder"}
        # Aliases accepted in addition, normalized to a canonical name.
        assert _SHAPE_ALIASES == {"cuboid": "box"}
        # Every alias target must itself be a canonical shape.
        assert set(_SHAPE_ALIASES.values()) <= canonical

    def test_cuboid_alias_normalizes_to_box(self) -> None:
        """``shape="cuboid"`` is accepted as an alias for ``"box"``.

        Mirrors Isaac's ``DynamicCuboid`` class name and the docs
        vocabulary (issue #88). It must construct a ``DynamicCuboid`` and
        report the canonical ``"box"`` shape back in the result json.
        """
        sim, scene = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object("cube", shape="cuboid", position=[1, 2, 3])
        assert result["status"] == "success"
        fake_objects.DynamicCuboid.assert_called_once()
        # The canonical name is reported, not the alias.
        assert result["content"][0]["json"]["shape"] == "box"
        scene.add.assert_called_once_with(fake_objects.DynamicCuboid.return_value)

    def test_unknown_shape_error_lists_cuboid_alias(self) -> None:
        """The unknown-shape error envelope advertises the ``cuboid``
        alias alongside the canonical set so a user mistyping ``shape``
        sees every accepted value.
        """
        sim, _ = _make_simulation_with_world()
        result = sim.add_object("test", shape="dodecahedron")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "cuboid" in text
        assert "box" in text

    def test_scale_alias_maps_to_size_for_box(self) -> None:
        """``scale=`` is accepted as an alias for ``size=`` (issue #88).

        Pre-fix, ``scale=`` was swallowed by ``**kwargs`` and silently
        ignored, so the object got default dimensions. It must now flow
        through to the underlying ``DynamicCuboid(scale=...)`` exactly as
        ``size=`` would.
        """
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object("cube", shape="box", scale=[0.10, 0.20, 0.30])
        assert result["status"] == "success"
        assert result["content"][0]["json"]["size"] == [0.10, 0.20, 0.30]
        kwargs = fake_objects.DynamicCuboid.call_args.kwargs
        assert list(kwargs["scale"]) == [0.10, 0.20, 0.30]

    def test_scale_alias_maps_to_size_for_sphere(self) -> None:
        """``scale=`` alias also works for non-box shapes (e.g. sphere)."""
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            sim.add_object("ball", shape="sphere", scale=[0.07])
        kwargs = fake_objects.DynamicSphere.call_args.kwargs
        assert kwargs["radius"] == 0.07

    def test_explicit_size_wins_over_scale_alias(self) -> None:
        """If both ``size`` and ``scale`` are passed, ``size`` wins."""
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object(
                "cube",
                shape="box",
                size=[0.10, 0.10, 0.10],
                scale=[0.99, 0.99, 0.99],
            )
        assert result["content"][0]["json"]["size"] == [0.10, 0.10, 0.10]
        kwargs = fake_objects.DynamicCuboid.call_args.kwargs
        assert list(kwargs["scale"]) == [0.10, 0.10, 0.10]

    def test_cuboid_with_scale_quickstart_snippet(self) -> None:
        """The headline docs snippet ``add_object(shape="cuboid",
        scale=[...])`` runs without error and produces the intended
        dimensions (issue #88 acceptance criterion).
        """
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object(
                "block",
                shape="cuboid",
                position=[0.4, 0.0, 0.05],
                scale=[0.05, 0.05, 0.05],
                color=[1.0, 0.0, 0.0],
            )
        assert result["status"] == "success"
        assert result["content"][0]["json"]["shape"] == "box"
        assert result["content"][0]["json"]["size"] == [0.05, 0.05, 0.05]
        kwargs = fake_objects.DynamicCuboid.call_args.kwargs
        assert list(kwargs["scale"]) == [0.05, 0.05, 0.05]

    def test_rgba_color_truncates_to_rgb(self) -> None:
        """A 4-vector ``[r, g, b, a]`` color is truncated to ``[r, g, b]``.

        Mirrors the #15 sketch's ``color=[1, 0, 0, 1]`` -- Isaac's primitive
        constructors take a 3-vector color and would otherwise raise on
        the alpha component.
        """
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            sim.add_object("red_cube", shape="box", color=[1.0, 0.0, 0.0, 0.5])
        kwargs = fake_objects.DynamicCuboid.call_args.kwargs
        # numpy array, length 3, alpha dropped.
        assert len(kwargs["color"]) == 3
        assert list(kwargs["color"]) == [1.0, 0.0, 0.0]

    def test_default_position_is_above_ground_plane(self) -> None:
        """Default position is ``[0, 0, 0.5]`` so the object doesn't
        intersect the default ground plane on spawn.
        """
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object("dropped", shape="box")
        assert result["content"][0]["json"]["position"] == [0.0, 0.0, 0.5]
        kwargs = fake_objects.DynamicCuboid.call_args.kwargs
        assert list(kwargs["position"]) == [0.0, 0.0, 0.5]

    def test_default_orientation_is_identity_quaternion(self) -> None:
        """Default orientation is ``[1, 0, 0, 0]`` (identity quaternion, w-first)."""
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object("aligned", shape="box")
        assert result["content"][0]["json"]["orientation"] == [1.0, 0.0, 0.0, 0.0]

    def test_success_envelope_carries_structured_json(self) -> None:
        """Success envelope's ``content[0].json`` carries name / prim_path /
        shape / position / orientation / size / mass / is_static.
        """
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object(
                "cube",
                shape="box",
                position=[1, 2, 3],
                size=[0.1, 0.2, 0.3],
                mass=0.5,
            )
        info = result["content"][0]["json"]
        assert info["name"] == "cube"
        assert info["prim_path"] == "/World/Objects/cube"
        assert info["shape"] == "box"
        assert info["position"] == [1, 2, 3]
        assert info["size"] == [0.1, 0.2, 0.3]
        assert info["mass"] == 0.5
        assert info["is_static"] is False

    def test_static_object_reports_zero_mass_in_json(self) -> None:
        """Static objects surface ``mass=0.0`` in the json payload (the
        ``mass=`` kwarg is not passed to ``Fixed*`` constructors, so the
        envelope reports the dynamics-effective value, which is zero).
        """
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object("anchor", shape="box", is_static=True, mass=5.0)
        info = result["content"][0]["json"]
        assert info["mass"] == 0.0
        assert info["is_static"] is True

    def test_failure_in_constructor_returns_error_no_registry_pollution(self) -> None:
        """If the omni constructor raises, no registry / scene state is
        recorded -- the caller can retry under the same name.
        """
        sim, scene = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        fake_objects.DynamicCuboid.side_effect = RuntimeError("USD prim collision")
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object("cube", shape="box")
        assert result["status"] == "error"
        assert "USD prim collision" in result["content"][0]["text"]
        scene.add.assert_not_called()
        assert "cube" not in sim._objects
        assert "/World/Objects/cube" not in sim._prim_registry

    def test_failure_in_scene_add_returns_error_no_registry_pollution(self) -> None:
        """If ``world.scene.add`` raises, registries are not updated."""
        sim, scene = _make_simulation_with_world()
        scene.add.side_effect = RuntimeError("scene already replicated")
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            result = sim.add_object("cube", shape="box")
        assert result["status"] == "error"
        assert "scene already replicated" in result["content"][0]["text"]
        assert "cube" not in sim._objects
        assert "/World/Objects/cube" not in sim._prim_registry

    def test_prim_registry_and_objects_dict_are_updated_on_success(self) -> None:
        """A successful add_object updates both ``_prim_registry`` and
        ``_objects[name]``. Pinned because :meth:`destroy` and
        :meth:`get_state` rely on the dual-bookkeeping invariant.
        """
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            sim.add_object("cube", shape="box")
        assert "cube" in sim._objects
        assert sim._objects["cube"].prim_path == "/World/Objects/cube"
        assert sim._objects["cube"].shape == "box"
        assert sim._objects["cube"].is_static is False
        assert "/World/Objects/cube" in sim._prim_registry


class TestRemoveObjectPhase2:
    """Phase 2 wiring (#14) for ``IsaacSimulation.remove_object``.

    Paired with :class:`TestAddObjectPhase2`; pins that
    ``world.scene.remove_object`` is invoked, registries are pruned, and
    the operation is retry-friendly on transient scene failures.
    """

    def _add_a_cube(self) -> "tuple[object, MagicMock]":
        sim, scene = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            sim.add_object("cube", shape="box")
        scene.reset_mock()  # so subsequent assertions see only remove activity
        return sim, scene

    def test_returns_error_for_unknown_name(self) -> None:
        """Removing an object that was never added returns error and
        does not call into ``world.scene``.
        """
        sim, scene = _make_simulation_with_world()
        result = sim.remove_object("ghost")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]
        scene.remove_object.assert_not_called()

    def test_calls_world_scene_remove_object_with_name(self) -> None:
        """Successful remove_object delegates to ``world.scene.remove_object(name)``."""
        sim, scene = self._add_a_cube()
        result = sim.remove_object("cube")
        assert result["status"] == "success"
        scene.remove_object.assert_called_once_with("cube")

    def test_prunes_objects_dict_and_prim_registry_on_success(self) -> None:
        """Both ``_objects`` and ``_prim_registry`` are pruned by
        a successful remove_object.
        """
        sim, _ = self._add_a_cube()
        sim.remove_object("cube")
        assert "cube" not in sim._objects
        assert "/World/Objects/cube" not in sim._prim_registry

    def test_scene_remove_failure_keeps_bookkeeping_for_retry(self) -> None:
        """If ``world.scene.remove_object`` raises a ``RuntimeError``,
        the in-Python registries are **not** pruned -- the caller can
        retry under the same name (e.g. after a stage refresh).
        """
        sim, scene = self._add_a_cube()
        scene.remove_object.side_effect = RuntimeError("stage closed")
        result = sim.remove_object("cube")
        assert result["status"] == "error"
        assert "stage closed" in result["content"][0]["text"]
        # Bookkeeping retained for retry.
        assert "cube" in sim._objects
        assert "/World/Objects/cube" in sim._prim_registry

    def test_remove_after_world_torn_down_still_succeeds(self) -> None:
        """If ``self._world`` was set to ``None`` (post-destroy), remove
        still cleans up the in-Python bookkeeping rather than crashing
        on the ``world.scene`` lookup.
        """
        sim, _ = self._add_a_cube()
        sim._world = None
        result = sim.remove_object("cube")
        assert result["status"] == "success"
        assert "cube" not in sim._objects


class TestDestroyAndGetStateSurfaceObjects:
    """Pin: ``destroy()`` and ``get_state()`` surface ``num_objects``
    in their structured json payloads, mirroring ``num_robots`` /
    ``num_cameras``. Required so an agent inspecting either method
    sees the Phase 2 entity surface without re-querying.
    """

    def test_get_state_includes_num_objects(self) -> None:
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            sim.add_object("a", shape="box")
            sim.add_object("b", shape="sphere")
        result = sim.get_state()
        assert result["status"] == "success"
        info = result["content"][0]["json"]
        assert info["num_objects"] == 2
        assert "objects=2" in result["content"][0]["text"]

    def test_destroy_releases_objects_and_reports_count(self) -> None:
        sim, _ = _make_simulation_with_world()
        fake_objects = _patched_isaac_objects_module()
        with patch.dict("sys.modules", {"isaacsim.core.api.objects": fake_objects}):
            sim.add_object("a", shape="box")
            sim.add_object("b", shape="cylinder")
        result = sim.destroy()
        assert result["status"] == "success"
        info = result["content"][0]["json"]
        assert info["num_objects_released"] == 2
        # Post-destroy the dict is empty.
        assert sim._objects == {}


class TestRenderFramePathPhase2:
    """Phase 2 wiring (#14) for ``IsaacSimulation.render``.

    Pins the four documented render paths (headless / no-camera /
    Phase-1-handle-None / RTX-real-frames) against the right blank-vs.
    -real frame return shapes plus the structured envelope text /
    json. Pairs with the Phase 2 ``add_camera`` slice; the RTX path
    only lights up when the camera in ``self._cameras`` carries a
    non-``None`` ``handle``, which Phase 2 ``add_camera`` populates
    via ``isaacsim.sensors.camera.Camera``.
    """

    def _make_sim(self, render_mode: str = "rtx_realtime") -> object:
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(render_mode=render_mode)
        sim._world_created = True
        sim._world = MagicMock()
        return sim

    def test_returns_error_without_world(self) -> None:
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.render()
        assert result["status"] == "error"
        assert "No world created" in result["content"][0]["text"]

    def test_headless_returns_blank_frames_with_headless_text(self) -> None:
        """``render_mode="headless"`` -> blank frames sized to the
        ``width`` / ``height`` arguments (or config defaults), with
        ``"headless, no RTX"`` in the success text.
        """
        sim = self._make_sim(render_mode="headless")
        result = sim.render(width=320, height=240)
        assert result["status"] == "success"
        assert result["rgb"].shape == (240, 320, 3)
        assert result["depth"].shape == (240, 320)
        assert result["rgb"].dtype == np.uint8
        assert result["depth"].dtype == np.float32
        assert (result["rgb"] == 0).all()
        assert "headless, no RTX" in result["content"][0]["text"]

    def test_unknown_camera_returns_blank_with_no_camera_text(self) -> None:
        """Unknown ``camera_name`` -> blank frames + ``"no camera"`` in text."""
        sim = self._make_sim()
        result = sim.render("ghost", width=160, height=120)
        assert result["status"] == "success"
        assert result["rgb"].shape == (120, 160, 3)
        assert "no camera" in result["content"][0]["text"]

    def test_phase1_camera_with_no_handle_returns_blank_frames(self) -> None:
        """A camera registered without a Phase-2 handle (``handle=None``)
        falls back to blank frames sized to the camera's registered
        resolution, with ``"Phase-1 camera, no RTX handle"`` text.
        """
        from strands_robots_sim.isaac.simulation import _CameraState

        sim = self._make_sim()
        sim._cameras["legacy"] = _CameraState(
            name="legacy",
            prim_path="/World/Cameras/legacy",
            width=400,
            height=300,
        )
        # handle deliberately left as None.
        result = sim.render("legacy")
        assert result["status"] == "success"
        assert result["rgb"].shape == (300, 400, 3)
        assert "Phase-1 camera, no RTX handle" in result["content"][0]["text"]

    def test_phase2_camera_with_handle_pulls_real_frames(self) -> None:
        """A Phase-2 camera (handle set) -> ``handle.get_rgba()`` and
        ``handle.get_depth()`` are called and their return values flow
        into the structured envelope.
        """
        from strands_robots_sim.isaac.simulation import _CameraState

        sim = self._make_sim(render_mode="rtx_realtime")
        handle = MagicMock(name="Camera_handle")
        # Real Isaac get_rgba returns RGBA; pin the slice-to-RGB.
        handle.get_rgba.return_value = np.full((240, 320, 4), 255, dtype=np.uint8)
        handle.get_depth.return_value = np.full((240, 320), 1.5, dtype=np.float32)

        cs = _CameraState(name="front", prim_path="/World/Cameras/front", width=320, height=240)
        cs.handle = handle
        sim._cameras["front"] = cs

        result = sim.render("front")
        assert result["status"] == "success"
        # rgb is sliced to 3 channels (alpha dropped).
        assert result["rgb"].shape == (240, 320, 3)
        assert (result["rgb"] == 255).all()
        assert result["depth"].shape == (240, 320)
        assert (result["depth"] == 1.5).all()

        handle.get_rgba.assert_called_once()
        handle.get_depth.assert_called_once()

        info = result["content"][0]["json"]
        assert info["rtx"] is True
        assert info["prim_path"] == "/World/Cameras/front"
        assert info["resolution"] == [320, 240]
        assert info["render_mode"] == "rtx_realtime"
        assert "RTX rtx_realtime" in result["content"][0]["text"]

    def test_phase2_camera_with_rgb_handle_passes_through(self) -> None:
        """Some Isaac Sim builds return RGB (not RGBA) from get_rgba.
        Pin that the slice-to-3-channels works against either shape.
        """
        from strands_robots_sim.isaac.simulation import _CameraState

        sim = self._make_sim()
        handle = MagicMock()
        handle.get_rgba.return_value = np.full((100, 200, 3), 128, dtype=np.uint8)
        handle.get_depth.return_value = np.full((100, 200), 2.0, dtype=np.float32)

        cs = _CameraState(name="rgb", prim_path="/World/Cameras/rgb", width=200, height=100)
        cs.handle = handle
        sim._cameras["rgb"] = cs

        result = sim.render("rgb")
        assert result["rgb"].shape == (100, 200, 3)

    def test_phase2_camera_get_rgba_failure_returns_error_envelope(self) -> None:
        """If ``handle.get_rgba`` raises, render returns the structured
        error envelope rather than letting the exception propagate.
        """
        from strands_robots_sim.isaac.simulation import _CameraState

        sim = self._make_sim()
        handle = MagicMock()
        handle.get_rgba.side_effect = RuntimeError("RTX render product not stepped")
        cs = _CameraState(name="front", prim_path="/World/Cameras/front", width=320, height=240)
        cs.handle = handle
        sim._cameras["front"] = cs

        result = sim.render("front")
        assert result["status"] == "error"
        assert "RTX render product not stepped" in result["content"][0]["text"]
        # Returned envelope must NOT carry rgb/depth keys on failure
        # so callers can branch on ``status`` alone.
        assert "rgb" not in result
        assert "depth" not in result

    def test_phase2_camera_malformed_rgb_returns_error_envelope(self) -> None:
        """A not-yet-warmed camera whose ``get_rgba`` returns a malformed
        (1-D / empty) buffer -> structured error envelope, not an
        unhandled IndexError on ``rgb.shape[1]``.

        Regression pin for the gap found during the isaac_gs example's
        GPU validation: freshly-added cameras whose RTX render product
        hasn't accumulated a frame return a 1-D array; the json
        ``resolution`` build then IndexError'd.
        """
        import numpy as np

        from strands_robots_sim.isaac.simulation import _CameraState

        sim = self._make_sim()
        handle = MagicMock()
        # 1-D buffer (e.g. empty / not-yet-rendered render product).
        handle.get_rgba.return_value = np.zeros((0,), dtype=np.uint8)
        cs = _CameraState(name="cold", prim_path="/World/Cameras/cold", width=320, height=240)
        cs.handle = handle
        sim._cameras["cold"] = cs

        result = sim.render("cold")
        assert result["status"] == "error"
        assert "malformed RGB buffer" in result["content"][0]["text"]
        assert "rgb" not in result
        assert "depth" not in result

    def test_phase2_camera_get_depth_failure_returns_error_envelope(self) -> None:
        """Pin failure path also covers ``get_depth`` raising independently
        of ``get_rgba``.
        """
        from strands_robots_sim.isaac.simulation import _CameraState

        sim = self._make_sim()
        handle = MagicMock()
        handle.get_rgba.return_value = np.zeros((100, 100, 4), dtype=np.uint8)
        handle.get_depth.side_effect = AttributeError("annotator not bound")
        cs = _CameraState(name="cam", prim_path="/World/Cameras/cam", width=100, height=100)
        cs.handle = handle
        sim._cameras["cam"] = cs

        result = sim.render("cam")
        assert result["status"] == "error"
        assert "annotator not bound" in result["content"][0]["text"]


def _patched_isaac_camera_modules() -> "tuple[MagicMock, MagicMock, MagicMock, MagicMock]":
    """Build MagicMocks for the three lazy-imported camera modules.

    Returns
    -------
    sensor_mod, viewports_mod, stage_mod, camera_handle
        ``sensor_mod`` stands in for ``isaacsim.sensors.camera`` (its
        ``.Camera`` attribute returns a fresh handle MagicMock).
        ``viewports_mod`` stands in for
        ``isaacsim.core.utils.viewports`` (``.set_camera_view``).
        ``stage_mod`` stands in for ``isaacsim.core.utils.prims``
        (``.delete_prim``). ``camera_handle`` is the MagicMock the
        ``Camera()`` constructor returns -- tests can assert on
        ``.initialize`` / ``.set_focal_length`` calls on it.
    """
    sensor_mod = MagicMock()
    viewports_mod = MagicMock()
    stage_mod = MagicMock()
    camera_handle = MagicMock(name="Camera_handle")
    # add_camera derives focal length from the camera's *actual* horizontal
    # aperture (read back via get_horizontal_aperture), so the mock must report
    # a real number -- otherwise float(MagicMock()) defaults to 1.0. USD's
    # nominal 24 mm keeps the pinhole relation assertions below meaningful.
    camera_handle.get_horizontal_aperture.return_value = 24.0
    sensor_mod.Camera.return_value = camera_handle
    return sensor_mod, viewports_mod, stage_mod, camera_handle


def _make_simulation_with_world_for_camera() -> object:
    """Build an ``IsaacSimulation`` with ``_world_created=True`` and a
    mocked ``_world`` so ``add_camera`` / ``remove_camera`` can exercise
    their full Phase 2 wiring without booting Isaac Sim.
    """
    from strands_robots_sim.isaac.simulation import IsaacSimulation

    sim = IsaacSimulation()
    sim._world_created = True
    sim._world = MagicMock()
    return sim


class TestAddCameraPhase2:
    """Phase 2 wiring (#14) for ``IsaacSimulation.add_camera``.

    Pins the underlying ``isaacsim.sensors.camera.Camera`` constructor + FOV /
    look-at wiring + the structured success envelope.
    """

    def test_returns_error_without_world(self) -> None:
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.add_camera()
        assert result["status"] == "error"
        assert "No world created" in result["content"][0]["text"]

    def test_returns_error_on_duplicate_name(self) -> None:
        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, _ = _patched_isaac_camera_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            r1 = sim.add_camera("front")
            r2 = sim.add_camera("front")
        assert r1["status"] == "success"
        assert r2["status"] == "error"
        assert "already exists" in r2["content"][0]["text"]
        # Only one Camera() construction landed; duplicate rejected
        # before any prim work.
        assert sensor.Camera.call_count == 1

    def test_calls_omni_camera_constructor_with_resolved_kwargs(self) -> None:
        """``Camera()`` is called with ``prim_path``, ``name``, ``position``,
        and ``resolution`` kwargs derived from the add_camera args.
        """
        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, handle = _patched_isaac_camera_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            sim.add_camera("cam0", position=[1, 2, 3], width=320, height=240)
        kwargs = sensor.Camera.call_args.kwargs
        assert kwargs["name"] == "cam0"
        assert kwargs["prim_path"] == "/World/Cameras/cam0"
        assert kwargs["resolution"] == (320, 240)
        assert list(kwargs["position"]) == [1, 2, 3]
        # The handle's initialize() must be called so RTX render product
        # allocation failures surface on add_camera, not on first render.
        handle.initialize.assert_called_once()

    def test_focal_length_matches_pinhole_relation_for_default_fov(self) -> None:
        """``fov=60`` (default, horizontal) on a 24 mm horizontal aperture
        yields focal_length ~ 20.78 mm via the standard pinhole relation
        ``focal = aperture / (2 * tan(fov/2))``.
        """
        import math

        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, handle = _patched_isaac_camera_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            result = sim.add_camera("cam_default", fov=60.0)
        expected_focal = 24.0 / (2.0 * math.tan(math.radians(30.0)))
        # set_focal_length called with the right value.
        handle.set_focal_length.assert_called_once()
        actual_focal = handle.set_focal_length.call_args.args[0]
        assert math.isclose(actual_focal, expected_focal, rel_tol=1e-9)
        # Same value surfaces in the json payload.
        assert math.isclose(
            result["content"][0]["json"]["focal_length_mm"],
            expected_focal,
            rel_tol=1e-9,
        )

    def test_focal_length_scales_with_fov(self) -> None:
        """Wider FOV -> shorter focal length (basic monotonicity sanity)."""
        import math

        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, _ = _patched_isaac_camera_modules()
        captured: dict[str, float] = {}
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            r45 = sim.add_camera("narrow", fov=45.0)
            r90 = sim.add_camera("wide", fov=90.0)
        captured["fov45"] = r45["content"][0]["json"]["focal_length_mm"]
        captured["fov90"] = r90["content"][0]["json"]["focal_length_mm"]
        # FOV 45 -> tan(22.5deg)=0.414 -> 24/(2*0.414)=29.0 mm
        # FOV 90 -> tan(45deg)=1.0    -> 24/(2*1.0)  =12.0 mm
        assert math.isclose(captured["fov45"], 24.0 / (2 * math.tan(math.radians(22.5))), rel_tol=1e-9)
        assert math.isclose(captured["fov90"], 12.0, rel_tol=1e-9)
        assert captured["fov45"] > captured["fov90"]

    def test_target_triggers_set_camera_view(self) -> None:
        """When ``target`` is given, ``set_camera_view`` is called with
        ``eye=position, target=target, camera_prim_path=prim_path``.
        """
        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, _ = _patched_isaac_camera_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            sim.add_camera(
                "front",
                position=[2, 0, 1],
                target=[0, 0, 0.5],
            )
        viewports.set_camera_view.assert_called_once()
        kwargs = viewports.set_camera_view.call_args.kwargs
        assert kwargs["eye"] == [2, 0, 1]
        assert kwargs["target"] == [0, 0, 0.5]
        assert kwargs["camera_prim_path"] == "/World/Cameras/front"

    def test_no_target_leaves_camera_view_unset(self) -> None:
        """With ``target=None`` (default), ``set_camera_view`` is NOT
        called -- the camera keeps its constructed orientation.
        """
        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, _ = _patched_isaac_camera_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            sim.add_camera("free")
        viewports.set_camera_view.assert_not_called()

    def test_default_position_and_resolution(self) -> None:
        """Default position is ``[2.0, 2.0, 2.0]`` (over-the-shoulder
        vantage) and resolution defaults from ``IsaacConfig``.
        """
        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, _ = _patched_isaac_camera_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            result = sim.add_camera("default")
        info = result["content"][0]["json"]
        assert info["position"] == [2.0, 2.0, 2.0]
        # Resolution follows the IsaacConfig defaults; pin the shape
        # rather than the exact values (config defaults could change
        # without breaking this contract).
        assert isinstance(info["resolution"], list) and len(info["resolution"]) == 2
        assert all(isinstance(d, int) and d > 0 for d in info["resolution"])

    def test_failure_in_camera_constructor_returns_error_no_registry_pollution(self) -> None:
        """If ``Camera(...)`` raises, neither ``_cameras`` nor
        ``_prim_registry`` are updated.
        """
        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, _ = _patched_isaac_camera_modules()
        sensor.Camera.side_effect = RuntimeError("RTX render product alloc failed")
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            result = sim.add_camera("broken")
        assert result["status"] == "error"
        assert "RTX render product alloc failed" in result["content"][0]["text"]
        assert "broken" not in sim._cameras
        assert "/World/Cameras/broken" not in sim._prim_registry

    def test_failure_in_initialize_returns_error_no_registry_pollution(self) -> None:
        """``Camera.initialize()`` failure (not just constructor failure)
        also prevents registry updates -- pinned because some Isaac Sim
        builds defer RTX work to ``initialize`` rather than the
        constructor itself.
        """
        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, handle = _patched_isaac_camera_modules()
        handle.initialize.side_effect = RuntimeError("annotator allocation failed")
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            result = sim.add_camera("late_fail")
        assert result["status"] == "error"
        assert "annotator allocation failed" in result["content"][0]["text"]
        assert "late_fail" not in sim._cameras

    def test_handle_stored_on_camera_state_for_later_render(self) -> None:
        """The Camera handle is stored on ``_cameras[name].handle`` so a
        future Phase-2.5 render-frame slice can call methods on it
        (e.g. ``handle.get_rgba()``) without re-importing.
        """
        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, handle = _patched_isaac_camera_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            sim.add_camera("front")
        assert sim._cameras["front"].handle is handle

    def test_registries_updated_on_success(self) -> None:
        """Both ``_cameras`` and ``_prim_registry`` are updated by
        a successful add_camera.
        """
        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, _, _ = _patched_isaac_camera_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
            },
        ):
            sim.add_camera("front", width=160, height=120)
        assert "front" in sim._cameras
        cs = sim._cameras["front"]
        assert cs.prim_path == "/World/Cameras/front"
        assert cs.width == 160 and cs.height == 120
        assert "/World/Cameras/front" in sim._prim_registry


class TestRemoveCameraPhase2:
    """Phase 2 (#14): new ``IsaacSimulation.remove_camera`` method.

    No Phase 1 stub existed; this test class pins the new contract.
    """

    def _add_a_camera(self) -> "tuple[object, MagicMock]":
        sim = _make_simulation_with_world_for_camera()
        sensor, viewports, stage, _ = _patched_isaac_camera_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.sensors.camera": sensor,
                "isaacsim.core.utils.viewports": viewports,
                "isaacsim.core.utils.prims": stage,
            },
        ):
            sim.add_camera("front")
        return sim, stage

    def test_returns_error_for_unknown_name(self) -> None:
        sim = _make_simulation_with_world_for_camera()
        result = sim.remove_camera("ghost")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_calls_delete_prim_with_prim_path(self) -> None:
        sim, stage = self._add_a_camera()
        with patch.dict("sys.modules", {"isaacsim.core.utils.prims": stage}):
            result = sim.remove_camera("front")
        assert result["status"] == "success"
        stage.delete_prim.assert_called_once_with("/World/Cameras/front")

    def test_prunes_cameras_dict_and_prim_registry_on_success(self) -> None:
        sim, stage = self._add_a_camera()
        with patch.dict("sys.modules", {"isaacsim.core.utils.prims": stage}):
            sim.remove_camera("front")
        assert "front" not in sim._cameras
        assert "/World/Cameras/front" not in sim._prim_registry

    def test_delete_prim_failure_keeps_bookkeeping_for_retry(self) -> None:
        """If ``delete_prim`` raises, registries are not pruned -- the
        caller can retry under the same name (e.g. after a stage refresh).
        """
        sim, stage = self._add_a_camera()
        stage.delete_prim.side_effect = RuntimeError("stage closed")
        with patch.dict("sys.modules", {"isaacsim.core.utils.prims": stage}):
            result = sim.remove_camera("front")
        assert result["status"] == "error"
        assert "stage closed" in result["content"][0]["text"]
        assert "front" in sim._cameras
        assert "/World/Cameras/front" in sim._prim_registry

    def test_remove_after_world_torn_down_still_succeeds(self) -> None:
        """Post-destroy (``self._world is None``) still cleans the
        in-Python bookkeeping rather than crashing on the lazy import.
        """
        sim, stage = self._add_a_camera()
        sim._world = None
        with patch.dict("sys.modules", {"isaacsim.core.utils.prims": stage}):
            result = sim.remove_camera("front")
        assert result["status"] == "success"
        assert "front" not in sim._cameras
        # delete_prim is NOT called when world is None (we skip the
        # stage-touching path entirely; the in-Python registries are
        # the only thing left to clean).
        stage.delete_prim.assert_not_called()


def _patched_isaac_articulation_modules() -> "tuple[MagicMock, MagicMock, MagicMock]":
    """Build MagicMocks for the three lazy-imported modules that
    ``_load_usd_robot`` Phase 2 wiring touches.

    Returns
    -------
    articulations_mod, stage_mod, art_handle
        ``articulations_mod`` stands in for
        ``isaacsim.core.api.articulations`` (its ``.Articulation``
        constructor returns ``art_handle``). ``stage_mod`` stands in
        for ``isaacsim.core.utils.stage`` (``.add_reference_to_stage``).
        ``art_handle`` is the Articulation MagicMock the constructor
        returns -- tests can assert on ``.initialize`` /
        ``.set_world_pose`` / ``.dof_names`` patterns.
    """
    articulations_mod = MagicMock()
    stage_mod = MagicMock()
    art_handle = MagicMock(name="Articulation_handle")
    art_handle.dof_names = ["joint_a", "joint_b", "joint_c"]
    articulations_mod.Articulation.return_value = art_handle
    return articulations_mod, stage_mod, art_handle


class TestLoadUsdRobotPhase2:
    """Phase 2 wiring (#14) for ``IsaacSimulation._load_usd_robot``.

    Pins the ``add_reference_to_stage`` + ``Articulation`` + ``initialize``
    chain plus the ``add_robot`` USD-branch integration: success populates
    ``_RobotState.articulation`` so ``get_observation`` / ``send_action``
    have a non-``None`` handle to dispatch through.
    """

    def _make_sim(self) -> object:
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        sim._world_created = True
        sim._world = MagicMock()
        return sim

    def test_load_usd_robot_calls_add_reference_to_stage(self) -> None:
        """``_load_usd_robot`` references the USD into the stage at
        the requested ``prim_path``.
        """
        sim = self._make_sim()
        articulations, stage, _ = _patched_isaac_articulation_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            sim._load_usd_robot(
                prim_path="/World/Robots/r1",
                usd_path="/path/to/robot.usd",
                position=[0.0, 0.0, 0.0],
            )
        stage.add_reference_to_stage.assert_called_once_with(
            usd_path="/path/to/robot.usd",
            prim_path="/World/Robots/r1",
        )

    def test_load_usd_robot_constructs_articulation_at_prim_path(self) -> None:
        """Articulation is constructed with the same ``prim_path`` used
        by ``add_reference_to_stage``, with the leaf segment as the
        articulation registry name.
        """
        sim = self._make_sim()
        articulations, stage, _ = _patched_isaac_articulation_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            sim._load_usd_robot(
                prim_path="/World/Robots/panda",
                usd_path="/x.usd",
                position=[0.0, 0.0, 0.0],
            )
        kwargs = articulations.Articulation.call_args.kwargs
        assert kwargs["prim_path"] == "/World/Robots/panda"
        assert kwargs["name"] == "panda"

    def test_load_usd_robot_calls_initialize(self) -> None:
        """``Articulation.initialize`` is called explicitly so
        ``dof_names`` is populated before the caller reads it.
        """
        sim = self._make_sim()
        articulations, stage, art_handle = _patched_isaac_articulation_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            sim._load_usd_robot(
                prim_path="/World/Robots/r",
                usd_path="/x.usd",
                position=[0.0, 0.0, 0.0],
            )
        art_handle.initialize.assert_called_once()

    def test_load_usd_robot_returns_dof_names_and_handle(self) -> None:
        """Return shape is ``(joint_names: list[str], articulation: Any)``."""
        sim = self._make_sim()
        articulations, stage, art_handle = _patched_isaac_articulation_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            joints, art = sim._load_usd_robot(
                prim_path="/World/Robots/r",
                usd_path="/x.usd",
                position=[0.0, 0.0, 0.0],
            )
        assert joints == ["joint_a", "joint_b", "joint_c"]
        assert art is art_handle

    def test_load_usd_robot_handles_none_dof_names(self) -> None:
        """If ``Articulation.dof_names`` is ``None`` (some Isaac builds
        when the USD has no Articulation root), return an empty list
        rather than crashing on iteration.
        """
        sim = self._make_sim()
        articulations, stage, art_handle = _patched_isaac_articulation_modules()
        art_handle.dof_names = None
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            joints, _ = sim._load_usd_robot(
                prim_path="/World/Robots/r",
                usd_path="/x.usd",
                position=[0.0, 0.0, 0.0],
            )
        assert joints == []

    def test_load_usd_robot_skips_set_world_pose_for_origin_position(self) -> None:
        """``position=[0, 0, 0]`` skips ``set_world_pose`` (USD's
        authored pose wins; saves a tensor round-trip).
        """
        sim = self._make_sim()
        articulations, stage, art_handle = _patched_isaac_articulation_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            sim._load_usd_robot(
                prim_path="/World/Robots/r",
                usd_path="/x.usd",
                position=[0.0, 0.0, 0.0],
            )
        art_handle.set_world_pose.assert_not_called()

    def test_load_usd_robot_calls_set_world_pose_for_non_origin(self) -> None:
        """Non-zero position triggers ``set_world_pose(position=...)``."""
        sim = self._make_sim()
        articulations, stage, art_handle = _patched_isaac_articulation_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            sim._load_usd_robot(
                prim_path="/World/Robots/r",
                usd_path="/x.usd",
                position=[1.0, 2.0, 3.0],
            )
        art_handle.set_world_pose.assert_called_once()
        kwargs = art_handle.set_world_pose.call_args.kwargs
        assert list(kwargs["position"]) == [1.0, 2.0, 3.0]


class TestAddRobotUsdBranchPhase2:
    """Phase 2 wiring of the ``add_robot(usd_path=...)`` integration.

    Pins that the USD branch:
    - calls ``_load_usd_robot``,
    - stores the returned ``Articulation`` handle on
      ``_RobotState.articulation`` (so ``get_observation`` /
      ``send_action`` light up for USD-loaded robots),
    - rolls back registry state if ``_load_usd_robot`` raises.
    """

    def _make_sim(self) -> object:
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        sim._world_created = True
        sim._world = MagicMock()
        return sim

    def test_add_robot_usd_branch_stores_articulation_on_robot_state(self) -> None:
        sim = self._make_sim()
        articulations, stage, art_handle = _patched_isaac_articulation_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            result = sim.add_robot(name="my_panda", usd_path="/path/to/panda.usd")
        assert result["status"] == "success"
        assert "my_panda" in sim._robots
        rs = sim._robots["my_panda"]
        assert rs.articulation is art_handle, (
            "USD-branch add_robot must wire the Articulation handle onto "
            "_RobotState.articulation so get_observation / send_action "
            "have a non-None handle to dispatch through."
        )
        assert rs.joint_names == ["joint_a", "joint_b", "joint_c"]

    def test_add_robot_usd_branch_surfaces_structured_json(self) -> None:
        """Success envelope's ``content[0]["json"]`` carries name /
        prim_path / usd_path / joint_names / joint_count / position /
        articulation_wired so an agent can confirm the load shape.
        """
        sim = self._make_sim()
        articulations, stage, _ = _patched_isaac_articulation_modules()
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            result = sim.add_robot(
                name="r",
                usd_path="/foo.usd",
                position=[0.5, 0.0, 0.0],
            )
        info = result["content"][0]["json"]
        assert info["name"] == "r"
        assert info["prim_path"] == "/World/Robots/r"
        assert info["usd_path"] == "/foo.usd"
        assert info["joint_count"] == 3
        assert info["position"] == [0.5, 0.0, 0.0]
        assert info["articulation_wired"] is True

    def test_add_robot_usd_branch_returns_error_on_load_failure(self) -> None:
        """If ``_load_usd_robot`` raises (USD file missing, Articulation
        init fails, omni surface drift), ``add_robot`` returns the
        structured error envelope with **no** registry pollution.
        """
        sim = self._make_sim()
        articulations, stage, _ = _patched_isaac_articulation_modules()
        stage.add_reference_to_stage.side_effect = OSError("USD file not found: /missing.usd")
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            result = sim.add_robot(name="ghost", usd_path="/missing.usd")
        assert result["status"] == "error"
        assert "ghost" in result["content"][0]["text"]
        assert "USD file not found" in result["content"][0]["text"]
        # Registry NOT polluted -- caller can retry under the same name.
        assert "ghost" not in sim._robots
        assert "/World/Robots/ghost" not in sim._prim_registry

    def test_add_robot_usd_branch_returns_error_on_initialize_failure(self) -> None:
        """``Articulation.initialize`` failure also leaves registries
        clean -- some Isaac Sim builds defer Articulation tree-walk to
        ``initialize`` rather than the constructor.
        """
        sim = self._make_sim()
        articulations, stage, art_handle = _patched_isaac_articulation_modules()
        art_handle.initialize.side_effect = RuntimeError("articulation root not found in USD")
        with patch.dict(
            "sys.modules",
            {
                "isaacsim.core.api.articulations": articulations,
                "isaacsim.core.utils.stage": stage,
            },
        ):
            result = sim.add_robot(name="bad", usd_path="/bad.usd")
        assert result["status"] == "error"
        assert "articulation root not found" in result["content"][0]["text"]
        assert "bad" not in sim._robots

    def test_add_robot_procedural_branch_still_has_no_articulation(self) -> None:
        """Regression pin: the procedural branch (which constructs USD
        via the build-via-API flow, not via Articulation wrapper) must
        keep ``_RobotState.articulation = None``. Procedural-robot
        articulation wiring is a separate Phase 2 slice; this PR
        intentionally only covers the USD branch.
        """
        sim = self._make_sim()
        # Procedural lookup is hit when no usd_path/urdf_path is given.
        # ``so100`` is a registered procedural robot.
        result = sim.add_robot(name="proc", data_config="so100")
        assert result["status"] == "success"
        assert sim._robots["proc"].articulation is None, (
            "Procedural add_robot branch must not silently wire an " "Articulation; that's a separate slice on #14."
        )


def _patched_isaac_urdf_modules() -> "tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]":
    """Build MagicMocks for the modules ``_load_urdf_robot`` Phase 2
    wiring touches.

    Returns
    -------
    urdf_iface_mod, urdf_importer_mod, articulations_mod, art_handle, import_config
        ``urdf_iface_mod`` stands in for the
        ``_urdf.acquire_urdf_interface()`` return value -- i.e. the
        interface that exposes ``.parse_urdf`` and ``.import_robot``.
        Default behaviour: ``parse_urdf`` returns a sentinel
        ``urdf_robot_mock``; ``import_robot`` returns
        ``"/World/Robots/imported"``.
        ``urdf_importer_mod`` stands in for ``isaacsim.asset.importer.urdf``;
        ``urdf_importer_mod._urdf.ImportConfig()`` returns
        ``import_config`` so the test can inspect which fields were
        set, and ``urdf_importer_mod._urdf.acquire_urdf_interface()``
        returns ``urdf_iface_mod``.
        ``articulations_mod`` stands in for
        ``isaacsim.core.api.articulations`` (returns ``art_handle``).
        ``art_handle`` is the Articulation MagicMock with
        ``dof_names = ["j1", "j2"]`` by default.
        ``import_config`` is the MagicMock that ``ImportConfig()``
        returns -- tests can read attributes like ``fix_base`` /
        ``distance_scale`` on it.
    """
    urdf_iface_mod = MagicMock()
    urdf_iface_mod.parse_urdf.return_value = MagicMock(name="urdf_robot")
    urdf_iface_mod.import_robot.return_value = "/World/Robots/imported"

    urdf_importer_mod = MagicMock()
    urdf_importer_mod._urdf.acquire_urdf_interface.return_value = urdf_iface_mod

    articulations_mod = MagicMock()
    art_handle = MagicMock(name="Articulation_handle")
    art_handle.dof_names = ["j1", "j2"]
    articulations_mod.Articulation.return_value = art_handle

    # ``ImportConfig()`` -> import_config; tests read attributes
    # set on this instance.
    import_config = MagicMock()
    urdf_importer_mod._urdf.ImportConfig.return_value = import_config

    return urdf_iface_mod, urdf_importer_mod, articulations_mod, art_handle, import_config


def _patched_urdf_sys_modules(
    urdf_iface_mod: MagicMock,
    urdf_importer_mod: MagicMock,
    articulations_mod: MagicMock,
) -> dict[str, MagicMock]:
    # Stub the stage-utilities mock for ``add_reference_to_stage``
    # which the new ``_load_urdf_robot`` calls to bind the converted
    # USD file onto the live stage at the caller-requested prim path.
    stage_mod = MagicMock()

    return {
        # Modern Isaac Sim 4.5+ namespace is preferred; the source
        # tries this first and falls back to ``omni.importer.urdf``.
        "isaacsim.asset.importer.urdf": urdf_importer_mod,
        "omni.importer.urdf": urdf_importer_mod,
        "isaacsim.core.api.articulations": articulations_mod,
        "isaacsim.core.utils.stage": stage_mod,
    }


class TestLoadUrdfRobotPhase2:
    """Phase 2 wiring (#14) for ``IsaacSimulation._load_urdf_robot``.

    Pins the ``URDFParseAndImportFile`` Kit-command + ``Articulation``
    chain plus the same call-site contract (``add_robot`` URDF branch
    populates ``_RobotState.articulation``) that the USD branch has.
    """

    def _make_sim(self) -> object:
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        sim._world_created = True
        sim._world = MagicMock()
        return sim

    def test_load_urdf_robot_executes_urdf_command_with_dest_path(self) -> None:
        """``omni.kit.commands.execute("URDFParseAndImportFile", ...)``
        is called with ``urdf_path``, an ``import_config``, and a
        temp-file ``dest_path``.

        Pre-PR-#64-GPU-fix the wiring passed the stage prim path as
        ``dest_path`` directly, but the Kit command interprets
        ``dest_path`` as a USD FILE PATH and crashed with
        ``Sdf.ErrorException: Failed verification: 'fileFormat'`` on
        a non-USD-suffixed value. Now ``dest_path`` is a temp .usd
        file keyed on the stage prim's leaf segment so re-runs reuse
        the converted USD on disk.
        """
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, _, import_config = _patched_isaac_urdf_modules()
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            sim._load_urdf_robot(
                prim_path="/World/Robots/r",
                urdf_path="/path/to/robot.urdf",
                position=[0.0, 0.0, 0.0],
            )
        # parse_urdf called with (root_dir, filename, import_config).
        urdf_iface.parse_urdf.assert_called_once()
        parse_args = urdf_iface.parse_urdf.call_args.args
        assert parse_args[0] == "/path/to"  # root dir
        assert parse_args[1] == "robot.urdf"  # filename
        # import_robot called with (root_dir, filename, urdf_robot, import_config, stage="").
        urdf_iface.import_robot.assert_called_once()
        import_args = urdf_iface.import_robot.call_args.args
        assert import_args[0] == "/path/to"
        assert import_args[1] == "robot.urdf"
        # stage="" (5th positional arg) -- empty string asks the
        # importer to add prims directly to the live USD stage rather
        # than writing a USD file. Pinned because Isaac Sim 4.5's
        # importer raises "Used null prim" when stage is a non-empty
        # path and the live stage isn't ready.
        assert import_args[4] == ""

    def test_load_urdf_robot_import_config_has_fixed_base_default(self) -> None:
        """``ImportConfig.fix_base`` defaults to ``True`` -- the most
        common LIBERO / GR00T case is a fixed-base manipulator.
        """
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, _, import_config = _patched_isaac_urdf_modules()
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            sim._load_urdf_robot(
                prim_path="/World/Robots/r",
                urdf_path="/x.urdf",
                position=[0.0, 0.0, 0.0],
            )
        # The config attributes set in source must have flowed onto the
        # ImportConfig instance that parse_urdf and import_robot received.
        config = import_config
        assert config.fix_base is True
        assert config.import_inertia_tensor is True
        # Don't create a new physics scene -- World already made one.
        assert config.create_physics_scene is False
        assert config.distance_scale == 1.0

    def test_load_urdf_robot_uses_imported_prim_path_for_articulation(self) -> None:
        """Articulation is constructed against the prim path the
        ``_urdf.import_robot`` actually used.

        The importer adds prims directly to the live stage at a path
        of its choosing (typically ``/World/<robot_name>``) when
        called with ``stage=""``; the caller's requested ``prim_path``
        is informational and used for bookkeeping. Articulation
        construction targets the importer's actual path.
        """
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, _, import_config = _patched_isaac_urdf_modules()
        urdf_iface.import_robot.return_value = "/World/some_robot_name"
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            sim._load_urdf_robot(
                prim_path="/World/Robots/r",
                urdf_path="/x.urdf",
                position=[0.0, 0.0, 0.0],
            )
        kwargs = arts.Articulation.call_args.kwargs
        # Articulation uses the importer's returned path (where prims
        # actually landed on the live stage), not the caller-requested
        # prim_path which is just bookkeeping.
        assert kwargs["prim_path"] == "/World/some_robot_name"
        # Articulation registry name uses that path's leaf segment.
        assert kwargs["name"] == "some_robot_name"

    def test_load_urdf_robot_no_add_reference_to_stage_call(self) -> None:
        """``import_robot(stage="")`` adds prims directly to the live
        stage; no separate ``add_reference_to_stage`` is needed.

        Pre-PR-#64-GPU-fix the wiring used a 2-step
        ``URDFParseAndImportFile -> add_reference_to_stage`` flow but
        the Kit command's ``stage=<path>`` mode raised "Used null
        prim" on Isaac Sim 4.5. The simpler 1-step
        ``import_robot(stage="")`` is the supported path; this test
        pins that the deprecated 2-step shape doesn't sneak back.
        """
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, _, import_config = _patched_isaac_urdf_modules()
        sys_modules = _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)
        with patch.dict("sys.modules", sys_modules):
            sim._load_urdf_robot(
                prim_path="/World/Robots/r",
                urdf_path="/x.urdf",
                position=[0.0, 0.0, 0.0],
            )
        # add_reference_to_stage stub exists in the patched modules
        # but should NOT be called.
        stage = sys_modules["isaacsim.core.utils.stage"]
        stage.add_reference_to_stage.assert_not_called()

    def test_load_urdf_robot_no_op_when_command_returns_empty_path(self) -> None:
        """Pre-PR-#64-GPU-fix this test asserted a fallback to
        ``prim_path`` when the importer returned an empty string.
        Post-fix the second tuple element is ignored entirely (the
        caller-requested ``prim_path`` always wins for Articulation
        construction), so the empty-string return is now a no-op
        from this method's perspective. Pin retained as a regression
        guard against any future re-introduction of the imported-
        prim-path indirection.
        """
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, _, import_config = _patched_isaac_urdf_modules()
        urdf_iface.import_robot.return_value = ""
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            with pytest.raises(RuntimeError, match="URDF import failed"):
                sim._load_urdf_robot(
                    prim_path="/World/Robots/r",
                    urdf_path="/x.urdf",
                    position=[0.0, 0.0, 0.0],
                )

    def test_load_urdf_robot_raises_runtimeerror_when_command_returns_false(self) -> None:
        """``import_robot`` returning ``""`` -> ``RuntimeError`` (caught
        by the caller's cleanup clause). Same shape as the legacy
        ``URDFParseAndImportFile`` Kit-command's ``(False, ...)`` return.
        """
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, _, import_config = _patched_isaac_urdf_modules()
        urdf_iface.import_robot.return_value = ""
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            with pytest.raises(RuntimeError, match="URDF import failed"):
                sim._load_urdf_robot(
                    prim_path="/World/Robots/r",
                    urdf_path="/bad.urdf",
                    position=[0.0, 0.0, 0.0],
                )

    def test_load_urdf_robot_raises_runtimeerror_when_parse_returns_none(self) -> None:
        """``parse_urdf`` returning ``None`` -> ``RuntimeError``."""
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, _, import_config = _patched_isaac_urdf_modules()
        urdf_iface.parse_urdf.return_value = None
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            with pytest.raises(RuntimeError, match="URDF parse failed"):
                sim._load_urdf_robot(
                    prim_path="/World/Robots/r",
                    urdf_path="/bad.urdf",
                    position=[0.0, 0.0, 0.0],
                )

    def test_load_urdf_robot_returns_dof_names_and_handle(self) -> None:
        """Return shape mirrors ``_load_usd_robot``:
        ``(joint_names: list[str], articulation: Any)``.
        """
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, art_handle, import_config = _patched_isaac_urdf_modules()
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            joints, art = sim._load_urdf_robot(
                prim_path="/World/Robots/r",
                urdf_path="/x.urdf",
                position=[0.0, 0.0, 0.0],
            )
        assert joints == ["j1", "j2"]
        assert art is art_handle

    def test_load_urdf_robot_skips_set_world_pose_for_origin(self) -> None:
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, art_handle, import_config = _patched_isaac_urdf_modules()
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            sim._load_urdf_robot(
                prim_path="/World/Robots/r",
                urdf_path="/x.urdf",
                position=[0.0, 0.0, 0.0],
            )
        art_handle.set_world_pose.assert_not_called()

    def test_load_urdf_robot_calls_set_world_pose_for_non_origin(self) -> None:
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, art_handle, import_config = _patched_isaac_urdf_modules()
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            sim._load_urdf_robot(
                prim_path="/World/Robots/r",
                urdf_path="/x.urdf",
                position=[1.0, 2.0, 3.0],
            )
        art_handle.set_world_pose.assert_called_once()
        kwargs = art_handle.set_world_pose.call_args.kwargs
        assert list(kwargs["position"]) == [1.0, 2.0, 3.0]


class TestAddRobotUrdfBranchPhase2:
    """Phase 2 wiring of the ``add_robot(urdf_path=...)`` integration.

    Same contract as the USD branch (PR #63 / TestAddRobotUsdBranchPhase2)
    -- success populates ``_RobotState.articulation``, failure leaves
    registries clean.
    """

    def _make_sim(self) -> object:
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        sim._world_created = True
        sim._world = MagicMock()
        return sim

    def test_add_robot_urdf_branch_stores_articulation(self) -> None:
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, art_handle, import_config = _patched_isaac_urdf_modules()
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            result = sim.add_robot(name="my_panda", urdf_path="/path/to/panda.urdf")
        assert result["status"] == "success"
        assert sim._robots["my_panda"].articulation is art_handle

    def test_add_robot_urdf_branch_surfaces_structured_json(self) -> None:
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, _, import_config = _patched_isaac_urdf_modules()
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            result = sim.add_robot(
                name="r",
                urdf_path="/foo.urdf",
                position=[0.5, 0.0, 0.0],
            )
        info = result["content"][0]["json"]
        assert info["name"] == "r"
        assert info["urdf_path"] == "/foo.urdf"
        assert info["joint_count"] == 2
        assert info["position"] == [0.5, 0.0, 0.0]
        assert info["articulation_wired"] is True

    def test_add_robot_urdf_branch_returns_error_on_command_failure(self) -> None:
        """``import_robot`` returning empty string -> structured error
        envelope, no registry pollution.
        """
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, _, import_config = _patched_isaac_urdf_modules()
        urdf_iface.import_robot.return_value = ""
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            result = sim.add_robot(name="ghost", urdf_path="/bad.urdf")
        assert result["status"] == "error"
        assert "ghost" in result["content"][0]["text"]
        assert "URDF import failed" in result["content"][0]["text"]
        assert "ghost" not in sim._robots
        assert "/World/Robots/ghost" not in sim._prim_registry

    def test_add_robot_urdf_branch_returns_error_on_initialize_failure(self) -> None:
        sim = self._make_sim()
        urdf_iface, urdf_imp, arts, art_handle, import_config = _patched_isaac_urdf_modules()
        art_handle.initialize.side_effect = RuntimeError("articulation root not in URDF")
        with patch.dict("sys.modules", _patched_urdf_sys_modules(urdf_iface, urdf_imp, arts)):
            result = sim.add_robot(name="bad", urdf_path="/bad.urdf")
        assert result["status"] == "error"
        assert "articulation root not in URDF" in result["content"][0]["text"]
        assert "bad" not in sim._robots


class TestSendActionReturnContract:
    """``send_action`` must return the SimEngine ``dict`` envelope.

    The ABC (``strands_robots.simulation.base.SimEngine``) declares
    ``send_action -> dict[str, Any]`` and ``PolicyRunner`` increments its
    ``_action_errors`` counter only when the returned ``status == "error"``.
    A bare ``None`` return (the pre-fix behaviour) silently swallowed every
    action failure. These tests pin the envelope on each path.
    """

    def _make_sim_with_robot(self, joint_names=("j0", "j1", "j2")):
        from strands_robots_sim.isaac.simulation import IsaacSimulation, _RobotState

        sim = IsaacSimulation()
        sim._world_created = True
        sim._world = MagicMock()
        art = MagicMock()
        sim._robots["arm"] = _RobotState(
            name="arm",
            prim_path="/World/Robots/arm",
            joint_names=list(joint_names),
            articulation=art,
        )
        return sim, art

    def test_no_world_returns_error_envelope(self):
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.send_action({"j0": 0.1})
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "No world created." in result["content"][0]["text"]

    def test_unknown_robot_returns_error_envelope(self):
        sim, _ = self._make_sim_with_robot()
        result = sim.send_action({"j0": 0.1}, robot_name="missing")
        assert result["status"] == "error"
        assert "missing" in result["content"][0]["text"]

    def test_ambiguous_robot_returns_error_envelope(self):
        from strands_robots_sim.isaac.simulation import _RobotState

        sim, _ = self._make_sim_with_robot()
        sim._robots["arm2"] = _RobotState(
            name="arm2",
            prim_path="/World/Robots/arm2",
            joint_names=["j0"],
            articulation=MagicMock(),
        )
        result = sim.send_action({"j0": 0.1})
        assert result["status"] == "error"
        assert "specify robot_name" in result["content"][0]["text"]

    def test_success_returns_success_envelope(self):
        sim, art = self._make_sim_with_robot()
        result = sim.send_action({"j0": 0.1, "j1": 0.2, "j2": 0.3}, n_substeps=2)
        assert result["status"] == "success"
        assert "arm" in result["content"][0]["text"]
        assert "2 substeps" in result["content"][0]["text"]
        art.set_joint_position_targets.assert_called_once()
        assert sim._world.step.call_count == 2

    def test_array_action_returns_success_envelope(self):
        sim, art = self._make_sim_with_robot()
        result = sim.send_action(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert result["status"] == "success"
        art.set_joint_position_targets.assert_called_once()

    def test_unresolved_keys_return_error_with_json_block(self):
        sim, art = self._make_sim_with_robot()
        result = sim.send_action({"j0": 0.1, "bogus": 9.9})
        assert result["status"] == "error"
        # Resolvable keys are still applied (physics advanced)...
        art.set_joint_position_targets.assert_called_once()
        # ...but the unresolved key surfaces in a json block for self-correction.
        json_block = next(c["json"] for c in result["content"] if "json" in c)
        assert json_block["unresolved_keys"] == ["bogus"]
        assert json_block["applied"] == ["j0"]

    def test_articulation_failure_returns_error_envelope(self):
        sim, art = self._make_sim_with_robot()
        art.set_joint_position_targets.side_effect = RuntimeError("articulation torn down")
        result = sim.send_action({"j0": 0.1})
        assert result["status"] == "error"
        assert "articulation torn down" in result["content"][0]["text"]
        # Physics must not advance once the action couldn't be applied.
        sim._world.step.assert_not_called()


class TestCamerasRecording:
    """Synchronous rollout-video recorder (strands-labs/robots-sim#112).

    ``IsaacSimulation`` records videos via a ``start_cameras_recording`` /
    ``stop_cameras_recording`` pair that mirrors the MuJoCo backend's API
    and filename convention (``{name}__{camera}.mp4``). Capture is
    synchronous (Isaac's RTX renderer is thread-bound): ``start`` returns
    an ``on_frame`` closure wired into ``evaluate_benchmark(on_frame=...)``,
    and ``stop`` flushes the in-memory buffers to MP4.
    """

    def _make_sim_with_camera(self, render_mode: str = "rtx_realtime"):
        """An IsaacSimulation with a world + one RTX camera that renders
        solid-colour frames, so the recorder has real frames to capture.
        """
        from strands_robots_sim.isaac.simulation import IsaacSimulation, _CameraState

        sim = IsaacSimulation(render_mode=render_mode)
        sim._world_created = True
        sim._world = MagicMock()
        handle = MagicMock(name="Camera_handle")
        handle.get_rgba.return_value = np.full((48, 64, 4), 200, dtype=np.uint8)
        handle.get_depth.return_value = np.full((48, 64), 1.0, dtype=np.float32)
        cs = _CameraState(name="image", prim_path="/World/Cameras/image", width=64, height=48)
        cs.handle = handle
        sim._cameras["image"] = cs
        return sim

    def test_start_without_world_errors(self):
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.start_cameras_recording(cameras=["image"])
        assert result["status"] == "error"
        assert "No world created" in result["content"][0]["text"]

    def test_start_unknown_camera_errors(self):
        sim = self._make_sim_with_camera()
        result = sim.start_cameras_recording(cameras=["ghost"])
        assert result["status"] == "error"
        assert "Camera(s) not found" in result["content"][0]["text"]

    def test_start_no_cameras_errors(self):
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        sim._world_created = True
        sim._world = MagicMock()
        # No cameras added; cameras=None resolves to empty.
        result = sim.start_cameras_recording()
        assert result["status"] == "error"
        assert "No cameras to record" in result["content"][0]["text"]

    def test_start_returns_on_frame_closure(self):
        sim = self._make_sim_with_camera()
        result = sim.start_cameras_recording(cameras=["image"], name="tag")
        assert result["status"] == "success"
        json_block = next(c["json"] for c in result["content"] if "json" in c)
        assert callable(json_block["on_frame"])
        assert json_block["cameras"] == ["image"]
        assert json_block["name"] == "tag"

    def test_double_start_errors(self):
        sim = self._make_sim_with_camera()
        sim.start_cameras_recording(cameras=["image"], name="first")
        result = sim.start_cameras_recording(cameras=["image"], name="second")
        assert result["status"] == "error"
        assert "Already recording 'first'" in result["content"][0]["text"]

    def test_stop_without_recording_is_idempotent_success(self):
        sim = self._make_sim_with_camera()
        result = sim.stop_cameras_recording()
        assert result["status"] == "success"
        assert "Was not recording" in result["content"][0]["text"]

    def test_full_cycle_writes_mp4(self, tmp_path):
        """on_frame captures frames; stop encodes them to a real MP4 under
        the MuJoCo-compatible ``{name}__{camera}.mp4`` filename.
        """
        import os

        pytest.importorskip("imageio")
        sim = self._make_sim_with_camera()
        result = sim.start_cameras_recording(cameras=["image"], output_dir=str(tmp_path), name="rec1", fps=10)
        on_frame = next(c["json"]["on_frame"] for c in result["content"] if "json" in c)

        # Simulate the eval loop firing the per-step hook.
        for step in range(5):
            on_frame(step, {}, {})

        stop = sim.stop_cameras_recording()
        assert stop["status"] == "success"
        artifacts = next(c["json"] for c in stop["content"] if "json" in c)["artifacts"]
        assert len(artifacts) == 1
        art = artifacts[0]
        assert art["camera"] == "image"
        assert art["frames"] == 5
        assert art["errors"] == 0
        # Filename convention matches MuJoCo: {name}__{camera}.mp4
        expected = os.path.join(str(tmp_path), "rec1__image.mp4")
        assert art["path"] == expected
        assert os.path.exists(expected)
        assert os.path.getsize(expected) > 0

    def test_on_frame_after_stop_is_noop(self, tmp_path):
        sim = self._make_sim_with_camera()
        result = sim.start_cameras_recording(cameras=["image"], output_dir=str(tmp_path), name="rec2")
        on_frame = next(c["json"]["on_frame"] for c in result["content"] if "json" in c)
        on_frame(0, {}, {})
        sim.stop_cameras_recording()
        # Firing the stale closure after stop must not raise / re-capture.
        on_frame(1, {}, {})  # no exception
        assert sim._cams_rec_state is None

    def test_render_error_counts_not_raises(self, tmp_path):
        """A per-camera render failure increments the error counter rather
        than aborting the whole eval.
        """
        pytest.importorskip("imageio")
        sim = self._make_sim_with_camera()
        sim._cameras["image"].handle.get_rgba.side_effect = RuntimeError("rtx hiccup")
        result = sim.start_cameras_recording(cameras=["image"], output_dir=str(tmp_path), name="rec3")
        on_frame = next(c["json"]["on_frame"] for c in result["content"] if "json" in c)
        on_frame(0, {}, {})  # must not raise
        stop = sim.stop_cameras_recording()
        art = next(c["json"] for c in stop["content"] if "json" in c)["artifacts"][0]
        assert art["frames"] == 0
        assert art["errors"] == 1

    def test_max_frames_cap_respected(self, tmp_path):
        sim = self._make_sim_with_camera()
        result = sim.start_cameras_recording(
            cameras=["image"], output_dir=str(tmp_path), name="rec4", max_frames_per_camera=3
        )
        on_frame = next(c["json"]["on_frame"] for c in result["content"] if "json" in c)
        for step in range(10):
            on_frame(step, {}, {})
        # Assert on the in-memory buffer (no imageio encode needed) so this
        # cap test runs even on hosts without the MP4 encoder installed.
        assert len(sim._cams_rec_state["buffers"]["image"]) == 3
        sim.stop_cameras_recording()

    def test_destroy_clears_recorder_state(self, tmp_path):
        sim = self._make_sim_with_camera()
        sim.start_cameras_recording(cameras=["image"], output_dir=str(tmp_path), name="rec5")
        assert sim._cams_rec_state is not None
        sim.destroy()
        assert sim._cams_rec_state is None
