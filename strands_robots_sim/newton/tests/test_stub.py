"""Newton stub-shape tests.

These pin the Stage-3 packaging contract: the Newton backend resolves
through ``[project.entry-points."strands_robots.backends"]``, the
subpackage imports without pulling warp/newton, and all SimEngine
methods on the stub raise ``NotImplementedError`` so callers fail fast
until R11 ships the real implementation.

Run with:: pytest strands_robots_sim/newton/tests/test_stub.py -v
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys

import pytest

_PYPROJECT = pathlib.Path(__file__).resolve().parents[3] / "pyproject.toml"


class TestEntryPointDeclaration:
    """Validate that the Newton/Warp entry points are declared in pyproject."""

    def test_pyproject_exists(self):
        assert _PYPROJECT.exists(), f"pyproject.toml not found at {_PYPROJECT}"

    def test_newton_entry_point_declared(self):
        content = _PYPROJECT.read_text()
        assert 'newton = "strands_robots_sim.newton.simulation:NewtonSimulation"' in content, (
            'Expected `newton = "strands_robots_sim.newton.simulation:NewtonSimulation"` '
            'under [project.entry-points."strands_robots.backends"] in pyproject.toml.'
        )

    def test_warp_alias_entry_point_declared(self):
        content = _PYPROJECT.read_text()
        assert 'warp = "strands_robots_sim.newton.simulation:NewtonSimulation"' in content, (
            "Expected `warp` alias entry point alongside `newton` so users can write either "
            '`create_simulation("newton")` or `create_simulation("warp")`.'
        )

    def test_newton_extra_declared(self):
        content = _PYPROJECT.read_text()
        assert "\nnewton = [" in content, (
            "Expected `newton = [...]` under [project.optional-dependencies] declaring " "warp-lang + newton-physics."
        )

    def test_newton_extra_pins_warp_and_newton_physics(self):
        content = _PYPROJECT.read_text()
        idx = content.find("\nnewton = [")
        assert idx != -1, "[newton] extras block not found"
        block_end = content.find("]", idx)
        block = content[idx:block_end]
        assert "warp-lang" in block, "[newton] extras must include warp-lang>=1.12"
        assert "newton-physics" in block, "[newton] extras must include newton-physics>=1.0"

    def test_all_extra_includes_isaac_and_newton(self):
        content = _PYPROJECT.read_text()
        idx = content.find("\nall = [")
        assert idx != -1, "[all] extras block not found"
        # The block has nested `[...]` literals (e.g. `strands-robots-sim[isaac]`),
        # so naive `find("]")` from `idx` would match the inner bracket first.
        # Find the outermost `]` by walking line-by-line until we hit the
        # closing bracket on its own line.
        block_lines: list[str] = []
        for line in content[idx:].splitlines():
            block_lines.append(line)
            if line.strip() == "]":
                break
        block = "\n".join(block_lines)
        assert "strands-robots-sim[isaac]" in block, "[all] must expand to [isaac]"
        assert "strands-robots-sim[newton]" in block, "[all] must expand to [newton]"

    def test_dependencies_pin_strands_robots_only(self):
        """Heavy runtime deps live upstream now -- ours should only list strands-robots."""
        content = _PYPROJECT.read_text()
        # Locate the [project] dependencies = [...] block.
        idx = content.find("\ndependencies = [")
        assert idx != -1, "[project] dependencies block not found"
        block_end = content.find("]", idx)
        block = content[idx:block_end]
        assert "strands-robots>=0.4.0" in block, (
            "[project.dependencies] must pin strands-robots>=0.4.0 (the version that walks "
            "entry-points to discover plugin backends)."
        )
        # Heavy deps that used to live here must NOT be back.
        forbidden = ("lerobot", "torch", "opencv-python-headless", "msgpack", "pyzmq")
        for dep in forbidden:
            assert dep not in block, (
                f"[project.dependencies] must not list {dep!r} -- it now comes in transitively "
                "via strands-robots>=0.4.0."
            )

    def test_entry_points_visible_via_importlib_metadata_when_installed(self):
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):
                backend_eps = list(eps.select(group="strands_robots.backends"))
            else:
                backend_eps = eps.get("strands_robots.backends", [])
        except Exception as exc:  # pragma: no cover - defensive
            pytest.skip(f"importlib.metadata unavailable: {exc}")

        if not backend_eps:
            pytest.skip(
                "Package not installed (no entry points discoverable). "
                "Run `pip install -e .` to validate this assertion locally."
            )

        names = {ep.name for ep in backend_eps}
        if "newton" not in names and "warp" not in names:
            pytest.skip(
                "Package installed but entry-point cache is stale -- reinstall after "
                "pyproject.toml change: `pip install -e . --force-reinstall --no-deps`."
            )

        for ep in backend_eps:
            if ep.name in {"newton", "warp"}:
                assert ep.value == "strands_robots_sim.newton.simulation:NewtonSimulation", (
                    f"Entry point {ep.name!r} resolves to {ep.value!r}; expected "
                    "'strands_robots_sim.newton.simulation:NewtonSimulation'."
                )


class TestLazyImportSurface:
    """Validate the PEP 562 lazy-import contract on the ``newton`` subpackage."""

    def test_import_newton_does_not_load_warp_or_newton_physics(self):
        """Importing ``strands_robots_sim.newton`` adds zero ``warp.*`` / ``newton.*`` modules."""
        before = {k for k in sys.modules if k.startswith(("warp", "newton")) and not k.startswith("newton_test")}
        import strands_robots_sim.newton  # noqa: F401

        added = {
            k for k in sys.modules if k.startswith(("warp", "newton")) and not k.startswith("newton_test")
        } - before
        # Filter out our own subpackage modules (they obviously become present
        # after the import above).
        added = {k for k in added if not k.startswith("strands_robots_sim")}
        # `newton_test` and similar third-party names with a `newton` prefix
        # would accidentally match; the prefix filter above already excludes
        # the obvious cases, but if a future stdlib happens to start with
        # `newton` we'd want to know.
        assert added == set(), (
            f"Importing strands_robots_sim.newton loaded warp/newton modules: {sorted(added)}. "
            "The PEP 562 lazy stub must defer warp/newton resolution until an attribute is accessed."
        )

    def test_newton_subpackage_exposes_lazy_attrs_in___all__(self):
        import strands_robots_sim.newton as newton_pkg

        assert "NewtonSimulation" in newton_pkg.__all__

    def test_unknown_attr_raises_attributeerror(self):
        import strands_robots_sim.newton as newton_pkg

        with pytest.raises(AttributeError, match="no attribute 'NotARealClass'"):
            _ = newton_pkg.NotARealClass

    def test_dunder_getattr_is_present(self):
        import strands_robots_sim.newton as newton_pkg

        assert hasattr(
            newton_pkg, "__getattr__"
        ), "PEP 562 module-level __getattr__ must be defined for lazy import to work."
        assert callable(newton_pkg.__getattr__)


class TestNewtonSimulationStub:
    """Pin the SimEngine-shaped contract of the stub class."""

    def test_class_resolves_via_lazy_attr(self):
        from strands_robots_sim.newton import NewtonSimulation

        assert NewtonSimulation.__name__ == "NewtonSimulation"

    def test_class_is_simengine_subclass(self):
        """The stub must be a SimEngine subclass so the factory accepts it."""
        from strands_robots_sim.newton.simulation import NewtonSimulation, SimEngine

        assert issubclass(NewtonSimulation, SimEngine)

    def test_constructor_accepts_no_args(self):
        from strands_robots_sim.newton import NewtonSimulation

        sim = NewtonSimulation()
        assert sim is not None

    def test_constructor_passes_kwargs_through(self):
        from strands_robots_sim.newton import NewtonSimulation

        sim = NewtonSimulation(num_envs=1024, device="cuda:0")
        assert sim._init_kwargs == {"num_envs": 1024, "device": "cuda:0"}

    def test_is_available_returns_tuple(self):
        from strands_robots_sim.newton import NewtonSimulation

        result = NewtonSimulation.is_available()
        assert isinstance(result, tuple)
        assert len(result) == 2
        ok, reason = result
        assert isinstance(ok, bool)
        assert reason is None or isinstance(reason, str)

    def test_is_available_reason_actionable_when_unavailable(self):
        """If the deps aren't installed, reason should hint at the install command."""
        from strands_robots_sim.newton import NewtonSimulation

        ok, reason = NewtonSimulation.is_available()
        if not ok:
            assert reason is not None
            assert "newton" in reason.lower() or "warp" in reason.lower()

    @pytest.mark.parametrize(
        "method_name,args",
        [
            ("create_world", ()),
            ("destroy", ()),
            ("reset", ()),
            ("step", ()),
            ("get_state", ()),
            ("add_robot", ("r1",)),
            ("remove_robot", ("r1",)),
            ("list_robots", ()),
            ("robot_joint_names", ("r1",)),
            ("add_object", ("o1",)),
            ("remove_object", ("o1",)),
            ("get_observation", ()),
            ("send_action", (None,)),
            ("render", ()),
        ],
    )
    def test_simengine_methods_raise_not_implemented(self, method_name, args):
        from strands_robots_sim.newton import NewtonSimulation

        sim = NewtonSimulation()
        with pytest.raises(NotImplementedError, match="pending R11|R11"):
            getattr(sim, method_name)(*args)
