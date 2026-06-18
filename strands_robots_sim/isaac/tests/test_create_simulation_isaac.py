"""Regression tests for the documented Isaac quickstart path (#97).

`docs/index.md`, `README.md`, `docs/getting-started/quickstart.md`, and
`docs/simulation/overview.md` all previously opened with::

    import strands_robots_sim
    from strands_robots.simulation import create_simulation
    sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)

That snippet is **broken** against every released ``strands-robots`` this
package can install: the pinned floor (``strands-robots>=0.3.8,<0.4``) does
not walk the ``strands_robots.backends`` entry-point group from its
``create_simulation`` factory, so the call raises
``ValueError: Unknown simulation backend: 'isaac'`` (issue #97). The docs
now use the supported direct constructor::

    from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig
    sim = IsaacSimulation(IsaacConfig(render_mode="rtx_realtime", headless=True))

These tests pin both halves of that contract so neither can silently drift:

1. ``TestDocumentedDirectConstructor`` — the path the docs actually show
   must keep working on a CPU-only box (no ``omni.*`` import at construct
   time), accepting both the ``IsaacConfig`` and the kwargs forms.
2. ``TestCreateSimulationIsaacDiscovery`` — encodes the *current* upstream
   reality. While the pinned ``strands-robots`` lacks the entry-point
   walker, ``create_simulation("isaac")`` must raise the documented
   ``ValueError``; the moment a future upstream gains the walker, the same
   call must resolve to ``IsaacSimulation`` (and this test flips to assert
   that). Either way the docs and code stay in lockstep.

Run with::

    pytest strands_robots_sim/isaac/tests/test_create_simulation_isaac.py -v
"""

from __future__ import annotations

import sys

import pytest


class TestDocumentedDirectConstructor:
    """The constructor the quickstart docs show must keep working (#97)."""

    def test_isaac_subpackage_exports_constructor_symbols(self):
        """``from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig`` works."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        assert IsaacSimulation is not None
        assert IsaacConfig is not None

    def test_construct_from_isaac_config(self):
        """The documented ``IsaacSimulation(IsaacConfig(...))`` form constructs."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        sim = IsaacSimulation(IsaacConfig(render_mode="rtx_realtime", headless=True))
        assert sim._config.render_mode == "rtx_realtime"
        assert sim._config.headless is True

    def test_construct_from_kwargs(self):
        """The kwargs form ``IsaacSimulation(render_mode=..., headless=...)`` constructs.

        These are the same kwargs the docs note will one day flow through
        ``create_simulation("isaac", ...)`` into ``IsaacConfig``.
        """
        from strands_robots_sim.isaac import IsaacSimulation

        sim = IsaacSimulation(render_mode="rtx_pathtracing", headless=True)
        assert sim._config.render_mode == "rtx_pathtracing"
        assert sim._config.headless is True

    def test_constructor_is_cpu_safe(self):
        """Constructing ``IsaacSimulation`` must not import any ``omni.*`` module.

        The quickstart is meant to be copy-pasteable on a CPU-only dev box
        up to the ``create_world()`` call. If construction eagerly imported
        ``omni`` the snippet would explode at line 1 on every non-GPU host.
        """
        before = {k for k in sys.modules if k.startswith("omni")}

        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        IsaacSimulation(IsaacConfig(headless=True))

        added = {k for k in sys.modules if k.startswith("omni")} - before
        assert added == set(), f"Constructing IsaacSimulation imported omni modules: {sorted(added)}"

    def test_is_a_simengine_subclass(self):
        """``IsaacSimulation`` is a ``SimEngine`` so it's drop-in for the agent loop."""
        # ``strands-robots`` is the runtime dep but is intentionally NOT
        # installed in the lint/test hatch env (skip-install=true). Skip
        # cleanly there; the assertion runs in any env that has it.
        sim_mod = pytest.importorskip("strands_robots.simulation")

        from strands_robots_sim.isaac import IsaacSimulation

        assert issubclass(IsaacSimulation, sim_mod.SimEngine)


def _isaac_via_factory():
    """Call ``create_simulation('isaac')`` and return (sim, error).

    Exactly one of the two is non-None. Construction kwargs match the
    documented quickstart (headless so no display / GPU is needed for the
    resolution step itself).
    """
    from strands_robots.simulation import create_simulation

    try:
        sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)
        return sim, None
    except Exception as exc:  # noqa: BLE001 - we classify it below
        return None, exc


class TestCreateSimulationIsaacDiscovery:
    """Pin the ``create_simulation('isaac')`` contract against the pinned upstream (#97).

    The pinned ``strands-robots>=0.3.8,<0.4`` has no entry-point walker, so
    the factory cannot resolve ``"isaac"``. This test asserts the documented
    failure mode today and auto-flips to assert success once an upstream
    release gains the walker (so the docs' "one day this collapses to
    ``create_simulation('isaac')``" promise is itself guarded).
    """

    def test_factory_either_resolves_isaac_or_raises_the_documented_error(self):
        # ``strands-robots`` provides ``create_simulation`` but is not
        # installed in the lint/test hatch env (skip-install=true). Skip
        # cleanly there; runs anywhere the runtime dep is present.
        pytest.importorskip("strands_robots.simulation")

        import strands_robots_sim  # noqa: F401 - parity with the (former) doc snippet

        sim, err = _isaac_via_factory()

        if err is None:
            # Upstream gained the entry-point walker: the docs' forward-looking
            # promise is now real. Guard that "isaac" resolves to *our* class.
            from strands_robots_sim.isaac import IsaacSimulation

            assert isinstance(sim, IsaacSimulation), (
                f"create_simulation('isaac') resolved to {type(sim)!r}; expected IsaacSimulation. "
                "An upstream strands-robots now walks strands_robots.backends but routed "
                "'isaac' to the wrong class."
            )
            return

        # No walker yet: the call must fail with the exact documented error
        # so the quickstart docs (which use the direct constructor) stay honest.
        assert isinstance(err, ValueError), (
            f"create_simulation('isaac') raised {type(err).__name__}: {err!r}; expected ValueError. "
            "If upstream now resolves 'isaac', this test auto-detects it via the err-is-None branch."
        )
        assert "isaac" in str(err).lower(), (
            f"create_simulation('isaac') raised an unexpected ValueError: {err!r}. "
            "Expected the 'Unknown simulation backend: isaac' message."
        )

    def test_entry_point_is_registered_even_though_factory_cannot_use_it_yet(self):
        """The ``isaac`` entry point is declared/discoverable regardless of the walker.

        This is the forward-compatible plumbing: once upstream walks the
        group, no change to this package is required (see docs/architecture.md).
        Skips cleanly if the package isn't pip-installed in this env.
        """
        import importlib.metadata

        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            backend_eps = list(eps.select(group="strands_robots.backends"))
        else:  # pragma: no cover - Python <3.10 shape
            backend_eps = eps.get("strands_robots.backends", [])

        names = {ep.name for ep in backend_eps}
        if not backend_eps or "isaac" not in names:
            pytest.skip(
                "strands-robots-sim not pip-installed (or entry-point cache stale). "
                "Run `pip install -e .` to validate the entry point locally."
            )

        isaac_ep = next(ep for ep in backend_eps if ep.name == "isaac")
        assert isaac_ep.value == "strands_robots_sim.isaac.simulation:IsaacSimulation"
