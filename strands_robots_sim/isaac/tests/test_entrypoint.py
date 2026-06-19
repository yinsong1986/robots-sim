"""Entry-point + lazy-import tests for the Isaac backend skeleton.

Pins the three pieces of packaging surface that the Isaac backend
relies on:

1. The ``[isaac]`` extra is declared in ``pyproject.toml`` and pulls in
   only the genuinely pip-installable companion dep (``usd-core``) for
   the supported Isaac Sim 6.0 runtime. Isaac Sim itself is installed
   out-of-band (Launcher / Isaac Lab / NGC Docker), so the extra must
   **not** pin ``isaacsim`` / ``isaaclab`` (see #108).
2. The ``isaac`` entry point under ``strands_robots.backends``
   resolves to ``strands_robots_sim.isaac.simulation:IsaacSimulation``.
   ``isaac`` is the *only* backend name declared; the docs must not
   advertise any other alias (e.g. ``isaac_sim``) that packaging does
   not back.
3. ``import strands_robots_sim.isaac`` is a PEP 562 lazy stub: it loads
   without pulling any ``omni.*`` modules into ``sys.modules``.

Contracts that depend on the simulation module itself (``SimEngine``
subclassing, abstract-method completeness, ``is_available()`` return
shape, no-GPU constructor) are covered in ``test_unit.py``.

Run with:: pytest strands_robots_sim/isaac/tests/test_entrypoint.py -v
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import tomllib

import pytest

_PYPROJECT = pathlib.Path(__file__).resolve().parents[3] / "pyproject.toml"


class TestEntryPointDeclaration:
    """Validate that ``strands_robots.backends`` entry points are declared."""

    def test_pyproject_exists(self):
        assert _PYPROJECT.exists(), f"pyproject.toml not found at {_PYPROJECT}"

    def test_isaac_entry_point_declared_in_pyproject(self):
        """``isaac`` entry point points at the simulation module."""
        content = _PYPROJECT.read_text()
        assert 'isaac = "strands_robots_sim.isaac.simulation:IsaacSimulation"' in content, (
            'Expected `isaac = "strands_robots_sim.isaac.simulation:IsaacSimulation"` '
            'under [project.entry-points."strands_robots.backends"] in pyproject.toml.'
        )

    def test_isaac_is_the_only_backend_entry_point(self):
        """Exactly one backend name (``isaac``) is declared in pyproject.toml.

        Drift guard for #95: the docs previously advertised an ``isaac_sim``
        alias under ``strands_robots.backends`` that ``pyproject.toml`` never
        declared, so ``create_simulation("isaac_sim")`` would hit an
        unknown-backend error. Pin the declared name set to exactly
        ``{"isaac"}`` so any new alias must be added to packaging (and this
        test) deliberately rather than drifting in via docs alone.
        """
        data = tomllib.loads(_PYPROJECT.read_text())
        backends = data.get("project", {}).get("entry-points", {}).get("strands_robots.backends", {})
        declared = set(backends)
        assert declared == {"isaac"}, (
            f"strands_robots.backends declares {sorted(declared)}; expected "
            "exactly {'isaac'}. If you add a backend alias (e.g. isaac_sim), "
            "declare it in pyproject.toml AND update the docs + this test so "
            "docs and packaging cannot drift (see #95)."
        )

    def test_docs_do_not_advertise_undeclared_backend_aliases(self):
        """Docs must not list a backend entry-point name packaging omits.

        Drift guard for #95: four docs files and this module's docstring
        previously claimed an ``isaac_sim`` entry point. After the fix the
        only backend name anywhere is ``isaac``. This asserts no doc reintroduces
        an ``isaac_sim`` *entry-point* claim. The unrelated Isaac Lab install
        path token ``_isaac_sim`` (e.g. ``_isaac_sim/setup_python_env.sh``) is
        explicitly allowed.
        """
        repo_root = _PYPROJECT.parent
        docs_dir = repo_root / "docs"
        if not docs_dir.is_dir():
            pytest.skip("docs/ not present in this checkout")

        offenders = []
        for md in docs_dir.rglob("*.md"):
            for lineno, line in enumerate(md.read_text().splitlines(), 1):
                if "isaac_sim" not in line:
                    continue
                # Allow the Isaac Lab install-path token, not an entry point.
                if "_isaac_sim" in line or "setup_python_env" in line:
                    continue
                offenders.append(f"{md.relative_to(repo_root)}:{lineno}: {line.strip()}")

        assert not offenders, (
            "Docs advertise an `isaac_sim` backend name that pyproject.toml does "
            "not declare (see #95). Offending lines:\n" + "\n".join(offenders)
        )

    def test_isaac_extra_declared_in_pyproject(self):
        """``[project.optional-dependencies] isaac = [...]`` extra exists."""
        content = _PYPROJECT.read_text()
        assert "\nisaac = [" in content or "\nisaac=[" in content, (
            "Expected `isaac = [...]` under [project.optional-dependencies] declaring "
            "the pip-installable companion dep for Isaac Sim 6.0 (usd-core)."
        )

    def test_isaac_extra_pins_only_pip_installable_helpers(self):
        """``[isaac]`` ships only the genuinely pip-installable companion deps.

        Option A of #108: Isaac Sim is installed out-of-band (Omniverse
        Launcher / Isaac Lab / NGC Docker image), which ships a complete,
        bootable Kit. The extra therefore carries only ``usd-core`` and must
        NOT pin the ``isaacsim`` PyPI metapackage (nor ``isaaclab``):

          * Pinning ``isaacsim`` contradicted every other surface (README,
            docs/, _install.py) that says Isaac Sim is "not on PyPI".
          * NVIDIA's ``isaacsim[all]`` metapackage is incomplete on its own
            (omits the ``isaacsim-extscache-*`` packages), so ``SimulationApp``
            aborts at boot with an ``omni.ext`` dependency-resolution error.

        This pins the extra to the Option-A contract so the contradiction
        cannot drift back in.
        """
        content = _PYPROJECT.read_text()
        # crude but durable: locate the [isaac] block and check its body
        idx = content.find("\nisaac = [")
        assert idx != -1, "[isaac] extras block not found"
        block_end = content.find("]", idx)
        block = content[idx:block_end]
        assert "usd-core" in block, "[isaac] extras must include usd-core (the pip-installable USD runtime)"
        assert "isaacsim" not in block, (
            "[isaac] extras must NOT pin `isaacsim` — Isaac Sim is installed "
            "out-of-band (Launcher / Isaac Lab / NGC Docker), and the PyPI "
            "`isaacsim` metapackage is incomplete (omits isaacsim-extscache-*), "
            "so SimulationApp won't boot. See #108."
        )
        assert "isaaclab" not in block, (
            "[isaac] extras must NOT pin `isaaclab` — it is provided by the "
            "out-of-band Isaac Lab / Isaac Sim install (the fleet-RL path is a "
            "source install: `./isaaclab.sh -i`), not by this pip extra. Also, "
            "the only `isaaclab` on the index is the pre-release `3.0.0b2`, so a "
            "`>=3.0,<4.0` pin made the whole [isaac] extra unsatisfiable (PEP 440 "
            "excludes pre-releases by default). See #108 and #107."
        )

    def test_entry_points_visible_via_importlib_metadata_when_installed(self):
        """If the package is pip-installed in this env, entry points are discoverable."""
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
        if "isaac" not in names:
            pytest.skip(
                "Package installed but entry-point cache is stale -- reinstall after "
                "pyproject.toml change: `pip install -e . --force-reinstall --no-deps`."
            )

        for ep in backend_eps:
            if ep.name == "isaac":
                assert ep.value == "strands_robots_sim.isaac.simulation:IsaacSimulation", (
                    f"Entry point {ep.name!r} resolves to {ep.value!r}; expected "
                    "'strands_robots_sim.isaac.simulation:IsaacSimulation'."
                )


class TestLazyImportSurface:
    """Validate the PEP 562 lazy-import contract on the ``isaac`` subpackage."""

    def test_import_isaac_does_not_load_omni(self):
        """Importing ``strands_robots_sim.isaac`` adds zero ``omni.*`` modules."""
        import sys

        before = {k for k in sys.modules if k.startswith("omni")}
        import strands_robots_sim.isaac  # noqa: F401

        added = {k for k in sys.modules if k.startswith("omni")} - before
        assert added == set(), (
            f"Importing strands_robots_sim.isaac loaded omni modules: {sorted(added)}. "
            "The PEP 562 lazy stub must defer `omni` resolution until an attribute is accessed."
        )

    def test_isaac_subpackage_exposes_lazy_attrs_in___all__(self):
        """``__all__`` advertises the planned public surface."""
        import strands_robots_sim.isaac as isaac_pkg

        assert "IsaacSimulation" in isaac_pkg.__all__
        assert "IsaacConfig" in isaac_pkg.__all__

    def test_unknown_attr_raises_attributeerror(self):
        """Unknown attribute access raises AttributeError, not ImportError."""
        import strands_robots_sim.isaac as isaac_pkg

        with pytest.raises(AttributeError, match="no attribute 'NotARealClass'"):
            _ = isaac_pkg.NotARealClass

    def test_dunder_getattr_is_present(self):
        """The PEP 562 hook is defined at module level."""
        import strands_robots_sim.isaac as isaac_pkg

        assert hasattr(
            isaac_pkg, "__getattr__"
        ), "PEP 562 module-level __getattr__ must be defined for lazy import to work."
        assert callable(isaac_pkg.__getattr__)


class TestNewtonRemoved:
    """Pin the Isaac-only re-scope (#8, #89): no Newton/Warp surface remains.

    The package was re-scoped to ship Isaac Sim as the only heavy backend.
    Newton/Warp was dropped from #8 and all Newton tracking issues closed,
    but the code + packaging lagged behind. These assertions pin the
    contract so Newton can't silently drift back into the package, the
    extras, or the entry points.
    """

    def test_no_newton_package_dir(self):
        """The ``strands_robots_sim/newton/`` package is gone."""
        newton_dir = pathlib.Path(__file__).resolve().parents[2] / "newton"
        assert not newton_dir.exists(), (
            f"strands_robots_sim/newton/ still exists at {newton_dir}; the "
            "Isaac-only re-scope (#89) removes the Newton backend package."
        )

    def test_no_newton_or_warp_entry_points_in_pyproject(self):
        """No ``newton``/``warp`` entries under strands_robots.backends."""
        content = _PYPROJECT.read_text()
        assert "NewtonSimulation" not in content, (
            "pyproject.toml still references NewtonSimulation; the Isaac-only "
            "re-scope (#89) removes the newton/warp backend entry points."
        )
        assert "strands_robots_sim.newton" not in content, (
            "pyproject.toml still imports from strands_robots_sim.newton; "
            "the Newton backend package was removed (#89)."
        )

    def test_no_newton_extra_in_pyproject(self):
        """No ``[newton]`` optional-dependencies extra remains."""
        content = _PYPROJECT.read_text()
        assert "\nnewton = [" not in content and "\nnewton=[" not in content, (
            "pyproject.toml still declares a `newton = [...]` extra; the " "Isaac-only re-scope (#89) removes it."
        )
        # The `all` extra must not pull in the dropped [newton] extra.
        assert "strands-robots-sim[newton]" not in content, (
            "The `all` extra still references `strands-robots-sim[newton]`; "
            "it must reference only `[isaac]` after the re-scope (#89)."
        )

    def test_no_warp_lang_dependency(self):
        """``warp-lang``/``newton-physics`` deps are gone from packaging."""
        content = _PYPROJECT.read_text()
        assert "warp-lang" not in content, (
            "pyproject.toml still pins warp-lang (the Newton backend dep); " "removed by the Isaac-only re-scope (#89)."
        )
        assert "newton-physics" not in content, "pyproject.toml still pins newton-physics; removed by #89."

    def test_installed_entry_points_exclude_newton_and_warp(self):
        """If pip-installed, no newton/warp backend entry points resolve."""
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
        assert "newton" not in names and "warp" not in names, (
            f"Installed backend entry points still expose Newton/Warp: {sorted(names)}. "
            "Reinstall after the #89 re-scope: "
            "`pip install -e . --force-reinstall --no-deps`."
        )
