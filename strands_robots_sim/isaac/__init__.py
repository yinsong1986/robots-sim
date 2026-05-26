"""strands_robots_sim.isaac -- GPU-native Isaac Sim simulation backend.

This subpackage will provide :class:`IsaacSimulation`, a ``SimEngine`` backend
built on **NVIDIA Isaac Sim / Omniverse** for photorealistic rendering,
synthetic data generation, and GPU-batched sensor simulation.

Usage (once :mod:`strands_robots_sim.isaac.simulation` and
:mod:`strands_robots_sim.isaac.config` land in subsequent PRs)::

    from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig
    config = IsaacConfig(num_envs=1, headless=True)
    sim = IsaacSimulation(config)
    ok, msg = IsaacSimulation.is_available()

This module ships in PR-1 of the #31 split (see issue #42); the lazy stubs
below are wired to the planned module layout so the `[isaac]` extra and
``strands_robots.backends`` entry points already declared in
``pyproject.toml`` resolve to the right import paths once the simulation +
config modules land in PR-2 / PR-4. ``import strands_robots_sim.isaac`` adds
zero ``omni.*`` modules to ``sys.modules`` -- by design, so a CPU-only CI
host can introspect the entry-point graph without paying any GPU import
cost.

Requires NVIDIA Isaac Sim 2024.x+ (not pip-installable).
Install via Omniverse Launcher or ``nvcr.io/nvidia/isaac-sim:4.5.0``.
"""

from __future__ import annotations

__all__ = ["IsaacSimulation", "IsaacConfig"]


def _lazy_isaac_simulation():
    """Lazy import to avoid pulling omni/Isaac at module-import time."""
    from strands_robots_sim.isaac.simulation import IsaacSimulation

    return IsaacSimulation


def _lazy_isaac_config():
    """Lazy import to avoid pulling dataclass internals at import time."""
    from strands_robots_sim.isaac.config import IsaacConfig

    return IsaacConfig


def __getattr__(name: str):
    """PEP 562 lazy attribute access.

    Returns the real classes if the corresponding submodule is importable,
    otherwise raises ``ImportError`` with a hint pointing at the missing
    PR in the #31 split.
    """
    if name == "IsaacSimulation":
        return _lazy_isaac_simulation()
    if name == "IsaacConfig":
        return _lazy_isaac_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
