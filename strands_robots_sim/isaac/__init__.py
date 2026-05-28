"""strands_robots_sim.isaac -- GPU-native Isaac Sim simulation backend.

This subpackage provides :class:`IsaacSimulation`, a ``SimEngine`` backend
built on **NVIDIA Isaac Sim / Omniverse** for photorealistic rendering,
synthetic data generation, and GPU-batched sensor simulation.

Usage::

    from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig
    config = IsaacConfig(num_envs=1, headless=True)
    sim = IsaacSimulation(config)
    ok, msg = IsaacSimulation.is_available()

Requires NVIDIA Isaac Sim 2024.x+ (not pip-installable). Install via
Omniverse Launcher, Isaac Lab, or the NGC docker image. The exact
supported image tag and install commands live in
:mod:`strands_robots_sim.isaac._install` -- update there, not here.
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
    """PEP 562 lazy attribute access."""
    if name == "IsaacSimulation":
        return _lazy_isaac_simulation()
    if name == "IsaacConfig":
        return _lazy_isaac_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
