"""strands_robots_sim.newton -- GPU-native Newton / Warp simulation backend.

This subpackage provides :class:`NewtonSimulation`, a ``SimEngine`` backend
built on **NVIDIA Warp + newton-physics** for differentiable, GPU-native
rigid-body and articulated dynamics. It complements the Isaac Sim backend
(``strands_robots_sim.isaac``) by trading photorealistic rendering for
massively-parallel physics throughput suitable for RL training.

Status
------
**Stub** — the simulation class is registered with the upstream
``strands_robots.backends`` entry-point group so ``create_simulation("newton")``
resolves correctly, but the SimEngine methods raise ``NotImplementedError``
("pending R11"). Full implementation lands in R11 (issue tracked under
the umbrella plan #8).

Usage (post-R11)::

    from strands_robots_sim.newton import NewtonSimulation
    sim = NewtonSimulation()
    ok, msg = sim.is_available()
    if ok:
        sim.create_world()
        ...

Until R11 lands, the only useful surface here is :meth:`NewtonSimulation.is_available`
which probes whether ``warp`` and ``newton`` can be imported on this host.

Install the runtime deps via the ``[newton]`` extra::

    pip install 'strands-robots-sim[newton]'   # warp-lang>=1.12, newton-physics>=1.0
"""

from __future__ import annotations

__all__ = ["NewtonSimulation"]


def _lazy_newton_simulation():
    """Lazy import to avoid pulling warp/newton at module-import time."""
    from strands_robots_sim.newton.simulation import NewtonSimulation

    return NewtonSimulation


def __getattr__(name: str):
    """PEP 562 lazy attribute access."""
    if name == "NewtonSimulation":
        return _lazy_newton_simulation()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
