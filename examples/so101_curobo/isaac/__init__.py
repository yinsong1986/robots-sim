# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim backend registration for the SO-101 demo (issue #67 T1).

Importing this package and calling :func:`register` wires ``IsaacSimulation``
into ``strands_robots.simulation.create_simulation`` under the name ``"isaac"``
(aliases ``isaac_sim``/``isaacsim``/``nvidia``) **at runtime**, without editing
the shared library's factory source. ``scene.make_sim("isaac")`` then resolves.

Use only inside the Isaac Sim Python environment (Python 3.10 venv with
``isaacsim`` installed); elsewhere :func:`register` is a no-op so the demo
degrades to MuJoCo.

Why this lives in the example (not ``strands_robots_sim.isaac``)
----------------------------------------------------------------
This is an *example-local* SimEngine adapter, intentionally kept separate
from the library backend at ``strands_robots_sim/isaac/`` -- it is **tracked
technical debt**, to be consolidated once the prerequisites land:

* The library backend targets the **Isaac Sim 5.x** ``omni.isaac.kit``
  namespace and is still a Phase-2 skeleton (the data-plane wiring lands in
  PRs #61 add_camera / #62 render / #63 USD-load / #64 URDF-load). This
  adapter targets the **Isaac Sim 4.5** ``isaacsim.*`` namespace -- the only
  Isaac actually installed on the dev box -- and runs the demo end-to-end
  today (GPU-validated).
* The two also differ in surface: the library takes ``IsaacConfig`` +
  ``add_robot(usd_path=/urdf_path=)``; this adapter implements the
  ``create_world(timestep=, gravity=, ground_plane=)`` + ``add_robot(
  urdf_path=)`` shape the example/collector use, plus the main-thread "pump"
  pattern the Gradio UI needs.

Consolidation into ``strands_robots_sim.isaac`` (abstracting the
4.5/5.x namespace split behind ``is_available()``/``ensure_app``) is tracked
as a follow-up to be done **after** PRs #61-64 and #68 merge -- see issue #69.
"""

from __future__ import annotations

import logging

from .simulation import IsaacSimulation, ensure_app, isaac_available

logger = logging.getLogger("so101_curobo.isaac")

__all__ = ["IsaacSimulation", "ensure_app", "isaac_available", "register"]

_REGISTERED = False


def register(force: bool = False) -> bool:
    """Register the Isaac backend with the strands_robots factory.

    Returns True if the backend is now registered (or already was), False if
    Isaac Sim isn't importable here. Safe to call multiple times.
    """
    global _REGISTERED
    if _REGISTERED and not force:
        return True
    if not isaac_available():
        logger.info("Isaac Sim not importable; skipping backend registration (will use MuJoCo).")
        return False
    from strands_robots.simulation.factory import register_backend

    register_backend(
        "isaac",
        lambda: IsaacSimulation,
        aliases=["isaac_sim", "isaacsim", "nvidia"],
        force=True,  # idempotent across re-imports / re-registration
    )
    _REGISTERED = True
    logger.info("Registered Isaac Sim backend: create_simulation('isaac') is now available.")
    return True
