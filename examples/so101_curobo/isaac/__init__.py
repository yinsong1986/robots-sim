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
