"""Newton / Warp simulation backend -- stub awaiting R11 implementation.

This module contains :class:`NewtonSimulation`, a ``SimEngine``-shaped
stub that resolves the ``newton`` and ``warp`` entry points declared in
``pyproject.toml`` without requiring the real implementation (which
lands in R11 of the umbrella plan, #8).

The shape mirrors :class:`strands_robots_sim.isaac.simulation.IsaacSimulation`
so that ``create_simulation("newton")`` returns an object with the same
SimEngine surface as the Isaac backend; every operational method raises
``NotImplementedError("pending R11")`` so callers fail fast and clearly.

The class is intentionally lightweight: the only method with real logic
is :meth:`NewtonSimulation.is_available`, which is what
``create_simulation("newton")`` would consult to decide whether the
backend is usable on this host.

Architecture (post-R11)
-----------------------
- All Warp / newton-physics imports are lazy (not at module level).
- The class will manage a Warp simulation device, articulation handles,
  and contact buffers.
- Rendering is intentionally out of scope for this backend; pair with
  the Isaac backend for photorealistic frames or use the upstream
  MuJoCo backend for offscreen rendering.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


try:
    from strands_robots.simulation.base import SimEngine
except (ImportError, ModuleNotFoundError):
    # Fallback: strands-robots < 0.4.0 doesn't have simulation.base yet.
    # Provide a minimal ABC stub so the class can still be defined.
    # Mirrors the isaac/simulation.py fallback ABC; same surface.
    from abc import ABC, abstractmethod

    class SimEngine(ABC):  # type: ignore[no-redef]
        """Minimal fallback ABC when strands-robots.simulation.base is unavailable."""

        @abstractmethod
        def create_world(self, **kwargs): ...
        @abstractmethod
        def destroy(self): ...
        @abstractmethod
        def reset(self): ...
        @abstractmethod
        def step(self, n_steps: int = 1): ...
        @abstractmethod
        def get_state(self): ...
        @abstractmethod
        def add_robot(self, name: str, **kwargs): ...
        @abstractmethod
        def remove_robot(self, name: str): ...
        @abstractmethod
        def list_robots(self) -> list: ...
        @abstractmethod
        def robot_joint_names(self, robot_name: str) -> list: ...
        @abstractmethod
        def add_object(self, name: str, **kwargs): ...
        @abstractmethod
        def remove_object(self, name: str): ...
        @abstractmethod
        def get_observation(self, robot_name=None, *, skip_images=False): ...
        @abstractmethod
        def send_action(self, action, robot_name=None, n_substeps: int = 1): ...
        @abstractmethod
        def render(self, camera_name: str = "default", width=None, height=None): ...

        def cleanup(self) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.cleanup()


_PENDING = "Newton/Warp backend implementation is pending R11 (see #8)."


class NewtonSimulation(SimEngine):
    """GPU-native simulation backend built on NVIDIA Warp + newton-physics.

    **Status: stub.** The class resolves the ``newton`` and ``warp``
    entry points declared under ``[project.entry-points."strands_robots.backends"]``
    so the upstream factory can hand callers an object of the right type;
    every operational method raises :class:`NotImplementedError` until
    the R11 implementation lands.

    Use :meth:`is_available` to check whether the host has the runtime
    deps (``warp-lang``, ``newton-physics``) installed -- that is the
    only contract this stub honours.

    Examples
    --------
    >>> sim = NewtonSimulation()
    >>> ok, msg = sim.is_available()
    >>> ok  # True iff `pip install 'strands-robots-sim[newton]'` succeeded
    False
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Stash any kwargs the caller passes (e.g. ``num_envs=1024``) so
        # the eventual R11 implementation can validate them up-front.
        # Not used today; the stub stays inert until a SimEngine method
        # is invoked, at which point we raise.
        self._init_args = args
        self._init_kwargs = kwargs
        logger.info("NewtonSimulation stub instantiated; %s", _PENDING)

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        """Check whether the Newton/Warp runtime is importable.

        Probes ``warp`` and ``newton`` without importing them so the
        function is cheap to call repeatedly and has no side effects.

        Returns
        -------
        tuple[bool, str | None]
            ``(True, None)`` if both ``warp`` and ``newton`` are
            importable on this host; ``(False, reason)`` otherwise. The
            reason string is suitable for surfacing to a user or agent
            (e.g. it includes the pip command that would install the
            missing dep).
        """
        import importlib.util

        # warp-lang exposes the ``warp`` import name.
        try:
            warp_spec = importlib.util.find_spec("warp")
        except ModuleNotFoundError:
            warp_spec = None
        if warp_spec is None:
            return (
                False,
                "warp-lang not installed. Run "
                "`pip install 'strands-robots-sim[newton]'` to add Warp "
                "and newton-physics.",
            )

        # newton-physics exposes the ``newton`` import name.
        try:
            newton_spec = importlib.util.find_spec("newton")
        except ModuleNotFoundError:
            newton_spec = None
        if newton_spec is None:
            return (
                False,
                "newton-physics not installed. Run " "`pip install 'strands-robots-sim[newton]'` to add it.",
            )

        return True, None

    # ------------------------------------------------------------------
    # SimEngine surface -- everything below raises until R11.
    # ------------------------------------------------------------------

    def create_world(self, **kwargs: Any) -> Any:
        raise NotImplementedError(_PENDING)

    def destroy(self) -> Any:
        raise NotImplementedError(_PENDING)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(_PENDING)

    def step(self, n_steps: int = 1) -> Any:
        raise NotImplementedError(_PENDING)

    def get_state(self) -> Any:
        raise NotImplementedError(_PENDING)

    def add_robot(self, name: str, **kwargs: Any) -> Any:
        raise NotImplementedError(_PENDING)

    def remove_robot(self, name: str) -> Any:
        raise NotImplementedError(_PENDING)

    def list_robots(self) -> list:
        raise NotImplementedError(_PENDING)

    def robot_joint_names(self, robot_name: str) -> list:
        raise NotImplementedError(_PENDING)

    def add_object(self, name: str, **kwargs: Any) -> Any:
        raise NotImplementedError(_PENDING)

    def remove_object(self, name: str) -> Any:
        raise NotImplementedError(_PENDING)

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> Any:
        raise NotImplementedError(_PENDING)

    def send_action(self, action: Any, robot_name: str | None = None, n_substeps: int = 1) -> Any:
        raise NotImplementedError(_PENDING)

    def render(self, camera_name: str = "default", width: int | None = None, height: int | None = None) -> Any:
        raise NotImplementedError(_PENDING)
