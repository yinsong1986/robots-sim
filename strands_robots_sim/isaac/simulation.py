"""Isaac Sim simulation backend -- GPU-native SimEngine implementation.

This module contains :class:`IsaacSimulation`, the primary implementation
of the ``SimEngine`` ABC for the NVIDIA Isaac Sim backend.

Architecture:
    - All heavy omni/Isaac imports are lazy (not at module level)
    - The class manages an Isaac Sim ``World``, ``Articulation`` handles,
      and RTX camera instances
    - Multi-env replication uses ``omni.isaac.cloner.Cloner``
    - SimulationApp is a process-wide singleton (never create more than one)
    - Rendering delegates to Isaac Sim's RTX pipeline

Thread safety:
    - ``step()``, ``send_action()``, and ``get_observation()`` acquire
      ``self._lock`` to prevent data races
    - ``step()`` must not run concurrently with ``add_robot()``

Environment variables:
    - STRANDS_ISAAC_HEADLESS: Override headless mode (true/false)
    - STRANDS_ISAAC_RTX_PATHTRACING: Enable RTX pathtracing (true/false)
    - STRANDS_ISAAC_NUCLEUS_URL: Override Nucleus asset server URL
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, TypedDict

import numpy as np


class SimulationAppLaunchConfig(TypedDict, total=False):
    """Typed shape for ``omni.isaac.kit.SimulationApp`` launch config.

    All keys optional; SimulationApp accepts an open-ended dict and any
    additional keys are forwarded to Kit unchanged. The keys below are the
    well-known ones documented by NVIDIA across Isaac Sim 4.x / 5.x and are
    the ones a Strands tool would realistically expose to an agent.

    See: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html

    Keys
    ----
    headless : bool
        Run without GUI. Required True on cloud / CI runners.
    renderer : str
        ``"RayTracedLighting"`` or ``"PathTracing"``.
    width, height : int
        Viewport resolution in pixels.
    physics_gpu : int
        CUDA device index for PhysX.
    active_gpu : int
        CUDA device index for rendering.
    multi_gpu : bool
        Enable multi-GPU rendering.
    sync_loads : bool
        Block until USD assets finish loading.
    hide_ui : bool
        Hide Kit's editor UI in non-headless mode.
    anti_aliasing : int
        Anti-aliasing level (0-3).
    """

    headless: bool
    renderer: str
    width: int
    height: int
    physics_gpu: int
    active_gpu: int
    multi_gpu: bool
    sync_loads: bool
    hide_ui: bool
    anti_aliasing: int


try:
    from strands_robots.simulation.base import SimEngine
except (ImportError, ModuleNotFoundError):
    # Fallback: strands-robots < 0.4.0 doesn't have simulation.base yet.
    # Provide a minimal ABC stub so the class can still be defined.
    from abc import ABC, abstractmethod

    class SimEngine(ABC):  # type: ignore[no-redef]
        """Minimal fallback ABC when strands-robots.simulation.base is unavailable.

        Mirrors the abstract surface of the real
        ``strands_robots.simulation.base.SimEngine`` so subclasses fail fast at
        instantiation if a method is missing -- same contract whether
        ``strands_robots`` is installed or not.
        """

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


from strands_robots_sim.isaac.config import IsaacConfig  # noqa: E402  # late import: must follow SimEngine fallback def

logger = logging.getLogger(__name__)

# Module-level singleton tracking for SimulationApp
_SIMULATION_APP: Any = None
_SIMULATION_APP_LOCK = threading.Lock()


def _get_or_create_simulation_app(
    headless: bool = True,
    launch_config: SimulationAppLaunchConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Get or create the process-wide SimulationApp singleton.

    Isaac Sim's SimulationApp can only be created ONCE per process.
    This function ensures that constraint is respected.

    Parameters
    ----------
    headless : bool
        Run without GUI.
    launch_config : SimulationAppLaunchConfig, optional
        Typed launch config dict forwarded to ``omni.isaac.kit.SimulationApp``.
        See :class:`SimulationAppLaunchConfig` for documented keys
        (``renderer``, ``width``, ``height``, ``physics_gpu``,
        ``active_gpu``, ``multi_gpu``, ``sync_loads``, ``hide_ui``,
        ``anti_aliasing``). The explicit ``headless`` argument always
        wins over any ``"headless"`` key in ``launch_config``.
    **kwargs
        Additional SimulationApp launch keys (escape hatch for Kit
        options not in :class:`SimulationAppLaunchConfig`). Merged on
        top of ``launch_config``; ``headless`` argument still wins.

    Returns
    -------
    SimulationApp instance.

    Raises
    ------
    ImportError
        If omni.isaac.kit is not available.
    """
    global _SIMULATION_APP

    with _SIMULATION_APP_LOCK:
        if _SIMULATION_APP is not None:
            return _SIMULATION_APP

        try:
            from omni.isaac.kit import SimulationApp  # type: ignore[import-not-found]
        except ImportError as e:
            from strands_robots_sim.isaac._install import not_available_import_error

            raise ImportError(not_available_import_error()) from e

        # Layer order: typed launch_config base, then **kwargs escape hatch,
        # then explicit headless argument (always wins so the caller's
        # intent is unambiguous).
        merged: dict[str, Any] = dict(launch_config or {})
        merged.update(kwargs)
        merged["headless"] = headless
        _SIMULATION_APP = SimulationApp(merged)
        logger.info(
            "SimulationApp created (headless=%s). " "Note: this is a process-wide singleton.",
            headless,
        )
        return _SIMULATION_APP


class _RobotState:
    """Internal bookkeeping for a robot in the Isaac simulation."""

    def __init__(
        self,
        name: str,
        prim_path: str,
        joint_names: list[str],
        articulation: Any = None,
    ):
        self.name = name
        self.prim_path = prim_path
        self.joint_names = joint_names
        self.articulation = articulation


class _CameraState:
    """Internal bookkeeping for a camera in the Isaac simulation."""

    def __init__(self, name: str, prim_path: str, width: int, height: int):
        self.name = name
        self.prim_path = prim_path
        self.width = width
        self.height = height
        self.handle: Any = None


class _ObjectState:
    """Internal bookkeeping for an object (shape primitive) in the Isaac simulation.

    ``handle`` is the ``omni.isaac.core.objects.{Dynamic,Fixed}{Cuboid,Sphere,
    Cylinder,Capsule}`` instance returned by :meth:`IsaacSimulation.add_object`.
    The handle is what got registered with ``world.scene.add()`` and is the
    keyhole ``world.scene.remove_object(name)`` later uses for deletion. Held
    here so :meth:`IsaacSimulation.remove_object` doesn't have to round-trip
    through ``world.scene.get_object()`` (which can raise on a torn-down
    stage) just to find the prim.
    """

    def __init__(
        self,
        name: str,
        prim_path: str,
        shape: str,
        is_static: bool,
        handle: Any = None,
    ):
        self.name = name
        self.prim_path = prim_path
        self.shape = shape
        self.is_static = is_static
        self.handle = handle


class IsaacSimulation(SimEngine):
    """GPU-native simulation backend built on NVIDIA Isaac Sim.

    Implements the ``SimEngine`` ABC. Provides photorealistic rendering,
    RTX sensors, USD scene management, and fleet replication via Cloner.

    Parameters
    ----------
    config : IsaacConfig or None
        Configuration. If None, defaults are used.
    **kwargs
        Shortcut kwargs merged into config (e.g. ``num_envs=1024``).

    Examples
    --------
    >>> sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
    >>> ok, msg = IsaacSimulation.is_available()
    >>> if ok:
    ...     sim.create_world()
    ...     sim.add_robot("so100")
    ...     sim.step(100)
    ...     sim.destroy()
    """

    def __init__(self, config: IsaacConfig | None = None, **kwargs: Any) -> None:
        # Merge shortcut kwargs into config. Unknown kwargs are rejected
        # eagerly (rather than silently dropped) so a typo like
        # ``IsaacSimulation(headles=False)`` surfaces at construction time
        # instead of producing a default-config sim with no warning.
        import dataclasses

        if config is None:
            # IsaacConfig is a dataclass; passing an unknown kwarg raises
            # TypeError("__init__() got an unexpected keyword argument ...")
            # naturally. Both branches now have symmetric strictness.
            config = IsaacConfig(**kwargs)
        elif kwargs:
            fields = {f.name for f in dataclasses.fields(config)}
            unknown = sorted(set(kwargs) - fields)
            if unknown:
                raise TypeError(
                    f"IsaacSimulation got unexpected kwargs: {unknown}. " f"Known IsaacConfig fields: {sorted(fields)}."
                )
            config = dataclasses.replace(config, **kwargs)
        self._config = config

        # Simulation state (all lazy-initialized)
        self._app: Any = None
        self._world: Any = None

        # World state
        self._world_created = False
        self._replicated = False
        self._num_envs_active = 1
        self._sim_time = 0.0
        self._step_count = 0

        # Entity tracking
        self._robots: dict[str, _RobotState] = {}
        self._cameras: dict[str, _CameraState] = {}
        self._objects: dict[str, _ObjectState] = {}
        self._prim_registry: list[str] = []  # track all created prims for cleanup

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            "IsaacSimulation initialized: num_envs=%d, device=%s, headless=%s",
            config.num_envs,
            config.device,
            config.headless,
        )

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        """Check if Isaac Sim is available on this system.

        Returns
        -------
        tuple[bool, str | None]
            (available, reason_if_not). If available is True, reason is None.
        """
        # Probe what create_world() actually needs: omni.isaac.kit.SimulationApp.
        # The bare ``omni`` namespace is a PEP 420 namespace package shared by
        # omni.ui, omni.usd, partial Omniverse SDK installs, and Isaac-Lab
        # pre-bootstrap states -- its mere presence is not a reliable signal
        # that Isaac Sim is usable. ``importlib.util.find_spec`` checks the
        # specific submodule without importing it (no side effects); it
        # raises ModuleNotFoundError when a parent package along the dotted
        # path is missing, which we treat the same as "not available".
        import importlib.util

        try:
            kit_spec = importlib.util.find_spec("omni.isaac.kit")
        except ModuleNotFoundError:
            kit_spec = None
        if kit_spec is None:
            from strands_robots_sim.isaac._install import not_importable_reason

            return False, not_importable_reason()

        # Isaac requires CUDA
        try:
            import torch

            if not torch.cuda.is_available():
                return False, ("CUDA device not detected. Isaac Sim requires an NVIDIA GPU " "with CUDA support.")
        except ImportError:
            return False, ("PyTorch not installed. Isaac Sim requires torch with CUDA support.")

        return True, None

    @property
    def config(self) -> IsaacConfig:
        """Current configuration (read-only)."""
        return self._config

    # --- SimEngine: World Lifecycle ----------------------------------------

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
    ) -> dict[str, Any]:
        """Create a new simulation world in Isaac Sim.

        Initializes the SimulationApp (singleton), creates a USD stage,
        configures physics, and optionally adds a ground plane.

        Parameters
        ----------
        timestep : float, optional
            Override physics_dt from config.
        gravity : list[float], optional
            Override gravity vector from config. [gx, gy, gz].
        ground_plane : bool
            Whether to add a ground plane. Default True.

        Returns
        -------
        dict
            Status dict with world info.
        """
        with self._lock:
            if self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "World already created. Call destroy() first."}],
                }

            try:
                # Create/get SimulationApp singleton
                self._app = _get_or_create_simulation_app(headless=self._config.headless)

                # Now safe to import Isaac core modules
                from omni.isaac.core import World  # type: ignore[import-not-found]

                dt = timestep if timestep is not None else self._config.physics_dt
                grav = gravity if gravity is not None else list(self._config.gravity)

                # Create World
                self._world = World(
                    stage_units_in_meters=1.0,
                    physics_dt=dt,
                    rendering_dt=self._config.rendering_dt,
                )

                # Set gravity
                # Isaac Sim 5.1: set_gravity takes a scalar magnitude, not a vector.
                # Extract the Z-component (convention: gravity points along -Z).
                gravity_magnitude = grav[2] if isinstance(grav, (list, tuple)) else grav
                self._world.get_physics_context().set_gravity(gravity_magnitude)

                # Add ground plane
                if ground_plane and self._config.ground_plane:
                    self._world.scene.add_default_ground_plane()
                    self._prim_registry.append(f"{self._config.stage_path}/defaultGroundPlane")

                # Reset world to initialize
                self._world.reset()

                self._world_created = True
                self._sim_time = 0.0
                self._step_count = 0

                logger.info(
                    "World created: dt=%.5f, gravity=%s, headless=%s",
                    dt,
                    grav,
                    self._config.headless,
                )

                # Surface a structured snapshot of the freshly-created
                # environment alongside the human-readable text. Agents
                # spinning up a sim can introspect device / dt / scene
                # config without re-querying via get_state().
                world_info = {
                    "physics_dt": dt,
                    "rendering_dt": self._config.rendering_dt,
                    "gravity": list(grav) if isinstance(grav, (list, tuple)) else [0.0, 0.0, float(grav)],
                    "ground_plane": bool(ground_plane and self._config.ground_plane),
                    "stage_path": self._config.stage_path,
                    "stage_units_in_meters": 1.0,
                    "device": self._config.device,
                    "headless": self._config.headless,
                    "render_mode": self._config.render_mode,
                    "num_envs": self._config.num_envs,
                    "num_envs_active": self._num_envs_active,
                    "replicated": self._replicated,
                    "sim_time": self._sim_time,
                    "step_count": self._step_count,
                }

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Isaac Sim world created. "
                                f"dt={dt:.5f}, gravity={grav}, "
                                f"device={self._config.device}, "
                                f"headless={self._config.headless}"
                            ),
                            "json": world_info,
                        }
                    ],
                }

            except ImportError as e:
                return {
                    "status": "error",
                    "content": [
                        {"text": (f"Isaac Sim import failed: {e}. " "Ensure Isaac Sim is installed and accessible.")}
                    ],
                }
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:
                # Cleanup on partial failure. Narrow to what World() /
                # set_gravity / add_default_ground_plane / reset actually
                # raise on Isaac: RuntimeError (Carb / sim init), ValueError
                # (USD prim shape mismatches, e.g. set_init_state on the
                # ground plane), OSError (USD/Nucleus IO), AttributeError
                # (omni surface drift across SDK versions), TypeError
                # (Isaac Sim 5.1 ``set_gravity`` rejects non-scalar input
                # — see #52; defence in depth for similar argument-shape
                # surface drift on neighbouring physics-context calls).
                # Programming bugs (NameError, ImportError-not-already-
                # caught above) propagate.
                self._world = None
                logger.error("Failed to create Isaac world: %s", e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to create world: {e}"}],
                }

    def destroy(self) -> dict[str, Any]:
        """Destroy the simulation world and release resources.

        Note: SimulationApp is NOT shut down (it is process-wide).
        Only the World/Stage are cleared.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "No world to destroy."}],
                }

            # Capture pre-teardown counts so the structured json payload
            # surfaces what was actually released (the agent's get_state()
            # window is gone after destroy() returns).
            num_robots_released = len(self._robots)
            num_cameras_released = len(self._cameras)
            num_objects_released = len(self._objects)
            num_prims_released = len(self._prim_registry)
            num_envs_released = self._num_envs_active
            sim_time_at_destroy = self._sim_time
            step_count_at_destroy = self._step_count

            try:
                if self._world is not None:
                    self._world.stop()
                    self._world.clear_instance()
                    self._world = None
            except (RuntimeError, OSError, AttributeError) as e:
                # World.stop() / clear_instance() can raise on partial init
                # or on a torn-down stage; AttributeError covers omni surface
                # drift across versions. Logged at WARNING because we still
                # mark the world destroyed below; programming bugs propagate.
                logger.warning("World cleanup warning: %s", e)

            # Clear entity tracking
            self._robots.clear()
            self._cameras.clear()
            self._objects.clear()
            self._prim_registry.clear()

            # Reset state
            self._world_created = False
            self._replicated = False
            self._num_envs_active = 1
            self._sim_time = 0.0
            self._step_count = 0

            logger.info("World destroyed. SimulationApp remains (process-wide singleton).")

            # Surface a structured snapshot of what teardown released
            # alongside the human-readable text. Mirrors the json content
            # block convention used by get_state() (L624) and create_world()
            # (L455) so an agent inspecting destroy() can confirm what was
            # actually torn down without re-querying.
            destroy_info = {
                "num_robots_released": num_robots_released,
                "num_cameras_released": num_cameras_released,
                "num_objects_released": num_objects_released,
                "num_prims_released": num_prims_released,
                "num_envs_released": num_envs_released,
                "sim_time_at_destroy": sim_time_at_destroy,
                "step_count_at_destroy": step_count_at_destroy,
                "stage_path": self._config.stage_path,
                "simulation_app_alive": True,  # singleton survives destroy()
            }

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "Isaac Sim world destroyed. All resources released. "
                            "SimulationApp singleton remains active."
                        ),
                        "json": destroy_info,
                    }
                ],
            }

    def reset(self, env_ids: list[int] | None = None) -> dict[str, Any]:
        """Reset simulation to initial state.

        Parameters
        ----------
        env_ids : list[int], optional
            Specific environment indices to reset. If None, reset all.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._world is not None:
                self._world.reset()

            self._sim_time = 0.0
            self._step_count = 0

            if env_ids is None:
                msg = "Full reset complete."
            else:
                msg = f"Partial reset complete for {len(env_ids)} envs."

            return {"status": "success", "content": [{"text": msg}]}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by n physics steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to take. Default 1.

        Returns
        -------
        dict
            Status dict with timing info.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._world is None:
                return {"status": "error", "content": [{"text": "World not initialized."}]}

            t0 = time.perf_counter()

            for _ in range(n_steps):
                self._world.step(render=self._config.render_mode != "headless")
                self._sim_time += self._config.physics_dt
                self._step_count += 1

            elapsed = time.perf_counter() - t0
            steps_per_sec = n_steps / elapsed if elapsed > 0 else float("inf")

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Stepped {n_steps}x. "
                            f"sim_time={self._sim_time:.4f}s, "
                            f"wall={elapsed * 1000:.1f}ms, "
                            f"{steps_per_sec:.0f} steps/sec"
                        )
                    }
                ],
            }

    def get_state(self) -> dict[str, Any]:
        """Get full simulation state summary.

        Returns
        -------
        dict
            Status dict with state information.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            state_data = {
                "sim_time": self._sim_time,
                "step_count": self._step_count,
                "num_envs": self._num_envs_active,
                "num_robots": len(self._robots),
                "num_cameras": len(self._cameras),
                "num_objects": len(self._objects),
                "stage_path": self._config.stage_path,
                "device": self._config.device,
                "headless": self._config.headless,
                "render_mode": self._config.render_mode,
            }

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"State: t={self._sim_time:.4f}s, "
                            f"step={self._step_count}, "
                            f"envs={self._num_envs_active}, "
                            f"robots={len(self._robots)}, "
                            f"cameras={len(self._cameras)}, "
                            f"objects={len(self._objects)}"
                        ),
                        "json": state_data,
                    }
                ],
            }

    # --- SimEngine: Robot Management ----------------------------------------

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        mjcf_path: str | None = None,
        usd_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation.

        Parameters
        ----------
        name : str
            Robot identifier (also used for procedural lookup).
        urdf_path : str, optional
            Path to URDF file.
        mjcf_path : str, optional
            Path to MJCF file.
        usd_path : str, optional
            Path to USD file (native Isaac format).
        data_config : str, optional
            Named data config for procedural lookup.
        position : list[float], optional
            Base position [x, y, z].
        orientation : list[float], optional
            Base orientation as quaternion [w, x, y, z].

        Returns
        -------
        dict
            Status dict with robot info.
        """
        with self._lock:
            if not self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "No world created. Call create_world() first."}],
                }

            if name in self._robots:
                return {
                    "status": "error",
                    "content": [{"text": f"Robot '{name}' already exists."}],
                }

            if self._replicated:
                return {
                    "status": "error",
                    "content": [{"text": "Cannot add robots after replicate(). Call destroy() first."}],
                }

            pos = position or [0.0, 0.0, 0.0]
            prim_path = f"{self._config.stage_path}/Robots/{name}"

            # Try procedural first
            lookup_name = data_config or name
            try:
                from strands_robots_sim.isaac.procedural import get_procedural_robot

                procedural = get_procedural_robot(lookup_name)
            except ImportError:
                procedural = None

            if procedural is not None:
                # Build procedurally via USD API
                joint_names = procedural.joint_names
                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                )
                self._robots[name] = robot_state

                logger.info("Added robot '%s' (procedural, %d joints)", name, len(joint_names))
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Robot '{name}' added (procedural: {procedural.name}, "
                                f"{len(joint_names)} joints: {joint_names})"
                            )
                        }
                    ],
                }

            elif usd_path is not None:
                # Load from USD (native Isaac format)
                joint_names = self._load_usd_robot(prim_path, usd_path, pos)
                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                )
                self._robots[name] = robot_state

                return {
                    "status": "success",
                    "content": [{"text": (f"Robot '{name}' added (USD: {usd_path}, " f"{len(joint_names)} joints)")}],
                }

            elif urdf_path is not None:
                # Convert URDF to USD and load
                joint_names = self._load_urdf_robot(prim_path, urdf_path, pos)
                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                )
                self._robots[name] = robot_state

                return {
                    "status": "success",
                    "content": [{"text": (f"Robot '{name}' added (URDF: {urdf_path}, " f"{len(joint_names)} joints)")}],
                }

            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Robot '{lookup_name}' not found in procedural registry "
                                "and no usd_path/urdf_path provided. "
                                "Available procedural robots: so100, panda, unitree_g1"
                            )
                        }
                    ],
                }

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        size: list[float] | None = None,
        color: list[float] | None = None,
        mass: float = 0.1,
        is_static: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add an object (shape primitive) to the scene.

        Phase 2 wiring (#14): instantiates the underlying USD prim via
        ``omni.isaac.core.objects.{Dynamic,Fixed}{Cuboid,Sphere,Cylinder,
        Capsule}`` and registers it with ``world.scene.add()``. In Phase 1
        this method silently returned ``status: "success"`` without
        creating any prim; that path is gone -- callers that previously
        relied on the silent-no-op shape will now see a real geometry on
        the stage and a ``DynamicXxx``/``FixedXxx`` handle in the
        scene.

        Parameters
        ----------
        name : str
            Object identifier. Must be unique across the simulation; a
            duplicate is rejected with a structured error envelope rather
            than silently overwriting the existing prim.
        shape : str
            Shape type: ``"box"`` (default), ``"sphere"``, ``"capsule"``,
            ``"cylinder"``. Anything else returns a structured error
            envelope listing the valid set.
        position : list[float], optional
            World-space position ``[x, y, z]`` in meters. Default
            ``[0.0, 0.0, 0.5]`` (50 cm above origin so an object dropped
            with the default ground plane doesn't intersect it).
        orientation : list[float], optional
            World-space orientation as a quaternion ``[w, x, y, z]``.
            Default ``[1.0, 0.0, 0.0, 0.0]`` (identity).
        size : list[float], optional
            Shape dimensions in meters. Conventions per shape:

            * ``box``:      ``[width, height, depth]`` (default ``[0.05, 0.05, 0.05]``).
            * ``sphere``:   ``[radius]`` (default ``[0.05]``).
            * ``cylinder``: ``[radius, height]`` (default ``[0.05, 0.10]``).
            * ``capsule``:  ``[radius, height]`` (default ``[0.05, 0.10]``).

            Lists shorter than the convention fall back to defaults for
            the missing trailing components.
        color : list[float], optional
            RGB color ``[r, g, b]`` in ``[0, 1]``. RGBA lists (length 4)
            are accepted; alpha is dropped (Isaac's primitive constructors
            take RGB only). ``None`` -> default white.
        mass : float
            Mass in kg. Default 0.1. Ignored when ``is_static=True``.
        is_static : bool
            If ``True``, the prim is constructed via ``Fixed{Cuboid,
            Sphere, Cylinder, Capsule}`` and stays pinned in space. If
            ``False`` (default), uses the ``Dynamic*`` counterpart and
            participates in physics with ``mass``.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text", "json"}]}``
            envelope. ``json`` carries the resolved ``prim_path``,
            ``shape``, ``position``, ``orientation``, ``size``,
            ``mass``, and ``is_static`` so an agent can confirm what
            actually landed on the stage without re-querying.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            # Validate shape
            valid_shapes = ("box", "sphere", "capsule", "cylinder")
            if shape not in valid_shapes:
                return {
                    "status": "error",
                    "content": [{"text": f"Unknown shape: {shape!r}. Valid: {valid_shapes}"}],
                }

            if name in self._objects:
                return {
                    "status": "error",
                    "content": [{"text": f"Object '{name}' already exists."}],
                }

            pos = list(position) if position is not None else [0.0, 0.0, 0.5]
            orient = list(orientation) if orientation is not None else [1.0, 0.0, 0.0, 0.0]
            size_in = list(size) if size is not None else None
            prim_path = f"{self._config.stage_path}/Objects/{name}"

            try:
                handle, resolved_size = self._create_shape_prim(
                    shape=shape,
                    prim_path=prim_path,
                    name=name,
                    position=pos,
                    orientation=orient,
                    size=size_in,
                    color=color,
                    mass=mass,
                    is_static=is_static,
                )
                # ``world.scene.add`` registers the wrapper so that
                # ``world.reset()`` re-initialises it on the same
                # ``post_reset`` callback as the ground plane and
                # robots. The return value is the (possibly wrapped)
                # handle Isaac uses internally; we keep our own ref in
                # ``_objects[name]`` so ``remove_object`` doesn't have
                # to round-trip through ``scene.get_object`` later.
                self._world.scene.add(handle)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                # Cleanup-clause shape mirrors create_world (line 467):
                # RuntimeError (Carb / sim init), ValueError (USD prim
                # shape mismatches, e.g. negative scale on a Dynamic*),
                # OSError (USD/Nucleus IO), AttributeError (omni surface
                # drift), TypeError (signature drift across SDK versions
                # -- see #52 for the gravity precedent), ImportError
                # (omni.isaac.core.objects unavailable on a partial
                # Isaac install). Programming bugs propagate.
                logger.error(
                    "Failed to add object '%s' (shape=%s, static=%s): %s",
                    name,
                    shape,
                    is_static,
                    e,
                )
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to add object '{name}' ({shape}): {e}"}],
                }

            self._prim_registry.append(prim_path)
            self._objects[name] = _ObjectState(
                name=name,
                prim_path=prim_path,
                shape=shape,
                is_static=is_static,
                handle=handle,
            )

            obj_info = {
                "name": name,
                "prim_path": prim_path,
                "shape": shape,
                "position": pos,
                "orientation": orient,
                "size": resolved_size,
                "mass": float(mass) if not is_static else 0.0,
                "is_static": bool(is_static),
            }
            logger.info(
                "Added object '%s' (shape=%s, pos=%s, mass=%.3f, static=%s)",
                name,
                shape,
                pos,
                mass,
                is_static,
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Object '{name}' added (shape={shape}, pos={pos}).",
                        "json": obj_info,
                    }
                ],
            }

    def _create_shape_prim(
        self,
        *,
        shape: str,
        prim_path: str,
        name: str,
        position: list[float],
        orientation: list[float],
        size: list[float] | None,
        color: list[float] | None,
        mass: float,
        is_static: bool,
    ) -> tuple[Any, list[float]]:
        """Construct the omni.isaac.core.objects shape wrapper.

        Returns the handle plus the resolved ``size`` list (defaults
        applied per shape) so :meth:`add_object` can surface the
        actually-used dimensions in its structured json payload.

        Lazy-imports ``omni.isaac.core.objects`` so the module loads
        cleanly without Isaac Sim installed (the call site only ever
        runs after :meth:`create_world` has booted ``SimulationApp``).
        """
        import numpy as np  # type: ignore[import-not-found]
        from omni.isaac.core.objects import (  # type: ignore[import-not-found]
            DynamicCapsule,
            DynamicCuboid,
            DynamicCylinder,
            DynamicSphere,
            FixedCapsule,
            FixedCuboid,
            FixedCylinder,
            FixedSphere,
        )

        common: dict[str, Any] = {
            "prim_path": prim_path,
            "name": name,
            "position": np.asarray(position, dtype=float),
            "orientation": np.asarray(orientation, dtype=float),
        }
        if color is not None:
            # RGBA -> RGB: Isaac's primitive constructors take a 3-vector
            # color; alpha would silently raise a shape mismatch deeper
            # in USD. Truncate here so RGBA-style examples (e.g. the #15
            # sketch's ``[1, 0, 0, 1]``) work transparently.
            rgb = list(color)[:3]
            common["color"] = np.asarray(rgb, dtype=float)
        if not is_static:
            common["mass"] = float(mass)

        if shape == "box":
            cls = FixedCuboid if is_static else DynamicCuboid
            # Per-component fallback to honour the docstring contract
            # ("Lists shorter than the convention fall back to defaults
            # for the missing trailing components"). Mirrors the
            # cylinder / capsule pattern below; previously this branch
            # was all-or-nothing, so e.g. ``size=[0.10]`` silently fell
            # back to ``[0.05, 0.05, 0.05]`` instead of the documented
            # ``[0.10, 0.05, 0.05]`` -- caught in PR #60 review.
            size_list = list(size) if size else []
            sx = float(size_list[0]) if len(size_list) >= 1 else 0.05
            sy = float(size_list[1]) if len(size_list) >= 2 else 0.05
            sz = float(size_list[2]) if len(size_list) >= 3 else 0.05
            scale = [sx, sy, sz]
            common["scale"] = np.asarray(scale, dtype=float)
            return cls(**common), scale
        if shape == "sphere":
            cls = FixedSphere if is_static else DynamicSphere
            radius = float(size[0]) if size and len(size) >= 1 else 0.05
            return cls(radius=radius, **common), [radius]
        if shape == "cylinder":
            cls = FixedCylinder if is_static else DynamicCylinder
            radius = float(size[0]) if size and len(size) >= 1 else 0.05
            height = float(size[1]) if size and len(size) >= 2 else 0.10
            return cls(radius=radius, height=height, **common), [radius, height]
        if shape == "capsule":
            cls = FixedCapsule if is_static else DynamicCapsule
            radius = float(size[0]) if size and len(size) >= 1 else 0.05
            height = float(size[1]) if size and len(size) >= 2 else 0.10
            return cls(radius=radius, height=height, **common), [radius, height]
        # Unreachable: shape was validated by add_object before this call;
        # raise loudly if a future caller bypasses that guard.
        raise ValueError(f"Unknown shape: {shape!r}")

    # --- SimEngine: Introspection / Removal ---------------------------------

    def list_robots(self) -> list[str]:
        """Return ordered list of robot names currently in the world.

        Returns
        -------
        list[str]
            Robot names in insertion order. Empty if no robots have been
            added (or after :meth:`destroy`).
        """
        with self._lock:
            return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Return ordered joint names for ``robot_name``.

        Parameters
        ----------
        robot_name : str
            Robot identifier previously passed to :meth:`add_robot`.

        Returns
        -------
        list[str]
            Joint names in articulation order, or an empty list if
            ``robot_name`` is not present (matches the silent-empty
            convention used by :meth:`get_observation` for unknown robots).
        """
        with self._lock:
            if robot_name not in self._robots:
                return []
            return list(self._robots[robot_name].joint_names)

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot from the simulation.

        Drops the robot's bookkeeping entry and prunes any prims rooted at
        the robot's prim path from ``self._prim_registry``. The actual USD
        prim deletion is delegated to :meth:`destroy` / world teardown in
        Phase 1; only the in-Python registry is updated here.

        Parameters
        ----------
        name : str
            Robot identifier previously passed to :meth:`add_robot`.

        Returns
        -------
        dict
            Status dict in the standard ``{"status", "content": [{"text"}]}``
            shape used by mutating methods on this class.
        """
        with self._lock:
            if name not in self._robots:
                return {
                    "status": "error",
                    "content": [{"text": f"Robot '{name}' not found."}],
                }
            prim_path = self._robots[name].prim_path
            self._prim_registry = [p for p in self._prim_registry if not p.startswith(prim_path)]
            del self._robots[name]
            logger.info("Removed robot '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Robot '{name}' removed."}],
            }

    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object from the scene.

        Phase 2 wiring (#14): paired with :meth:`add_object`'s prim
        creation. Calls ``world.scene.remove_object(name)`` to actually
        delete the USD prim, then prunes the in-Python registries
        (``_objects`` + ``_prim_registry``). In Phase 1 this method only
        updated the in-Python registry; that is no longer the case --
        the prim is gone from the stage when this returns.

        Parameters
        ----------
        name : str
            Object identifier previously passed to :meth:`add_object`.

        Returns
        -------
        dict
            Status dict in the standard ``{"status", "content": [{"text"}]}``
            shape used by mutating methods on this class. Returns ``error``
            if the object is unknown to ``_objects``; this is the only
            authoritative source -- ``_prim_registry`` is cleanup-time
            bookkeeping that does not distinguish robots from objects.
        """
        with self._lock:
            if name not in self._objects:
                return {
                    "status": "error",
                    "content": [{"text": f"Object '{name}' not found."}],
                }

            prim_path = self._objects[name].prim_path

            # Delete the prim from the world's scene. Wrapped in the same
            # cleanup-clause shape as add_object since the failure modes
            # mirror it: scene.remove_object can RuntimeError on a torn-
            # down stage, AttributeError on omni surface drift, etc.
            try:
                if self._world is not None:
                    self._world.scene.remove_object(name)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:
                logger.error("Failed to remove object '%s' (prim=%s): %s", name, prim_path, e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to remove object '{name}': {e}"}],
                }

            # Now drop our bookkeeping. The order matters: we only want
            # to forget the object after the scene call succeeded so a
            # transient ``RuntimeError`` from ``scene.remove_object``
            # leaves a retry-friendly state.
            del self._objects[name]
            if prim_path in self._prim_registry:
                self._prim_registry.remove(prim_path)

            logger.info("Removed object '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Object '{name}' removed."}],
            }

    # --- SimEngine: Observation / Action ------------------------------------

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """Get observation for a robot.

        Parameters
        ----------
        robot_name : str, optional
            Robot to observe. Auto-resolves if only one robot exists.
        skip_images : bool
            Skip camera rendering. Default False.

        Returns
        -------
        dict
            Observation with joint positions keyed by joint name. An empty dict
            indicates one of four diagnostically-distinct conditions, each of
            which is logged before return so silent failures are visible in
            operational logs:

            * ``world not yet created`` -- DEBUG (expected pre-init state).
            * ``ambiguous robot_name=None with multiple robots`` -- WARNING.
            * ``unknown robot_name`` (typo / not-yet-added) -- WARNING.
            * ``robot present but Articulation handle not yet initialised``
              (Phase 1 procedural / load stub) -- DEBUG via the inner except;
              the dict returns empty because no joint positions are reachable.

            The return shape is preserved as a plain dict (rather than the
            ``{"status": ..., "content": [...]}`` shape used by mutating
            methods on this class) because callers consume joint positions
            keyed by joint name; the four silent-``{}`` modes are distinguished
            in logs rather than in the return value.
        """
        with self._lock:
            if not self._world_created:
                # Expected pre-init state; many callers probe before
                # create_world() to feature-detect, so DEBUG-only.
                logger.debug(
                    "get_observation(robot_name=%r): world not yet created",
                    robot_name,
                )
                return {}

            # Resolve robot
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                else:
                    logger.warning(
                        "get_observation(robot_name=None): ambiguous -- "
                        "%d robots present (%s); pass robot_name explicitly. "
                        "Returning empty observation.",
                        len(self._robots),
                        sorted(self._robots),
                    )
                    return {}

            if robot_name not in self._robots:
                logger.warning(
                    "get_observation(robot_name=%r): unknown robot. Known: %s. " "Returning empty observation.",
                    robot_name,
                    sorted(self._robots),
                )
                return {}

            robot = self._robots[robot_name]
            obs: dict[str, Any] = {}

            # Get joint state from Articulation handle
            if robot.articulation is not None:
                try:
                    joint_positions = robot.articulation.get_joint_positions()
                    if joint_positions is not None:
                        positions = (
                            joint_positions.cpu().numpy()
                            if hasattr(joint_positions, "cpu")
                            else np.array(joint_positions)
                        )
                        for i, jname in enumerate(robot.joint_names):
                            if i < len(positions):
                                obs[jname] = float(positions[i])
                except (RuntimeError, ValueError, AttributeError, TypeError) as e:
                    # Articulation handle may raise RuntimeError on a not-yet
                    # -initialized world, AttributeError on torch-tensor surface
                    # drift, ValueError/TypeError on np coercion. Programming
                    # bugs propagate.
                    logger.debug("Failed to get joint positions: %s", e)

            return obs

    def send_action(
        self,
        action: dict[str, Any] | np.ndarray | list,
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> None:
        """Apply action and advance physics.

        Parameters
        ----------
        action : dict or array-like
            Joint targets. If dict, keyed by joint name.
        robot_name : str, optional
            Robot to control.
        n_substeps : int
            Physics sub-steps after applying action. Default 1.
        """
        with self._lock:
            if not self._world_created or self._world is None:
                return

            # Resolve robot
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                else:
                    return

            if robot_name not in self._robots:
                return

            robot = self._robots[robot_name]

            # Convert action to array
            if isinstance(action, dict):
                action_array = np.zeros(len(robot.joint_names), dtype=np.float32)
                for i, jname in enumerate(robot.joint_names):
                    if jname in action:
                        action_array[i] = float(action[jname])
            elif isinstance(action, np.ndarray):
                action_array = action.astype(np.float32).flatten()
            else:
                action_array = np.array(action, dtype=np.float32)

            # Apply to articulation
            if robot.articulation is not None:
                try:
                    robot.articulation.set_joint_position_targets(action_array)
                except (RuntimeError, ValueError, AttributeError) as e:
                    # set_joint_position_targets raises RuntimeError on a
                    # torn-down articulation, ValueError on shape mismatch,
                    # AttributeError on omni surface drift. Programming bugs
                    # (NameError, KeyError) propagate.
                    logger.debug("Failed to set joint targets: %s", e)

            # Step physics
            for _ in range(n_substeps):
                self._world.step(render=False)
                self._sim_time += self._config.physics_dt
                self._step_count += 1

    # --- SimEngine: Rendering -----------------------------------------------

    def render(
        self,
        camera_name: str = "default",
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """Render a camera view using Isaac Sim's RTX pipeline.

        Phase 2 wiring (#14): when a camera registered via
        :meth:`add_camera` carries a non-``None`` ``handle`` (i.e. the
        Phase-2 ``omni.isaac.sensor.Camera`` was successfully constructed)
        and the simulation isn't in ``headless`` render mode, this method
        pulls real frames via ``handle.get_rgba()`` + ``handle.get_depth()``.
        Otherwise returns blank frames -- four documented fallback paths,
        each tagged in the success envelope's text so a caller / agent
        can tell which path was taken without inspecting array contents:

        * ``Rendered (headless, no RTX)`` -- ``IsaacConfig.render_mode``
          is ``"headless"``; RTX path-tracing is unavailable. Most CI
          and GR00T server flows hit this path.
        * ``Rendered (no camera)`` -- ``camera_name`` is unknown to
          ``self._cameras``. Caller probably forgot to call ``add_camera``
          (or typo'd the name).
        * ``Rendered (Phase-1 camera, no RTX handle)`` -- the camera
          exists in ``self._cameras`` but its ``handle`` is ``None``.
          Happens when the camera was added before the
          ``add_camera`` Phase-2 wiring landed (or when the camera
          construction failed but bookkeeping was still seeded -- not
          possible after PR #61, but kept as a defensive fallback).
        * ``Rendered (RTX <render_mode>)`` -- Phase-2 path: real
          frames pulled from the Camera handle. ``rgb`` / ``depth``
          are the actual array shapes returned by Isaac (matching
          the camera's resolved resolution; not necessarily the
          ``width`` / ``height`` arguments passed to this method,
          which are only used to size the blank-frame fallbacks).

        Parameters
        ----------
        camera_name : str
            Camera identifier previously passed to :meth:`add_camera`.
            Default ``"default"``.
        width : int, optional
            Frame width for blank-frame fallbacks. Default from
            ``IsaacConfig.camera_width``. Ignored on the RTX path
            (the camera's own resolution wins).
        height : int, optional
            Frame height for blank-frame fallbacks. Default from
            ``IsaacConfig.camera_height``. Ignored on the RTX path.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text"}], "rgb", "depth"}``
            envelope. ``rgb`` is a uint8 ``(H, W, 3)`` array; ``depth``
            is a float32 ``(H, W)`` array. On the RTX path, ``content``
            also carries a ``json`` block with the resolved camera
            ``resolution``, ``prim_path``, and the boolean ``rtx`` flag
            so an agent can route on the path without parsing text.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            w = width or self._config.camera_width
            h = height or self._config.camera_height

            if self._config.render_mode == "headless":
                # Return blank frames in headless mode. Most CI flows
                # land here; Isaac's RTX path-tracer is unavailable.
                return {
                    "status": "success",
                    "rgb": np.zeros((h, w, 3), dtype=np.uint8),
                    "depth": np.zeros((h, w), dtype=np.float32),
                    "content": [{"text": f"Rendered (headless, no RTX): {w}x{h}"}],
                }

            if camera_name not in self._cameras:
                # No camera configured — return blank. Caller probably
                # forgot to call add_camera or typo'd the name.
                return {
                    "status": "success",
                    "rgb": np.zeros((h, w, 3), dtype=np.uint8),
                    "depth": np.zeros((h, w), dtype=np.float32),
                    "content": [{"text": f"Rendered (no camera): {w}x{h}"}],
                }

            cam = self._cameras[camera_name]

            if cam.handle is None:
                # Phase-1 camera (no Phase-2 handle was attached).
                # Defensive fallback: blank frames sized to the camera's
                # registered resolution rather than the method's
                # ``width`` / ``height`` arguments, since the camera's
                # resolution is what the caller asked for at add_camera.
                return {
                    "status": "success",
                    "rgb": np.zeros((cam.height, cam.width, 3), dtype=np.uint8),
                    "depth": np.zeros((cam.height, cam.width), dtype=np.float32),
                    "content": [{"text": (f"Rendered (Phase-1 camera, no RTX handle): " f"{cam.width}x{cam.height}")}],
                }

            # Phase-2 RTX path: pull real frames from the Camera handle.
            try:
                rgba = cam.handle.get_rgba()
                # ``get_rgba`` returns either ``(H, W, 4)`` or
                # ``(H, W, 3)`` depending on the Isaac Sim build. Slice
                # to RGB defensively so the returned shape is stable
                # for downstream agents.
                rgb = np.asarray(rgba)[..., :3]
                # A camera whose RTX render product hasn't accumulated a
                # frame yet (e.g. added after the last world step, not
                # warmed up) returns a malformed / empty buffer -- a 1-D
                # or 0-size array rather than ``(H, W, C)``. Guard so we
                # raise a structured RuntimeError (caught below) instead
                # of an unhandled IndexError when building the json
                # ``resolution`` from ``rgb.shape[1]``. Caught during the
                # isaac_gs example's GPU validation with multiple
                # freshly-added cameras.
                if rgb.ndim < 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
                    raise RuntimeError(
                        f"camera {camera_name!r} returned a malformed RGB buffer "
                        f"(shape {np.asarray(rgba).shape}); the RTX render product "
                        "likely hasn't accumulated a frame yet -- step the world a "
                        "few times after add_camera before rendering."
                    )
                depth_raw = cam.handle.get_depth()
                if depth_raw is None:
                    # Camera was constructed without the depth annotator
                    # (Isaac Sim ships rgba on by default but depth is
                    # opt-in via ``Camera.add_distance_to_image_plane_to_frame()``;
                    # PR #61's add_camera enables it post-initialize, but
                    # an older sim or a manually-attached Phase-1 camera
                    # state may not). Surface a zero-depth array sized to
                    # rgb so callers see a stable shape, plus a WARNING
                    # so misconfigured cameras don't silently produce
                    # zero-depth telemetry.
                    logger.warning(
                        "Camera '%s': get_depth() returned None (depth annotator not enabled). "
                        "Returning zero-depth array; "
                        "check add_distance_to_image_plane_to_frame() in add_camera.",
                        camera_name,
                    )
                    depth = np.zeros(rgb.shape[:2], dtype=np.float32)
                else:
                    depth = np.asarray(depth_raw)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:
                # Cleanup-clause shape mirrors create_world (#52
                # precedent). The Camera handle's ``get_rgba`` /
                # ``get_depth`` can raise on a not-yet-stepped world
                # (RTX render product hasn't accumulated samples) or
                # surface drift; surface as the structured error
                # envelope rather than letting the exception propagate.
                logger.error("Failed to render camera '%s': %s", camera_name, e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to render camera '{camera_name}': {e}"}],
                }

            render_info = {
                "rtx": True,
                "prim_path": cam.prim_path,
                "resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
                "render_mode": self._config.render_mode,
            }
            return {
                "status": "success",
                "rgb": rgb,
                "depth": depth,
                "content": [
                    {
                        "text": (f"Rendered (RTX {self._config.render_mode}): " f"{cam.width}x{cam.height}"),
                        "json": render_info,
                    }
                ],
            }

    def add_camera(
        self,
        name: str = "default",
        position: list[float] | None = None,
        target: list[float] | None = None,
        width: int | None = None,
        height: int | None = None,
        fov: float = 60.0,
    ) -> dict[str, Any]:
        """Add an RTX camera to the scene.

        Phase 2 wiring (#14): instantiates the underlying USD camera prim
        via ``omni.isaac.sensor.Camera`` and stores the handle on the
        ``_CameraState`` for later retrieval by :meth:`render`. In Phase 1
        this method silently returned ``status: "success"`` without
        creating any prim; that path is gone -- callers will now see a
        real camera prim on the stage and a Camera handle in
        ``self._cameras[name].handle``.

        ``render`` continues to return blank frames in Phase 2 because
        the actual ``camera.get_rgba()`` / annotator wiring is a separate
        slice. The Camera prim is the prerequisite, not the full frame
        path.

        Parameters
        ----------
        name : str
            Camera identifier. Default ``"default"``. Must be unique
            across the simulation; a duplicate is rejected with a
            structured error envelope.
        position : list[float], optional
            World-space position ``[x, y, z]`` in meters. Default
            ``[2.0, 2.0, 2.0]`` (an over-the-shoulder vantage that
            sees the default ground plane and any objects above it).
        target : list[float], optional
            World-space look-at point ``[x, y, z]``. If provided, the
            camera is oriented so its forward axis points at ``target``
            via ``omni.isaac.core.utils.viewports.set_camera_view``.
            If ``None``, the camera keeps its constructed orientation
            (identity).
        width : int, optional
            Image width in pixels. Defaults to ``IsaacConfig.camera_width``.
        height : int, optional
            Image height in pixels. Defaults to ``IsaacConfig.camera_height``.
        fov : float
            Horizontal field of view in degrees. Default 60.0. Mapped
            onto ``Camera.set_focal_length`` using the standard pinhole
            relation ``focal_length = horizontal_aperture / (2 * tan(fov/2))``
            with the USD-default 24 mm horizontal aperture.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text", "json"}]}``
            envelope. ``json`` carries the resolved ``prim_path``,
            ``position``, ``target``, ``resolution``, ``fov``, and the
            computed ``focal_length`` so an agent can confirm the
            camera setup without re-querying.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if name in self._cameras:
                return {
                    "status": "error",
                    "content": [{"text": f"Camera '{name}' already exists."}],
                }

            w = int(width or self._config.camera_width)
            h = int(height or self._config.camera_height)
            pos = list(position) if position is not None else [2.0, 2.0, 2.0]
            tgt = list(target) if target is not None else None
            fov_deg = float(fov)

            prim_path = f"{self._config.stage_path}/Cameras/{name}"

            try:
                handle, focal_length_mm = self._create_camera_prim(
                    name=name,
                    prim_path=prim_path,
                    position=pos,
                    target=tgt,
                    width=w,
                    height=h,
                    fov_deg=fov_deg,
                )
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                # Cleanup-clause shape mirrors create_world (#52 precedent)
                # and add_object: the constructor or initialise / look-at
                # call either succeeds and updates registries, or fails
                # with a structured envelope and updates neither.
                logger.error("Failed to add camera '%s' (prim=%s): %s", name, prim_path, e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to add camera '{name}': {e}"}],
                }

            self._prim_registry.append(prim_path)
            cam_state = _CameraState(name=name, prim_path=prim_path, width=w, height=h)
            cam_state.handle = handle
            self._cameras[name] = cam_state

            cam_info = {
                "name": name,
                "prim_path": prim_path,
                "position": pos,
                "target": tgt,
                "resolution": [w, h],
                "fov": fov_deg,
                "focal_length_mm": focal_length_mm,
            }
            logger.info(
                "Added camera '%s' at pos=%s target=%s res=%dx%d fov=%.1f",
                name,
                pos,
                tgt,
                w,
                h,
                fov_deg,
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": (f"Camera '{name}' added at {pos}, " f"resolution={w}x{h}, fov={fov_deg}"),
                        "json": cam_info,
                    }
                ],
            }

    def remove_camera(self, name: str) -> dict[str, Any]:
        """Remove a camera from the scene.

        Phase 2 wiring (#14): paired with :meth:`add_camera`'s prim
        creation. Deletes the underlying USD camera prim via
        ``omni.isaac.core.utils.prims.delete_prim`` and prunes the
        in-Python registries. New method (no Phase 1 stub existed).

        Parameters
        ----------
        name : str
            Camera identifier previously passed to :meth:`add_camera`.

        Returns
        -------
        dict
            Status dict in the standard ``{"status", "content": [{"text"}]}``
            shape used by mutating methods on this class. Returns ``error``
            if the camera is unknown to ``_cameras``.
        """
        with self._lock:
            if name not in self._cameras:
                return {
                    "status": "error",
                    "content": [{"text": f"Camera '{name}' not found."}],
                }

            prim_path = self._cameras[name].prim_path

            # Cameras aren't added via ``world.scene.add`` (they're
            # standalone USD prims, not articulations or shape wrappers)
            # so removal goes via the stage utility rather than
            # ``world.scene.remove_object``. Wrapped in the same except
            # tuple as add_camera so a transient stage error returns the
            # structured envelope and leaves bookkeeping intact for
            # retry.
            try:
                if self._world is not None:
                    from omni.isaac.core.utils.prims import (  # type: ignore[import-not-found]
                        delete_prim,
                    )

                    delete_prim(prim_path)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                logger.error("Failed to remove camera '%s' (prim=%s): %s", name, prim_path, e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to remove camera '{name}': {e}"}],
                }

            del self._cameras[name]
            if prim_path in self._prim_registry:
                self._prim_registry.remove(prim_path)

            logger.info("Removed camera '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Camera '{name}' removed."}],
            }

    def _create_camera_prim(
        self,
        *,
        name: str,
        prim_path: str,
        position: list[float],
        target: list[float] | None,
        width: int,
        height: int,
        fov_deg: float,
    ) -> tuple[Any, float]:
        """Construct the omni.isaac.sensor.Camera prim + apply look-at + FOV.

        Returns the camera handle plus the resolved focal length in mm
        so :meth:`add_camera` can surface the actually-used focal length
        in its structured json payload.

        Lazy-imports both ``omni.isaac.sensor.Camera`` and
        ``omni.isaac.core.utils.viewports.set_camera_view`` so the
        module loads cleanly without Isaac Sim installed (the call
        site only runs after :meth:`create_world` has booted
        ``SimulationApp``).
        """
        import math

        import numpy as np  # type: ignore[import-not-found]
        from omni.isaac.sensor import Camera  # type: ignore[import-not-found]

        camera = Camera(
            prim_path=prim_path,
            name=name,
            position=np.asarray(position, dtype=float),
            resolution=(int(width), int(height)),
        )
        # ``initialize`` allocates the RTX render product + annotators.
        # Some Camera builds defer this to first ``get_rgba()`` call;
        # call it explicitly so an init-time failure surfaces here
        # (and gets caught by the cleanup clause in add_camera) rather
        # than silently on the first render attempt.
        camera.initialize()

        # Map FOV (deg, horizontal) to focal length (mm) using the
        # standard pinhole lens relation:
        #
        #     focal_length = horizontal_aperture / (2 * tan(fov / 2))
        #
        # The horizontal aperture MUST be the camera's actual aperture,
        # read back from the prim -- assuming a nominal 24 mm is wrong on
        # Isaac's Camera (its default aperture + unit convention yield
        # fx≈6348 px at 640 px, i.e. a ~6° telephoto, instead of the
        # intended 60° / fx≈554). Deriving the focal length from the
        # read-back aperture makes the resulting pixel intrinsics
        # fx = width / (2*tan(fov/2)) exactly, independent of the
        # aperture's absolute value or units.
        try:
            horizontal_aperture_mm = float(camera.get_horizontal_aperture())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            horizontal_aperture_mm = 24.0
        focal_length_mm = horizontal_aperture_mm / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        camera.set_focal_length(focal_length_mm)

        # Enable the depth annotator on the RTX render product. Isaac
        # Sim's Camera ships with rgba enabled by default but depth
        # is opt-in via this method; without it, ``camera.get_depth()``
        # returns ``None`` with a "Annotator 'distance_to_image_plane'
        # not found" warning -- which then crashes downstream
        # ``np.asarray`` calls in ``render()``. Caught during PR #61
        # GPU validation against the Isaac Sim 4.5 docker image.
        # ``add_distance_to_image_plane_to_frame`` is idempotent on
        # repeat calls so this is safe even if the camera has already
        # been initialized with depth elsewhere.
        try:
            camera.add_distance_to_image_plane_to_frame()
        except (AttributeError, RuntimeError):
            # Older Isaac Sim builds expose this under a different name
            # (``add_depth_to_frame``). Try the fallback before giving
            # up; downstream ``get_depth`` will still return ``None``
            # but ``render()``'s defensive None-handling (PR #62) will
            # cover it.
            try:
                camera.add_depth_to_frame()
            except (AttributeError, RuntimeError):
                logger.debug(
                    "Camera %s: depth annotator not enabled; ``get_depth()`` will return None",
                    name,
                )

        # Apply look-at after focal-length so the camera's forward axis
        # is correctly oriented at the target. ``set_camera_view`` works
        # on any USD camera prim by path; no Camera-specific API.
        if target is not None:
            from omni.isaac.core.utils.viewports import (  # type: ignore[import-not-found]
                set_camera_view,
            )

            set_camera_view(eye=position, target=target, camera_prim_path=prim_path)

        return camera, focal_length_mm

    # --- Isaac-specific: Fleet Replication -----------------------------------

    def replicate(self, num_envs: int | None = None) -> dict[str, Any]:
        """Replicate the current scene into parallel environments.

        Uses ``omni.isaac.cloner.Cloner`` for GPU-efficient replication.

        Parameters
        ----------
        num_envs : int, optional
            Number of environments. Defaults to config.num_envs.

        Returns
        -------
        dict
            Status dict with replication info.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if not self._robots:
                return {
                    "status": "error",
                    "content": [{"text": "Add at least one robot first."}],
                }

            n = num_envs or self._config.num_envs

            t0 = time.perf_counter()
            # In full implementation: use omni.isaac.cloner.Cloner
            # to replicate the scene N times
            self._replicated = True
            self._num_envs_active = n
            elapsed = time.perf_counter() - t0

            logger.info("Replicated to %d envs in %.2fs", n, elapsed)

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Replicated to {n} environments. "
                            f"Build time: {elapsed * 1000:.0f}ms. "
                            f"Device: {self._config.device}."
                        ),
                        "json": {
                            "num_envs": n,
                            "build_time_ms": elapsed * 1000,
                        },
                    }
                ],
            }

    # --- Private Implementation ----------------------------------------------

    def _load_usd_robot(self, prim_path: str, usd_path: str, position: list[float]) -> list[str]:
        """Load a robot from USD file. Returns joint names."""
        # In full implementation: use omni.isaac.core to add USD reference
        # and extract Articulation joint names
        logger.info("Loading USD robot from %s at %s", usd_path, prim_path)
        return []

    def _load_urdf_robot(self, prim_path: str, urdf_path: str, position: list[float]) -> list[str]:
        """Load a robot from URDF (converted to USD). Returns joint names."""
        # In full implementation: use omni.isaac.urdf to convert and load
        logger.info("Loading URDF robot from %s at %s", urdf_path, prim_path)
        return []

    def cleanup(self) -> None:
        """Release all resources.

        Callers must invoke this explicitly (or use the class as a context
        manager). There is intentionally no ``__del__`` finalizer: at
        interpreter shutdown the ``threading`` / ``logger`` / ``omni``
        modules can already be partially torn down, and acquiring
        ``self._lock`` from a finalizer is unsafe. Relying on GC for
        Isaac Sim cleanup also leaks the ``World``/USD stage on the
        common case where the GC scheduler defers the finalizer past
        the SimulationApp shutdown.
        """
        if self._world_created:
            self.destroy()

    def __enter__(self) -> IsaacSimulation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __repr__(self) -> str:
        return (
            f"IsaacSimulation("
            f"num_envs={self._config.num_envs}, "
            f"device={self._config.device!r}, "
            f"headless={self._config.headless}, "
            f"world={'created' if self._world_created else 'none'})"
        )
