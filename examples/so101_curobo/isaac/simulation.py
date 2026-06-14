# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim backend for the SO-101 demo (issue #67 T1).

``IsaacSimulation`` implements the ``strands_robots.simulation.base.SimEngine``
surface on top of NVIDIA Isaac Sim's ``isaacsim.core.api`` (validated on Isaac
Sim 4.5.0 / Python 3.10 / CUDA 12 torch / driver 550 + an L4). It is the
concrete backend behind ``create_simulation("isaac")`` (registered at runtime in
:mod:`examples.so101_curobo.isaac` -- no edit to the shared library factory).

It deliberately mirrors the MuJoCo ``SimEngine`` contract the example relies on
so ``scene.build_pick_place_scene`` / ``collector`` run unchanged:

    create_world, add_robot(urdf_path=...), add_object, add_camera,
    get_observation, set_joint_positions, send_action, step, move_object,
    render, robot_joint_names, list_robots, reset, destroy, randomize

**Runtime requirements (hard):** Isaac Sim only runs after a ``SimulationApp``
is constructed *before* any ``omni``/``isaacsim`` core import, with
``OMNI_KIT_ACCEPT_EULA=YES`` and -- on a box with duplicate Vulkan ICDs --
``VK_ICD_FILENAMES`` pinned to a single NVIDIA ICD. :func:`ensure_app` does the
boot; :func:`register` wires the factory loader. Both are no-ops/raise-clean
when Isaac isn't installed, so the rest of the example degrades to MuJoCo.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("so101_curobo.isaac")

# Populated by ensure_app(); the live SimulationApp handle (kept alive process-wide).
_APP = None

# Minimum NATIVE render width for RTX cameras. The default RTX pipeline runs the
# DLSS temporal upscaler, which renders internally at ~half the output width and
# upscales. Below ~300px internal resolution DLSS falls back to a temporal-
# accumulation path that smears a moving arm into a translucent "ghost" (the
# long-standing front/oblique-view bug). Rendering at >= 640px wide keeps the
# DLSS internal resolution above that threshold so every frame is crisp on its
# own; captured frames are downscaled to the caller's requested size.
_MIN_RENDER_PX = 640


def isaac_available() -> bool:
    """True if the Isaac Sim python packages are importable."""
    import importlib.util

    try:
        return importlib.util.find_spec("isaacsim") is not None
    except Exception:  # noqa: BLE001
        return False


def ensure_app(headless: bool = True):
    """Boot (once) the Isaac Sim ``SimulationApp`` and return it.

    Must run before importing any ``isaacsim.core`` / ``omni`` module. Sets the
    EULA + a single Vulkan ICD if the caller hasn't, so a fresh process "just
    works" on this box. Idempotent: subsequent calls return the live app.
    """
    global _APP
    if _APP is not None:
        return _APP
    import os

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    # The box ships duplicate nvidia Vulkan ICDs (/etc + /usr/share); Isaac's RTX
    # renderer then "Failed to create any GPU devices". Pin a single ICD if the
    # caller hasn't already (validated fix on the L4 / driver 550).
    if not os.environ.get("VK_ICD_FILENAMES") and os.path.exists("/etc/vulkan/icd.d/nvidia_icd.json"):
        os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"

    # The Isaac kit installs a custom import finder that shadows several pip
    # packages with its own (older) bundled copies (omni.kit.pip_archive's
    # pip_prebundle). lerobot's import chain pulls boto3/botocore, and the
    # bundled botocore is too old (missing DEFAULT_CHECKSUM_ALGORITHM), so
    # `lerobot` fails to import *inside* the kit -> the collector reports
    # "lerobot missing". Pre-importing the venv copies here (BEFORE the kit
    # boots) pins them in sys.modules so the kit finder can't replace them.
    for _mod in ("botocore", "botocore.httpchecksum", "boto3"):
        try:
            __import__(_mod)
        except Exception:  # noqa: BLE001 - best-effort; only matters if lerobot is present
            logger.debug("pre-import of %s failed (non-fatal)", _mod, exc_info=True)

    from isaacsim import SimulationApp  # noqa: PLC0415 - must follow env setup

    logger.info("Booting Isaac Sim SimulationApp (headless=%s)...", headless)
    _APP = SimulationApp({"headless": headless})
    logger.info("Isaac Sim SimulationApp ready.")
    return _APP


def _ok(text: str, **extra: Any) -> Dict[str, Any]:
    return {"status": "success", "content": [{"text": text}], **extra}


def _err(text: str) -> Dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _env_int(name: str, default: int) -> int:
    """Read a small positive int from the environment (fallback to ``default``)."""
    import os

    try:
        v = int(os.environ.get(name, ""))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


class _Robot:
    """Bookkeeping for one articulated robot added to the Isaac stage."""

    def __init__(self, name: str, prim_path: str, articulation, joint_names: List[str]):
        self.name = name
        self.prim_path = prim_path
        self.articulation = articulation  # isaacsim.core.prims.SingleArticulation
        self.joint_names = joint_names


class IsaacSimulation:
    """Isaac Sim implementation of the SimEngine surface (issue #67 T1).

    One Isaac ``World`` per instance. The arm is loaded from a URDF (the same
    one cuRobo plans with), objects are core-API cuboids, cameras are RTX
    ``Camera`` sensors. Kinematic control (``set_joint_positions``) drives the
    articulation directly; ``send_action`` writes position targets and steps.
    """

    name = "isaac"

    def __init__(
        self,
        tool_name: str = "isaac",
        default_timestep: float = 0.002,
        default_width: int = 320,
        default_height: int = 240,
        headless: bool = True,
        **_ignored: Any,
    ):
        # Boot the app FIRST (before any isaacsim.core import below).
        ensure_app(headless=headless)
        self.tool_name = tool_name
        self.default_timestep = default_timestep
        self.default_width = default_width
        self.default_height = default_height

        self._world = None
        self._robots: Dict[str, _Robot] = {}
        self._objects: Dict[str, Any] = {}  # name -> core prim (cuboid)
        self._cameras: Dict[str, Any] = {}  # name -> (Camera, origin_robot)
        # Requested output size per camera (w, h). RTX cameras are rendered at a
        # larger NATIVE resolution (>= _MIN_RENDER_PX wide) so the DLSS temporal
        # upscaler doesn't ghost a moving arm (it ghosts below ~300px internal
        # res); every captured frame is downscaled back to this requested size.
        self._cam_out_size: Dict[str, tuple] = {}
        self._frame_dump_n: Dict[str, int] = {}  # SO101_DUMP_FRAMES debug counter
        self._timestep = default_timestep

        # Isaac's renderer + physics may only be driven from the thread that
        # created SimulationApp (the main thread). A web UI (Gradio) calls into
        # the sim from worker threads, where world.step(render=True) deadlocks.
        # So: the main thread runs pump() (steps + renders + caches frames and
        # joint state); worker-thread reads return the cache, and worker-thread
        # actions are enqueued for the pump to apply. main_tid identifies the
        # owning thread; when called ON it we run inline (no queue).
        import queue as _queue
        import threading as _threading

        self._main_tid = _threading.get_ident()
        self._lock = _threading.RLock()
        self._action_q: "_queue.Queue" = _queue.Queue()
        # Whole-job queue: a worker thread submits a full record/plan callable via
        # run_on_main(); the pump runs it inline on the main thread (so the job's
        # per-frame sim calls run directly, not via per-call round-trips).
        self._main_jobs: "_queue.Queue" = _queue.Queue()
        self._frame_cache: Dict[str, Any] = {}
        self._joint_cache: Dict[str, Dict[str, float]] = {}
        self._pump_cameras = True  # render cameras in the pump loop
        self._pump_running = False  # True while run_pump_forever owns the renderer
        # DLSS-convergence tick counts (renders per captured frame). The ghost
        # fix needs the renderer to settle on a held-static pose, but 8 ticks
        # PER FRAME is the dominant cost for long trajectories in the live UI
        # (e.g. a 355-frame cuRobo replay -> thousands of renders -> minutes).
        # During a continuous trajectory the renderer stays warm and the pose
        # changes little frame-to-frame, so fewer ticks converge cleanly:
        #   _record_converge -- per recorded frame (worker capture)
        #   _idle_converge   -- live-preview refresh when the sim is idle
        # Both are env-tunable for headroom on slower GPUs.
        self._record_converge = _env_int("SO101_RECORD_CONVERGE", 6)
        self._idle_converge = _env_int("SO101_IDLE_CONVERGE", 4)

    def _on_main_thread(self) -> bool:
        import threading

        return threading.get_ident() == self._main_tid

    # --- world lifecycle ----------------------------------------------------

    def create_world(
        self,
        timestep: Optional[float] = None,
        gravity: Optional[List[float]] = None,
        ground_plane: bool = True,
    ) -> Dict[str, Any]:
        if self._world is not None:
            return _err("World already exists. Use destroy() first, or reset().")
        from isaacsim.core.api import World

        self._timestep = float(timestep or self.default_timestep)
        self._world = World(stage_units_in_meters=1.0, physics_dt=self._timestep, rendering_dt=self._timestep)
        if ground_plane:
            self._world.scene.add_default_ground_plane()
        self._add_lighting()
        self._configure_renderer()
        self._world.reset()
        hz = (1.0 / self._timestep) if self._timestep else 0.0
        return _ok(f"Isaac world created (dt={self._timestep}s, {hz:.0f} Hz physics, ground={ground_plane}).")

    def _configure_renderer(self) -> None:
        """Best-effort RTX settings for a stable real-time image.

        These carb settings (RaytracedLighting, FXAA, no temporal denoiser)
        nudge RTX toward a single-frame-stable image, but note the RTX pipeline
        re-asserts ``/rtx/post/aa/op`` back to DLSS (3) on every render tick, so
        they do NOT by themselves stop the moving-arm "ghost". The actual ghost
        fix is rendering cameras at a high native resolution (>= ``_MIN_RENDER_PX``
        wide) so the DLSS upscaler stays out of its temporal-ghost regime, plus
        ``_converge_render`` holding the pose static while it settles -- see
        ``add_camera`` / ``_grab_frame`` / ``_converge_render``.
        """
        try:
            import carb

            s = carb.settings.get_settings()
            # Real-time raster-RT path (single-frame stable) rather than path tracing.
            s.set("/rtx/rendermode", "RaytracedLighting")
            # Kill temporal accumulation/history so a moving scene doesn't smear.
            s.set("/rtx/directLighting/sampledLighting/enabled", True)
            s.set("/rtx/raytracing/subframes", 1)
            s.set("/rtx/pathtracing/totalSpp", 1)
            s.set("/rtx/sceneDb/ambientLightIntensity", 1.0)
            # AA: 1 = FXAA (spatial only). DLSS/TAA (the default, op=3) accumulate
            # across frames and smear a moving arm into a translucent "ghost".
            s.set("/rtx/post/aa/op", 1)
            s.set("/rtx/post/dlss/execMode", 0)
            s.set("/rtx/post/taa/enabled", False)
            # Disable the temporal/animation denoiser reuse as well.
            s.set("/rtx/directLighting/denoiser/enabled", False)
            s.set("/rtx/raytracing/lightcache/spatialCache/enabled", False)
        except Exception:  # noqa: BLE001 - settings are a visual nicety
            logger.debug("renderer config skipped", exc_info=True)

    def _add_lighting(self) -> None:
        """Add a dome + key light so RTX camera frames aren't black.

        Unlike MuJoCo (which has implicit headlight/ambient), an Isaac stage is
        unlit by default -- without this, ``get_rgba()`` returns near-black
        frames and the UI preview looks empty.
        """
        try:
            import omni.usd
            from pxr import Sdf, UsdLux

            stage = omni.usd.get_context().get_stage()
            # Soft ambient fill from all directions.
            dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/lights/dome"))
            dome.CreateIntensityAttr(800.0)
            # Directional key light for shape/shading.
            distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/lights/key"))
            distant.CreateIntensityAttr(2500.0)
            distant.CreateAngleAttr(1.0)
            from pxr import Gf, UsdGeom

            UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 25.0))
            # Front fill light from the -Y side (toward the front/oblique cameras)
            # so the arm's camera-facing side isn't left in shadow / silhouette.
            fill = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/lights/fill"))
            fill.CreateIntensityAttr(1500.0)
            fill.CreateAngleAttr(1.0)
            UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-60.0, 0.0, 180.0))
        except Exception:  # noqa: BLE001 - lighting is a visual nicety
            logger.debug("Could not add scene lighting", exc_info=True)

    def destroy(self) -> Dict[str, Any]:
        if self._world is None:
            return _ok("No world to destroy.")
        try:
            self._world.stop()
        except Exception:  # noqa: BLE001
            pass
        self._world = None
        self._robots.clear()
        self._objects.clear()
        self._cameras.clear()
        return _ok("Isaac world destroyed.")

    def reset(self) -> Dict[str, Any]:
        if self._world is None:
            return _err("No world. Call create_world() first.")
        self._world.reset()
        return _ok("Isaac world reset.")

    def step(self, n_steps: int = 1) -> Dict[str, Any]:
        if self._world is None:
            return _err("No world. Call create_world() first.")
        n = max(0, int(n_steps))
        if self._on_main_thread():
            for _ in range(n):
                self._world.step(render=False)
            return _ok(f"+{n} steps.")
        # Worker thread: the main-thread pump owns stepping. Enqueue n steps and
        # wait until the pump has drained them, so the caller's per-waypoint
        # timing (e.g. the collector) is preserved without driving Isaac here.
        import threading
        import time

        done = threading.Event()
        remaining = {"n": n}

        def _one_step():
            self._world.step(render=False)
            remaining["n"] -= 1
            if remaining["n"] <= 0:
                done.set()

        for _ in range(n):
            self._action_q.put(_one_step)
        done.wait(timeout=10.0)
        time.sleep(0)
        return _ok(f"+{n} steps (queued).")

    def get_state(self) -> Dict[str, Any]:
        if self._world is None:
            return _err("No world. Call create_world() first.")
        return _ok(
            f"Isaac state: robots={len(self._robots)} objects={len(self._objects)} "
            f"cameras={len(self._cameras)} dt={self._timestep}s"
        )

    def physics_timestep(self) -> Optional[float]:
        return self._timestep if self._world is not None else None

    # --- main-thread pump (for web UI: render off-main-thread deadlocks) -----

    def pump(self, render: bool = True) -> None:
        """Drain queued actions, step once, refresh caches. MAIN THREAD ONLY.

        A web UI calls get_observation/send_action from worker threads where
        Isaac's renderer/physics deadlock. Those calls instead enqueue actions
        and read cached frames; this pump (run on the owning main thread) is the
        single place that actually advances the sim and renders the cameras.
        """
        if self._world is None:
            return
        # 1. Apply any actions queued by worker threads, counting them.
        n_actions = 0
        while not self._action_q.empty():
            try:
                fn = self._action_q.get_nowait()
            except Exception:  # noqa: BLE001
                break
            try:
                fn()
                n_actions += 1
            except Exception:  # noqa: BLE001
                logger.debug("queued action failed", exc_info=True)
        # 2. Advance the sim, then converge the renderer on the (now static) pose.
        # world.step(render=True) advances physics each tick, so converging with a
        # plain step-loop keeps the arm drifting and leaves a faint temporal ghost;
        # _converge_render holds the pose static while DLSS settles.
        #
        # When worker actions ran this tick (n_actions>0) they include the
        # recording capture, which does its OWN _converge_render + grab. Doing a
        # second idle converge here just doubles the render load and serializes
        # behind the capture -- the dominant cost that made long (cuRobo) episodes
        # take many minutes in the live UI. So only render here when the sim is
        # IDLE (no queued work): that keeps the live preview fresh between
        # episodes without competing with the recorder mid-episode.
        if n_actions == 0 and render:
            self._converge_render(self._idle_converge)
        # 3. Refresh joint-state cache for every robot.
        for rname, r in self._robots.items():
            try:
                q = r.articulation.get_joint_positions()
                if q is not None:
                    self._joint_cache[rname] = {jn: float(v) for jn, v in zip(r.joint_names, list(q))}
            except Exception:  # noqa: BLE001
                pass
        # 4. Refresh camera frame cache for the live preview -- only when we
        # actually rendered this tick (idle path). When actions ran, the capture
        # already published its frames to the cache; re-grabbing here would be a
        # wasted readback per camera every recorded frame.
        if render and n_actions == 0 and self._pump_cameras:
            new_frames: Dict[str, Any] = {}
            for cname, cam in self._cameras.items():
                try:
                    img = self._grab_frame(cname, cam)
                    if img is not None:
                        new_frames[cname] = img
                except Exception:  # noqa: BLE001
                    pass
            for cname, img in new_frames.items():
                self._frame_cache[cname] = img

    def run_pump_forever(self, stop_event=None, render_every: int = 3) -> None:
        """Block on the MAIN THREAD running pump() in a loop (UI launched elsewhere).

        Drains queued worker actions (an executing episode) every iteration so
        the episode runs at full speed, and refreshes the live preview only
        every ``render_every`` IDLE iterations. A short sleep when idle keeps the
        renderer from running flat out and pegging the CPU -- which otherwise
        starves the Gradio HTTP thread so the page never loads.
        """
        import time

        i = 0
        self._pump_running = True
        try:
            while stop_event is None or not stop_event.is_set():
                # A whole-job submission (UI record/plan) takes priority: run it
                # inline on this main thread. The job drives the sim directly
                # (no per-frame round-trips); the preview just freezes for its
                # duration, which is the right trade for a fast, reliable record.
                try:
                    job = self._main_jobs.get_nowait()
                except Exception:  # noqa: BLE001 - empty queue
                    job = None
                if job is not None:
                    job()
                    i = 0
                    continue
                busy = not self._action_q.empty()
                if busy:
                    # Episode executing via per-call queue: drain as fast as possible.
                    self.pump(render=False)
                    i = 0
                    continue
                # Idle: occasional preview refresh, then yield the CPU so the web
                # server thread is scheduled and the page stays responsive.
                self.pump(render=(i % max(1, render_every) == 0))
                i += 1
                time.sleep(0.02)
        finally:
            self._pump_running = False

    def run_on_main(self, fn, timeout: Optional[float] = None):
        """Run ``fn()`` on the MAIN THREAD (the pump owner) and return its result.

        A web UI calls record/plan jobs from a Gradio worker thread. Driving the
        episode from there means every per-frame ``set_joint_positions`` / ``step``
        / ``get_observation`` round-trips through the action queue to the pump --
        slow and deadlock-prone for a long (355-frame) trajectory. Instead, submit
        the WHOLE job here: the pump runs it inline on the main thread, so inside
        ``fn`` ``_on_main_thread()`` is True and the collector drives the sim
        directly (exactly like the headless smoke path -- fast, no round-trips).
        While the job runs, the pump's normal loop is paused. Re-raises any
        exception from ``fn`` on the caller's thread.

        If already on the main thread, runs ``fn`` immediately.
        """
        if self._on_main_thread():
            return fn()
        import threading

        done = threading.Event()
        box: Dict[str, Any] = {}

        def _job():
            try:
                box["result"] = fn()
            except BaseException as exc:  # noqa: BLE001 - surfaced to caller below
                box["exc"] = exc
            finally:
                done.set()

        self._main_jobs.put(_job)
        if not done.wait(timeout=timeout):
            raise TimeoutError("run_on_main timed out waiting for the main-thread pump.")
        if "exc" in box:
            raise box["exc"]
        return box.get("result")

    # --- robots -------------------------------------------------------------

    def add_robot(
        self,
        name: str,
        urdf_path: Optional[str] = None,
        data_config: Optional[str] = None,
        position: Optional[List[float]] = None,
        orientation: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        if self._world is None:
            return _err("No world. Call create_world() first.")
        if name in self._robots:
            return _err(f"Robot {name!r} already exists.")
        if not urdf_path:
            return _err("Isaac backend requires urdf_path (the cuRobo-matched SO-101 URDF).")
        if not os.path.exists(urdf_path):
            return _err(f"File not found: {urdf_path}")

        position = position or [0.0, 0.0, 0.0]
        orientation = orientation or [1.0, 0.0, 0.0, 0.0]
        prim_path = f"/World/{name}"

        try:
            prim_path = self._import_urdf(urdf_path, prim_path)
        except Exception as exc:  # noqa: BLE001
            return _err(f"URDF import failed: {type(exc).__name__}: {exc}")

        # Wrap as an articulation and register it with the world so physics + the
        # joint API initialise on the next reset. The URDF importer already placed
        # the robot (fix_base bolts the base at the origin, matching the scene);
        # we bind the SingleArticulation to the discovered articulation-root prim.
        from isaacsim.core.prims import SingleArticulation

        art = SingleArticulation(prim_path=prim_path, name=name)
        self._world.scene.add(art)
        self._world.reset()  # initialises the articulation handle (dof_names, etc.)

        joint_names = [str(j) for j in (art.dof_names or [])]
        self._robots[name] = _Robot(name, prim_path, art, joint_names)
        return _ok(
            f"Robot {name!r} added from URDF ({len(joint_names)} joints: "
            f"{joint_names[:6]}{'...' if len(joint_names) > 6 else ''}) at {position}."
        )

    def _import_urdf(self, urdf_path: str, dest_prim: str) -> str:
        """Import a URDF onto the stage; return the articulation-root prim path.

        Uses the kit-command import path (parse -> import), then locates the
        prim carrying ``UsdPhysics.ArticulationRootAPI`` -- the import command's
        own return value is unreliable on Isaac 4.5 (it raises "Used null prim"
        on the final placement step even when the robot imported fine), so we
        scan the stage instead. NOTE: mesh files referenced by the URDF must
        resolve (the ``assets/`` dir next to the URDF); missing meshes make the
        importer skip those links and no articulation is built.
        """
        import omni.kit.commands
        import omni.usd
        from pxr import UsdPhysics

        _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
        cfg.merge_fixed_joints = False
        cfg.fix_base = True  # tabletop arm: base is bolted down
        cfg.make_default_prim = False
        cfg.self_collision = False
        cfg.distance_scale = 1.0
        cfg.create_physics_scene = False  # World already owns the physics scene

        _, robot_model = omni.kit.commands.execute("URDFParseFile", urdf_path=urdf_path, import_config=cfg)
        # import_robot raises on its final step but still places the robot; ignore.
        try:
            omni.kit.commands.execute(
                "URDFImportRobot", urdf_path=urdf_path, urdf_robot=robot_model, import_config=cfg, dest_path=""
            )
        except Exception:  # noqa: BLE001 - "Used null prim" is benign on 4.5
            logger.debug("URDFImportRobot raised (benign on 4.5); scanning stage for root", exc_info=True)

        stage = omni.usd.get_context().get_stage()
        roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if not roots:
            raise RuntimeError(
                "URDF imported but no ArticulationRootAPI found -- usually the URDF's "
                "mesh assets did not resolve (need the 'assets/' dir next to the URDF). "
                "Check the importer warnings for 'Failed to resolve mesh'."
            )
        return roots[0]

    def remove_robot(self, name: str) -> Dict[str, Any]:
        if name not in self._robots:
            return _err(f"Robot {name!r} not found.")
        try:
            self._world.scene.remove_object(name)
        except Exception:  # noqa: BLE001
            pass
        del self._robots[name]
        return _ok(f"Robot {name!r} removed.")

    def list_robots(self) -> List[str]:
        return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> List[str]:
        r = self._robots.get(robot_name)
        return list(r.joint_names) if r else []

    # --- objects ------------------------------------------------------------

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: Optional[List[float]] = None,
        orientation: Optional[List[float]] = None,
        size: Optional[List[float]] = None,
        color: Optional[List[float]] = None,
        mass: float = 0.1,
        is_static: bool = False,
        mesh_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self._world is None:
            return _err("No world. Call create_world() first.")
        import numpy as np
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid

        position = position or [0.0, 0.0, 0.0]
        size = size or [0.05, 0.05, 0.05]
        color = color or [0.5, 0.5, 0.5, 1.0]
        # The example passes half-extents (MuJoCo convention); Isaac cuboids take
        # full-extent `scale`. Double so the cube matches the MuJoCo scene size.
        scale = np.array([2.0 * float(s) for s in size[:3]], dtype=float)
        prim_path = f"/World/{name}"
        rgb = np.array(color[:3], dtype=float)
        try:
            cls = FixedCuboid if is_static else DynamicCuboid
            common = dict(
                prim_path=prim_path, name=name, position=np.array(position[:3], dtype=float), scale=scale, color=rgb
            )
            obj = cls(**common) if is_static else cls(mass=float(mass), **common)
            self._world.scene.add(obj)
            self._objects[name] = obj
        except Exception as exc:  # noqa: BLE001
            return _err(f"add_object failed: {type(exc).__name__}: {exc}")
        return _ok(f"'{name}' added: {shape} at {position} ({'static' if is_static else f'{mass}kg'}).")

    def remove_object(self, name: str) -> Dict[str, Any]:
        if name not in self._objects:
            return _err(f"Object {name!r} not found.")
        try:
            self._world.scene.remove_object(name)
        except Exception:  # noqa: BLE001
            pass
        del self._objects[name]
        return _ok(f"'{name}' removed.")

    def move_object(
        self, name: str, position: Optional[List[float]] = None, orientation: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        obj = self._objects.get(name)
        if obj is None:
            return _err(f"Object {name!r} not found.")
        import numpy as np

        try:
            pos = np.array(position[:3], dtype=float) if position else None
            ori = np.array(orientation[:4], dtype=float) if orientation else None
            obj.set_world_pose(position=pos, orientation=ori)
            # Zero velocity so a teleport doesn't fling a dynamic body.
            if hasattr(obj, "set_linear_velocity"):
                obj.set_linear_velocity(np.zeros(3))
            if hasattr(obj, "set_angular_velocity"):
                obj.set_angular_velocity(np.zeros(3))
        except Exception as exc:  # noqa: BLE001
            return _err(f"move_object failed: {type(exc).__name__}: {exc}")
        return _ok(f"'{name}' moved to {position or 'same'}.")

    def _object_position(self, name: str) -> Optional[List[float]]:
        obj = self._objects.get(name)
        if obj is None:
            return None
        try:
            pos, _ = obj.get_world_pose()
            return [float(x) for x in pos]
        except Exception:  # noqa: BLE001
            return None

    def gripper_frame_pos(self, robot_name: Optional[str] = None) -> Optional[List[float]]:
        """World position of the robot's gripper/tool link, read from the USD stage.

        The collector's kinematic grasp-attach needs the end-effector world
        position to decide when the gripper is close enough to the cube to
        attach it. MuJoCo exposes this via ``mj_data``; on Isaac we read the
        link prim's world transform. Prefers a ``gripper_frame``/``tool`` link,
        then any ``gripper``/``moving_jaw`` link, under the robot's prim.
        """
        if robot_name is None:
            robot_name = next(iter(self._robots), None)
        r = self._robots.get(robot_name) if robot_name else None
        if r is None:
            return None
        import os as _os

        _dbg = bool(_os.environ.get("SO101_GRASP_DBG"))
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            # r.prim_path is the articulation-root prim, which for the SO-101
            # URDF is a leaf joint (e.g. /so101_new_calib/root_joint) -- its
            # subtree has no link prims. Walk up to the top-level robot prim
            # (the first path component under the pseudo-root) and search its
            # whole subtree for the gripper/tool link.
            from pxr import Sdf

            sdf_path = Sdf.Path(r.prim_path)
            top = sdf_path
            while top.GetParentPath() != Sdf.Path.absoluteRootPath and top.GetParentPath() != Sdf.Path.emptyPath:
                top = top.GetParentPath()
            root = stage.GetPrimAtPath(top)
            if not root or not root.IsValid():
                if _dbg:
                    logger.info("[grasp-dbg] gripper_frame_pos: root invalid for top=%r (from %r)", top, r.prim_path)
                return None
            preferred = None
            fallback = None
            for p in Usd.PrimRange(root):
                if not p.IsA(UsdGeom.Xformable):
                    continue
                ln = p.GetName().lower()
                if "gripper_frame" in ln or "tool" in ln:
                    preferred = p
                    break
                if "moving_jaw" in ln or "gripper" in ln:
                    fallback = fallback or p
            prim = preferred or fallback
            if prim is None:
                if _dbg:
                    logger.info("[grasp-dbg] gripper_frame_pos: no EE link under %r", r.prim_path)
                return None
            t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            return [float(t[0]), float(t[1]), float(t[2])]
        except Exception as exc:  # noqa: BLE001
            if _dbg:
                logger.info("[grasp-dbg] gripper_frame_pos EXC: %s: %s", type(exc).__name__, exc)
            logger.debug("gripper_frame_pos failed", exc_info=True)
            return None

    # --- cameras ------------------------------------------------------------

    def add_camera(
        self,
        name: str,
        position: Optional[List[float]] = None,
        target: Optional[List[float]] = None,
        fov: float = 60.0,
        width: int = 320,
        height: int = 240,
        parent_body: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._world is None:
            return _err("No world. Call create_world() first.")
        if name in self._cameras:
            return _err(f"Camera {name!r} already exists.")
        import numpy as np
        from isaacsim.sensors.camera import Camera

        pos = np.array((position or [1.0, 1.0, 1.0])[:3], dtype=float)
        tgt = np.array((target or [0.0, 0.0, 0.0])[:3], dtype=float)
        orient = self._look_at_quat(pos, tgt)
        out_w, out_h = int(width), int(height)
        # Render at a higher NATIVE resolution if the requested output is small,
        # so the DLSS upscaler stays above its temporal-ghost threshold. Preserve
        # the requested aspect ratio; downscale captures back to (out_w, out_h).
        if out_w < _MIN_RENDER_PX:
            scale = _MIN_RENDER_PX / float(out_w)
            render_w = _MIN_RENDER_PX
            render_h = int(round(out_h * scale))
        else:
            render_w, render_h = out_w, out_h
        try:
            cam = Camera(
                prim_path=f"/World/cameras/{name}",
                name=name,
                position=pos,
                orientation=orient,  # wxyz, world frame
                frequency=20,
                resolution=(render_w, render_h),
            )
            cam.initialize()
            # Apply the requested horizontal FOV via focal length (Isaac cameras
            # default to ~24mm/very-wide, which makes the close tabletop framing
            # look too zoomed-in / distorted). focal = aperture / (2 tan(fov/2)).
            try:
                aperture = float(cam.get_horizontal_aperture())
            except Exception:  # noqa: BLE001
                aperture = 20.955
            focal = aperture / (2.0 * math.tan(math.radians(float(fov)) / 2.0))
            try:
                cam.set_focal_length(float(focal))
            except Exception:  # noqa: BLE001
                logger.debug("set_focal_length failed", exc_info=True)
            self._cameras[name] = cam
            self._cam_out_size[name] = (out_w, out_h)
            # RTX cameras need a few rendered frames before get_rgba() returns
            # pixels (the annotator buffer is empty until the renderer has ticked).
            for _ in range(3):
                self._world.step(render=True)
        except Exception as exc:  # noqa: BLE001
            return _err(f"add_camera failed: {type(exc).__name__}: {exc}")
        return _ok(
            f"Camera {name!r} added at {pos.tolist()} " f"(render {render_w}x{render_h} -> output {out_w}x{out_h})."
        )

    def _converge_render(self, n: int = 8) -> None:
        """Render ``n`` ticks while HOLDING the robots at their current pose.

        ``world.step(render=True)`` advances physics every tick, so a kinematic
        arm keeps drifting (gravity/settling) while we try to converge the DLSS
        temporal upscaler -> the moving target leaves a faint ghost. Re-asserting
        each robot's joint positions (and zeroing velocities) before every render
        freezes the pose so DLSS converges on a single, static image.
        """
        if self._world is None:
            return
        import numpy as np

        for _ in range(max(1, n)):
            for r in self._robots.values():
                try:
                    q = r.articulation.get_joint_positions()
                    if q is not None:
                        qa = np.asarray(q, dtype=float)
                        r.articulation.set_joint_positions(qa)
                        try:
                            r.articulation.set_joint_velocities(np.zeros_like(qa))
                        except Exception:  # noqa: BLE001
                            pass
                except Exception:  # noqa: BLE001
                    pass
            self._world.step(render=True)

    def _grab_frame(self, cname: str, cam) -> Optional[Any]:
        """Capture ``cam`` as an RGB uint8 array at the camera's requested output
        size. The RTX camera renders at a higher native resolution (to keep DLSS
        out of its temporal-ghost regime); this downscales the result back to the
        size the caller asked for. Returns None if no frame is available yet.
        """
        import numpy as np

        frame = cam.get_rgba()
        if frame is None or not getattr(frame, "size", 0):
            return None
        img = np.asarray(frame)[:, :, :3].astype("uint8")
        out = self._cam_out_size.get(cname)
        if out is not None:
            ow, oh = out
            if img.shape[1] != ow or img.shape[0] != oh:
                img = self._resize_rgb(img, ow, oh)
        _dbg = os.environ.get("SO101_DUMP_FRAMES")
        if _dbg:
            try:
                import imageio.v3 as _iio

                n = self._frame_dump_n.get(cname, 0)
                self._frame_dump_n[cname] = n + 1
                _iio.imwrite(os.path.join(_dbg, f"raw_{cname}_{n:03d}.png"), img)
            except Exception:  # noqa: BLE001
                pass
        return img

    @staticmethod
    def _resize_rgb(img, out_w: int, out_h: int):
        """Downscale an HxWx3 uint8 array to (out_h, out_w). Uses cv2/PIL if
        present, else a fast NumPy area-average / nearest fallback (no new deps).
        """
        import numpy as np

        try:
            import cv2  # type: ignore

            return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        except Exception:  # noqa: BLE001
            pass
        try:
            from PIL import Image  # type: ignore

            return np.asarray(Image.fromarray(img).resize((out_w, out_h), Image.BILINEAR))
        except Exception:  # noqa: BLE001
            pass
        # NumPy fallback: integer-factor area average when possible, else nearest.
        h, w = img.shape[:2]
        if w % out_w == 0 and h % out_h == 0:
            fx, fy = w // out_w, h // out_h
            return img.reshape(out_h, fy, out_w, fx, 3).mean(axis=(1, 3)).astype("uint8")
        ys = (np.arange(out_h) * (h / out_h)).astype(int).clip(0, h - 1)
        xs = (np.arange(out_w) * (w / out_w)).astype(int).clip(0, w - 1)
        return img[ys][:, xs]

    @staticmethod
    def _look_at_quat(eye, target):
        """World-frame wxyz quaternion aiming an Isaac camera from ``eye`` at ``target``.

        Isaac/USD cameras look down their local -Z with +Y up. The robust way to
        orient one on this Isaac build is via Euler angles (roll=0, pitch, yaw)
        derived from the look direction, fed through Isaac's own
        ``euler_angles_to_quats`` (validated: straight-down and angled views both
        frame the scene correctly; a hand-rolled rotation-matrix path did not).
        """
        import math

        import isaacsim.core.utils.numpy.rotations as rot_utils
        import numpy as np

        f = np.array(target, dtype=float) - np.array(eye, dtype=float)
        n = np.linalg.norm(f)
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        f /= n
        yaw = math.atan2(f[1], f[0])
        pitch = math.asin(-f[2])  # looking downward -> negative world-Z component
        return rot_utils.euler_angles_to_quats(np.array([0.0, math.degrees(pitch), math.degrees(yaw)]), degrees=True)

    # --- observation / action ----------------------------------------------

    def get_observation(self, robot_name: Optional[str] = None, *, skip_images: bool = False) -> Dict[str, Any]:
        if self._world is None or not self._robots:
            return {}
        if robot_name is None:
            robot_name = next(iter(self._robots))
        r = self._robots.get(robot_name)
        if r is None:
            return {}
        obs: Dict[str, Any] = {}
        on_main = self._on_main_thread()
        # Joint state: read live on the main thread, else use the pump's cache.
        if on_main:
            try:
                q = r.articulation.get_joint_positions()
                for jn, val in zip(r.joint_names, list(q)):
                    obs[jn] = float(val)
            except Exception:  # noqa: BLE001
                logger.debug("joint read failed", exc_info=True)
        else:
            obs.update(self._joint_cache.get(robot_name, {}))
        if not skip_images:
            # Drive the renderer inline whenever we're on the main thread -- we
            # ARE the renderer-owning thread there (either the pump itself, or a
            # whole-job submitted via run_on_main that the pump is executing
            # inline), so there's no racing-renderer concern. Only true WORKER
            # threads must enqueue a capture for the pump. (Previously this also
            # gated on ``not self._pump_running``, which deadlocked a run_on_main
            # job: on_main=True but pump_running=True -> it enqueued a _capture
            # and waited for the pump, which was busy running this very job.)
            drive_inline = on_main
            if drive_inline:
                # On the main thread we may drive the renderer directly. Hold the
                # pose static and converge the DLSS upscaler (at the native render
                # resolution this clears any temporal smear of the moving arm).
                if self._cameras:
                    try:
                        self._converge_render(self._record_converge)
                    except Exception:  # noqa: BLE001
                        logger.debug("converge render failed", exc_info=True)
                for cname, cam in self._cameras.items():
                    try:
                        img = self._grab_frame(cname, cam)
                        if img is not None:
                            obs[cname] = img
                            self._frame_cache[cname] = img  # keep the cache fresh too
                    except Exception:  # noqa: BLE001
                        logger.debug("camera %s render failed", cname, exc_info=True)
            else:
                # Worker thread (or pump active): we cannot drive the renderer
                # here (deadlock). Enqueue a synchronous render+capture for the
                # main-thread pump and WAIT for it, so the returned frame matches
                # the CURRENT pose. (Returning the async cache instead yields a
                # pose/render mismatch -> the moving arm looks like a "ghost".)
                import threading

                if self._pump_running:
                    done = threading.Event()
                    result: Dict[str, Any] = {}

                    def _capture():
                        try:
                            # Hold the just-applied pose static and converge the
                            # renderer (cameras render at a high native resolution
                            # so DLSS doesn't ghost a moving arm), then capture.
                            # _record_converge (default 3) keeps long trajectories
                            # fast in the UI; the warm renderer converges in a few
                            # ticks frame-to-frame.
                            self._converge_render(self._record_converge)
                            for cname, cam in self._cameras.items():
                                img = self._grab_frame(cname, cam)
                                if img is not None:
                                    result[cname] = img
                        finally:
                            done.set()

                    self._action_q.put(_capture)
                    if done.wait(timeout=5.0):
                        obs.update(result)
                        self._frame_cache.update(result)
                    else:
                        # Pump didn't service in time; fall back to last cache.
                        for cname, img in self._frame_cache.items():
                            obs.setdefault(cname, img)
                else:
                    # No pump running: serve the most recent cached frames.
                    for cname, img in self._frame_cache.items():
                        obs[cname] = img
        return obs

    def set_joint_positions(
        self,
        positions: Any = None,
        robot_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._world is None or not self._robots:
            return _err("No world/robot.")
        if positions is None:
            return _err("'positions' is required.")
        if robot_name is None:
            robot_name = next(iter(self._robots))
        r = self._robots.get(robot_name)
        if r is None:
            return _err(f"Robot {robot_name!r} not found.")
        import numpy as np

        def _apply():
            if isinstance(positions, dict):
                cur = list(r.articulation.get_joint_positions())
                idx = {jn: i for i, jn in enumerate(r.joint_names)}
                for jn, v in positions.items():
                    if jn in idx:
                        cur[idx[jn]] = float(v)
                r.articulation.set_joint_positions(np.array(cur, dtype=float))
            else:
                r.articulation.set_joint_positions(np.array(positions, dtype=float))

        if self._on_main_thread():
            _apply()
            return _ok("Set joint positions (main).")
        # Worker thread: enqueue for the main-thread pump to apply.
        self._action_q.put(_apply)
        return _ok("Set joint positions (queued).")

    def send_action(self, action: Dict[str, Any], robot_name: Optional[str] = None, n_substeps: int = 1) -> None:
        if self._world is None or not self._robots:
            return
        if robot_name is None:
            robot_name = next(iter(self._robots))
        r = self._robots.get(robot_name)
        if r is None:
            return
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        idx = {jn: i for i, jn in enumerate(r.joint_names)}

        def _apply():
            targets = list(r.articulation.get_joint_positions())
            for jn, v in action.items():
                if jn in idx:
                    targets[idx[jn]] = float(v)
            try:
                r.articulation.apply_action(ArticulationAction(joint_positions=np.array(targets, dtype=float)))
            except Exception:  # noqa: BLE001
                logger.debug("apply_action failed; falling back to set_joint_positions", exc_info=True)
                r.articulation.set_joint_positions(np.array(targets, dtype=float))

        if self._on_main_thread():
            _apply()
            for _ in range(max(1, int(n_substeps))):
                self._world.step(render=False)
        else:
            # Worker thread: enqueue the write; the pump advances physics.
            self._action_q.put(_apply)

    # --- rendering ----------------------------------------------------------

    def render(
        self, camera_name: str = "default", width: Optional[int] = None, height: Optional[int] = None
    ) -> Dict[str, Any]:
        cam = self._cameras.get(camera_name)
        if cam is None:
            return _err(f"Camera {camera_name!r} not found. Available: {list(self._cameras)}")
        try:
            self._converge_render(8)
            img = self._grab_frame(camera_name, cam)
            if img is None:
                return _err(f"Camera {camera_name!r} produced no frame yet.")
            return {
                "status": "success",
                "content": [{"text": f"rendered {img.shape} from {camera_name!r}"}],
                "image": img,
            }
        except Exception as exc:  # noqa: BLE001
            return _err(f"Render failed: {type(exc).__name__}: {exc}")

    # --- optional -----------------------------------------------------------

    def randomize(self, **kwargs: Any) -> Dict[str, Any]:
        # Domain randomization not yet wired for Isaac; no-op so the collector's
        # best-effort randomize() call doesn't break the loop.
        return _ok("randomize: no-op on Isaac backend (not yet implemented).")

    def cleanup(self) -> None:
        self.destroy()
