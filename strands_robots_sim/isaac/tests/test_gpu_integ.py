"""GPU integration tests for Isaac Sim backend.

These tests require:
  - NVIDIA GPU with CUDA
  - Isaac Sim 6.0+ installed (Python 3.12)
  - Environment variable: STRANDS_GPU_TEST=1

Gated behind @pytest.mark.gpu -- skipped in CI by default.

Run with: STRANDS_GPU_TEST=1 pytest strands_robots_sim/isaac/tests/test_gpu_integ.py -v
"""

from __future__ import annotations

import os

import pytest

# Gate all tests on STRANDS_GPU_TEST=1
pytestmark = pytest.mark.gpu

_GPU_AVAILABLE = os.environ.get("STRANDS_GPU_TEST", "0") == "1"


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="STRANDS_GPU_TEST=1 not set")
class TestIsaacGPUIntegration:
    """GPU integration tests requiring real Isaac Sim."""

    def test_create_world_and_step(self):
        """Create world, step 100 frames, verify state."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        config = IsaacConfig(num_envs=1, headless=True)
        sim = IsaacSimulation(config)

        available, msg = IsaacSimulation.is_available()
        if not available:
            pytest.skip(f"Isaac Sim not available: {msg}")

        result = sim.create_world()
        assert result["status"] == "success"

        result = sim.step(100)
        assert result["status"] == "success"

        state = sim.get_state()
        assert state["status"] == "success"
        assert state["content"][0].get("json", {}).get("step_count") == 100

        sim.destroy()

    @pytest.mark.xfail(
        reason=(
            "Phase 1 limitation: procedural builder registers metadata but does not "
            "yet create USD prims / Articulation handle on the stage. "
            "get_observation() returns {} via the documented "
            "'robot present but Articulation handle not yet initialised' code path "
            "(simulation.py:1001). Will pass once Phase 2 lands the procedural USD "
            "prim builder."
        ),
        strict=False,
    )
    def test_add_procedural_robot(self):
        """Add SO-100 procedurally and verify joint state shape."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        config = IsaacConfig(num_envs=1, headless=True)
        sim = IsaacSimulation(config)

        available, msg = IsaacSimulation.is_available()
        if not available:
            pytest.skip(f"Isaac Sim not available: {msg}")

        sim.create_world()
        result = sim.add_robot("so100")
        assert result["status"] == "success"
        assert "6 joints" in result["content"][0]["text"]

        sim.step(10)
        obs = sim.get_observation("so100")
        assert isinstance(obs, dict)
        # Should have 6 joint values
        assert len(obs) == 6

        sim.destroy()

    def test_render_produces_image(self):
        """render() should produce an RGB array."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        config = IsaacConfig(num_envs=1, headless=True, render_mode="rtx_realtime")
        sim = IsaacSimulation(config)

        available, msg = IsaacSimulation.is_available()
        if not available:
            pytest.skip(f"Isaac Sim not available: {msg}")

        sim.create_world()
        sim.add_camera("cam1", position=[2, 2, 2])
        sim.step(10)

        result = sim.render("cam1")
        assert result["status"] == "success"
        assert "rgb" in result
        rgb = result["rgb"]
        assert rgb.shape[2] == 3  # RGB channels
        assert rgb.dtype.name == "uint8"

        sim.destroy()

    def test_replicate_fleet(self):
        """replicate() should create parallel envs."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        config = IsaacConfig(num_envs=16, headless=True)
        sim = IsaacSimulation(config)

        available, msg = IsaacSimulation.is_available()
        if not available:
            pytest.skip(f"Isaac Sim not available: {msg}")

        sim.create_world()
        sim.add_robot("so100")
        result = sim.replicate(16)
        assert result["status"] == "success"
        assert "16" in result["content"][0]["text"]

        sim.destroy()

    def test_libero_run_isaac_lifecycle_smoke(self):
        """Smoke-test the lifecycle ``examples/libero/run_isaac.py`` exercises.

        Pins the contract from `#73 <https://github.com/strands-labs/robots-sim/issues/73>`_:
        ``IsaacSimulation`` boots SimulationApp, creates a world, loads the
        bundled Franka USD via ``add_robot(usd_path=...)``, attaches an
        RTX camera, steps physics, and tears down cleanly. This is the
        full lifecycle the LIBERO Isaac example walks through *up to*
        ``evaluate_benchmark`` -- the latter additionally depends on the
        LIBERO benchmark suite being importable inside Isaac's bundled
        Python (``strands-robots`` interpreter constraint, tracked
        separately under
        `#71 <https://github.com/strands-labs/robots-sim/issues/71>`_),
        which this smoke deliberately doesn't exercise.

        Validated against ``nvcr.io/nvidia/isaac-sim:4.5.0`` on a 4×L4
        host during PR validation; runs in ~3 minutes end-to-end (the
        bulk of which is SimulationApp startup, not anything testable).
        """
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        available, msg = IsaacSimulation.is_available()
        if not available:
            pytest.skip(f"Isaac Sim not available: {msg}")

        # Resolve the bundled-asset URL via the modern-then-legacy
        # fallback so this test mirrors what the example scripts'
        # ``_resolve_robot_asset`` does. Both namespaces are imported
        # lazily; either resolves on Isaac Sim 6.0 post-bootstrap.
        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            r = sim.create_world()
            assert r.get("status") == "success", f"create_world: {r}"

            try:
                from isaacsim.storage.native import (  # type: ignore[import-not-found]
                    get_assets_root_path,
                )
            except ImportError:
                from omni.isaac.nucleus import (  # type: ignore[import-not-found]
                    get_assets_root_path,
                )
            assets_root = get_assets_root_path()
            assert assets_root, "get_assets_root_path() returned empty"

            franka_usd = f"{assets_root}/Isaac/Robots/Franka/franka.usd"
            r = sim.add_robot(name="robot", usd_path=franka_usd)
            assert r.get("status") == "success", f"add_robot: {r}"

            r = sim.add_camera(
                name="image",
                position=[2.0, 0.0, 1.5],
                target=[0.0, 0.0, 0.5],
                fov=60.0,
            )
            assert r.get("status") == "success", f"add_camera: {r}"

            r = sim.step(5)
            assert r.get("status") == "success", f"step: {r}"
        finally:
            sim.destroy()

    def test_send_action_on_real_usd_articulation(self):
        """``send_action`` must drive a real USD articulation on Isaac Sim 6.0.

        Regression smoke for
        `#123 <https://github.com/strands-labs/robots-sim/issues/123>`_:
        the pre-fix code called ``SingleArticulation.set_joint_position_targets``,
        which does not exist on Isaac Sim 6.0's
        ``isaacsim.core.prims.SingleArticulation`` -- every ``send_action``
        returned ``{'status': 'error', ...: "... has no attribute
        'set_joint_position_targets'"}``. The fix routes through
        ``apply_action(ArticulationAction(joint_positions=...))``. This test
        loads the bundled USD Franka, resets + steps (so it's not an
        init-timing artefact), then exercises both the dict and list action
        forms exactly as the issue reproduction does, asserting a success
        envelope and a non-empty flat-dict observation.
        """
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        available, msg = IsaacSimulation.is_available()
        if not available:
            pytest.skip(f"Isaac Sim not available: {msg}")

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            r = sim.create_world()
            assert r.get("status") == "success", f"create_world: {r}"

            try:
                from isaacsim.storage.native import (  # type: ignore[import-not-found]
                    get_assets_root_path,
                )
            except ImportError:
                from omni.isaac.nucleus import (  # type: ignore[import-not-found]
                    get_assets_root_path,
                )
            assets_root = get_assets_root_path()
            assert assets_root, "get_assets_root_path() returned empty"

            franka_usd = f"{assets_root}/Isaac/Robots/Franka/franka.usd"
            r = sim.add_robot(name="robot", usd_path=franka_usd)
            assert r.get("status") == "success", f"add_robot: {r}"

            # Reset + step so the articulation is fully initialised -- mirrors
            # the issue's "reproduced after reset() + step(5)" note.
            sim.reset()
            sim.step(5)

            joint_names = sim.robot_joint_names("robot")
            assert isinstance(joint_names, list) and joint_names, f"joint names: {joint_names}"

            # dict form -- the documented world-building loop.
            r = sim.send_action({n: 0.0 for n in joint_names}, robot_name="robot")
            assert r.get("status") == "success", f"send_action(dict): {r}"

            # list form -- same path, flat array in joint order.
            r = sim.send_action([0.0] * len(joint_names), robot_name="robot")
            assert r.get("status") == "success", f"send_action(list): {r}"

            # get_observation returns a flat {joint_name: position} dict.
            obs = sim.get_observation(robot_name="robot")
            assert isinstance(obs, dict) and obs, f"get_observation: {obs!r}"
            assert set(obs) <= set(joint_names), f"unexpected obs keys: {sorted(obs)}"
            assert all(isinstance(v, float) for v in obs.values()), f"obs values: {obs}"
        finally:
            sim.destroy()


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="STRANDS_GPU_TEST=1 not set")
class TestIsaacDestroyCreateDeterminism:
    """destroy() + create_world() must produce a stable USD stage.

    Reproduces the determinism regression: without a stage clear in
    destroy(), a second create_world() + add_robot() on the same
    process-wide SimulationApp builds onto the leftover prims and Isaac
    auto-suffixes the colliding physics-material path, so the prim set
    drifts run-to-run. Asserts the two cycles yield identical prim paths.
    """

    def _prim_paths(self) -> list[str]:
        import omni.usd  # type: ignore[import-not-found]

        stage = omni.usd.get_context().get_stage()
        return sorted(str(p.GetPath()) for p in stage.Traverse())

    def test_destroy_create_cycle_prim_paths_stable(self):
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        available, msg = IsaacSimulation.is_available()
        if not available:
            pytest.skip(f"Isaac Sim not available: {msg}")

        config = IsaacConfig(num_envs=1, headless=True)

        def one_cycle() -> list[str]:
            sim = IsaacSimulation(config)
            r = sim.create_world(timestep=1 / 120, gravity=[0, 0, -9.81])
            assert r["status"] == "success", f"create_world: {r}"
            r = sim.add_robot("so100", position=[0, 0, 0])
            assert r["status"] == "success", f"add_robot: {r}"
            sim.step(5)
            paths = self._prim_paths()
            sim.destroy()
            return paths

        paths1 = one_cycle()
        paths2 = one_cycle()

        # The second cycle must not gain a deduped physics_material_1 (or any
        # other auto-suffixed leftover) from a dirty stage.
        drift = set(paths2) - set(paths1)
        assert not drift, f"prim paths drifted across destroy()/create_world(): {drift}"
        assert paths1 == paths2, "prim path sets differ between identical cycles"
