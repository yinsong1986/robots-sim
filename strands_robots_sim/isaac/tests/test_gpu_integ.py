"""GPU integration tests for Isaac Sim backend.

These tests require:
  - NVIDIA GPU with CUDA
  - Isaac Sim 2024.x+ installed
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
