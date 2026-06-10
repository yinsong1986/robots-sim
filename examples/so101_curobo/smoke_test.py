# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI smoke test for the SO-101 cuRobo demo (issue #67 T10, acceptance #4).

Exercises the **import + MuJoCo + scripted-planner + collection** path with no
GPU-only deps: builds the SO-101 scene, plans a scripted pick-and-place, and
records a small LeRobot dataset (state + action only, ``record_images=False`` so
no GL/EGL is needed), then reloads it locally. Skips cleanly when MuJoCo /
strands_robots / lerobot aren't installed, so it is safe in any CI.

Run standalone:   python -m examples.so101_curobo.smoke_test
Run under pytest: pytest examples/so101_curobo/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile


def _deps_ok() -> tuple[bool, str]:
    import importlib.util as u

    for mod in ("mujoco", "strands_robots", "lerobot"):
        if u.find_spec(mod) is None:
            return False, f"{mod} not installed"
    from examples.so101_curobo.collector import lerobot_available

    if not lerobot_available():
        return False, "lerobot dataset support unavailable"
    return True, ""


def run_smoke(n_episodes: int = 2, tmp_root: str | None = None) -> dict:
    """Build -> plan -> record -> reload. Returns a summary dict; raises on failure."""
    from examples.so101_curobo.controller import SO101CuroboDemo

    root = tmp_root or tempfile.mkdtemp(prefix="so101_curobo_smoke_")
    demo = SO101CuroboDemo(
        backend="mujoco",
        repo_id="local/so101_curobo_smoke",
        root=root,
        prefer_planner="scripted",  # cuRobo not required for the smoke
        record_images=False,  # no GL/EGL -> CPU/CI friendly
    ).build()

    summary = demo.record_dataset(n_episodes=n_episodes, randomize=False)
    assert summary.get("status") == "success", summary
    assert summary["episodes"] == n_episodes, summary
    assert summary["total_frames"] > 0, summary

    ds, start, length = demo.collector.load_back(episode=0)
    assert length > 0, f"episode 0 empty (len={length})"
    feats = set(ds.features.keys())
    assert "observation.state" in feats and "action" in feats, feats

    demo.close()
    return {
        "root": root,
        "episodes": summary["episodes"],
        "total_frames": summary["total_frames"],
        "planner": summary["planner"],
        "ep0_frames": length,
        "features": sorted(feats),
        "success_rate": summary["success_rate"],
    }


def test_smoke_mujoco_pickplace():
    """pytest entry — skips when deps are absent."""
    import pytest

    ok, why = _deps_ok()
    if not ok:
        pytest.skip(f"so101_curobo smoke skipped: {why}")
    out = run_smoke(n_episodes=2)
    assert out["total_frames"] > 0
    assert out["ep0_frames"] > 0


def main() -> int:
    ok, why = _deps_ok()
    if not ok:
        print(f"SKIP: {why}")
        return 0
    out = run_smoke(n_episodes=2)
    print("SMOKE OK:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
