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


def run_rerun(repo_id: str = "local/so101_curobo_rerun_smoke", root: str | None = None) -> dict:
    """Record the SAME dataset twice (issue #143 repro).

    The first run creates the dataset dir; the second must NOT raise
    ``FileExistsError`` from ``LeRobotDataset.create``. Defaults ``root=None`` so
    this exercises the HF-cache default path (the exact case the documented
    ``app.py`` command hits, which #143 reported failing on the 2nd run).
    """
    from examples.so101_curobo.controller import SO101CuroboDemo

    def _once() -> dict:
        demo = SO101CuroboDemo(
            backend="mujoco",
            repo_id=repo_id,
            root=root,
            prefer_planner="scripted",
            record_images=False,
        ).build()
        summary = demo.record_dataset(n_episodes=1, randomize=False)
        demo.close()
        return summary

    first = _once()
    assert first.get("status") == "success", first
    # Second run with the SAME repo_id/root must also succeed (idempotent re-run).
    second = _once()
    assert second.get("status") == "success", second
    assert second["total_frames"] > 0, second
    return {"first": first, "second": second}


def test_smoke_mujoco_pickplace():
    """pytest entry — skips when deps are absent."""
    import pytest

    ok, why = _deps_ok()
    if not ok:
        pytest.skip(f"so101_curobo smoke skipped: {why}")
    out = run_smoke(n_episodes=2)
    assert out["total_frames"] > 0
    assert out["ep0_frames"] > 0


def test_rerun_default_root_is_idempotent():
    """Regression for #143: re-running with the default repo-id/root must not raise.

    Before the fix, the 2nd ``record_dataset`` with the default (HF-cache) root
    raised ``FileExistsError`` because ``LeRobotDataset.create`` refuses an
    existing dir. The collector now clears a prior dataset dir at the resolved
    effective root, so the documented command is idempotently re-runnable.
    """
    import pytest

    ok, why = _deps_ok()
    if not ok:
        pytest.skip(f"so101_curobo rerun smoke skipped: {why}")
    out = run_rerun()
    assert out["second"]["total_frames"] > 0


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
