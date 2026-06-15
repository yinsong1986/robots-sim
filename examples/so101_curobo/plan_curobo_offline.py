#!/usr/bin/env python3
"""Offline cuRobo trajectory planner for the SO-101 pick-place demo.

Runs in a venv that has **cuRobo** (and ``strands_robots`` / this repo on the
PYTHONPATH) -- NOT the Isaac venv. cuRobo and Isaac Sim 4.5 can't share a
process: cuRobo's collision kernels need ``warp-lang >= 1.14`` (for
``wp.func(module=)``) while the Isaac kit bundles warp 1.5 which lacks it, so an
in-kit ``import curobo`` collision path raises and the demo silently falls back
to the scripted planner. So plan offline here, dump the JointTrajectory to JSON,
and have the Isaac run replay it via ``PrecomputedPlanner`` (which imports
neither cuRobo nor warp, so it runs inside the kit):

    # 1) plan offline in a cuRobo-capable venv:
    python examples/so101_curobo/plan_curobo_offline.py \\
        --urdf <so101.urdf> --asset <asset_dir> \\
        --cube-xy 0.2 0.2 --place-xy 0.0 0.25 --out curobo_traj.json
    # 2) replay it on the Isaac backend:
    python -m examples.so101_curobo.app --backend isaac --planner curobo \\
        --curobo-urdf <so101.urdf> --curobo-traj curobo_traj.json --smoke

Ensure the repo + ``strands_robots`` are importable, e.g.
``PYTHONPATH=<robots-sim>:<robots>`` (this file does not hardcode paths).
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=os.environ.get("SO101_URDF"))
    ap.add_argument("--asset", default=os.environ.get("SO101_ASSET"))
    ap.add_argument("--cube-xy", nargs=2, type=float, default=[0.20, 0.20])
    ap.add_argument("--place-xy", nargs=2, type=float, default=[0.0, 0.25])
    ap.add_argument(
        "--grasp-z",
        type=float,
        default=None,
        help="Tool-frame Z (m) at the grasp/close pose. Lower = the gripper descends "
        "further onto the cube so the fingers straddle it (vs hovering above). "
        "Defaults to the planner's grasp_z.",
    )
    ap.add_argument("--approach", type=float, default=None, help="Approach/lift clearance height (m) above the table.")
    ap.add_argument("--table-z", type=float, default=None, help="Table surface Z (m).")
    ap.add_argument(
        "--joint-names",
        nargs="*",
        default=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    )
    ap.add_argument("--start-q", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from examples.so101_curobo.planner import CUROBO_AVAILABLE, CuroboMotionPlanner

    if not CUROBO_AVAILABLE:
        raise SystemExit("cuRobo not importable in this venv. Run from the curobo venv.")

    jn = list(args.joint_names)
    start_q = args.start_q if args.start_q is not None else [0.0] * len(jn)

    planner = CuroboMotionPlanner(urdf_path=args.urdf, asset_path=args.asset)
    # Only forward grasp-pose overrides that were actually passed, so the
    # planner's own defaults still apply otherwise.
    pp_kwargs = {}
    if args.grasp_z is not None:
        pp_kwargs["grasp_z"] = args.grasp_z
    if args.approach is not None:
        pp_kwargs["approach"] = args.approach
    if args.table_z is not None:
        pp_kwargs["table_z"] = args.table_z
    traj = planner.plan_pick_place(
        joint_names=jn,
        start_q=start_q,
        gripper_joint=jn[-1],
        cube_xy=list(args.cube_xy),
        place_xy=list(args.place_xy),
        **pp_kwargs,
    )

    payload = {
        "joint_names": list(traj.joint_names),
        "waypoints": [dict(wp) for wp in traj.waypoints],
        "phases": list(traj.phases),
        "planner": traj.planner,
        "meta": dict(traj.meta or {}),
        # Record the scene the plan targets so the replayer can sanity-check it.
        "plan_for": {"cube_xy": list(args.cube_xy), "place_xy": list(args.place_xy), "start_q": list(start_q)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"WROTE {args.out}: planner={traj.planner} waypoints={len(traj.waypoints)} joints={traj.joint_names}")


if __name__ == "__main__":
    main()
