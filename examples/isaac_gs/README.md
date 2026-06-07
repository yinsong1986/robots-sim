# Isaac Sim + 3D Gaussian Splatting hybrid render — strands-robots-sim example

The Isaac-Sim companion to [`examples/mujoco_gs`](../mujoco_gs/). Same core
idea — a simulated robot composited against a photoreal 3D Gaussian Splatting
(3DGS) background with per-pixel, depth-aware occlusion — but with a
**deliberately different motivation**.

## Why this exists (and how it differs from `mujoco_gs`)

`mujoco_gs` exists because MuJoCo's renderer **isn't** photoreal, so it
composites the robot against a 3DGS scene to *gain* photorealism.

Isaac Sim's RTX renderer is **already** photoreal. So this example isn't about
fixing a renderer — it's about the **digital-twin / real2sim** use case:

> Drop an RTX-rendered **simulated** robot into a **real-world-captured 3DGS
> environment**, with correct depth-aware occlusion, so the sim robot looks
> like it's standing in the captured real scene.

That's the genuinely Isaac-flavoured angle for 3DGS compositing: merging sim
physics + RTX rendering with captured reality, rather than patching a
non-photoreal rasteriser.

| Aspect | `mujoco_gs` | `isaac_gs` (this) |
|---|---|---|
| Physics + foreground render | MuJoCo offscreen | **Isaac Sim RTX** |
| Why composite over 3DGS | gain photorealism | place sim robot in a *captured-real* scene |
| Robot | SO-101 (MuJoCo MJCF) | **real Franka Panda USD** (9 DoF) |
| Foreground depth | MuJoCo `render_depth` | Isaac camera `get_depth()` |
| Background renderer | `mujoco_gs.backgrounds` | **reused verbatim** from `mujoco_gs.backgrounds` |
| UI | live Gradio MJPEG | **render-stills / clip** (RTX isn't real-time-cheap) |

## What's reused vs. new

The background renderers (`PanoramaBackground` procedural default,
`GsplatBackground` for real `.ply` / `.spz` captures) are backend-agnostic and
**reused verbatim** from `examples.mujoco_gs.backgrounds` — they only need a
pinhole `CameraParams` (intrinsics + world pose) and numpy. This example
supplies those `CameraParams` from the Isaac RTX camera instead of MuJoCo's
`mj_data`.

| Module | New / reused |
|---|---|
| `camera_utils.py` | **new** — `CameraParams` from the Isaac `Camera` handle (`get_intrinsics_matrix()` / `get_world_pose()`) + `render_rgb_and_depth` via `sim.render()` |
| `compositor.py` | **new** — `IsaacHybridCompositor`: z-composite Isaac RGB+depth over the background (the maths mirrors `mujoco_gs`'s but pulls Isaac frames) |
| `scene.py` | **new** — real Franka USD + red cube + RTX camera |
| `render_demo.py` | **new** — render-stills / clip entry point |
| `backgrounds.py` | **reused** from `examples/mujoco_gs` (imported, not copied) |

## Install + run

```bash
pip install 'strands-robots-sim[isaac]'          # + a working Isaac Sim (RTX GPU)

# Procedural panorama background (zero ML deps), default Franka:
python -m examples.isaac_gs.render_demo --frames 1 --out rollouts/isaac_gs

# Sweep the arm across frames to show it moving on the backdrop:
python -m examples.isaac_gs.render_demo --frames 12 --wave

# Real captured 3DGS background (digital-twin use case; needs gsplat + a .ply):
pip install gsplat
python -m examples.isaac_gs.render_demo --gsplat-ply /path/to/kitchen.ply
```

Frames are written as PNGs under the output dir; a grep-stable summary line
(`isaac_gs  frames=N  robot=...  out=...  backend=isaac`) closes the run.

## Runtime dependencies

This example exercises the Phase-2 camera + render wiring on `IsaacSimulation`:

| Need | Rides on |
|---|---|
| Camera intrinsics / pose / handle | [PR #61](https://github.com/strands-labs/robots-sim/pull/61) `add_camera` |
| RGB + metric depth frames | [PR #62](https://github.com/strands-labs/robots-sim/pull/62) `render` |
| Real Franka articulation | [PR #63](https://github.com/strands-labs/robots-sim/pull/63) `add_robot(usd_path=)` |

Until those merge, `render` returns blank frames / the robot loads as a
no-op stub on a stock `main` build. **Draft until they land.**

## GPU-validated

Run on Isaac Sim 4.5 (`nvcr.io/nvidia/isaac-sim:4.5.0`, NVIDIA L4) against a
local integration of #61 + #62 + #63:

```
Scene built: robot=robot (9 joints), camera=front, objects=['cube']
→ composited frame: 640x480, 13,736 unique colors, gradient backdrop present
  (Franka RTX foreground z-composited over the procedural panorama)
```

(Isaac Sim 4.5 segfaults on its atexit cleanup after the frame is written —
a known Isaac issue, unrelated to this example's correctness.)

## Deliberate scope cuts

* **Render-stills / clip, not a live Gradio view.** Isaac's RTX renderer isn't
  real-time-cheap the way MuJoCo's offscreen renderer is, and the
  `SimulationApp` boot is heavyweight (~200 s). A live-view / agent-driven
  variant can layer on once the per-frame RTX cost is budgeted.
* **DC-term GS color only** (inherited from the reused `GsplatBackground`) — no
  view-dependent spherical-harmonics.
* **No view-dependent background relighting** — the captured 3DGS scene is a
  fixed backdrop; the sim robot is lit by Isaac's RTX scene lights.

## License

Apache-2.0, same as the rest of `strands-robots-sim`.
