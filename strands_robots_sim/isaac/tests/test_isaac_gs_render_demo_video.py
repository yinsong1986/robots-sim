"""Output-parity tests for ``examples/isaac_gs/render_demo.py``'s clip writer.

`isaac_gs` is the digital-twin sibling of ``examples/mujoco_gs``, but it
historically produced only PNG stills while ``mujoco_gs/libero_groot.py``
produced an MP4 (see strands-labs/robots-sim#160). ``render_demo.py`` now
assembles the rendered frames into a video clip for a multi-frame run, at
output parity with ``mujoco_gs``'s ``imageio`` libx264 encode.

These tests pin the *frame-assembly* surface only — the pure, GPU-free
helpers (``_want_video`` / ``_video_path`` / ``_encode_clip``). The full
render path needs Isaac Sim + an RTX GPU and is exercised out-of-band
(see the example README's "GPU-validated" section), so it is not unit
tested here. The module is loaded directly from the examples tree via
``importlib`` so importing it does **not** pull in any ``omni.*`` / Isaac
modules (``main()`` performs the Isaac import lazily, at call time).

Run with::

    pytest strands_robots_sim/isaac/tests/test_isaac_gs_render_demo_video.py -v
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_RENDER_DEMO = _REPO_ROOT / "examples" / "isaac_gs" / "render_demo.py"


def _load_render_demo():
    spec = importlib.util.spec_from_file_location("isaac_gs_render_demo", _RENDER_DEMO)
    assert spec and spec.loader, f"cannot load {_RENDER_DEMO}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rd():
    return _load_render_demo()


def test_render_demo_file_exists():
    assert _RENDER_DEMO.is_file(), f"missing {_RENDER_DEMO}"


def test_clip_flags_present(rd):
    """The clip-control flags from #160 are wired into the parser."""
    parser = rd._build_parser()
    opts = {a.option_strings[0] for a in parser._actions if a.option_strings}
    for flag in ("--mp4", "--gif", "--out-video", "--fps", "--no-stills"):
        assert flag in opts, f"{flag} missing from render_demo CLI"


def test_single_frame_is_stills_only(rd):
    """A bare single-frame run has nothing to animate -> PNG only."""
    args = rd._build_parser().parse_args([])
    assert rd._want_video(args) is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--frames", "2"],
        ["--frames", "24"],
        ["--wave"],
    ],
)
def test_multiframe_runs_emit_a_clip(rd, argv):
    """Multi-frame / --wave runs assemble a clip by default (mujoco_gs parity)."""
    args = rd._build_parser().parse_args(argv)
    assert rd._want_video(args) is True


@pytest.mark.parametrize("flag", ["--mp4", "--gif", "--out-video=/tmp/clip.mp4"])
def test_explicit_clip_flag_forces_video_even_for_one_frame(rd, flag):
    args = rd._build_parser().parse_args(["--frames", "1", flag])
    assert rd._want_video(args) is True


def test_video_path_defaults_to_mp4_under_out_dir(rd, tmp_path):
    args = rd._build_parser().parse_args(["--frames", "4"])
    path = rd._video_path(args, str(tmp_path), "TS")
    assert path == str(tmp_path / "TS--isaac_gs.mp4")


def test_video_path_gif_extension(rd, tmp_path):
    args = rd._build_parser().parse_args(["--frames", "4", "--gif"])
    path = rd._video_path(args, str(tmp_path), "TS")
    assert path.endswith("TS--isaac_gs.gif")


def test_video_path_out_video_override_wins(rd, tmp_path):
    target = tmp_path / "nested" / "custom.mp4"
    args = rd._build_parser().parse_args(["--frames", "4", "--out-video", str(target)])
    path = rd._video_path(args, str(tmp_path), "TS")
    assert path == str(target)
    assert target.parent.is_dir(), "out-video parent dir should be created"


def _fake_frames(n: int = 6, h: int = 48, w: int = 64):
    rng = np.random.default_rng(0)
    return [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(n)]


def test_encode_clip_writes_mp4(rd, tmp_path):
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    out = tmp_path / "clip.mp4"
    rd._encode_clip(_fake_frames(), str(out), fps=10)
    assert out.is_file() and out.stat().st_size > 0


def test_encode_clip_writes_gif(rd, tmp_path):
    pytest.importorskip("imageio")
    out = tmp_path / "clip.gif"
    rd._encode_clip(_fake_frames(), str(out), fps=10)
    assert out.is_file() and out.stat().st_size > 0
