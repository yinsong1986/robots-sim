"""Test suite for the Newton/Warp backend stub package.

Pre-R11 the only behaviour worth pinning is:
* the package imports cleanly (no warp/newton at module-load time),
* the entry-point declared in ``pyproject.toml`` resolves to
  :class:`NewtonSimulation`,
* :meth:`NewtonSimulation.is_available` returns the expected
  ``(bool, str | None)`` shape,
* every other SimEngine method raises ``NotImplementedError``.

The R11 PR will replace this stub-shape suite with real Newton tests
gated on ``STRANDS_GPU_TEST=1`` (mirroring ``isaac/tests/test_gpu_integ.py``).
"""
