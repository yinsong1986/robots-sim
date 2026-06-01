# `strands_robots_sim/isaac/tests/`

Test suite for the Isaac Sim backend, split into two tiers by what
each tier needs to run.

## Two tiers

| Tier | Files | Needs | Where it runs |
|---|---|---|---|
| **CPU-only** | `test_unit.py`, `test_entrypoint.py`, `test_loaders.py`, `test_procedural_g1_dof.py`, `test_procedural_kinematic_guard.py`, `test_phase1_doc_banner.py`, `test_get_observation_diagnostic_logs.py` | Python 3.12 + `[isaac]` extras (`usd-core`, `warp-lang`, `pytest`); zero `omni.*` imports thanks to PEP 562 lazy stub | Every push / PR — main `test-lint.yml` workflow |
| **GPU** | `test_gpu_integ.py` (`@pytest.mark.gpu`) | NVIDIA RTX GPU, Isaac Sim 5.x installed, `STRANDS_GPU_TEST=1` env var | Nightly only — `nightly-gpu.yml` workflow on a self-hosted runner |

Marker source of truth: `pyproject.toml`'s
`[tool.pytest.ini_options].markers` declares only `gpu`. Anything that
needs a real Isaac Sim runtime gets that marker and is automatically
excluded from `pytest -m "not gpu"`-style invocations.

## Running locally

```bash
# CPU-only suite (the default in main CI):
pytest strands_robots_sim/isaac/tests/ \
    --ignore=strands_robots_sim/isaac/tests/test_gpu_integ.py

# Or just the unit subset:
pytest strands_robots_sim/isaac/tests/test_unit.py -v

# GPU suite (requires Isaac Sim):
STRANDS_GPU_TEST=1 pytest strands_robots_sim/isaac/tests/test_gpu_integ.py -v
```

Note that `STRANDS_GPU_TEST=1` is an additional gate inside
`test_gpu_integ.py` itself: even with `pytest -m gpu`, the test cases
no-op if the env var isn't set. This lets a developer enable the marker
without accidentally launching a `SimulationApp` on a CI machine that
happens to have the marker selected.

## Nightly GPU CI

[`.github/workflows/nightly-gpu.yml`](../../../.github/workflows/nightly-gpu.yml)
schedules the GPU tier nightly (03:17 UTC) and exposes a manual trigger
via the Actions UI's `workflow_dispatch`. The workflow:

1. Probes Isaac Sim availability before invoking `pytest`. If the
   runner is misprovisioned or the SDK can't be imported, the workflow
   exits with a yellow skip rather than a red failure -- so a runner
   issue surfaces visually but doesn't break the badge.
2. Runs `pytest -m gpu --tb=short` with `STRANDS_GPU_TEST=1` and
   uploads `test-results/junit-isaac-gpu.xml` + `pytest.log` as a
   workflow artifact (30-day retention).
3. On failure, surfaces the first 5 `FAILED` / `ERROR` / `E` /
   traceback-arrow lines into the run's `$GITHUB_STEP_SUMMARY` so
   triage doesn't require opening the log artifact.

Failure notifications use GitHub Actions' default email-to-watchers
path; if the org switches to a Slack / Discord webhook later, only the
summary step needs to change.

## Adding a test that needs Isaac Sim runtime

Mark the test (or the whole module) with `gpu` and gate any
`SimulationApp`-touching code on `STRANDS_GPU_TEST=1`:

```python
import os

import pytest

pytestmark = pytest.mark.gpu

_GPU_AVAILABLE = os.environ.get("STRANDS_GPU_TEST", "0") == "1"


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="STRANDS_GPU_TEST=1 not set")
def test_my_runtime_thing() -> None:
    from strands_robots_sim.isaac import IsaacSimulation

    available, msg = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {msg}")
    # ... real instantiation ...
```

The double-gate (marker + env var + `is_available()` skip) is by
design: each layer independently protects a different failure mode.
