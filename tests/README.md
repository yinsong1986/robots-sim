# tests/

`strands-robots-sim` is currently a re-scoped plugin host with no runtime
code of its own (see [`../strands_robots_sim/__init__.py`](../strands_robots_sim/__init__.py)),
so this directory is intentionally empty.

Backend-specific test suites land alongside the backends:

| Backend | Tracking issue | Tests appear under |
|---|---|---|
| Isaac Sim | [#14](https://github.com/strands-labs/robots-sim/issues/14) (R7) | `tests/isaac/` |
| Newton / Warp | [#18](https://github.com/strands-labs/robots-sim/issues/18) (R11) | `tests/newton/` |

GPU CI is tracked separately in
[#17](https://github.com/strands-labs/robots-sim/issues/17) (R10, Isaac) and
[#21](https://github.com/strands-labs/robots-sim/issues/21) (R14, Newton).

Tests for the LIBERO benchmark, MuJoCo backend, and GR00T policy live in
[`strands-labs/robots`](https://github.com/strands-labs/robots) — see
[`examples/MIGRATION.md`](../examples/MIGRATION.md) for the old → new mapping.
