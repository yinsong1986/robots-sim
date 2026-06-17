# Contributing

Issues and PRs welcome. Track work on the
[Strands Labs - Robots project board](https://github.com/orgs/strands-labs/projects/2);
it is the source of truth for roadmap and follow-ups.

- [GitHub Issues](https://github.com/strands-labs/robots-sim/issues)
- [Pull Requests](https://github.com/strands-labs/robots-sim/pulls)
- [Umbrella roadmap (#8)](https://github.com/strands-labs/robots-sim/issues/8)

## Repo split (read this first)

`strands-robots-sim` is a **plugin** for `strands-robots`. The split is
deliberate and load-bearing for the user experience — please keep
contributions on the right side of the line:

| Lives upstream (`strands-labs/robots`) | Lives here (`strands-labs/robots-sim`) |
|---|---|
| `Simulation` AgentTool, `SimEngine` ABC, `create_simulation` factory | `IsaacSimulation`, `IsaacConfig`, procedural builders, URDF/MJCF/USD loaders |
| MuJoCo backend | Isaac Sim backend |
| Robot catalog (`robots.json`) | (none — robots are name-resolved upstream) |
| Policy providers (GR00T, LeRobot, cuRobo) | (none — backend-agnostic) |
| Hardware (`HardwareRobot`, mesh, IoT, device-connect) | (none) |
| LIBERO MuJoCo example drivers | LIBERO Isaac example drivers |

If a change extends the `SimEngine` ABC or modifies the agent-facing
`Simulation` AgentTool, file the PR against `strands-labs/robots` first;
this repo follows.

## Dev setup

```bash
git clone https://github.com/strands-labs/robots-sim
cd robots-sim
pip install -e '.[isaac,dev]'

hatch run lint                              # black --check + isort --check + flake8
hatch run format                            # black + isort
hatch run test                              # pytest strands_robots_sim/isaac/tests/
```

GPU integration tests are gated on an env var:

```bash
STRANDS_GPU_TEST=1 pytest strands_robots_sim/isaac/tests/test_gpu_integ.py -v
```

These require Isaac Sim installed on the host. They will not run in CI
unless the `STRANDS_GPU_TEST` env var is set; the GPU runner wiring is
tracked under the umbrella roadmap
[#8](https://github.com/strands-labs/robots-sim/issues/8).

Python 3.10+ required (mirroring Isaac Sim 4.5's bundled interpreter).

## Conventions

- **Conventional commits.** `feat(scope):`, `fix(scope):`, `docs(scope):`,
  `refactor(scope):`, `test(scope):`, etc. The `scope` is one of `isaac`,
  `examples`, `docs`, `ci`, `pyproject`.
- **PR titles mirror the commit message** — see the
  [PR list](https://github.com/strands-labs/robots-sim/pulls) for
  examples.
- **Close issues** with `closes #N` in the commit body so GitHub auto-links.
- **No emojis in commit messages, PR titles, or PR comments** unless the
  surrounding repo culture clearly uses them.

## Lint / format

Code style is enforced by `black` (line-length 120), `isort` (black
profile), and `flake8`. The lint target also catches drift in the
`examples/` files because they get copy-pasted into PR docstrings + the
matrix table.

## Testing

Test slices live alongside the code:

```
strands_robots_sim/isaac/tests/
├── test_unit.py                            # mocked, no GPU
├── test_entrypoint.py                      # entry-point + lazy-import surface
├── test_get_observation_diagnostic_logs.py # WARNING/DEBUG level pins
├── test_procedural_g1_dof.py               # G1 DOF-count drift pin
├── test_procedural_kinematic_guard.py      # fail-first kinematic-tree pin
├── test_loaders.py                         # URDF / MJCF / USD round-trip + robosuite parity
└── test_gpu_integ.py                       # gated on STRANDS_GPU_TEST=1
```

Two house rules:

- **Every PR adds or updates a test.** A behavior change with no test
  delta will get review pushback. The exception: pure docs / lint PRs.
- **Tests must run without a GPU** (excluding `test_gpu_integ.py`). Mock
  `omni.*` imports with `unittest.mock.patch` if you need to cross the
  Isaac boundary in unit tests.

## Documentation

Doc edits go in `docs/` (this directory). The site is built with MkDocs
Material:

```bash
pip install -r docs/requirements.txt
mkdocs serve                                # live-reload at http://127.0.0.1:8000
mkdocs build --strict                       # CI's check; should print zero warnings
```

The `Docs` GitHub Actions workflow runs `mkdocs build --strict` on every
PR touching `docs/**` or `mkdocs.yml`. Broken nav links / missing pages
fail the PR.

When you add a page:

1. Author it under `docs/<section>/<page>.md`.
2. Add it to the `nav:` block in [`mkdocs.yml`](https://github.com/strands-labs/robots-sim/blob/main/mkdocs.yml).
3. Cross-link from the closest sibling pages.
4. `mkdocs build --strict` locally before pushing.

## Coordinating with `strands-labs/robots`

Many follow-ups in this repo are gated on upstream PRs. The pattern:

- **R<n>** issues here track work on this repo's side of a split task.
- They reference upstream issues / PRs by number.
- Land the upstream piece first, rebase this side on the merged upstream,
  open the PR here.

The umbrella tracker [`#8`](https://github.com/strands-labs/robots-sim/issues/8)
maps stages → versions → which issues block which.

## Reporting security issues

**Do not** open a public issue for a security vulnerability. See
[Security](security.md) for the disclosure process (AWS VDP / HackerOne).
