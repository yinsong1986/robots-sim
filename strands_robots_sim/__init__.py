"""strands-robots-sim — heavy NVIDIA simulation backends for strands-robots.

As of 0.2.0 this package is a re-scoped plugin host. The legacy ``SimEnv``,
``SteppedSimEnv``, Libero-direct environment layer, GR00T policy client, and
``gr00t_inference`` AgentTool have all been removed — that lightweight
MuJoCo + LIBERO + GR00T code path now lives in
`strands-labs/robots <https://github.com/strands-labs/robots>`_, accessible
via the ``Simulation`` AgentTool, the ``LiberoAdapter`` benchmark plugin, and
``strands_robots.tools.gr00t_inference``.

This module is currently a no-op stub. The heavy GPU-only Isaac Sim
backend (``IsaacSimulation``) registers itself through ``strands-robots``
entry points; see the umbrella issue
https://github.com/strands-labs/robots-sim/issues/8.

See ``examples/MIGRATION.md`` for the old-API → new-API mapping.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("strands-robots-sim")
except PackageNotFoundError:
    # Editable install before metadata is generated, or running from a
    # working tree without ``pip install -e .`` having been run yet.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

_LEGACY_REMOVED = {
    "SimEnv": (
        "`SimEnv` was removed in strands-robots-sim 0.2.0. "
        "Use `Simulation(...).evaluate_benchmark(benchmark_name='libero-<suite>-<task>', ...)` "
        "from `strands-robots` instead. See examples/MIGRATION.md."
    ),
    "SteppedSimEnv": (
        "`SteppedSimEnv` was removed in strands-robots-sim 0.2.0. "
        "Use `Simulation.start_policy(...)` + poll `get_state` / `render` "
        "from `strands-robots` instead. See examples/MIGRATION.md."
    ),
    "gr00t_inference": (
        "`gr00t_inference` was removed in strands-robots-sim 0.2.0. "
        "Use `from strands_robots.tools.gr00t_inference import gr00t_inference` instead. "
        "See examples/MIGRATION.md."
    ),
    "Gr00tPolicy": (
        "`Gr00tPolicy` was removed in strands-robots-sim 0.2.0. "
        "Use `from strands_robots.policies.groot import Gr00tPolicy` instead. "
        "See examples/MIGRATION.md."
    ),
    "Policy": (
        "`Policy` was removed in strands-robots-sim 0.2.0. "
        "Use `from strands_robots.policies import Policy` instead. "
        "See examples/MIGRATION.md."
    ),
    "MockPolicy": (
        "`MockPolicy` was removed in strands-robots-sim 0.2.0. "
        "Use `from strands_robots.policies import MockPolicy` instead. "
        "See examples/MIGRATION.md."
    ),
    "create_policy": (
        "`create_policy` was removed in strands-robots-sim 0.2.0. "
        "Use `from strands_robots.policies import create_policy` instead. "
        "See examples/MIGRATION.md."
    ),
}


def __getattr__(name):  # PEP 562 module-level __getattr__
    """Surface a clear, actionable error for legacy import names.

    Raises ``ImportError`` (not ``AttributeError`` + ``DeprecationWarning``)
    so the message survives ``-W error::DeprecationWarning`` test envs that
    would otherwise mask the actionable hint with the warning's traceback.
    """
    if name in _LEGACY_REMOVED:
        raise ImportError(_LEGACY_REMOVED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
