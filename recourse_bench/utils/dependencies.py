"""Optional third-party dependencies for solver-backed methods.

Some methods need a heavy or separately licensed solver (Gurobi, Z3, pySMT,
clingo, CVXPY, ART, LIME). Those live in :ref:`extras <install-extras>` rather
than the base install, so the package stays small for the majority of users who
never run them.

An uninstalled extra must not make a method disappear: ``list_methods()`` is
meant to describe the benchmark, not the current machine, and a config naming a
solver-backed method should fail with a message that says how to fix it rather
than "unknown method". So an affected module guards its import and defers the
failure to construction time::

    with optional_dependency("apas", "gurobipy") as _gurobi:
        from recourse_bench.method.apas.support import prepare_apas_context

    @register("apas")
    class ApasMethod(MethodObject):
        def __init__(self, ...):
            _gurobi.require()

The class is registered either way; only ``__init__`` raises
:class:`~utils.exceptions.MissingDependencyError`. Import failures unrelated to
the optional package are never swallowed — they propagate as usual, so a real
bug in a support module still surfaces immediately.
"""

from __future__ import annotations

import importlib.util
from types import TracebackType

from recourse_bench.utils.exceptions import MissingDependencyError

# Importable module name -> (pyproject extra, PyPI distribution). The extra is
# what users type; the distribution is shown when it differs from the module so
# the message stays greppable against a `pip list`.
optional_packages: dict[str, tuple[str, str]] = {
    "gurobipy": ("gurobi", "gurobipy"),
    "z3": ("z3", "z3-solver"),
    "pysmt": ("smt", "PySMT"),
    "clingo": ("asp", "clingo"),
    "cvxpy": ("cvx", "cvxpy"),
    "art": ("art", "adversarial-robustness-toolbox"),
    "lime": ("lime", "lime"),
}


def _missing(component: str, module: str) -> MissingDependencyError:
    extra, distribution = optional_packages[module]
    if distribution != module:
        distribution = f"{distribution} ({module})"
    return MissingDependencyError(
        f"'{component}' requires {distribution}, which is not installed. "
        f"Install it with: pip install recourse_bench[{extra}]"
    )


def require_optional(component: str, module: str) -> None:
    """Raise unless an optional package is importable.

    For components that import the optional package lazily at call time rather
    than at module import. Calling this from ``__init__`` reports the missing
    extra up front, with the same message as the guarded-import path, instead of
    letting a bare ``ModuleNotFoundError`` surface mid-run. The package itself is
    not imported.

    Parameters
    ----------
    component : str
        Registry name of the component, used in the error message.
    module : str
        Top-level importable name of the optional package. Must be a key of
        :data:`optional_packages`.

    Returns
    -------
    None
        When the package is importable.

    Raises
    ------
    KeyError
        If ``module`` is not a known optional package.
    MissingDependencyError
        When the package is not installed.
    """
    if module not in optional_packages:
        raise KeyError(f"Unknown optional package: {module}")
    try:
        found = importlib.util.find_spec(module) is not None
    except ImportError as error:
        # find_spec returns None for a plainly absent module, but raises when a
        # parent package is missing or a finder rejects the name.
        raise _missing(component, module) from error
    if found:
        return
    raise _missing(component, module)


class optional_dependency:
    """Context manager guarding an import that needs an optional package.

    Wraps a module-level import so a missing optional package is recorded
    instead of raised, letting the module finish importing and the component
    stay registered. Call :meth:`require` from the component's ``__init__`` to
    turn a recorded failure into a
    :class:`~utils.exceptions.MissingDependencyError`.

    Parameters
    ----------
    component : str
        Registry name of the component, used in the error message.
    module : str
        Top-level importable name of the optional package (e.g. ``"gurobipy"``).
        Must be a key of :data:`optional_packages`.

    Raises
    ------
    KeyError
        If ``module`` is not a known optional package.

    Examples
    --------
    >>> with optional_dependency("cemsp", "z3") as _z3:
    ...     from recourse_bench.method.cemsp.support import build_solver
    >>> _z3.require()  # no-op when z3 is installed  # doctest: +SKIP
    """

    def __init__(self, component: str, module: str):
        if module not in optional_packages:
            raise KeyError(f"Unknown optional package: {module}")
        self.component = component
        self.module = module
        self.extra, self.distribution = optional_packages[module]
        self.error: ImportError | None = None

    @property
    def installed(self) -> bool:
        """Whether the optional package imported successfully."""
        return self.error is None

    def __enter__(self) -> optional_dependency:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if not isinstance(exc, ImportError):
            return False
        # Only swallow the absence of this optional package. An ImportError from
        # anywhere else is a real failure and must not be hidden behind a
        # "missing extra" message.
        name = exc.name or ""
        if name != self.module and not name.startswith(f"{self.module}."):
            return False
        self.error = exc
        return True

    def require(self) -> None:
        """Raise if the optional package was missing at import time.

        Returns
        -------
        None
            When the package is installed.

        Raises
        ------
        MissingDependencyError
            When it is not, naming the extra that provides it.
        """
        if self.error is None:
            return
        raise _missing(self.component, self.module) from self.error
