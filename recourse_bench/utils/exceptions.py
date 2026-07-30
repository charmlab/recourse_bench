from __future__ import annotations


class RecourseBenchError(Exception):
    """Base class for all errors raised by the recourse_bench library.

    Library code raises subclasses of this exception rather than calling
    ``sys.exit``/``SystemExit`` so that programmatic callers (notebooks,
    pipelines, tests) can catch and handle failures. Only the CLI entry
    points translate these into a process exit code.
    """


class ConfigError(RecourseBenchError):
    """Raised when an experiment configuration is missing or invalid."""


class MissingDependencyError(RecourseBenchError):
    """Raised when a component needs a package from an uninstalled extra.

    Methods backed by a heavy or separately licensed solver keep their
    third-party import optional: the method stays registered and discoverable,
    and this is raised only when one is actually constructed. See
    :mod:`recourse_bench.utils.dependencies`.
    """
