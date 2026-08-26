"""Functional, user-facing API for RecourseBench.

This module backs the canonical entry points most users interact with:

* :func:`run` / :func:`run_config_file` — run a whole experiment from a config.
* ``list_*`` — discover the registered component names.

Components themselves are constructed through the dynamically populated
namespaces (``recourse_bench.methods``, ``recourse_bench.datasets``,
``recourse_bench.models``, ``recourse_bench.evaluations``,
``recourse_bench.preprocessors``), which are built from the same registry by
:func:`build_namespace` and exposed on the package. For example
``recourse_bench.methods.wachter`` is the ``WachterMethod`` class; call it to
construct an instance.
"""

from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import yaml

from experiments import Experiment
from utils.registry import get_registry


def list_datasets() -> list[str]:
    """Return the names of all registered datasets."""
    return sorted(get_registry("dataset"))


def list_preprocess() -> list[str]:
    """Return the names of all registered preprocessing steps."""
    return sorted(get_registry("preprocess"))


def list_models() -> list[str]:
    """Return the names of all registered target models."""
    return sorted(get_registry("model"))


def list_methods() -> list[str]:
    """Return the names of all registered recourse methods."""
    return sorted(get_registry("method"))


def list_evaluations() -> list[str]:
    """Return the names of all registered evaluation metrics."""
    return sorted(get_registry("evaluation"))


def run(config: dict) -> pd.DataFrame:
    """Run a full experiment from a config dictionary.

    Thin functional facade over :class:`~experiments.Experiment`; equivalent to
    ``Experiment(config).run()``. Use :class:`~experiments.Experiment` directly
    when you also want the trained model, counterfactuals, or other artifacts.

    Parameters
    ----------
    config : dict
        Experiment configuration (see :class:`~experiments.Experiment`).

    Returns
    -------
    pandas.DataFrame
        The metrics table, with provenance under ``metrics.attrs``.
    """
    return Experiment(config).run()


def run_config_file(path: str) -> pd.DataFrame:
    """Load a YAML config from ``path`` and :func:`run` it.

    Parameters
    ----------
    path : str
        Path to a YAML experiment config.

    Returns
    -------
    pandas.DataFrame
        The metrics table.
    """
    return run(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def build_namespace(kind: str, fqname: str, doc: str) -> types.ModuleType:
    """Build a module-like namespace of registered classes for one kind.

    The returned object exposes each registered component as an attribute under
    its registry name (e.g. ``methods.wachter`` is the ``WachterMethod`` class),
    so it can be called to construct an instance. Populated dynamically from the
    registry, so it never drifts from the registered components.

    Parameters
    ----------
    kind : str
        Registry kind: ``"dataset"``, ``"preprocess"``, ``"model"``,
        ``"method"``, or ``"evaluation"``.
    fqname : str
        Fully qualified module name to assign (e.g. ``"recourse_bench.methods"``).
    doc : str
        Module docstring.

    Returns
    -------
    types.ModuleType
        A namespace whose attributes are the registered component classes.
    """
    namespace = types.ModuleType(fqname)
    namespace.__doc__ = doc
    registry = get_registry(kind)
    for name, cls in registry.items():
        setattr(namespace, name, cls)
    namespace.__all__ = sorted(registry)
    return namespace
