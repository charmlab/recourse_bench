Component registry
==================

.. currentmodule:: recourse_bench.utils.registry

Components register themselves with :func:`register` and are looked up by
:class:`~experiments.Experiment` via :func:`get_registry`.

.. autofunction:: register

.. autofunction:: get_registry

Renamed components keep working: :class:`~recourse_bench.experiments.Experiment`
passes every config name through :func:`resolve_name` before looking it up, so a
config written against an older release still runs and reports the new name
through a ``DeprecationWarning``.

.. autofunction:: resolve_name

.. autodata:: deprecated_names
