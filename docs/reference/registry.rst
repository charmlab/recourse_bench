Component registry
==================

.. currentmodule:: recourse_bench.utils.registry

Components register themselves with :func:`register` and are looked up by
:class:`~experiments.Experiment` via :func:`get_registry`.

.. autofunction:: register

.. autofunction:: get_registry
