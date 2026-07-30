Utilities
=========

Exceptions
----------

.. currentmodule:: recourse_bench.utils.exceptions

.. autoexception:: RecourseBenchError
   :show-inheritance:

.. autoexception:: ConfigError
   :show-inheritance:

.. autoexception:: MissingDependencyError
   :show-inheritance:

Optional dependencies
---------------------

.. automodule:: recourse_bench.utils.dependencies
   :no-members:

.. currentmodule:: recourse_bench.utils.dependencies

.. autoclass:: optional_dependency
   :members:

.. autofunction:: require_optional

Reproducibility
---------------

.. currentmodule:: recourse_bench.utils.seed

.. autofunction:: seed_context

Caching
-------

.. currentmodule:: recourse_bench.utils.caching

.. autofunction:: default_cache_dir

.. autofunction:: get_cache_dir

.. autofunction:: set_cache_dir

Logging
-------

.. currentmodule:: recourse_bench.utils.logger

.. autofunction:: setup_logger

Model helpers
-------------

.. currentmodule:: recourse_bench.model.model_utils

.. autofunction:: resolve_device

.. autofunction:: logits_to_prediction

Evaluation helpers
------------------

.. currentmodule:: recourse_bench.evaluation.evaluation_utils

.. autofunction:: distance

.. autofunction:: restore_features

Preprocessing helpers
---------------------

.. currentmodule:: recourse_bench.preprocess.preprocess_utils

.. autofunction:: resolve_feature_metadata

Benchmark suites
----------------

.. currentmodule:: recourse_bench.benchmark.run

.. autofunction:: run_benchmarks
