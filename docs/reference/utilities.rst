Utilities
=========

Exceptions
----------

.. currentmodule:: utils.exceptions

.. autoexception:: RecourseBenchError
   :show-inheritance:

.. autoexception:: ConfigError
   :show-inheritance:

Reproducibility
---------------

.. currentmodule:: utils.seed

.. autofunction:: seed_context

Caching
-------

.. currentmodule:: utils.caching

.. autofunction:: get_cache_dir

.. autofunction:: set_cache_dir

Logging
-------

.. currentmodule:: utils.logger

.. autofunction:: setup_logger

Model helpers
-------------

.. currentmodule:: model.model_utils

.. autofunction:: resolve_device

.. autofunction:: logits_to_prediction

Evaluation helpers
------------------

.. currentmodule:: evaluation.evaluation_utils

.. autofunction:: distance

.. autofunction:: restore_features

Preprocessing helpers
---------------------

.. currentmodule:: preprocess.preprocess_utils

.. autofunction:: resolve_feature_metadata

Benchmark suites
----------------

.. currentmodule:: benchmark.run

.. autofunction:: run_benchmarks
