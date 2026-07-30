Using RecourseBench
===================

This is the canonical, user-facing API. Import the package as ``rb`` and use the
named namespaces to construct components, or :func:`~recourse_bench.api.run` to
execute a whole experiment from a config.

.. code-block:: python

   import recourse_bench as rb

   metrics = rb.run(config)              # run an experiment from a config dict

   model  = rb.models.linear(seed=7)     # construct components by name
   method = rb.methods.wachter(target_model=model, seed=7, desired_class=1)
   data   = rb.datasets.credit()

Named namespaces
----------------

Each registered component is available as an attribute of a namespace, under its
registry name. The attribute *is* the component class — call it to construct an
instance. The namespaces are populated dynamically from the registry, so they
always reflect the currently registered components.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Namespace
     - Base class
     - Contains
   * - ``rb.datasets``
     - :class:`~dataset.dataset_object.DatasetObject`
     - Tabular datasets, e.g. ``rb.datasets.credit``
   * - ``rb.preprocessors``
     - :class:`~preprocess.preprocess_object.PreProcessObject`
     - Pipeline steps, e.g. ``rb.preprocessors.scale``
   * - ``rb.models``
     - :class:`~model.model_object.ModelObject`
     - Target classifiers, e.g. ``rb.models.linear``
   * - ``rb.methods``
     - :class:`~method.method_object.MethodObject`
     - Recourse methods, e.g. ``rb.methods.wachter``
   * - ``rb.evaluations``
     - :class:`~evaluation.evaluation_object.EvaluationObject`
     - Evaluation metrics, e.g. ``rb.evaluations.validity``

Constructor arguments match the corresponding base class (see :doc:`../extending`
for the full argument and method signatures). The lists below are a snapshot;
call ``rb.list_datasets()``, ``rb.list_methods()``, etc. for the authoritative,
up-to-date set. To iterate over components by name (e.g. for a sweep), use
``getattr``::

   for name in ["wachter", "dice", "gs"]:
       method = getattr(rb.methods, name)(target_model=model, seed=7)

Datasets — ``rb.datasets``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Available:* ``adult``, ``adult_cfrl``, ``adult_cfvae``, ``adult_cogs``,
``boston_housing``, ``breast_cancer``, ``compas``, ``compas_carla``,
``compas_clue``, ``credit``, ``credit_cchvae``, ``diabetes``, ``german``,
``german_roar``, ``german_sns``, ``hepatitis``, ``news_popularity``,
``synthetic_face``, ``toy_data`` (variants suffixed with a method name carry the
features/metadata that method expects).

* **Construct:** ``data = rb.datasets.credit()`` — no required arguments; the raw
  dataframe and feature metadata are loaded from the bundled offline data.
* **Output:** a :class:`~dataset.dataset_object.DatasetObject` in its *mutable*
  state. Preprocessing steps mutate it; once frozen the read interface is
  ``data.get(target=False)`` (feature columns as a ``DataFrame``),
  ``data.get(target=True)`` (the label column), ``len(data)``, ``data[idx]``, and
  ``data.attr(name)`` for feature metadata (type, mutability, actionability).

Preprocessors — ``rb.preprocessors``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Available:* ``balance``, ``encode``, ``finalize``, ``reorder``, ``scale``,
``split`` (a typical pipeline runs ``balance → encode → scale → split →
finalize``).

* **Construct:** ``step = rb.preprocessors.scale(scaling="normalize")`` —
  ``seed`` plus step-specific options.
* **Input → output:** ``step.transform(dataset)`` takes a mutable
  :class:`~dataset.dataset_object.DatasetObject` and returns the transformed
  dataset, or a *tuple* of datasets for steps that split (``split`` → train_set,
  test_set).

Models — ``rb.models``
^^^^^^^^^^^^^^^^^^^^^^^

*Available:* ``linear``, ``mlp``, ``mlp_bayesian``, ``random_forest``,
``sklearn_logistic_regression``.

* **Construct:** ``model = rb.models.linear(seed=7, device="cpu")``.
* **Inputs → outputs:** ``model.fit(train_set)`` trains on a frozen
  :class:`~dataset.dataset_object.DatasetObject`; ``model.predict(test_set)`` /
  ``model.predict_proba(test_set)`` return a ``(n_rows, n_classes)``
  ``torch.Tensor`` of logits / probabilities; ``model.get_prediction(X,
  proba=...)`` predicts on a feature ``DataFrame``. Differentiable models also
  support ``model(X)`` / ``model.forward(X)`` on a feature tensor.

Methods — ``rb.methods``
^^^^^^^^^^^^^^^^^^^^^^^^^

*Available:* ``apas``, ``arg_ensembling``, ``cchvae``, ``cemsp``, ``cfrl``,
``cfvae``, ``claproar``, ``clue``, ``cogs``, ``cols``, ``cruds``, ``cvas_proj``,
``dice``, ``diverse_dist``, ``face``, ``feature_tweak``, ``gravitational``,
``gs``, ``larr``, ``mace``, ``probe``, ``proplace``, ``rbr``, ``revise``,
``roar``, ``sns``, ``toy``, ``trex``, ``wachter``.

* **Construct:** ``method = rb.methods.wachter(target_model=model, seed=7,
  desired_class=1)`` — wraps a (to-be-)trained ``model``; ``desired_class``
  steers which class counterfactuals move toward (``None`` flips a binary
  label). Extra keyword arguments are method-specific hyperparameters.
* **Inputs → outputs:** ``method.fit(train_set)`` builds any auxiliary search
  structures; ``method.get_counterfactuals(factuals)`` takes a feature
  ``DataFrame`` and returns one with the same rows and columns, with ``NaN`` rows
  where no valid counterfactual was found. The inherited
  ``method.predict(test_set)`` runs that in batches and returns a *frozen*
  counterfactual :class:`~dataset.dataset_object.DatasetObject` carrying runtime,
  prediction, and target-label metadata (failed rows have ``NaN`` features and
  target ``-1``).

Evaluations — ``rb.evaluations``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Available:* ``constraint``, ``distance``, ``example``, ``knn``, ``runtime``,
``validity``, ``ynn``.

* **Construct:** ``metric = rb.evaluations.validity()`` — metric-specific options
  (e.g. a reference set or distance norm).
* **Input → output:** ``metric.evaluate(factuals, counterfactuals)`` takes the
  finalized factual dataset and the counterfactual dataset from
  ``method.predict`` and returns a *single-row* ``DataFrame`` of named metrics.
  :class:`~experiments.Experiment` concatenates these column-wise into the final
  metrics table.

Run an experiment
-----------------

.. currentmodule:: recourse_bench.api

.. autofunction:: run

.. autofunction:: run_config_file

For full control — including access to the trained model, the generated
counterfactuals, and run provenance — use the :class:`~experiments.Experiment`
class directly.

.. currentmodule:: recourse_bench.experiments

.. autoclass:: Experiment
   :members:
   :member-order: bysource

Discover components
-------------------

.. currentmodule:: recourse_bench.api

.. autofunction:: list_datasets

.. autofunction:: list_preprocessors

.. autofunction:: list_models

.. autofunction:: list_methods

.. autofunction:: list_evaluations
