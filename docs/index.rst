RecourseBench documentation
============================

**RecourseBench** is a modular framework for reproducible algorithmic recourse
evaluation. It composes five extensible component types — datasets,
preprocessing steps, target models, recourse methods, and evaluation metrics —
into experiments that are configured as data and run end to end.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      Install the package, run your first experiment from a YAML config, and
      use the Python API.

   .. grid-item-card:: API reference
      :link: reference/index
      :link-type: doc

      Full description of every public class and function: arguments, return
      values, and behaviour.

   .. grid-item-card:: Extending the framework
      :link: extending
      :link-type: doc

      Register new datasets, preprocessing, models, methods, and metrics.

   .. grid-item-card:: Components
      :link: reference/index
      :link-type: doc

      Browse the bundled datasets, models, methods, and evaluation metrics.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   extending
   reference/index
