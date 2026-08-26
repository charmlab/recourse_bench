.. _reference:

API reference
=============

The :doc:`using` page covers the canonical, user-facing API — the named
namespaces (``rb.methods``, ``rb.datasets``, ``rb.models``, ...) and
:func:`~recourse_bench.api.run`. The **Extending** section below documents the
abstract base classes you subclass to add your own datasets, preprocessing,
models, methods, or metrics; see also the narrative :doc:`../extending` guide.

.. toctree::
   :maxdepth: 2
   :caption: Using RecourseBench

   using

.. toctree::
   :maxdepth: 1
   :caption: Extending RecourseBench

   dataset
   preprocess
   model
   method
   evaluation
   registry

.. toctree::
   :maxdepth: 1
   :caption: Utilities

   utilities
