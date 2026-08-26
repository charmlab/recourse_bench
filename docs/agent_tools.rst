Agent tools
===========

RecourseBench has two agent-facing pieces:

* **Skills** tell Codex which workflow to follow.
* **The MCP server** gives MCP clients a small set of safe RecourseBench tools.

Skills guide the agent's reasoning and file changes. MCP tools expose
controlled actions, such as listing registered methods, validating a config, or
running one bounded check.

Skills
------

Use the RecourseBench skills below. They intentionally separate ordinary
library usage from source-tree development and paper reproduction.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: use-recourse-bench
      :class-card: sd-shadow-sm

      Use this when the task is to run RecourseBench as a library.

      **Typical requests**

      * list available datasets, models, methods, or metrics
      * build a YAML or Python experiment config
      * run one small experiment
      * inspect metrics and provenance

      **Core rule:** stay on the public ``recourse_bench as rb`` API.

   .. grid-item-card:: add-recourse-method
      :class-card: sd-shadow-sm

      Use this when the task is to add a new recourse method to the repo.

      **Typical requests**

      * port a paper or reference implementation
      * implement a ``MethodObject`` subclass
      * wire the method into the registry
      * add a smoke config and smoke driver

      **Core rule:** treat new methods as plug-ins, not framework rewrites.

   .. grid-item-card:: paper-experiment-tests
      :class-card: sd-shadow-sm

      Use this after implementing a method to check it against the paper.

      **Typical requests**

      * extract runnable tests from a paper's experiments and tables
      * build reproduction configs on the paper's dataset/model/metrics
      * fill missing artifacts (dataset variants, target models)
      * write a standardized reproduction log

      **Core rule:** matching metrics is evidence, not proof of faithfulness.

.. tab-set::

   .. tab-item:: Library usage

      The ``use-recourse-bench`` skill is for existing components. It verifies
      names from the live registries before composing a config.

      .. code-block:: python

         import recourse_bench as rb

         rb.list_datasets()
         rb.list_models()
         rb.list_methods()
         rb.list_evaluations()

         metrics = rb.run(config)
         print(metrics.to_string(index=False))

      Choose this path when no source files need to change.

   .. tab-item:: Method development

      The ``add-recourse-method`` skill is for source checkouts. It follows the
      RecourseBench method contract and validates the new method with a smoke
      run.

      .. code-block:: text

         method/<name>/<name>.py              # MethodObject subclass
         method/__init__.py                   # registry import
         experiment/<name>/smoke_config.yaml  # small health check
         experiment/<name>/smoke.py           # smoke driver

      Choose this path when adding an algorithmic recourse method.

   .. tab-item:: Paper reproduction

      The ``paper-experiment-tests`` skill runs with or after
      ``add-recourse-method``. It turns a paper's experiments into runnable
      RecourseBench tests and records how close the observed numbers come to
      the reported ones, feeding the human faithfulness review.

      .. code-block:: text

         experiment/<name>/_paper_tests/          # paper evidence + test plan
         experiment/<name>/<ds>_<model>_<name>_reproduce.yaml
         experiment/<name>/test_<name>_reproduce.py
         experiment/<name>/reproduce_logs.txt     # standardized report

      Choose this path when reproducing a paper's reported results for an
      already-implemented method.

MCP server
----------

The MCP server is optional. It is a local-first stdio server for agents and MCP
clients. It is not the primary user interface; for direct Python work, use the
public API in :doc:`getting_started`.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: What it does
      :class-card: sd-shadow-sm

      * discovers registered RecourseBench components
      * builds small example configs
      * validates config structure
      * runs one experiment or one smoke test
      * runs a bounded benchmark pack

   .. grid-item-card:: What it avoids
      :class-card: sd-shadow-sm

      * arbitrary Python execution
      * arbitrary file reads
      * full benchmark sweeps by default
      * writing or updating baselines
      * returning raw pandas or numpy objects

Install the optional dependency:

.. code-block:: bash

   pip install -e ".[mcp]"

Start the server:

.. code-block:: bash

   recourse-bench-mcp
   # or
   python -m recourse_bench.mcp_server

Example client entry:

.. code-block:: json

   {
     "mcpServers": {
       "recourse-bench": {
         "command": "recourse-bench-mcp"
       }
     }
   }

MCP tools
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - Purpose
   * - ``list_recourse_components``
     - List registered datasets, preprocessors, models, methods, or
       evaluations.
   * - ``get_example_config``
     - Build a small editable config from registered component names.
   * - ``validate_recourse_config``
     - Check config structure without running an experiment.
   * - ``run_recourse_experiment``
     - Run exactly one config through ``rb.run``.
   * - ``run_recourse_config_file``
     - Run one repo-local config file.
   * - ``run_smoke_test``
     - Run one method's ``experiment/<method>/smoke_config.yaml``.
   * - ``run_benchmark_pack``
     - Run a bounded pack such as ``small``.

.. important::

   The MCP server validates component names, restricts config and baseline
   paths, uses CPU-first examples, converts pandas/numpy values to JSON, and
   keeps benchmark packs bounded.
