---
name: use-recourse-bench
description: Use when you need to use RecourseBench as a library to run algorithmic-recourse / counterfactual-explanation experiments — installing it, importing it, writing Python or YAML configs, choosing datasets/preprocessors/models/methods/evaluations, running experiments, and inspecting metrics and provenance. Not for developing RecourseBench itself.
metadata:
  version: v0.1.0
---

# Use RecourseBench

You are using `recourse-bench` as a **library** to run algorithmic-recourse
(counterfactual-explanation) experiments. An experiment composes five component
kinds — a dataset, preprocessing steps, a target model, a recourse method, and
evaluation metrics — and produces a one-row metrics table.

Prefer the high-level public API (`import recourse_bench as rb`). Do **not**
import internal modules (e.g. `recourse_bench.experiments`,
`recourse_bench.dataset.*`, `recourse_bench.utils.*`) unless the user explicitly
asks for debugging or development — that work belongs to a separate skill (see
[Out of scope](#out-of-scope)).

## When to use this skill

- The user wants to run a recourse experiment or sweep configs.
- The user asks which datasets / models / methods / metrics are available.
- The user wants metrics for a given (dataset, model, method) combination.
- The user wants to read an existing YAML config and run it.

Some neighboring requests belong to other skills — e.g. adding a new
method/dataset/model → `add-recourse-method` (see [Out of scope](#out-of-scope)
for the full list). Switch only when the request is *entirely* such a task. If
it's mostly a using-task that brushes development in passing, finish the
using-work — don't bail.

## Install

RecourseBench is published on TestPyPI. Dependencies (numpy, pandas, torch, …)
live on the real PyPI, so include the extra index:

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    recourse-bench
```

In a notebook, use `%pip` (not `!pip`) so it installs into the running kernel,
then restart the kernel before importing.

## Canonical import

```python
import recourse_bench as rb
```

**Gotcha — directory shadowing.** Python puts the current directory first on the
import path. If you run from a directory that contains a `recourse_bench/` folder
(e.g. a source checkout of this repo), that local folder shadows the installed
package and `import recourse_bench` may fail or import the wrong code. When using
the installed package, run your Python from a directory that does **not** contain
a `recourse_bench/` folder (a scratch/work directory is fine).

## Public API

Use only these entry points for normal usage:

| Call | Purpose |
| --- | --- |
| `rb.run(config)` | Run a full experiment from a config `dict`; returns a `pandas.DataFrame` of metrics. |
| `rb.run_config_file(path)` | Load a YAML config from `path` and run it; returns the same metrics table. |
| `rb.list_datasets()` | Names of registered datasets. |
| `rb.list_preprocessors()` | Names of registered preprocessing steps. |
| `rb.list_models()` | Names of registered target models. |
| `rb.list_methods()` | Names of registered recourse methods. |
| `rb.list_evaluations()` | Names of registered evaluation metrics. |

The `list_*` functions are the **authoritative** source of valid component
names. Call them instead of hard-coding names from memory or docs.

## Config shape

A config is a dict (or the YAML equivalent) with four required sections —
`dataset`, `model`, `method`, `evaluation` — plus optional `preprocess`,
`name`, and `seed`. Each component is selected by its registry `name`; remaining
keys are that component's options. A top-level `seed` propagates to any
component that does not set its own.

```python
config = {
    "name": "demo",
    "seed": 7,
    "dataset": {"name": "toy_data"},
    "preprocess": [
        {"name": "scale", "scaling": "standardize", "range": True},
        {"name": "split", "split": 0.25},
    ],
    "model":  {"name": "linear", "device": "cpu", "epochs": 30},
    "method": {"name": "wachter", "desired_class": 1},
    "evaluation": [
        {"name": "validity"},
        {"name": "distance"},
    ],
}
```

## Typical workflow

This is a quick composition, not a research plan. Don't enumerate every option or
deliberate over combinations — pick sensible verified names, build the config, and
run. The steps:

1. **Choose a dataset** — pick a name from `rb.list_datasets()`.
2. **Choose preprocessors** — pick from `rb.list_preprocessors()` (a typical
   pipeline is `scale` → `split`). Optional.
3. **Choose a target model** — pick from `rb.list_models()`.
4. **Choose a recourse method** — pick from `rb.list_methods()`. Most methods
   take `desired_class` to steer which class counterfactuals move toward.
5. **Choose evaluation metrics** — pick from `rb.list_evaluations()`
   (e.g. `validity`, `distance`).
6. **Run the experiment**:

   ```python
   metrics = rb.run(config)          # or rb.run_config_file("path/to/config.yaml")
   ```

7. **Inspect the metrics** — `metrics` is a one-row `pandas.DataFrame`:

   ```python
   print(metrics.to_string(index=False))
   ```

8. **Inspect provenance** when present — reproducibility metadata is attached
   under `metrics.attrs`:

   ```python
   prov = metrics.attrs.get("provenance")
   if prov:
       for key in ("library_version", "config_hash", "git_revision",
                   "timestamp_utc", "seed"):
           print(key, "=", prov.get(key))
   ```

## Verification

Before building on RecourseBench, confirm the install works and the registries
are populated. This is cheap and runs no experiment:

```bash
python -c "import recourse_bench as rb; \
print('datasets:', len(rb.list_datasets())); \
print('models:', len(rb.list_models())); \
print('methods:', len(rb.list_methods())); \
print('evaluations:', len(rb.list_evaluations()))"
```

Each count should be non-zero. If the import fails, re-check the install step
(and, in a notebook, that the kernel was restarted after installing).

## Cost discipline

- **Do not run expensive benchmark sweeps by default.** Run a single, small
  experiment unless the user explicitly asks for a sweep or full evaluation.
- Default to the smallest sensible setup for a smoke check (e.g. the `toy_data`
  dataset, the `linear` model, a single method, a few epochs, CPU device).
- Some methods rely on heavy solvers or long optimization. If a method is slow
  or pulls extra dependencies, surface that to the user before running it at
  scale rather than launching a large job.

## Out of scope

This skill is for **using** RecourseBench as an installed library. These belong
elsewhere:

- Adding datasets, preprocessors, models, methods, or metrics → `add-recourse-method`.
- Running the full smoke-test suite → `recoursebench-smoke-tests`.
- Reproducing a paper end-to-end → `recoursebench-reproduction`.
- Repo setup, refactors, packaging, or debugging internal modules — work the
  repo checkout directly, not through this skill.

Redirect only when the request is *wholly* one of these. Don't abandon a
using-task partway because it touches development — finish what's in scope and
name the other skill for the rest.

## Completion criteria

You are done when:

- The package imports and all five `list_*` calls return non-empty lists.
- A config was assembled from **verified** component names (from the `list_*`
  calls), not guessed.
- `rb.run(...)` (or `rb.run_config_file(...)`) returned a metrics `DataFrame`,
  and you reported the metrics to the user.
- Provenance was reported when `metrics.attrs["provenance"]` is present.
- No internal modules were imported (unless the user explicitly requested
  debugging/development).
- No large or expensive sweep was run without explicit user approval.
