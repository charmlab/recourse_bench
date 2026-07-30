# Changelog

## 0.2.0

Naming is now consistent across the public API, and solver backends are
optional. Renames keep the old names working with a `DeprecationWarning`, with
one exception noted under *Breaking* below.

### Breaking

- `rb.list_preprocess()` is now `rb.list_preprocessors()`. This is the only
  rename without a compatibility alias — all five `list_*` helpers are plural.
- Solver backends moved out of the base install into extras (see *Install*).
  Methods needing one stay registered and listed, but constructing them without
  their extra raises `MissingDependencyError` instead of working.
- `requirements.txt` no longer pins a CUDA build of PyTorch. Install torch for
  your platform first; see the README.

### Renamed (old names still work, with a `DeprecationWarning`)

| Kind | Old | New |
| --- | --- | --- |
| Evaluation | `constraints`, `examples` | `constraint`, `example` |
| Dataset | `toydata` | `toy_data` |
| Model | `randomforest` | `random_forest` |
| `Experiment` | `.trainset()`, `.testset()` | `.train_set()`, `.test_set()` |
| Dataset flag | `trainset`, `testset` | `train_set`, `test_set` |
| `knn` argument | `refset` | `ref_set` |

Method names were left alone: `cchvae`, `proplace`, `cemsp`, `trex` and friends
are paper acronyms, not multi-word runs.

Parameters and locals throughout the codebase follow the same rename, so
`MethodObject.fit(train_set)` and `.predict(test_set)` now use snake_case.
Python does not bind parameter names for positional calls, so subclasses and
positional callers are unaffected.

### Install

Solver backends are now extras, one per package, plus `all-methods`:

| Extra | Package | Methods |
| --- | --- | --- |
| `gurobi` | gurobipy | `apas`, `proplace` |
| `z3` | z3-solver | `cemsp` |
| `smt` | PySMT | `mace` |
| `asp` | clingo | `arg_ensembling` |
| `cvx` | cvxpy | `cvas_proj` |
| `art` | adversarial-robustness-toolbox | `sns`, `trex` |
| `lime` | lime | `roar`, `larr` |

The base install drops from 17 to 10 dependencies and no longer pulls a
commercially licensed solver. `pip install recourse_bench[all-methods]` restores
every backend.

PyTorch is no longer pinned to a `+cu126` build, which previously made a CPU-only
install impossible. Installing the CPU build first takes the environment from
~4.5 GB to ~500 MB on Linux.

### Caching

Caches now default to the user cache directory — `$RECOURSE_BENCH_CACHE`, else
`$XDG_CACHE_HOME/recourse_bench`, else `~/.cache/recourse_bench` — instead of
`./cache/` in the working directory. A config's `caching.path` still wins.

### Fixed

- Component registration no longer depends on import order. `utils.registry`
  previously imported the five base classes on every `@register` call, which
  worked only because a solver's support module happened to import `model`
  early; with solvers optional that raised a circular-import `ImportError` at
  `import recourse_bench`.

## 0.1.2

- Namespace all packages under `recourse_bench.*`; ship the public API and the
  Sphinx documentation site.
