# Getting started

## Install

Install PyTorch for your machine first, then RecourseBench. Which torch build
you pick dominates the install size, so it is a deliberate step rather than
something this project decides for you.

```bash
conda create -n recoursebench python=3.12
conda activate recoursebench

# 1. PyTorch — pick ONE (skip on macOS/Windows, where PyPI wheels are CPU-only)
pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cpu     # ~500 MB
pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu126   # ~4.5 GB, needs an NVIDIA GPU

# 2. RecourseBench and the rest
pip install -r requirements.txt
# Optional: only needed for some methods (e.g. diverse_dist). Downgrades numpy.
pip install alibi==0.9.6
```

See <https://pytorch.org/get-started/locally/> for ROCm and other CUDA versions.
Use `--index-url`, not `--extra-index-url` — the latter leaves pip resolving
across both PyPI and the PyTorch index with no guarantee which one a given
package comes from.

(install-extras)=
### Solver extras

`requirements.txt` installs every solver backend, which is what you want for a
source checkout or to reproduce the full benchmark. Installing the package
itself keeps the solvers optional:

```bash
pip install recourse_bench                 # core, no solver backends
pip install recourse_bench[all-methods]    # + every solver backend
pip install recourse_bench[gurobi,lime]    # + just the backends you need
```

| Extra | Package | Methods |
| --- | --- | --- |
| `gurobi` | gurobipy (needs a Gurobi license) | `apas`, `proplace` |
| `z3` | z3-solver | `cemsp` |
| `smt` | PySMT | `mace` |
| `asp` | clingo | `arg_ensembling` |
| `cvx` | cvxpy | `cvas_proj` |
| `art` | adversarial-robustness-toolbox | `sns`, `trex` |
| `lime` | lime | `roar`, `larr` |

Every method stays registered and listed by `rb.list_methods()` whether or not
its backend is installed — constructing one without it raises
`MissingDependencyError` naming the extra to install.

**On Linux, skipping the torch step above gives you the CUDA build**: PyPI's
default `torch` wheel pulls ~3.1 GB of `nvidia-*` wheels, taking the install from
~500 MB to ~4.5 GB. If you do not have an NVIDIA GPU, that is ~4 GB you will
never use.

## Where the cache goes

Trained models and other derived artifacts are cached outside the working
directory, under (in order) `$RECOURSE_BENCH_CACHE`,
`$XDG_CACHE_HOME/recourse_bench`, or `~/.cache/recourse_bench`. Set
`caching.path` in a config to keep a run's cache alongside its outputs instead.

## Run an experiment from the CLI

After installation the `recourse-bench` entry point runs any experiment config:

```bash
recourse-bench -p experiment/toy/smoke_config.yaml
```

From a source checkout, `python experiments.py -p <config>` is equivalent.

## A minimal config

An experiment is fully described by a config dictionary (usually YAML). A single
top-level `seed` is propagated to every component that does not set its own.

```yaml
name: credit_linear_wachter_smoke
seed: 7                       # propagated to dataset/model/method/preprocess
logger:
  level: INFO
  path: ./logs/credit_linear_wachter_smoke.log
caching:
  path: ./cache/
dataset:
  name: credit
preprocess:
  - name: balance
    strategy: downsample
  - name: encode
    encoding: onehot
  - name: scale
    scaling: normalize
  - name: split
    split: 0.3
model:
  name: linear
  device: cpu
method:
  name: wachter
  device: cpu
  desired_class: 1
evaluation:
  - name: validity
  - name: distance
```

`finalize` is appended automatically if omitted, and must be the last
preprocessing step. `model.device` and `method.device` must match.

## The Python API

Import the package as `rb`. The quickest path is `rb.run(config)`:

```python
import yaml
from pathlib import Path
import recourse_bench as rb

config = yaml.safe_load(Path("experiment/toy/smoke_config.yaml").read_text())
metrics = rb.run(config)
print(metrics.to_string(index=False))
print(metrics.attrs["provenance"])     # library version, config hash, seed, ...
```

### Construct components by name

Each registered component is reachable through a named namespace; the attribute
*is* the class, so call it to construct an instance:

```python
model  = rb.models.linear(seed=7)
method = rb.methods.wachter(target_model=model, seed=7, desired_class=1)
data   = rb.datasets.credit()

rb.list_methods()      # discover what's available
```

See {doc}`reference/using` for the full list of namespaces.

### Full control with `Experiment`

When you also want the trained model, the counterfactuals, or provenance, use
the {class}`~experiments.Experiment` class:

```python
experiment = rb.Experiment(config)
metrics = experiment.run()

model = experiment.target_model()       # the trained classifier
cfs   = experiment.counterfactuals()    # the generated counterfactuals
train = experiment.train_set()
```

Configuration problems raise {class}`~utils.exceptions.ConfigError` (a subclass
of {class}`~utils.exceptions.RecourseBenchError`) instead of exiting the
process, so they can be caught programmatically.

## What `run()` does

`Experiment.run()` performs preprocessing, resolves the train/test datasets,
trains the target model, fits the recourse method, generates counterfactuals,
and concatenates evaluation outputs into one metrics dataframe. Failed
counterfactual rows are represented with `NaN` feature values and target `-1`.
